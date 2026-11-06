"""Shared solo-driving task. One native simulator per experiment process."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pystk2_gymnasium  # noqa: F401
from pystk2_gymnasium import AgentSpec

from .expert import Expert
from .metrics import DrivingMetrics, banana_contact, harmful_rows
from .native_telemetry import rescue_time_left

N_PATHS = 8
ACTION_KEYS = ("steer", "acceleration", "brake", "nitro", "fire")
ACTION_DIM = len(ACTION_KEYS)
SCHEMA_VERSION = 3


def scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def item_features(raw):
    # Match Team 4's hazard radar, road filtering, and forward-distance ordering.
    positions = np.asarray(raw["items_position"], dtype=np.float32).reshape(-1, 3)
    types = np.asarray(raw["items_type"], dtype=int).reshape(-1)
    x, z = positions[:, 0], positions[:, 2]
    relevant = (
        np.isin(types, [1, 4])
        & (np.abs(x) <= 2)
        & (z >= 0)
        & (z <= 17)
        & (
            np.abs(scalar(raw["center_path_distance"]) + x)
            <= scalar(raw["paths_width"][0]) / 2
        )
    )
    indices = np.flatnonzero(relevant)
    indices = indices[np.argsort(z[indices], kind="stable")[:2]]
    output = np.zeros((2, 3), dtype=np.float32)
    for slot, index in enumerate(indices):
        output[slot] = (
            np.hypot(x[index], z[index]) / np.sqrt(293),
            np.arctan2(x[index], z[index]) / (np.pi / 2),
            1,
        )
    return output.ravel()


def encode(raw):
    def paths(key, scale, width):
        values = np.asarray(raw[key], dtype=np.float32).reshape(-1, width)[:N_PATHS]
        out = np.zeros((N_PATHS, width), dtype=np.float32)
        out[: len(values)] = values / scale
        return out.ravel()

    features = np.concatenate(
        [
            np.asarray(raw["velocity"]).ravel() / 25,
            np.asarray(raw["center_path"]).ravel() / 20,
            np.asarray(raw["center_path_distance"]).ravel() / 10,
            np.asarray(raw["max_steer_angle"]).ravel(),
            np.asarray(raw["distance_down_track"]).ravel() / 1000,
            paths("paths_start", 50, 3),
            paths("paths_end", 50, 3),
            paths("paths_width", 10, 1),
            np.asarray(raw["energy"]).ravel() / 10,
            np.eye(11, dtype=np.float32)[int(raw["powerup_type"])],
            np.array([scalar(raw["powerup_count"]) / 5], np.float32),
            item_features(raw),
        ]
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite observation")
    return np.clip(features, -10, 10)


def normalize_action(action):
    return np.array(
        [
            scalar(action[key]) if key == "steer" else 2 * scalar(action[key]) - 1
            for key in ACTION_KEYS
        ],
        dtype=np.float32,
    )


def decode_action(action):
    a = np.clip(np.asarray(action), -1, 1)
    if a.shape != (ACTION_DIM,):
        raise ValueError(f"Expected {ACTION_DIM} controls, got {a.shape}")
    return {
        key: np.array([a[i] if key == "steer" else (a[i] + 1) / 2], np.float32)
        if key in ("steer", "acceleration")
        else int(a[i] > 0)
        for i, key in enumerate(ACTION_KEYS)
    } | {"drift": 0, "rescue": 0}


class DrivingEnv:
    def __init__(
        self,
        track="lighthouse",
        max_steps=1500,
        render=False,
        width=960,
        height=540,
    ):
        self.track = track
        self.max_steps = max_steps
        graphics = {}
        if render:
            import pystk2

            config = pystk2.GraphicsConfig.sd()
            config.display = False
            config.screen_width, config.screen_height = width, height
            graphics = {
                "render_mode": "human",
                "graphics_config": config,
                "num_cameras": 1,
            }
        self.env = gym.make(
            "supertuxkart/full-v0",
            track=track,
            num_kart=1,
            laps=1,
            max_paths=N_PATHS,
            use_subprocess=False,
            step_size=0.1,
            max_episode_steps=max_steps,
            agent=AgentSpec(name="study", rank_start=0, use_ai=False),
            **graphics,
        )
        self.expert = Expert()
        self.raw = None

    def reset(self, seed=None, track=None):
        if track is not None:
            self.track = track
            self.env.unwrapped.default_track = track
        self.raw, info = self.env.reset(seed=seed)
        self.steps = 0
        self.previous_distance = 0.0
        self.max_progress = 0.0
        self.start_time = self.env.unwrapped.world.time
        self.harmful_row_count = len(
            harmful_rows(self.env.unwrapped.world.items, self.env.unwrapped.track)
        )
        self.harmful_contacts = 0
        self.driving_metrics = DrivingMetrics()
        self.expert.reset(self.raw)
        self._expert_action = normalize_action(self.expert.act(self.raw))
        return encode(self.raw), info

    def expert_action(self):
        """Cached label: original expert advances exactly once per observation."""
        return self._expert_action.copy()

    def step(self, action):
        kart = self.env.unwrapped.world.karts[0]
        previous = (int(kart.attachment.type), float(kart.attachment.time_left))
        before_time = float(self.env.unwrapped.world.time)
        executed = np.asarray(action, dtype=np.float32).copy()
        self.raw, _, terminated, truncated, info = self.env.step(
            decode_action(executed)
        )
        kart = self.env.unwrapped.world.karts[0]
        current = (int(kart.attachment.type), float(kart.attachment.time_left))
        self.harmful_contacts += int(
            banana_contact(
                previous, current, float(self.env.unwrapped.world.time) - before_time
            )
        )
        rescue_timer = rescue_time_left()
        self.driving_metrics.update(
            before_time,
            float(self.env.unwrapped.world.time),
            bool(kart.is_on_road),
            rescue_timer,
            float(kart.finish_time) if terminated else None,
        )
        info.update(self.driving_metrics.snapshot())
        info.update(
            is_on_road=bool(kart.is_on_road), rescue_active=rescue_timer is not None
        )
        return self._record_step(terminated, truncated, info, executed.tolist())

    def _record_step(self, terminated, truncated, info, executed):
        self.steps += 1
        distance = float(info["distance"])
        # Signed progress: reversing and driving forward cannot repeatedly earn progress.
        progress_delta = distance - self.previous_distance
        self.previous_distance = distance
        self.max_progress = max(self.max_progress, distance)
        reward = progress_delta / 10 - 0.1 + 10 * int(terminated)
        info.update(
            success=bool(terminated),
            elapsed_seconds=float(self.env.unwrapped.world.time - self.start_time),
            finish_time=float(self.env.unwrapped.world.karts[0].finish_time)
            if terminated
            else None,
            lateral_error=scalar(self.raw["center_path_distance"]),
            speed=float(np.linalg.norm(self.raw["velocity"])),
            progress=distance,
            max_progress=self.max_progress,
            executed_action=executed,
            track_length_m=float(self.env.unwrapped.track.length),
            harmful_contacts=self.harmful_contacts,
            harmful_row_count=self.harmful_row_count,
            harmful_contact_proportion=self.harmful_contacts / self.harmful_row_count
            if self.harmful_row_count
            else None,
            world_position=list(self.env.unwrapped.world.karts[0].location),
        )
        if not (terminated or truncated):
            self._expert_action = normalize_action(self.expert.act(self.raw))
        return encode(self.raw), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.env.close()
