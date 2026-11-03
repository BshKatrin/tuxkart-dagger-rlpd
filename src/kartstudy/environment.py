"""Shared solo-driving task. One native simulator per experiment process."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pystk2_gymnasium  # noqa: F401
from pystk2_gymnasium import AgentSpec

from .expert import Expert

N_PATHS = 8
ACTION_DIM = 3


def scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


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
        ]
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite observation")
    return np.clip(features, -10, 10)


def normalize_action(action):
    return np.array(
        [
            scalar(action["steer"]),
            2 * scalar(action["acceleration"]) - 1,
            2 * int(action["brake"]) - 1,
        ],
        dtype=np.float32,
    )


def decode_action(action):
    a = np.clip(np.asarray(action), -1, 1)
    return {
        "steer": np.array([a[0]], np.float32),
        "acceleration": np.array([(a[1] + 1) / 2], np.float32),
        "brake": int(a[2] > 0),
        "drift": 0,
        "nitro": 0,
        "rescue": 0,
        "fire": 0,
    }


class DrivingEnv:
    def __init__(self, track="lighthouse", max_steps=1500, perturb=False):
        self.track = track
        self.max_steps = max_steps
        self.perturb = perturb
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
        )
        self.experts = {False: Expert(), True: Expert(conservative=True)}
        self.raw = None

    def reset(self, seed=None):
        self.raw, info = self.env.reset(seed=seed)
        self.steps = 0
        self.previous_distance = 0.0
        self.max_progress = 0.0
        rng = np.random.default_rng(seed)
        self.perturb_distance = float(rng.uniform(250, 450))
        self.perturb_remaining = -1
        self.perturb_steer = float(rng.choice([-0.8, 0.8]))
        self.start_time = self.env.unwrapped.world.time
        return encode(self.raw), info

    def expert_action(self, conservative=False):
        return normalize_action(self.experts[bool(conservative)].act(self.raw))

    def step(self, action):
        executed = np.asarray(action, dtype=np.float32).copy()
        if (
            self.perturb
            and self.perturb_remaining == -1
            and self.previous_distance >= self.perturb_distance
        ):
            self.perturb_remaining = 5
        overridden = bool(self.perturb and self.perturb_remaining > 0)
        if overridden:
            executed[0] = self.perturb_steer
            self.perturb_remaining -= 1
        self.raw, _, terminated, truncated, info = self.env.step(
            decode_action(executed)
        )
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
            action_overridden=overridden,
            executed_action=executed.tolist(),
            world_position=list(self.env.unwrapped.world.karts[0].location),
        )
        return encode(self.raw), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.env.close()
