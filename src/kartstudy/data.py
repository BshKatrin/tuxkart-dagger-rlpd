"""Collect complete, seeded expert trajectories with immutable provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from rl_mind.core import Action
from rl_mind.data import Transitions

from .environment import DrivingEnv
from .expert import Expert


def load_transitions(path, episodes=None):
    with np.load(path) as data:
        mask = np.ones(len(data["obs"]), dtype=bool)
        if episodes is not None:
            mask = data["episode"] < episodes
        return Transitions(
            torch.from_numpy(data["obs"][mask]),
            Action(torch.from_numpy(data["action"][mask])),
            torch.from_numpy(data["reward"][mask]),
            torch.from_numpy(data["next_obs"][mask]),
            torch.from_numpy(data["terminated"][mask]),
        )


def dataset_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(path, track="lighthouse", episodes=20, conservative=False, max_steps=1500):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    env = DrivingEnv(track=track, max_steps=max_steps)
    rows = {
        k: []
        for k in [
            "obs",
            "action",
            "reward",
            "next_obs",
            "terminated",
            "truncated",
            "episode",
            "expert_action",
        ]
    }
    summaries = []
    start = time.perf_counter()
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=1000 + ep)
            rng = np.random.default_rng(1000 + ep)
            ret = 0.0
            for step in range(max_steps):
                label = env.expert_action(conservative)
                action = label.copy()
                action[:2] = np.clip(action[:2] + rng.normal(0, [0.035, 0.02]), -1, 1)
                nxt, reward, term, trunc, info = env.step(action)
                values = [obs, action, reward, nxt, term, trunc, ep, label]
                for key, value in zip(rows, values):
                    rows[key].append(value)
                ret += reward
                obs = nxt
                if term or trunc:
                    break
            summary = {
                "episode": ep,
                "seed": 1000 + ep,
                "success": term,
                "steps": step + 1,
                "return_": ret,
                "sim_seconds": info["elapsed_seconds"],
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        if not all(ep["success"] for ep in summaries):
            raise RuntimeError(
                "Expert did not finish all qualification/data laps; inspect before training"
            )
        arrays = {
            k: np.asarray(
                v,
                dtype=np.float32
                if k in ["obs", "action", "reward", "next_obs", "expert_action"]
                else bool
                if k in ["terminated", "truncated"]
                else np.int32,
            )
            for k, v in rows.items()
        }
        np.savez_compressed(path, **arrays)
        metadata = {
            "track": track,
            "episodes": summaries,
            "conservative": conservative,
            "expert": Expert.name,
            "source_commit": Expert.source_commit,
            "transitions": len(arrays["obs"]),
            "wall_seconds": time.perf_counter() - start,
            "sha256": dataset_hash(path),
            "max_steps": max_steps,
            "control_noise_std": [0.035, 0.02, 0.0],
            "train_episode_ids": list(range(episodes - 2)),
            "validation_episode_ids": list(range(episodes - 2, episodes)),
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--track", default="lighthouse")
    parser.add_argument("--episodes", type=int, default=22)
    parser.add_argument("--conservative", action="store_true")
    args = parser.parse_args()
    collect(args.output, args.track, args.episodes, args.conservative)
