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

from .environment import SCHEMA_VERSION, DrivingEnv
from .expert import Expert
from .tracks import TRAIN_TRACKS


def training_mask(data, metadata, episodes=None):
    """Select explicit training IDs per track; validation IDs are never eligible."""
    ids = metadata["train_episode_ids"]
    if episodes is not None:
        ids = [
            episode
            for track in metadata["tracks"]
            for episode in [
                row["episode"]
                for row in metadata["episodes"]
                if row["track"] == track and row["episode"] in ids
            ][:episodes]
        ]
    return np.isin(data["episode"], ids)


def load_transitions(path, episodes=None):
    metadata = json.loads(Path(path).with_suffix(".json").read_text())
    with np.load(path) as data:
        mask = training_mask(data, metadata, episodes)
        return Transitions(
            torch.from_numpy(data["obs"][mask]),
            Action(torch.from_numpy(data["action"][mask])),
            torch.from_numpy(data["reward"][mask]),
            torch.from_numpy(data["next_obs"][mask]),
            torch.from_numpy(data["terminated"][mask]),
        )


def dataset_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_dataset(path, tracks, episodes, max_steps):
    metadata = json.loads(Path(path).with_suffix(".json").read_text())
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("tracks") != list(tracks)
        or metadata.get("expert") != Expert.name
        or metadata.get("source_commit") != Expert.source_commit
        or metadata.get("control_noise") is not False
        or metadata.get("max_steps") != max_steps
        or metadata.get("sha256") != dataset_hash(path)
    ):
        raise ValueError("Dataset schema, tracks, expert, task or checksum mismatch")
    train_ids, val_ids = map(
        set, (metadata["train_episode_ids"], metadata["validation_episode_ids"])
    )
    if train_ids & val_ids or not train_ids or not val_ids:
        raise ValueError("Training and validation episodes must be distinct")
    summaries = metadata["episodes"]
    if not all(row["success"] for row in summaries):
        raise ValueError("Expert failed a demonstration lap")
    for track in tracks:
        count = sum(
            row["track"] == track and row["episode"] in train_ids for row in summaries
        )
        if not 1 <= episodes <= count:
            raise ValueError("Demonstration budget overlaps validation or is empty")
    with np.load(path) as data:
        if (
            data["obs"].shape[1:] != (84,)
            or data["action"].shape[1:] != (5,)
            or not np.array_equal(data["action"], data["expert_action"])
            or set(data["episode"].tolist()) != train_ids | val_ids
        ):
            raise ValueError(
                "Invalid observation/action schema or demonstration labels"
            )
    return metadata


def collect(path, tracks=TRAIN_TRACKS, episodes=22, max_steps=1500):
    if episodes < 3:
        raise ValueError("Need training episodes and two validation episodes")
    tracks = tuple(tracks)
    if not tracks or len(set(tracks)) != len(tracks):
        raise ValueError("Require distinct training tracks")
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    env = DrivingEnv(track=tracks[0], max_steps=max_steps)
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
        for track_index, track in enumerate(tracks):
            for local_episode in range(episodes):
                ep = track_index * episodes + local_episode
                obs, _ = env.reset(seed=1000 + local_episode, track=track)
                ret = 0.0
                for step in range(max_steps):
                    label = env.expert_action()
                    action = label.copy()
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
                    "seed": 1000 + local_episode,
                    "track": track,
                    "success": term,
                    "steps": step + 1,
                    "return_": ret,
                    "sim_seconds": info["elapsed_seconds"],
                }
                if not term:
                    raise RuntimeError(
                        f"Expert failed: {track}, seed {1000 + local_episode}"
                    )
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
            "tracks": list(tracks),
            "episodes": summaries,
            "schema_version": SCHEMA_VERSION,
            "expert": Expert.name,
            "source_commit": Expert.source_commit,
            "transitions": len(arrays["obs"]),
            "wall_seconds": time.perf_counter() - start,
            "sha256": dataset_hash(path),
            "max_steps": max_steps,
            "control_noise": False,
            "train_episode_ids": [
                i * episodes + ep
                for i in range(len(tracks))
                for ep in range(episodes - 2)
            ],
            "validation_episode_ids": [
                i * episodes + ep
                for i in range(len(tracks))
                for ep in range(episodes - 2, episodes)
            ],
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tracks", nargs="+", default=list(TRAIN_TRACKS))
    parser.add_argument("--episodes", type=int, default=22)
    args = parser.parse_args()
    collect(args.output, args.tracks, args.episodes)
