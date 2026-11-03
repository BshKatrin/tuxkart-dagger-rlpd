"""Driving evaluation and summaries with the training run as statistical unit."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def evaluate(
    env,
    actor,
    seeds,
    device: str = "cpu",
    perturb: bool = False,
    trajectory_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate deterministic actions on fixed seeds, without updating the actor.

    ``actor`` can be a torch actor exposing ``act(batch)`` or a callable accepting
    one numpy observation. Failed laps have ``lap_time=None``. The environment
    is reused sequentially and left at its last evaluated terminal state; callers
    must reset before resuming training. Evaluation perturbations belong to the
    environment and are restored after evaluation, including on exceptions.
    """
    records: list[dict[str, Any]] = []
    traces: dict[str, list] = defaultdict(list)
    prior_perturb = env.perturb
    was_training = actor.training if isinstance(actor, torch.nn.Module) else None
    if isinstance(actor, torch.nn.Module):
        actor.eval()
    env.perturb = perturb
    try:
        with torch.inference_mode():
            for episode_index, seed in enumerate(seeds):
                obs, _ = env.reset(seed=int(seed))
                total_reward, steps, lateral_sum = 0.0, 0, 0.0
                max_progress, max_distance = 0.0, 0.0
                overridden_steps = 0
                terminated = truncated = False
                info = {}
                while not (terminated or truncated):
                    if hasattr(actor, "act"):
                        batch = torch.as_tensor(
                            obs, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        action = actor.act(batch).squeeze(0).detach().cpu().numpy()
                    else:
                        action = np.asarray(actor(obs), dtype=np.float32)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    steps += 1
                    lateral = abs(float(info.get("lateral_error", 0.0)))
                    lateral_sum += lateral
                    max_progress = max(max_progress, float(info.get("progress", 0.0)))
                    max_distance = max(max_distance, float(info.get("distance", 0.0)))
                    overridden_steps += int(info.get("action_overridden", False))
                    if trajectory_path is not None:
                        traces["obs"].append(np.asarray(obs, dtype=np.float32))
                        traces["next_obs"].append(
                            np.asarray(next_obs, dtype=np.float32)
                        )
                        traces["commanded_action"].append(
                            np.asarray(action, dtype=np.float32)
                        )
                        traces["executed_action"].append(
                            np.asarray(
                                info.get("executed_action", action), dtype=np.float32
                            )
                        )
                        traces["world_position"].append(
                            np.asarray(
                                info.get("world_position", [np.nan] * 3),
                                dtype=np.float32,
                            )
                        )
                        traces["episode"].append(episode_index)
                        traces["seed"].append(int(seed))
                        traces["step"].append(steps)
                        traces["reward"].append(float(reward))
                        for key in (
                            "speed",
                            "lateral_error",
                            "progress",
                            "distance",
                            "elapsed_seconds",
                        ):
                            traces[key].append(float(info.get(key, float("nan"))))
                        traces["action_overridden"].append(
                            bool(info.get("action_overridden", False))
                        )
                        traces["terminated"].append(bool(terminated))
                        traces["truncated"].append(bool(truncated))
                    obs = next_obs
                success = bool(terminated and info.get("success", True))
                sim_seconds = info.get("elapsed_seconds")
                if sim_seconds is not None:
                    sim_seconds = float(sim_seconds)
                records.append(
                    {
                        "seed": int(seed),
                        "return": total_reward,
                        "success": success,
                        "lap_time": float(info.get("finish_time") or sim_seconds)
                        if success
                        else None,
                        "timeout": bool(truncated and not success),
                        "sim_seconds": sim_seconds,
                        "steps": steps,
                        "max_progress": max_progress,
                        "max_distance": max_distance,
                        "mean_abs_lateral_error": lateral_sum / max(1, steps),
                        "overridden_steps": overridden_steps,
                    }
                )
    finally:
        env.perturb = prior_perturb
        if isinstance(actor, torch.nn.Module):
            actor.train(bool(was_training))
    if trajectory_path is not None:
        path = Path(trajectory_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, **{key: np.asarray(value) for key, value in traces.items()}
        )
    return records


def _estimate(values: list[float], rng, bootstrap_samples: int) -> dict:
    values_array = np.asarray(values, dtype=float)
    finite = values_array[np.isfinite(values_array)]
    if len(finite) == 0:
        return {"mean": None, "ci95": None, "n_training_seeds": 0}
    interval = None
    if len(finite) >= 2:
        samples = rng.choice(
            finite, size=(bootstrap_samples, len(finite)), replace=True
        ).mean(axis=1)
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
    return {
        "mean": float(finite.mean()),
        "ci95": interval,
        "n_training_seeds": len(finite),
    }


def summarize_runs(
    root: str | Path,
    output: str | Path | None = None,
    bootstrap_samples: int = 2000,
) -> list[dict]:
    """Summarize each run's last checkpoint, bootstrapping *training seeds*.

    Evaluation rows must have ``step``, ``condition``, and ``episodes`` fields.
    Each run has a sibling ``config.json`` including ``seed``. All other config
    fields define experimental conditions, except output/runtime metadata below.
    Duplicate directories for the same config and training seed count once.
    Lap-time means are conditional on success and explicitly report the number
    of training seeds with any successful evaluation. No uncertainty interval is
    produced for a single training seed.
    """
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    excluded = {
        "seed",
        "device",
        "output",
        "out",
        "run_dir",
        "started_at",
        "created_at",
        "pid",
    }
    latest: dict[tuple[str, str, int], dict] = {}
    for path in sorted(Path(root).rglob("evaluations.jsonl")):
        config_path = path.parent / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        if "seed" not in config:
            continue
        comparable = {
            key: value for key, value in config.items() if key not in excluded
        }
        signature = json.dumps(comparable, sort_keys=True)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # An actively written final line can be incomplete.
            if not row.get("episodes"):
                continue
            condition = row.get("condition", "normal")
            key = signature, condition, int(config["seed"])
            if key not in latest or row["step"] >= latest[key]["step"]:
                latest[key] = {
                    **row,
                    "config": comparable,
                    "training_seed": int(config["seed"]),
                    "path": str(path.parent),
                }
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (signature, condition, _), row in latest.items():
        groups[(signature, condition)].append(row)
    rng = np.random.default_rng(2026)
    summaries = []
    for (_, condition), rows in sorted(groups.items()):
        per_seed = []
        for row in rows:
            eps = row["episodes"]
            lap_times = [
                e["lap_time"]
                for e in eps
                if e["success"] and e.get("lap_time") is not None
            ]
            per_seed.append(
                {
                    "seed": row["training_seed"],
                    "step": row["step"],
                    "episodes": len(eps),
                    "completion_rate": float(np.mean([e["success"] for e in eps])),
                    "conditional_lap_time": float(np.mean(lap_times))
                    if lap_times
                    else None,
                    "return": float(np.mean([e["return"] for e in eps])),
                    "mean_abs_lateral_error": float(
                        np.mean([e["mean_abs_lateral_error"] for e in eps])
                    ),
                    "run_dir": row["path"],
                }
            )
        evaluated_steps = [row["step"] for row in per_seed]
        summary = {
            "config": rows[0]["config"],
            "condition": condition,
            "per_seed": per_seed,
            "checkpoint_step_range": [min(evaluated_steps), max(evaluated_steps)],
            "same_checkpoint_budget": len(set(evaluated_steps)) == 1,
        }
        for metric in (
            "completion_rate",
            "conditional_lap_time",
            "return",
            "mean_abs_lateral_error",
        ):
            values = [row[metric] for row in per_seed if row[metric] is not None]
            summary[metric] = _estimate(values, rng, bootstrap_samples)
        summaries.append(summary)
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries
