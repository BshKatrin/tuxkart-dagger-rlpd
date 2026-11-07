"""Driving evaluation and summaries with the training run as statistical unit."""

from __future__ import annotations

import copy
import json
import math
import multiprocessing as mp
import pickle
import tempfile
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .environment import SCHEMA_VERSION, DrivingEnv


def evaluate(
    env,
    actor,
    seeds,
    device: str = "cpu",
    trajectory_path: str | Path | None = None,
    *,
    workers: int = 1,
    worker_timeout: float = 120.0,
    env_factory=None,
) -> list[dict[str, Any]]:
    """Evaluate deterministic actions on fixed seeds, without updating the actor.

    ``actor`` can be a torch actor exposing ``act(batch)`` or a callable accepting
    one numpy observation. Failed laps have ``lap_time=None``. The environment
    is reused sequentially and left at its last evaluated terminal state; callers
    must reset before resuming training.

    ``workers=2..4`` runs contiguous seed partitions in spawned processes, each
    with its own headless simulator and private frozen CPU actor. The parent
    environment, actor and RNG states are untouched. Records and trajectory
    episode indices retain input seed order. ``worker_timeout`` sets a shared
    deadline including startup; child teardown is separately bounded. A
    spawn-picklable ``env_factory`` can supply a custom environment; otherwise the headless DrivingEnv settings
    are copied. Parallel evaluation requires a deterministic torch actor with
    ``act`` and CPU inference. Initialize shared simulator assets before use.
    """
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Evaluation workers must be an integer from 1 to 4")
    if workers == 1:
        return _evaluate_serial(env, actor, seeds, device, trajectory_path)
    return _evaluate_parallel(
        env, actor, seeds, device, trajectory_path, workers, worker_timeout, env_factory
    )


def _evaluate_serial(env, actor, seeds, device="cpu", trajectory_path=None):
    records: list[dict[str, Any]] = []
    traces: dict[str, list] = defaultdict(list)
    was_training = actor.training if isinstance(actor, torch.nn.Module) else None
    if isinstance(actor, torch.nn.Module):
        actor.eval()
    try:
        with torch.inference_mode():
            for episode_index, seed in enumerate(seeds):
                obs, _ = env.reset(seed=int(seed))
                total_reward, steps, lateral_sum = 0.0, 0, 0.0
                max_progress, max_distance = 0.0, 0.0
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
                            "off_track_fraction",
                            "rescue_count",
                        ):
                            traces[key].append(float(info.get(key, float("nan"))))
                        traces["terminated"].append(bool(terminated))
                        traces["is_on_road"].append(bool(info.get("is_on_road", True)))
                        traces["rescue_active"].append(
                            bool(info.get("rescue_active", False))
                        )
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
                        "average_speed_m_s": float(
                            info["track_length_m"]
                            / (info.get("finish_time") or sim_seconds)
                        )
                        if success and "track_length_m" in info
                        else None,
                        "harmful_contact_proportion": info.get(
                            "harmful_contact_proportion"
                        ),
                        **{
                            key: info.get(key)
                            for key in (
                                "measurement_seconds",
                                "off_track_seconds",
                                "off_track_fraction",
                                "rescue_count",
                            )
                        },
                    }
                )
    finally:
        if isinstance(actor, torch.nn.Module):
            actor.train(bool(was_training))
    if trajectory_path is not None:
        path = Path(trajectory_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, **{key: np.asarray(value) for key, value in traces.items()}
        )
    return records


def _evaluation_worker(connection, factory, actor_payload, seeds, trajectory_path):
    try:
        torch.set_num_threads(1)
        # Ordinary bytes avoid Torch's shared-memory/socket transport on macOS.
        policy = pickle.loads(actor_payload).eval()
        env = factory()
        try:
            records = _evaluate_serial(env, policy, seeds, "cpu", trajectory_path)
        finally:
            env.close()
        connection.send(("ok", records))
    except Exception as exc:  # noqa: BLE001 — transport worker failures to the parent
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except (BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


def _stop_evaluation_worker(process):
    process.join(timeout=0.5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
    if process.is_alive():
        process.kill()
        process.join(timeout=0.5)
    if not process.is_alive():
        process.close()
        return True
    return False


def _evaluate_parallel(
    env, actor, seeds, device, trajectory_path, workers, worker_timeout, env_factory
):
    if device != "cpu":
        raise ValueError("Parallel evaluation requires CPU inference")
    if not isinstance(actor, torch.nn.Module) or not callable(
        getattr(actor, "act", None)
    ):
        raise TypeError("Parallel evaluation requires a torch actor exposing act")
    if (
        isinstance(worker_timeout, bool)
        or not math.isfinite(worker_timeout)
        or worker_timeout <= 0
    ):
        raise ValueError("Evaluation worker timeout must be finite and positive")
    if mp.parent_process() is not None:
        raise RuntimeError("Nested parallel evaluation is unsupported")
    if env_factory is None:
        if type(env) is not DrivingEnv or getattr(
            env.env.unwrapped, "render_mode", None
        ):
            raise ValueError(
                "Parallel evaluation requires a headless DrivingEnv or env_factory"
            )
        env_factory = partial(DrivingEnv, track=env.track, max_steps=env.max_steps)
    if not callable(env_factory):
        raise TypeError("env_factory must be a spawn-picklable callable")
    try:
        pickle.dumps(env_factory)
    except (AttributeError, TypeError, pickle.PicklingError) as exc:
        raise ValueError(
            "env_factory must be a top-level spawn-picklable callable"
        ) from exc
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        if trajectory_path is not None:
            path = Path(trajectory_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path)
        return []
    try:
        actor_payload = pickle.dumps(
            copy.deepcopy(actor).cpu().eval(), protocol=pickle.HIGHEST_PROTOCOL
        )
    except (AttributeError, TypeError, pickle.PicklingError) as exc:
        raise ValueError(
            "Parallel actor must be a top-level pickle-compatible module"
        ) from exc
    count = min(workers, len(seeds))
    partitions = [
        (len(seeds) * i // count, len(seeds) * (i + 1) // count) for i in range(count)
    ]
    context = mp.get_context("spawn")
    processes, connections = [], []
    results = {}
    scratch = tempfile.TemporaryDirectory(prefix="kartstudy-evaluation-", delete=False)
    temporary = Path(scratch.name)
    reaped = False
    try:
        try:
            deadline = time.monotonic() + worker_timeout
            for index, (start, end) in enumerate(partitions):
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Evaluation worker {index} timed out after {worker_timeout:g}s"
                    )
                parent, child = context.Pipe(duplex=False)
                connections.append(parent)
                part = (
                    temporary / f"part-{index}.npz"
                    if trajectory_path is not None
                    else None
                )
                process = context.Process(
                    target=_evaluation_worker,
                    args=(child, env_factory, actor_payload, seeds[start:end], part),
                    daemon=True,
                )
                try:
                    process.start()
                except Exception:
                    if process.pid is None:
                        process.close()
                    else:
                        processes.append(process)
                    raise
                finally:
                    child.close()
                processes.append(process)
            pending = set(range(count))
            while pending:
                for index in tuple(pending):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"Evaluation worker {index} timed out after {worker_timeout:g}s"
                        )
                    connection, process = connections[index], processes[index]
                    if connection.poll():
                        try:
                            kind, payload = connection.recv()
                        except (EOFError, OSError) as exc:
                            raise RuntimeError(
                                f"Evaluation worker {index} exited without a result"
                            ) from exc
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                f"Evaluation worker {index} timed out after {worker_timeout:g}s"
                            )
                        if kind != "ok":
                            raise RuntimeError(
                                f"Evaluation worker {index} failed: {payload}"
                            )
                        results[index] = payload
                        pending.remove(index)
                    elif not process.is_alive():
                        raise RuntimeError(
                            f"Evaluation worker {index} exited with code {process.exitcode}"
                        )
                if pending:
                    time.sleep(0.01)
        finally:
            for connection in connections:
                connection.close()
            reaped = True
            for process in processes:
                try:
                    if not _stop_evaluation_worker(process):
                        reaped = False
                except (OSError, ValueError, AssertionError):
                    reaped = False
            if not reaped:
                raise RuntimeError(
                    f"Could not reap evaluation workers; scratch retained at {temporary}"
                )
        if trajectory_path is not None:
            traces = defaultdict(list)
            for index, (start, _) in enumerate(partitions):
                with np.load(
                    temporary / f"part-{index}.npz", allow_pickle=False
                ) as part:
                    for key in part.files:
                        values = part[key]
                        traces[key].append(
                            values + start if key == "episode" else values
                        )
            path = Path(trajectory_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path, **{key: np.concatenate(values) for key, values in traces.items()}
            )
    finally:
        if reaped:
            scratch.cleanup()
    return [record for index in range(count) for record in results[index]]


def _mean_available(rows, metric):
    values = [row[metric] for row in rows if row.get(metric) is not None]
    return float(np.mean(values)) if values else None


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

    Evaluation rows must have ``step`` and ``episodes`` fields.
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
        if config.get("schema_version") != SCHEMA_VERSION or "seed" not in config:
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
            partitions = defaultdict(list)
            for episode in row["episodes"]:
                if "track" not in episode:
                    partitions["normal"].append(episode)
                else:
                    partitions[f"track/{episode['track']}"].append(episode)
                    partitions[f"{episode['split']}/{episode['difficulty']}"].append(
                        episode
                    )
                    partitions[f"{episode['split']}/all"].append(episode)
            for condition, episodes in partitions.items():
                key = signature, condition, int(config["seed"])
                if key not in latest or row["step"] >= latest[key]["step"]:
                    latest[key] = {
                        **row,
                        "episodes": episodes,
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
                    "average_speed_m_s": _mean_available(eps, "average_speed_m_s"),
                    "off_track_fraction": _mean_available(eps, "off_track_fraction"),
                    "rescue_count": _mean_available(eps, "rescue_count"),
                    "harmful_contact_proportion": _mean_available(
                        eps, "harmful_contact_proportion"
                    ),
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
            "average_speed_m_s",
            "off_track_fraction",
            "rescue_count",
            "harmful_contact_proportion",
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
