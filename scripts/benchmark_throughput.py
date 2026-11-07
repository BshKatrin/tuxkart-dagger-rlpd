"""Compare learner implementations on frozen training batches, without a simulator."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import heapq
import importlib.util
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import statistics
import sys
import time
from pathlib import Path

METHODS = {"sac": (2, False, 1), "sac_sym": (2, False, 1), "rlpd": (10, True, 4)}
CHUNK_ACTIONS = 32


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def variant_settings(device, optimization):
    return {
        "target_update": {
            "baseline": "loop",
            "optimized": "loop" if optimization == "metrics" else "grouped-foreach",
        },
        "metrics_reporting": {
            "baseline": "each-update",
            "optimized": "32-action-chunk-last"
            if optimization in ("metrics", "both")
            else "each-update",
        },
        "production_default_target_update": "grouped-foreach"
        if device == "cuda"
        else "loop",
    }


def configure_variant(optimized, baseline, device, optimization):
    # Every worker has freshly imported modules; the archived baseline stays untouched.
    if device == "cuda":
        if optimization == "metrics":
            optimized.foreach_soft_update = baseline.soft_update
    elif optimization in ("foreach", "both"):
        optimized.soft_update = optimized.foreach_soft_update
    return variant_settings(device, optimization)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline", type=Path)
    result.add_argument("--batches", type=Path)
    result.add_argument("--device", choices=("mps", "cuda"))
    result.add_argument("--method", choices=tuple(METHODS))
    result.add_argument(
        "--optimization", choices=("metrics", "foreach", "both"), default="metrics"
    )
    result.add_argument("--workers", type=int, choices=(1, 2), default=1)
    result.add_argument("--warmup-actions", type=int, default=32)
    result.add_argument("--block-actions", type=int, default=32)
    result.add_argument("--repeats", type=int)
    result.add_argument(
        "--profile",
        action="store_true",
        help="Export five post-measurement updates from one worker",
    )
    result.add_argument(
        "--queue-replay",
        type=Path,
        help="Estimate two-slot scheduling from six measured tiny-job durations",
    )
    result.add_argument("--timeout", type=float, default=300)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--background", type=Path, nargs="*", default=[])
    result.add_argument("--label", required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def validate(args):
    if any(
        value is None
        for value in (args.baseline, args.batches, args.device, args.method)
    ):
        raise ValueError(
            "A learner benchmark requires baseline, batches, device and method"
        )
    minimum = 5 if args.method == "rlpd" else 10
    args.repeats = minimum if args.repeats is None else args.repeats
    if any(
        not 32 <= value <= 1024 or value % 32
        for value in (args.warmup_actions, args.block_actions)
    ):
        raise ValueError(
            "Warmup and block actions must be multiples of 32, between 32 and 1024"
        )
    if not minimum <= args.repeats <= 16 or not 5 <= args.timeout <= 1800:
        raise ValueError(
            f"Use {minimum}–16 repeats and a 5–1800 second overall timeout"
        )
    if args.profile and args.workers != 1:
        raise ValueError("Profiling requires exactly one additional fixture worker")
    if not 0 <= args.seed < 2**32 or not args.label.strip():
        raise ValueError("Use a 32-bit nonnegative seed and a nonempty window label")
    provenance = json.loads(args.batches.with_suffix(".json").read_text())
    if (
        provenance.get("schema_version") != 1
        or provenance.get("phase") != "cached-training-update"
        or provenance.get("method") != args.method
        or provenance.get("sha256") != digest(args.batches)
        or not provenance.get("sources")
    ):
        raise ValueError(
            "Frozen batch provenance must identify method, phase, sources and checksum"
        )
    digest(args.baseline)
    return provenance


def tail_json(path):
    try:
        with path.open("rb") as source:
            if path.suffix == ".json":
                content = source.read(65537)
                return json.loads(content) if len(content) <= 65536 else None
            source.seek(max(0, path.stat().st_size - 65536))
            rows = source.read().splitlines()
        return json.loads(rows[-1]) if rows else None
    except (OSError, ValueError):
        return None


def background_state(manifests):
    snapshots = []
    for path in manifests:
        path = Path(path)
        row = {"manifest": str(path)}
        try:
            manifest = json.loads(path.read_text())
            row.update(pid=manifest.get("pid"), state=manifest.get("state"), jobs=[])
            for job in manifest.get("jobs", []):
                if job.get("state") != "running":
                    continue
                output = Path(job["output"])
                if not output.exists():
                    output = path.parent / job["name"]
                status = output / "status.json"
                evaluation = output / "evaluations.jsonl"
                last_evaluation = tail_json(evaluation)
                row["jobs"].append(
                    {
                        "name": job["name"],
                        "pid": job.get("pid"),
                        "status": tail_json(status),
                        "status_age_seconds": time.time() - status.stat().st_mtime
                        if status.exists()
                        else None,
                        "last_evaluation": {
                            key: last_evaluation.get(key)
                            for key in ("step", "updates", "final", "wall_seconds")
                        }
                        if last_evaluation
                        else None,
                        "evaluation_log_bytes": evaluation.stat().st_size
                        if evaluation.exists()
                        else None,
                    }
                )
        except (OSError, ValueError, KeyError):
            row["unavailable"] = True
        snapshots.append(row)
    return snapshots


def validate_arrays(arrays):
    import numpy as np

    required = {"obs", "actions", "reward", "next_obs", "terminated"}
    if not required <= set(arrays):
        raise ValueError(f"Missing frozen batch arrays: {required - set(arrays)}")
    obs = arrays["obs"]
    if (
        obs.ndim != 3
        or not 1 <= obs.shape[0] <= 256
        or obs.shape[1] != 256
        or not 1 <= obs.shape[2] <= 512
    ):
        raise ValueError("obs must have shape (1–256 batches,256,1–512 features)")
    expected = {
        "next_obs": obs.shape,
        "actions": (obs.shape[0], 256, 5),
        "reward": obs.shape[:2],
        "terminated": obs.shape[:2],
    }
    if any(arrays[key].shape != shape for key, shape in expected.items()):
        raise ValueError("Frozen training batch shapes disagree")
    if arrays["terminated"].dtype != np.bool_:
        raise ValueError("terminated must contain booleans")
    for key in required - {"terminated"}:
        if arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} must contain finite float32 values")
    if np.abs(arrays["actions"]).max() > 1:
        raise ValueError(
            "Cached actions must be normalized training replay actions in [-1,1]"
        )


def snapshot_digest(state):
    """Hash full tensor contents and typed containers without pickle identities."""
    import torch

    fingerprint = hashlib.sha256()

    def visit(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise FloatingPointError("Nonfinite tensor in final benchmark state")
            fingerprint.update(
                json.dumps(["tensor", str(tensor.dtype), list(tensor.shape)]).encode()
            )
            fingerprint.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            fingerprint.update(f"dict:{len(value)}".encode())
            for key in sorted(
                value, key=lambda item: (type(item).__name__, repr(item))
            ):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            fingerprint.update(f"{type(value).__name__}:{len(value)}".encode())
            for item in value:
                visit(item)
        else:
            fingerprint.update(
                json.dumps([type(value).__name__, value], allow_nan=False).encode()
            )

    visit(state)
    return fingerprint.hexdigest()


def qualify_learners(learners, rng_states):
    fingerprints = {}
    for name, learner in learners.items():
        fingerprints[name] = snapshot_digest(
            {
                "checkpoint": learner.checkpoint(),
                "gradients": {
                    key: parameter.grad for key, parameter in learner.named_parameters()
                },
                "rng": rng_states[name],
            }
        )
    if fingerprints["baseline"] != fingerprints["optimized"]:
        raise RuntimeError(
            f"Final checkpoint/optimizer/gradient/RNG qualification differs exactly: {fingerprints}"
        )
    return {
        "exact": True,
        "snapshot_sha256": fingerprints["baseline"],
        "compared": "full checkpoint, optimizer state, gradients, Python and Torch CPU/device RNG",
    }


def worker(config, index, messages, barrier):
    os.setsid()
    try:
        import random

        import numpy as np
        import torch
        from rl_mind.core import Action
        from rl_mind.data import Transitions

        from kartstudy import algorithms as optimized

        torch.set_num_threads(1)
        if config["device"] == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; no CPU fallback")
            synchronize = torch.cuda.synchronize
        else:
            if not torch.backends.mps.is_available():
                raise RuntimeError("Metal unavailable; no CPU fallback")
            synchronize = torch.mps.synchronize
        spec = importlib.util.spec_from_file_location(
            "benchmark_baseline", config["baseline"]
        )
        if digest(config["baseline"]) != config["baseline_sha256"]:
            raise RuntimeError("Baseline source changed after benchmark launch")
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        effective_settings = configure_variant(
            optimized, baseline, config["device"], config["optimization"]
        )
        if digest(optimized.__file__) != config["optimized_sha256"]:
            raise RuntimeError("Optimized source changed after benchmark launch")
        if digest(config["batches"]) != config["batches_sha256"]:
            raise RuntimeError("Frozen input batches changed after benchmark launch")
        with np.load(config["batches"], allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        validate_arrays(arrays)
        batches = [
            Transitions(
                torch.from_numpy(arrays["obs"][i]),
                Action(torch.from_numpy(arrays["actions"][i])),
                torch.from_numpy(arrays["reward"][i]),
                torch.from_numpy(arrays["next_obs"][i]),
                torch.from_numpy(arrays["terminated"][i]),
            )
            for i in range(len(arrays["obs"]))
        ]
        critics, normalization, utd = METHODS[config["method"]]
        seed = (config["seed"] + index) % 2**32
        learners = {}
        for name, module in (("baseline", baseline), ("optimized", optimized)):
            random.seed(seed)
            torch.manual_seed(seed)
            learners[name] = module.SAC(
                arrays["obs"].shape[2],
                hidden=128,
                device=config["device"],
                ensemble=critics,
                layer_norm=normalization,
            )
        if any(
            not torch.equal(value.cpu(), learners["optimized"].state_dict()[key].cpu())
            for key, value in learners["baseline"].state_dict().items()
        ):
            raise RuntimeError("Learners do not share identical initial parameters")

        def reset_rng(block_seed):
            random.seed(block_seed)
            torch.manual_seed(block_seed)

        def update_block(name, actions, offset):
            updates = actions * utd
            for i in range(updates):
                boundary = (i + 1) % (CHUNK_ACTIONS * utd) == 0
                kwargs = {"update_actor": (i + 1) % utd == 0}
                if name == "optimized" and config["optimization"] in (
                    "metrics",
                    "both",
                ):
                    kwargs["collect_metrics"] = boundary
                metrics = learners[name].update(
                    batches[(offset + i) % len(batches)], **kwargs
                )
                if boundary and (
                    not metrics
                    or not all(math.isfinite(value) for value in metrics.values())
                ):
                    raise FloatingPointError(f"Invalid {name} metrics: {metrics}")

        for name in learners:
            reset_rng(seed)
            update_block(name, config["warmup_actions"], 0)
        synchronize()
        barrier.wait(config["timeout"])
        final_rng = {}
        for repeat in range(config["repeats"]):
            order = (
                ("baseline", "optimized")
                if repeat % 2 == 0
                else ("optimized", "baseline")
            )
            for name in order:
                reset_rng((seed + repeat + 1) % 2**32)
                barrier.wait(config["timeout"])
                synchronize()
                start = time.time()
                clock = time.perf_counter()
                offset = (
                    config["warmup_actions"] + repeat * config["block_actions"]
                ) * utd
                update_block(name, config["block_actions"], offset)
                synchronize()
                end = time.time()
                elapsed = time.perf_counter() - clock
                if repeat == config["repeats"] - 1:
                    # No device state copies while a peer still measures this block.
                    barrier.wait(config["timeout"])
                    final_rng[name] = {
                        "python": random.getstate(),
                        "torch_cpu": torch.get_rng_state(),
                        "torch_device": torch.cuda.get_rng_state_all()
                        if config["device"] == "cuda"
                        else torch.mps.get_rng_state(),
                    }
                messages.put(
                    {
                        "kind": "block",
                        "worker": index,
                        "repeat": repeat,
                        "variant": name,
                        "started_at": start,
                        "finished_at": end,
                        "seconds": elapsed,
                        "critic_updates": config["block_actions"] * utd,
                        "equivalent_actions": config["block_actions"],
                    }
                )
        barrier.wait(config["timeout"])
        qualification = qualify_learners(learners, final_rng)
        barrier.wait(config["timeout"])
        profile_metadata = None
        if config["profile"]:
            activity = getattr(
                torch.profiler.ProfilerActivity, config["device"].upper(), None
            )
            activities = [torch.profiler.ProfilerActivity.CPU]
            if (
                activity is not None
                and activity in torch.profiler.supported_activities()
            ):
                activities.append(activity)
            with torch.profiler.profile(activities=activities) as profile:
                for i in range(5):
                    kwargs = {"update_actor": (i + 1) % utd == 0}
                    if config["optimization"] in ("metrics", "both"):
                        kwargs["collect_metrics"] = i == 4
                    metrics = learners["optimized"].update(
                        batches[i % len(batches)], **kwargs
                    )
                    profile.step()
                synchronize()
            if not metrics or not all(
                math.isfinite(value) for value in metrics.values()
            ):
                raise FloatingPointError("Invalid profiler fixture metrics")
            profile.export_chrome_trace(
                str(Path(config["output"]) / "profile-five-updates.json")
            )
            profile_metadata = {
                "phase": "five-post-measurement-updates",
                "activities": [str(value) for value in activities],
                "native_device_events": sum(
                    event.device_type != torch.autograd.DeviceType.CPU
                    for event in profile.events()
                ),
                "limitations": "CPU dispatch trace on Metal unless native activity is supported. CUDA kernel coverage depends on available CUPTI; CPU timing is not GPU kernel timing.",
            }
        messages.put(
            {
                "kind": "done",
                "worker": index,
                "runtime": {
                    "torch": torch.__version__,
                    "python": sys.version,
                    "device": config["device"],
                    "gpu": torch.cuda.get_device_name(0)
                    if config["device"] == "cuda"
                    else "Apple Metal",
                    "profile": profile_metadata,
                    "qualification": qualification,
                    **effective_settings,
                },
            }
        )
    except BaseException as exc:
        messages.put({"kind": "error", "worker": index, "error": repr(exc)})
        raise


def run_workers(config, target=worker):
    context = mp.get_context("spawn")
    messages = context.Queue()
    barrier = context.Barrier(config["workers"])
    processes, blocks, backgrounds, runtimes, done = [], [], [], [], set()
    deadline = time.monotonic() + config["timeout"]
    try:
        for index in range(config["workers"]):
            process = context.Process(
                target=target, args=(config, index, messages, barrier)
            )
            process.start()
            processes.append(process)
        while len(done) != config["workers"]:
            backgrounds.append(
                {
                    "at": time.time(),
                    "background": background_state(config["background"]),
                }
            )
            if time.monotonic() >= deadline:
                raise TimeoutError("Benchmark exceeded its overall timeout")
            try:
                message = messages.get(timeout=0.25)
            except queue.Empty:
                if any(
                    process.exitcode is not None and process.exitcode != 0
                    for process in processes
                ):
                    raise RuntimeError("Benchmark worker exited unsuccessfully")
                continue
            if message["kind"] == "error":
                raise RuntimeError(f"Worker {message['worker']}: {message['error']}")
            if message["kind"] == "done":
                if message["runtime"].get("qualification", {}).get("exact") is not True:
                    raise RuntimeError(
                        "Worker did not pass exact final state qualification"
                    )
                done.add(message["worker"])
                runtimes.append({"worker": message["worker"], **message["runtime"]})
            else:
                blocks.append(message)
        for process in processes:
            process.join(timeout=5)
            if process.exitcode != 0:
                raise RuntimeError("Benchmark worker did not exit successfully")
        return blocks, backgrounds, runtimes
    finally:
        for process in processes:
            if process.is_alive():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    process.kill()
                process.join(timeout=5)
        messages.close()
        messages.join_thread()


def aggregate(blocks, workers, repeats):
    rates = {"baseline": [], "optimized": []}
    windows = []
    for repeat in range(repeats):
        for variant, variant_rates in rates.items():
            group = [
                row
                for row in blocks
                if row["repeat"] == repeat and row["variant"] == variant
            ]
            if (
                len(group) != workers
                or len({row["worker"] for row in group}) != workers
            ):
                raise ValueError("Missing or duplicated benchmark block")
            start = min(row["started_at"] for row in group)
            end = max(row["finished_at"] for row in group)
            if not end > start:
                raise ValueError("Invalid benchmark timing window")
            rate = sum(row["equivalent_actions"] for row in group) / (end - start)
            variant_rates.append(rate)
            windows.append(
                {
                    "repeat": repeat,
                    "variant": variant,
                    "started_at": start,
                    "finished_at": end,
                    "aggregate_equivalent_actions_per_second": rate,
                }
            )
    return {
        "windows": windows,
        "rates": rates,
        "median_speedup": statistics.median(
            b / a for a, b in zip(rates["baseline"], rates["optimized"])
        ),
    }


def evaluation_crossings(samples):
    previous, crossings = {}, []
    for sample in samples:
        for manifest in sample["background"]:
            for job in manifest.get("jobs", []):
                key = manifest["manifest"], job["name"]
                signature = job["last_evaluation"], job["evaluation_log_bytes"]
                if key in previous and signature != previous[key]:
                    crossings.append(
                        {
                            "at": sample["at"],
                            "manifest": key[0],
                            "job": key[1],
                            "before": previous[key],
                            "after": signature,
                        }
                    )
                previous[key] = signature
    return crossings


def queue_replay(jobs):
    if len(jobs) != 6 or {row["method"] for row in jobs} != set(METHODS):
        raise ValueError("Queue replay requires six SAC/SACsym/RLPD tiny jobs")
    seeds = {row["seed"] for row in jobs}
    if len(seeds) != 2 or len({(row["method"], row["seed"]) for row in jobs}) != 6:
        raise ValueError("Queue replay requires the same three methods at two seeds")
    if any(not math.isfinite(row["seconds"]) or row["seconds"] <= 0 for row in jobs):
        raise ValueError("Measured job durations must be positive and finite")
    traces = {}
    priority = {"rlpd": 0, "sac_sym": 1, "sac": 2}
    orders = {
        "existing": sorted(
            jobs, key=lambda row: (row["seed"], tuple(METHODS).index(row["method"]))
        ),
        "long-first": sorted(
            jobs, key=lambda row: (priority[row["method"]], row["seed"])
        ),
    }
    for name, order in orders.items():
        slots, trace = [(0.0, 0), (0.0, 1)], []
        for job in order:
            start, slot = heapq.heappop(slots)
            finish = start + job["seconds"]
            trace.append(
                job
                | {
                    "slot": slot,
                    "started_at_seconds": start,
                    "finished_at_seconds": finish,
                }
            )
            heapq.heappush(slots, (finish, slot))
        traces[name] = {
            "makespan_seconds": max(finish for finish, _ in slots),
            "jobs": trace,
        }
    return {
        "phase": "shape-limited-queue-estimate",
        "additional_slots": 2,
        "strategies": traces,
        "limitations": "Uses the same six measured tiny-job durations without changing their contents. Concurrent contention, startup, evaluation and full-study scaling are not predicted; this is an estimate, not measured study speedup.",
    }


def main(argv=None):
    args = parser().parse_args(argv)
    if args.queue_replay:
        try:
            source = json.loads(args.queue_replay.read_text())
            result = queue_replay(source["jobs"])
            args.output.mkdir(parents=True, exist_ok=False)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser().error(str(exc))
        result.update(
            label=args.label,
            input_sha256=digest(args.queue_replay),
            input_provenance=source,
        )
        (args.output / "queue-estimate.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return
    try:
        provenance = validate(args)
        args.output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as exc:
        parser().error(str(exc))
    config = vars(args) | {
        "baseline": str(args.baseline.resolve()),
        "batches": str(args.batches.resolve()),
        "background": [str(path.resolve()) for path in args.background],
    }
    config["output"] = str(args.output.resolve())
    config.update(variant_settings(args.device, args.optimization))
    config["baseline_sha256"] = digest(args.baseline)
    config["batches_sha256"] = provenance["sha256"]
    optimized = Path(__file__).resolve().parents[1] / "src/kartstudy/algorithms.py"
    config["optimized_sha256"] = digest(optimized)
    report = {
        "schema_version": 1,
        "status": "failed",
        "exact_final_state_qualification": False,
        "phase": "cached-training-update",
        "label": args.label,
        "config": config,
        "input_provenance": provenance,
        "source_sha256": {
            "baseline": digest(args.baseline),
            "optimized": digest(optimized),
            "benchmark": digest(__file__),
        },
        "batch_size": 256,
        "hidden": 128,
        "finite_guard_action_cadence": 32,
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "limitations": "Cached learner updates exclude simulator rollout and evaluation. Additional workers run under the recorded background load; this does not establish an idle optimal worker count. Background status may lag; evaluation crossings are observable only when evaluation logs advance.",
    }
    try:
        blocks, backgrounds, runtimes = run_workers(config)
        report.update(
            status="passed",
            exact_final_state_qualification=True,
            blocks=blocks,
            background_samples=backgrounds,
            background_evaluation_crossings=evaluation_crossings(backgrounds),
            runtime=runtimes,
            additional_fixture_workers=args.workers,
            summary=aggregate(blocks, args.workers, args.repeats),
        )
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report["finished_at"] = dt.datetime.now(dt.UTC).isoformat()
        (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Benchmark saved: {args.output / 'benchmark.json'}")


if __name__ == "__main__":
    main()
