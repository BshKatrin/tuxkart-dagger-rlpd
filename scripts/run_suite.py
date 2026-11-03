"""Bounded local experiment queue; survives the interactive agent session."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from kartstudy.evaluation import summarize_runs
from kartstudy.train import defaults, write_json


def experiment_specs():
    specs = []
    # Interleave methods by seed so the first complete comparison arrives early.
    for seed in range(5):
        for algo in ["bc", "dagger", "sac", "sac_sym", "rlpd"]:
            specs.append(
                (
                    f"baseline_{algo}_s{seed}",
                    defaults()
                    | {"algo": algo, "seed": seed, "utd": 4 if algo == "rlpd" else 1},
                )
            )
    # Deliberately exploratory: compare to the baseline's 50k checkpoint.
    for condition in ["small_data", "conservative_expert"]:
        for seed in range(3):
            for algo in ["dagger", "rlpd"]:
                specs.append(
                    (
                        f"{condition}_{algo}_s{seed}",
                        defaults()
                        | {
                            "algo": algo,
                            "seed": seed,
                            "steps": 50000,
                            "utd": 4 if algo == "rlpd" else 1,
                            "demo_episodes": 2 if condition == "small_data" else 20,
                            "conservative": condition == "conservative_expert",
                            "dataset": "data/conservative.npz"
                            if condition == "conservative_expert"
                            else "data/expert.npz",
                        },
                    )
                )
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/study")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Use 1–4 workers")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "suite.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    jobs = []
    active = {}
    try:
        with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
            for path in [
                "src",
                "scripts",
                "tests",
                "configs",
                "pyproject.toml",
                "uv.lock",
                ".python-version",
                "TASK.md",
                "README.md",
            ]:
                if Path(path).exists():
                    archive.add(
                        path,
                        filter=lambda info: (
                            None if "__pycache__" in info.name else info
                        ),
                    )
        specs = experiment_specs()
        for name, config in specs:
            out = root / name
            config_file = root / "configs" / f"{name}.json"
            config_file.parent.mkdir(exist_ok=True)
            write_json(config_file, config)
            if out.exists():
                raise FileExistsError(f"Fresh suite required: {out} exists")
            jobs.append(
                {
                    "name": name,
                    "config": str(config_file),
                    "output": str(out),
                    "state": "queued",
                    "algo": config["algo"],
                    "seed": config["seed"],
                    "steps": config["steps"],
                }
            )
        manifest = {
            "pid": os.getpid(),
            "workers": args.workers,
            "started_at": dt.datetime.now(dt.UTC).isoformat(),
            "jobs": jobs,
        }
        while True:
            if stop:
                for process, _ in active.values():
                    if process.poll() is None:
                        process.terminate()
            for name, (process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                job = next(j for j in jobs if j["name"] == name)
                job.update(
                    state="completed" if code == 0 else "failed",
                    returncode=code,
                    finished_at=dt.datetime.now(dt.UTC).isoformat(),
                )
                del active[name]
                if code != 0:
                    stop = True
                    print(
                        f"Failed {name}: exit {code}; stopping queue. See log.",
                        flush=True,
                    )
                summarize_runs(root)
            # STOP means drain active workers, preserving queued work unlaunched.
            draining = stop or (root / "STOP").exists()
            if not draining:
                for job in jobs:
                    if len(active) >= args.workers:
                        break
                    if job["state"] != "queued":
                        continue
                    log = (root / f"{job['name']}.log").open("a")
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "kartstudy.train",
                            "--config",
                            job["config"],
                            "--output",
                            job["output"],
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=os.environ
                        | {
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                            "VECLIB_MAXIMUM_THREADS": "1",
                            "PYTHONUNBUFFERED": "1",
                        },
                    )
                    active[job["name"]] = (process, log)
                    job.update(
                        state="running",
                        pid=process.pid,
                        started_at=dt.datetime.now(dt.UTC).isoformat(),
                    )
                    print(f"Started {job['name']}, pid {process.pid}", flush=True)
            manifest["updated_at"] = dt.datetime.now(dt.UTC).isoformat()
            manifest["state"] = "draining" if draining else "running"
            write_json(root / "manifest.json", manifest)
            if not active and (draining or all(j["state"] != "queued" for j in jobs)):
                manifest["state"] = "stopped" if draining else "completed"
                write_json(root / "manifest.json", manifest)
                break
            time.sleep(2)
    finally:
        for process, log in active.values():
            if process.poll() is None:
                process.terminate()
            log.close()
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
