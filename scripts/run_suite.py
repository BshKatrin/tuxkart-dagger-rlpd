"""Bounded local experiment queue; survives the interactive agent session."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from kartstudy.data import validate_dataset
from kartstudy.evaluation import summarize_runs
from kartstudy.train import defaults, write_json


def experiment_specs(dataset=None):
    return [
        (
            f"shared_{algo}_s{seed}",
            defaults()
            | {"algo": algo, "seed": seed, "utd": 4 if algo == "rlpd" else 1}
            | ({"dataset": dataset} if dataset else {}),
        )
        for seed in range(5)
        for algo in ("bc", "dagger", "sac", "rlpd")
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/study-polar-v3")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dataset", default=defaults()["dataset"])
    parser.add_argument(
        "--qualification", default="runs/qualification-polar-v3/metrics.json"
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Use 1–4 workers")
    specs = experiment_specs(args.dataset)
    validate_dataset(
        args.dataset,
        specs[0][1]["tracks"],
        specs[0][1]["demo_episodes"],
        specs[0][1]["max_steps"],
    )
    qualification = json.loads(Path(args.qualification).read_text())
    if (
        qualification.get("schema_version") != specs[0][1]["schema_version"]
        or {row["track"] for row in qualification["tracks"]}
        != set(specs[0][1]["tracks"])
        or qualification["protocol"]["max_steps"] != specs[0][1]["max_steps"]
        or qualification["protocol"]["seeds"] != [0, 1, 2, 3]
        or not all(
            seed["success"]
            for row in qualification["tracks"]
            for seed in row["per_seed"]
        )
        or any(
            hashlib.sha256((Path("src/kartstudy") / name).read_bytes()).hexdigest()
            != digest
            for name, digest in qualification["source_sha256"].items()
        )
    ):
        raise ValueError("Expert qualification failed or uses different source/tracks")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.qualification, root / "qualification.json")
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
                "AGENTS.md",
                "docs",
            ]:
                if Path(path).exists():
                    archive.add(
                        path,
                        filter=lambda info: (
                            None if "__pycache__" in info.name else info
                        ),
                    )
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
            "schema_version": specs[0][1]["schema_version"],
            "dataset": args.dataset,
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
