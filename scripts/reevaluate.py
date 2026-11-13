"""Evaluate completed checkpoints in separate CPU simulator processes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from kartstudy.environment import SCHEMA_VERSION
from kartstudy.tracks import ALL_TRACKS, track_group

SUPPLEMENTAL_TRACKS = {"zengarden": ("held_out", "hard")}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def discover(roots):
    jobs, excluded, seen = [], [], set()
    for root in roots:
        root = Path(root)
        paths = (
            [root / "config.json"]
            if (root / "config.json").exists()
            else sorted(root.glob("*/config.json"))
        )
        for path in paths:
            config = json.loads(path.read_text())
            status_path = path.parent / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get("status") != "completed":
                excluded.append(
                    {"source": str(path.parent.resolve()), "status": status}
                )
                continue
            if config.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"Incompatible observation schema: {path}")
            key = config["algo"], config["seed"]
            if key in seen:
                raise ValueError(f"Duplicate method/training seed: {key}")
            seen.add(key)
            checkpoint = path.parent / "actor.pt"
            if not checkpoint.exists():
                checkpoint = path.parent / "checkpoint.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            jobs.append(
                {
                    "name": f"{key[0]}_s{key[1]}",
                    "source": str(path.parent.resolve()),
                    "checkpoint": str(checkpoint.resolve()),
                    "state": "queued",
                }
            )
    if not jobs:
        raise ValueError("No completed checkpoints")
    return jobs, excluded


def initialize_worker():
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def worker(job, root, tracks, seeds, trajectories):

    from kartstudy.environment import DrivingEnv
    from kartstudy.evaluation import evaluate
    from kartstudy.playback import load_actor

    start = time.perf_counter()
    out = Path(root) / job["name"]
    out.mkdir(exist_ok=False)
    if job["name"] == "expert":
        actor, step = None, 0
        config = {
            "algo": "expert",
            "seed": 0,
            "schema_version": SCHEMA_VERSION,
            "max_steps": 1500,
            "tracks": list(tracks),
        }
        provenance = {"training_seed": None, "expert": True}
    else:
        actor, config, step = load_actor(job["checkpoint"])
        if config["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Checkpoint schema differs from current task")
        checkpoint = Path(job["checkpoint"])
        with checkpoint.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        provenance = {
            "source": job["source"],
            "checkpoint": str(checkpoint),
            "sha256": digest,
            "checkpoint_step": step,
            "training_seed": config["seed"],
        }
    write_json(out / "config.json", config)
    write_json(out / "provenance.json", provenance)
    write_json(
        out / "status.json",
        {"status": "running", "pid": os.getpid(), "completed_tracks": 0, "episodes": 0},
    )
    records = []
    env = DrivingEnv(track=tracks[0], max_steps=config["max_steps"])
    try:
        for track in tracks:
            env.track = track
            env.env.unwrapped.default_track = track
            # The expert action reads the current native state on every step.
            policy = actor if actor is not None else lambda _: env.expert_action()
            rows = evaluate(
                env,
                policy,
                seeds,
                trajectory_path=(
                    out / f"trajectories_{track}.npz" if trajectories else None
                ),
            )
            split, difficulty = (
                SUPPLEMENTAL_TRACKS[track]
                if track in SUPPLEMENTAL_TRACKS else track_group(track)
            )
            rows = [
                dict(row, track=track, split=split, difficulty=difficulty)
                for row in rows
            ]
            records.extend(rows)
            with (out / "tracks.jsonl").open("a") as file:
                file.write(json.dumps({"track": track, "episodes": rows}) + "\n")
            write_json(
                out / "status.json",
                {
                    "status": "running",
                    "pid": os.getpid(),
                    "completed_tracks": len(records) // len(seeds),
                    "episodes": len(records),
                    "wall_seconds": time.perf_counter() - start,
                },
            )
        with (out / "evaluations.jsonl").open("x") as file:
            file.write(
                json.dumps(
                    {
                        "step": step,
                        "final": True,
                        "episodes": records,
                        "wall_seconds": time.perf_counter() - start,
                    }
                )
                + "\n"
            )
        status = {
            "status": "completed",
            "episodes": len(records),
            "steps": sum(row["steps"] for row in records),
            "wall_seconds": time.perf_counter() - start,
        }
        write_json(out / "status.json", status)
        return status
    except BaseException as error:
        write_json(out / "status.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--tracks", nargs="+", choices=(*ALL_TRACKS, *SUPPLEMENTAL_TRACKS), default=list(ALL_TRACKS)
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(20000, 20030))
    )
    parser.add_argument("--trajectories", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error("Use positive workers and unique seeds")
    jobs, excluded = discover(args.roots)
    jobs.append({"name": "expert", "state": "queued"})
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for path in ("src", "scripts", "docs", "pyproject.toml", "uv.lock"):
            archive.add(
                path, filter=lambda info: None if "__pycache__" in info.name else info
            )
    shutil.copyfile(__file__, root / "reevaluate.py")
    manifest = {
        "state": "running",
        "pid": os.getpid(),
        "workers": args.workers,
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "jobs": jobs,
        "excluded": excluded,
        "evaluation_seeds": args.seeds,
        "tracks": args.tracks,
        "schema_version": SCHEMA_VERSION,
        "trajectories": args.trajectories,
    }
    write_json(root / "manifest.json", manifest)
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
        initializer=initialize_worker,
    ) as pool:
        futures = {
            pool.submit(
                worker, job, str(root), args.tracks, args.seeds, args.trajectories
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                job.update(state="completed", result=future.result())
            except Exception as error:  # noqa: BLE001 -- persist worker failures
                job.update(state="failed", error=repr(error))
            print(json.dumps(job), flush=True)
            write_json(root / "manifest.json", manifest)
    manifest.update(
        state="failed"
        if any(job["state"] == "failed" for job in jobs)
        else "completed",
        finished_at=dt.datetime.now(dt.UTC).isoformat(),
    )
    write_json(root / "manifest.json", manifest)
    if manifest["state"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
