"""Fixed-seed expert diagnostics, independent of training evaluation logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .environment import SCHEMA_VERSION, DrivingEnv
from .expert import Expert
from .metrics import banana_contact, harmful_rows
from .tracks import TRAIN_TRACKS


def summarize(records):
    metrics = (
        "off_track_fraction",
        "rescue_count",
        "harmful_contact_proportion",
        "average_speed_m_s",
        "lap_time_seconds",
    )
    result = {
        "n_seeds": len(records),
        "completed": sum(row["success"] for row in records),
    }
    for metric in metrics:
        values = [row[metric] for row in records if row.get(metric) is not None]
        result[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "n": len(values),
        }
    return result


def evaluate_track(track, seeds, max_steps=1500, row_tolerance=3.0):
    env = DrivingEnv(track=track, max_steps=max_steps)
    records = []
    try:
        for seed in seeds:
            env.reset(seed=seed)
            base = env.env.unwrapped
            rows = harmful_rows(base.world.items, base.track, row_tolerance)
            events = []
            for step in range(max_steps):
                kart = base.world.karts[0]
                previous = (int(kart.attachment.type), float(kart.attachment.time_left))
                before_time = float(base.world.time)
                _, _, done, truncated, info = env.step(env.expert_action())
                kart = base.world.karts[0]
                current = (int(kart.attachment.type), float(kart.attachment.time_left))
                if banana_contact(
                    previous, current, float(base.world.time) - before_time
                ):
                    events.append(
                        {
                            "step": step + 1,
                            "sim_seconds": info["elapsed_seconds"],
                            "progress_m": info["progress"],
                            "position": [float(x) for x in kart.location],
                            "attachment_before": previous,
                            "attachment_after": current,
                        }
                    )
                if done or truncated:
                    break
            success = bool(info["success"])
            lap_time = info["finish_time"] if success else None
            record = {
                "seed": seed,
                "simulator_seed": int(base.config.seed),
                "kart": str(kart.kart_type),
                "success": success,
                "steps": step + 1,
                "elapsed_seconds": info["elapsed_seconds"],
                "progress_m": info["max_progress"],
                "track_length_m": float(base.track.length),
                "lap_time_seconds": lap_time,
                "average_speed_m_s": float(base.track.length / lap_time)
                if success
                else None,
                **{
                    key: info[key]
                    for key in (
                        "measurement_seconds",
                        "off_track_seconds",
                        "off_track_fraction",
                        "rescue_count",
                    )
                },
                "harmful_item_count": sum(len(row["items"]) for row in rows),
                "harmful_row_count": len(rows),
                "harmful_contacts": len(events),
                "harmful_contact_proportion": len(events) / len(rows) if rows else None,
                "full_lap_exposure": success,
                "harmful_rows": rows,
                "contact_events": events,
            }
            records.append(record)
            print(
                json.dumps(
                    {
                        "track": track,
                        **{
                            k: v
                            for k, v in record.items()
                            if k not in ("harmful_rows", "contact_events")
                        },
                    }
                ),
                flush=True,
            )
    finally:
        env.close()
    return {"track": track, "per_seed": records, "summary": summarize(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", default=list(TRAIN_TRACKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--row-tolerance", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.max_steps < 1
        or args.row_tolerance <= 0
        or len(set(args.seeds)) != len(args.seeds)
    ):
        parser.error("Require positive limits and distinct seeds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a fresh artifact before simulation; never overwrite a prior result.
    with args.output.open("x") as output:
        result = {
            "schema_version": SCHEMA_VERSION,
            "expert": Expert.name,
            "expert_source_commit": Expert.source_commit,
            "source_sha256": {
                name: hashlib.sha256(
                    (Path(__file__).parent / name).read_bytes()
                ).hexdigest()
                for name in (
                    "expert.py",
                    "metrics.py",
                    "native_telemetry.py",
                    "native_telemetry.cpp",
                    "environment.py",
                    "expert_evaluation.py",
                    "_team4/manifest.json",
                    *[
                        str(path.relative_to(Path(__file__).parent))
                        for path in sorted(
                            (Path(__file__).parent / "_team4").glob("*.py")
                        )
                    ],
                    "_team4/configuration.yaml",
                )
            },
            "versions": {
                name: version(name)
                for name in (
                    "pysupertuxkart2",
                    "pystk2-gymnasium",
                    "numpy",
                    "omegaconf",
                )
            },
            "protocol": {
                "seeds": args.seeds,
                "max_steps": args.max_steps,
                "step_seconds": 0.1,
                "controls": "steer, throttle, brake, nitro and fire; drift/rescue off",
                "row_tolerance_m": args.row_tolerance,
                "contact_detection": "attachment transitions; inferred at 0.1-second intervals",
                "contact_denominator": "all harmful rows on the track, including unreached rows on failed laps",
                "speed_and_lap_time": "successful laps only; arithmetic means of per-seed metrics",
                "off_track": "post-step native is_on_road, weighted by elapsed race time; countdown excluded; final interval clipped to finish time",
                "rescues": "distinct native rescue animations sampled every 0.1 s; explosion and cannon animations excluded",
            },
            "complete": False,
            "tracks": [],
        }
        for track in args.tracks:
            result["tracks"].append(
                evaluate_track(track, args.seeds, args.max_steps, args.row_tolerance)
            )
            output.seek(0)
            json.dump(result, output, indent=2, allow_nan=False)
            output.truncate()
            output.flush()
        result["complete"] = True
        output.seek(0)
        json.dump(result, output, indent=2, allow_nan=False)
        output.truncate()
        output.write("\n")


if __name__ == "__main__":
    main()
