"""Summarize 30-seed report trajectories without changing original evaluations.

    uv run --with matplotlib python report/scripts/prepare_performance.py

Reads runs/report-30-seeds-trajectories-20261113 and the existing road geometry.
Writes seed-metrics.json and audited per-seed sources/local metrics under
report/generated/performance-thirty-seeds. Figure generation:
    uv run --with matplotlib --with seaborn python report/scripts/performance_scorecards.py \
      report/generated/performance-thirty-seeds/seed-metrics.json \
      --output report/generated/performance-thirty-seeds --capped-time-bars
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from aggregate_performance import CAP, METHODS
from local_performance import completed_lengths
from track_time_loss import aggregate_episodes, digest, local_summary

ROOT = Path(__file__).resolve().parents[2]


def finite_mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def lap_metrics(records, seeds):
    failures = [e for episodes in records.values() for e in episodes if not e["success"]]
    horizon = failures[0]["sim_seconds"] if failures else CAP
    row = aggregate_episodes(records, seeds, horizon)
    capped = [float(np.mean([e["lap_time"] if e["success"] else CAP
                            for episodes in records.values() for e in episodes
                            if e["seed"] == seed])) for seed in seeds]
    return {"completion": row["completion_pct"], "capped": float(np.mean(capped)),
            "capped_sd": float(np.std(capped, ddof=1)), "capped_by_seed": capped,
            "offroad": row["offroad_pct"], "rescues": row["rescues_per_attempt"],
            "completed": row["completed"], "attempts": row["attempts"]}


def prepare(root, output, geometry):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["state"] != "completed" or not manifest["trajectories"]:
        raise ValueError("Need completed trajectory evaluations")
    seeds = manifest["evaluation_seeds"]
    if seeds != list(range(20000, 20030)):
        raise ValueError("Expected exactly 30 evaluation seeds")
    selection = json.loads((geometry / "sources.json").read_text())["tracks"]
    if {t["id"] for t in selection} != set(manifest["tracks"]):
        raise ValueError("Report track selection differs from evaluation")
    output.mkdir(parents=True, exist_ok=True)
    for track in selection:
        target = output / "data" / track["id"]
        target.mkdir(parents=True, exist_ok=True)
        link = target / "geometry.json"
        if not link.exists():
            link.symlink_to(geometry / "data" / track["id"] / "geometry.json")
    groups = {split: [] for split in ("Training", "Held out")}
    for training_seed in range(3):
        sources = {t["id"]: {} for t in selection}
        for method in METHODS:
            directory = root / ("expert" if method == "expert" else f"{method}_s{training_seed}")
            config = json.loads((directory / "config.json").read_text())
            rows = [json.loads(line) for line in (directory / "evaluations.jsonl").read_text().splitlines()]
            (final,) = rows
            if method != "expert" and (config["seed"] != training_seed or
                    config["algo"] != method or final["step"] != (0 if method == "bc" else 400000)):
                raise ValueError(f"Wrong checkpoint: {directory}")
            for track in selection:
                track_id = track["id"]
                episodes = [e for e in final["episodes"] if e["track"] == track_id]
                if sorted(e["seed"] for e in episodes) != seeds:
                    raise ValueError(f"Missing evaluation seeds: {directory}/{track_id}")
                path = directory / f"trajectories_{track_id}.npz"
                sources[track_id][method] = {
                    "path": str(path.relative_to(ROOT)), "sha256": digest(path),
                    "episodes": episodes, "origin": "30-seed report trajectory evaluation"}
        metadata = {"tracks": selection, "seeds": seeds, "sources": sources}
        (output / "sources.json").write_text(json.dumps(metadata))
        (output / f"sources-s{training_seed}.json").write_text(json.dumps(metadata))
        local_summary(SimpleNamespace(output=output))
        local = json.loads((output / "local-metrics.json").read_text())
        (output / f"local-metrics-s{training_seed}.json").write_text(json.dumps(local))
        lengths = completed_lengths(metadata)
        for split, entries in groups.items():
            tracks = [t["id"] for t in selection if t["split"] == split]
            for method in METHODS if training_seed == 0 else METHODS[1:]:
                records = {t: sources[t][method]["episodes"] for t in tracks}
                metrics = lap_metrics(records, seeds)
                def local_mean(field, local=local, method=method, tracks=tracks):
                    return finite_mean([finite_mean([
                        local["tracks"][t]["by_seed"][method][i][field] for t in tracks])
                        for i in range(len(seeds))])
                relative_lengths = [finite_mean([
                    100 * lengths[t][method][str(s)] / lengths[t]["expert_mean_metres"]
                    for t in tracks if str(s) in lengths[t][method]]) for s in seeds]
                entries.append(dict(method=method,
                    seed=None if method == "expert" else training_seed,
                    **metrics, gap=local_mean("signed_gap"), offset=local_mean("lateral"),
                    length=finite_mean(relative_lengths), matched=local_mean("segments")))
    for entries in groups.values():
        (expert,) = [r for r in entries if r["method"] == "expert"]
        for row in entries:
            row["capped_gap"] = row["capped"] - expert["capped"]
    (output / "seed-metrics.json").write_text(json.dumps(groups, indent=2, allow_nan=False))
    print(output / "seed-metrics.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/report-30-seeds-trajectories-20261113")
    parser.add_argument("--output", type=Path, default=ROOT / "report/generated/performance-thirty-seeds")
    parser.add_argument("--geometry", type=Path, default=ROOT / "report/generated/track-time-loss")
    args = parser.parse_args()
    prepare(args.root.resolve(), args.output.resolve(), args.geometry.resolve())


if __name__ == "__main__":
    main()
