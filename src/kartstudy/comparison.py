"""Matched completed-run comparisons with training runs as statistical units."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from .tracks import ALL_TRACKS, TRAIN_TRACKS, track_group

METHODS = ("bc", "dagger", "sac", "sac_sym", "rlpd")
LABELS = {
    "bc": "BC",
    "dagger": "DAgger",
    "sac": "Online SAC",
    "sac_sym": "SAC + symmetric",
    "rlpd": "RLPD",
}
COLORS = {
    "bc": "#ff9c40",
    "dagger": "#40baff",
    "sac": "#ff4040",
    "sac_sym": "#c240ff",
    "rlpd": "#44ff40",
}
METRICS = (
    "completion_rate",
    "off_track_fraction",
    "rescues_per_lap",
    "conditional_lap_time_s",
    "mean_return",
)
EVALUATION_SEEDS = (20000, 20001, 20002, 20003)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _validate_episodes(episodes, tracks):
    expected = {(track, seed) for track in tracks for seed in EVALUATION_SEEDS}
    observed = [(row["track"], row["seed"]) for row in episodes]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError(
            "Evaluation must contain each fixed track/seed pair exactly once"
        )
    for row in episodes:
        if (row["split"], row["difficulty"]) != track_group(row["track"]):
            raise ValueError("Evaluation split/difficulty does not match the study")
        seconds, off_track = row["measurement_seconds"], row["off_track_seconds"]
        if not (
            math.isfinite(seconds)
            and seconds > 0
            and math.isfinite(off_track)
            and 0 <= off_track <= seconds + 1e-5
        ):
            raise ValueError(
                "Evaluation requires positive race time and bounded off-track time"
            )
        if not isinstance(row["success"], bool):
            raise TypeError("Evaluation success must be a boolean")
        lap_time = row["lap_time"]
        if row["success"]:
            if lap_time is None or not math.isfinite(lap_time) or lap_time <= 0:
                raise ValueError("Successful evaluations require positive lap time")
        elif lap_time is not None:
            raise ValueError("Failed evaluations must not have successful lap time")
        rescues = row["rescue_count"]
        if (
            type(rescues) is not int
            or rescues < 0
            or not math.isfinite(row["return"])
        ):
            raise ValueError(
                "Evaluation requires nonnegative rescue counts and finite return"
            )


def lap_statistics(episodes):
    """Pool race time within one training run; keep failed laps in denominators."""
    if not episodes:
        raise ValueError("At least one evaluation lap is required")
    successful = [row["lap_time"] for row in episodes if row["success"]]
    seconds = sum(row["measurement_seconds"] for row in episodes)
    return {
        "completed": len(successful),
        "evaluation_laps": len(episodes),
        "measurement_seconds": seconds,
        "completion_rate": len(successful) / len(episodes),
        "off_track_fraction": sum(row["off_track_seconds"] for row in episodes)
        / seconds,
        "rescues_per_lap": sum(row["rescue_count"] for row in episodes) / len(episodes),
        "conditional_lap_time_s": float(np.mean(successful)) if successful else None,
        "mean_return": float(np.mean([row["return"] for row in episodes])),
    }


def _estimate(values, rng, bootstrap_samples):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    interval = None
    if len(values) > 1:
        samples = rng.choice(
            values, (bootstrap_samples, len(values)), replace=True
        ).mean(1)
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
    return {
        "mean": float(values.mean()) if len(values) else None,
        "ci95": interval,
        "n_training_seeds": len(values),
    }


def _conditions(episodes):
    groups = defaultdict(list)
    for row in episodes:
        groups[f"track/{row['track']}"].append(row)
        groups[f"{row['split']}/{row['difficulty']}"].append(row)
        groups[f"{row['split']}/all"].append(row)
    return groups


def _load_run(path):
    path = Path(path).resolve()
    names = ("config.json", "status.json", "provenance.json", "evaluations.jsonl")
    hashes = {name: _sha(path / name) for name in names}
    config, status, provenance = (_read(path / name) for name in names[:3])
    method, seed = config["algo"], config["seed"]
    if method not in METHODS or config["schema_version"] != 3:
        raise ValueError("Comparison requires supported schema-3 methods")
    if type(seed) is not int or seed < 0:
        raise ValueError("Training seed must be a nonnegative integer")
    if status["status"] != "completed":
        raise ValueError(f"Completed runs required: {path}")
    if (
        config["tracks"] != list(TRAIN_TRACKS)
        or config["steps"] != 400000
        or config["steps_per_track"] != 100000
    ):
        raise ValueError(
            "Comparison requires the frozen four-track 400,000-action protocol"
        )
    if config["bc_updates"] != 8000 or config["utd"] != (4 if method == "rlpd" else 1):
        raise ValueError(
            "Comparison requires matching fitting and method-specific update budgets"
        )
    if (config["demo_episodes"], config["max_steps"], config["eval_every"]) != (
        20,
        1500,
        40000,
    ):
        raise ValueError(
            "Comparison requires 20 training demos per track and the frozen assessment schedule"
        )
    expected_step = 0 if method == "bc" else config["steps"]
    if status["step"] != expected_step or any(
        status["track_steps"].get(track) != (0 if method == "bc" else 100000)
        for track in TRAIN_TRACKS
    ):
        raise ValueError("Run did not finish its exact action budget")
    if method == "bc" and status["updates"] != config["bc_updates"]:
        raise ValueError("BC did not finish its fitting update budget")
    if provenance["configuration"] != config:
        raise ValueError("Run configuration differs from its recorded provenance")
    dataset_sha = provenance["dataset_sha256"]
    if (
        not isinstance(dataset_sha, str)
        or len(dataset_sha) != 64
        or any(character not in "0123456789abcdef" for character in dataset_sha)
    ):
        raise ValueError("Dataset provenance checksum is required")
    rows = [
        json.loads(line)
        for line in (path / "evaluations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if any(type(row.get("final")) is not bool for row in rows):
        raise ValueError("Assessments require explicit boolean final flags")
    final = [row for row in rows if row.get("final")]
    if len(final) != 1 or final[0]["step"] != expected_step:
        raise ValueError("One final assessment at the completed budget is required")
    expected_steps = [0] if method == "bc" else list(range(0, 400001, 40000))
    if [row["step"] for row in rows] != expected_steps or rows[-1] != final[0]:
        raise ValueError("Assessment counters must match the complete frozen schedule")
    if final[0]["updates"] != status["updates"]:
        raise ValueError("Final assessment must match completed optimizer updates")
    _validate_episodes(final[0]["episodes"], ALL_TRACKS)
    curve = []
    seen = set()
    for row in rows:
        training = [
            episode for episode in row["episodes"] if episode["split"] == "train"
        ]
        _validate_episodes(training, TRAIN_TRACKS)
        if not row.get("final") and len(row["episodes"]) != len(training):
            raise ValueError(
                "Intermediate assessments must not contain held-out feedback"
            )
        x = row["updates"] if method == "bc" else row["step"]
        if (
            type(row["updates"]) is not int
            or row["updates"] < 0
            or not math.isfinite(row["wall_seconds"])
            or row["wall_seconds"] < 0
        ):
            raise ValueError(
                "Assessments require nonnegative updates and finite elapsed time"
            )
        if x in seen or not 0 <= x <= (8000 if method == "bc" else 400000):
            raise ValueError("Learning-curve counters must be unique and within budget")
        seen.add(x)
        curve.append(
            {
                "method": method,
                "training_seed": seed,
                "counter": "fitting_updates" if method == "bc" else "collected_actions",
                "x": x,
                "updates": row["updates"],
                "wall_seconds": row.get("wall_seconds"),
                **lap_statistics(training),
            }
        )
    if any(a["x"] >= b["x"] for a, b in pairwise(curve)):
        raise ValueError("Learning-curve counters must increase")
    if any(
        a["updates"] > b["updates"] or a["wall_seconds"] > b["wall_seconds"]
        for a, b in pairwise(curve)
    ):
        raise ValueError("Assessment updates and elapsed time must not decrease")
    if (
        not math.isfinite(status["wall_seconds"])
        or status["wall_seconds"] < final[0]["wall_seconds"]
    ):
        raise ValueError("Completed elapsed time must include final assessment")
    return {
        "path": str(path),
        "method": method,
        "seed": seed,
        "config": config,
        "status": status,
        "provenance": provenance,
        "input_sha256": hashes,
        "final": final[0],
        "curve": curve,
    }


def build_comparison(run_dirs, bootstrap_samples=2000):
    """Reject incomplete/mismatched inputs; aggregate each independent seed once."""
    if bootstrap_samples < 1:
        raise ValueError("Bootstrap sample count must be positive")
    runs = [_load_run(path) for path in run_dirs]
    if {run["method"] for run in runs} != set(METHODS):
        raise ValueError("Supply at least one completed run for all five methods")
    identities = [(run["method"], run["seed"]) for run in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate method/training seed would count one run twice")
    runs.sort(key=lambda run: (METHODS.index(run["method"]), run["seed"]))
    excluded = {"algo", "seed", "dataset", "utd"}
    configurations = [
        {key: value for key, value in run["config"].items() if key not in excluded}
        for run in runs
    ]
    if any(config != configurations[0] for config in configurations[1:]):
        raise ValueError("Run configurations do not match the retained study")
    if len({run["provenance"]["dataset_sha256"] for run in runs}) != 1:
        raise ValueError("Demonstration dataset checksums do not match")
    versions = [run["provenance"]["versions"] for run in runs]
    if any(version != versions[0] for version in versions[1:]):
        raise ValueError(
            "Simulator/framework versions differ; compare environments separately"
        )
    grouped = defaultdict(list)
    for run in runs:
        for condition, episodes in _conditions(run["final"]["episodes"]).items():
            grouped[run["method"], condition].append(
                {"seed": run["seed"], **lap_statistics(episodes)}
            )
    rng = np.random.default_rng(2026)
    groups = []
    for method in METHODS:
        for (group_method, condition), rows in sorted(grouped.items()):
            if group_method != method:
                continue
            groups.append(
                {
                    "method": method,
                    "condition": condition,
                    "training_seeds": sorted(row["seed"] for row in rows),
                    "completed": sum(row["completed"] for row in rows),
                    "evaluation_laps": sum(row["evaluation_laps"] for row in rows),
                    "per_seed": sorted(rows, key=lambda row: row["seed"]),
                    "metrics": {
                        metric: _estimate(
                            [row[metric] for row in rows], rng, bootstrap_samples
                        )
                        for metric in METRICS
                    },
                }
            )
    return {
        "format_version": 1,
        "methods": list(METHODS),
        "common_settings": configurations[0],
        "dataset_sha256": runs[0]["provenance"]["dataset_sha256"],
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "versions": versions[0],
        "aggregation": "Pool laps/race time within each training run, then average independent training runs equally; percentile intervals bootstrap training runs, never evaluation laps.",
        "limits": "A single training seed has no training-seed uncertainty interval. Intervals from a few seeds are exploratory. Lap time is conditional on successful completion; seeds with no successes have no lap-time estimate. Held-out results never select a checkpoint.",
        "analysis": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": 2026,
            "source_sha256": _sha(__file__),
        },
        "groups": groups,
        "learning_curves": [point for run in runs for point in run["curve"]],
        "runs": [
            {
                key: run[key]
                for key in (
                    "path",
                    "method",
                    "seed",
                    "config",
                    "status",
                    "provenance",
                    "input_sha256",
                )
            }
            for run in runs
        ],
    }


def _csv(path, rows):
    rows = list(rows)
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flat_groups(report):
    for group in report["groups"]:
        row = {
            key: group[key]
            for key in ("method", "condition", "completed", "evaluation_laps")
        }
        row["training_seeds"] = ";".join(map(str, group["training_seeds"]))
        for metric, estimate in group["metrics"].items():
            row[metric] = estimate["mean"]
            row[f"{metric}_ci95_low"] = (
                estimate["ci95"][0] if estimate["ci95"] else None
            )
            row[f"{metric}_ci95_high"] = (
                estimate["ci95"][1] if estimate["ci95"] else None
            )
            row[f"{metric}_n_training_seeds"] = estimate["n_training_seeds"]
        yield row


def _markdown(report):
    lines = [
        "# Five-method final comparison",
        "",
        report["aggregation"],
        "",
        report["limits"],
        "",
        "BC uses 8,000 fitting updates and zero collected actions; online methods use 400,000 collected actions. Elapsed time below includes assessment and checkpoint work.",
        "",
        "| Method | Split | Training seeds | Completed / laps | Off-track time | Rescues / lap | Successful lap time (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:
        if group["condition"] not in ("train/all", "held_out/all"):
            continue
        metrics = group["metrics"]
        lap_time = metrics["conditional_lap_time_s"]["mean"]
        lap_display = f"{lap_time:.2f}" if lap_time is not None else "—"
        lines.append(
            f"| {LABELS[group['method']]} | {group['condition']} | {len(group['training_seeds'])} | {group['completed']} / {group['evaluation_laps']} | {metrics['off_track_fraction']['mean']:.1%} | {metrics['rescues_per_lap']['mean']:.2f} | {lap_display} |"
        )
    lines += [
        "",
        "Detailed track and difficulty groups, success denominators and per-run estimates are in comparison.json and final-results.csv.",
        "",
        "| Method | Training seed | Actions | Optimizer updates | Elapsed seconds | Device |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for run in report["runs"]:
        status = run["status"]
        lines.append(
            f"| {LABELS[run['method']]} | {run['seed']} | {status['step']} | {status['updates']} | {status['wall_seconds']:.1f} | {run['config']['device']} |"
        )
    return "\n".join(lines) + "\n"


def _figures(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    lookup = {
        (group["method"], group["condition"]): group for group in report["groups"]
    }
    fields = (
        ("completion_rate", "Completion (%)", 100),
        ("off_track_fraction", "Off-track race time (%)", 100),
        ("rescues_per_lap", "Rescues per lap", 1),
        ("conditional_lap_time_s", "Successful lap time (s)", 1),
    )
    seed_counts = {
        method: len(lookup[method, "train/all"]["training_seeds"]) for method in METHODS
    }
    subtitle = ", ".join(f"{LABELS[m]} n={seed_counts[m]}" for m in METHODS)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    for row, split in enumerate(("train", "held_out")):
        for column, (metric, title, scale) in enumerate(fields):
            ax = axes[row, column]
            for index, method in enumerate(METHODS):
                group = lookup[method, f"{split}/all"]
                estimate = group["metrics"][metric]
                value = estimate["mean"]
                if value is None:
                    ax.text(
                        index, 0, "No successes", rotation=90, ha="center", va="bottom"
                    )
                    continue
                ax.bar(
                    index,
                    value * scale,
                    color=COLORS[method],
                    edgecolor="#333333",
                    linewidth=0.5,
                )
                if estimate["ci95"]:
                    low, high = estimate["ci95"]
                    ax.errorbar(
                        index,
                        value * scale,
                        yerr=[
                            [max(0, (value - low) * scale)],
                            [max(0, (high - value) * scale)],
                        ],
                        color="black",
                        capsize=3,
                    )
                if metric == "completion_rate":
                    ax.text(
                        index,
                        value * scale + 2,
                        f"{group['completed']}/{group['evaluation_laps']}",
                        ha="center",
                        fontsize=9,
                    )
            ax.set_xticks(
                range(5),
                [LABELS[m].replace(" + ", " +\n") for m in METHODS],
                rotation=35,
                ha="right",
            )
            ax.set_title(("Training: " if split == "train" else "Held out: ") + title)
            ax.set_ylim(bottom=0)
            if metric in ("completion_rate", "off_track_fraction"):
                ax.set_ylim(0, 112 if metric == "completion_rate" else 100)
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Final policy comparison — "
        + subtitle
        + "\nFixed evaluation seeds 20000–20003; lap time conditional on success; intervals bootstrap training seeds (omitted for n=1)",
        fontsize=12,
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"final-results.{suffix}", dpi=220)
    plt.close(fig)

    matrix = np.asarray(
        [
            [
                lookup[method, f"track/{track}"]["metrics"]["completion_rate"]["mean"]
                for method in METHODS
            ]
            for track in ALL_TRACKS
        ]
    )
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    for row, track in enumerate(ALL_TRACKS):
        for column, method in enumerate(METHODS):
            group = lookup[method, f"track/{track}"]
            ax.text(
                column,
                row,
                f"{group['completed']}/{group['evaluation_laps']}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > 0.7 else "black",
            )
    ax.set_xticks(range(5), [LABELS[m] for m in METHODS], rotation=20, ha="right")
    ax.set_yticks(
        range(len(ALL_TRACKS)), [track.replace("_", " ") for track in ALL_TRACKS]
    )
    ax.axhline(len(TRAIN_TRACKS) - 0.5, color="black", linewidth=2)
    ax.set_title(
        "Final completion by track — counts include every evaluation lap\nTraining tracks above line; held-out tracks below"
    )
    fig.colorbar(image, ax=ax, label="Mean completion per independent training run")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"completion-by-track.{suffix}", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, (metric, title, scale) in zip(axes.flat, fields):
        for method in METHODS:
            points = [
                point
                for point in report["learning_curves"]
                if point["method"] == method
            ]
            if method == "bc":
                values = [
                    point[metric] for point in points if point[metric] is not None
                ]
                if values:
                    ax.axhline(
                        np.mean(values) * scale,
                        color=COLORS[method],
                        linestyle="--",
                        label="BC: 8k fitting updates (reference)",
                    )
                continue
            grouped = defaultdict(list)
            for point in points:
                grouped[point["x"]].append(point[metric])
            xs = sorted(grouped)
            ys = [
                np.mean([value for value in grouped[x] if value is not None])
                if any(value is not None for value in grouped[x])
                else np.nan
                for x in xs
            ]
            ax.plot(
                xs,
                np.asarray(ys) * scale,
                marker="o",
                markersize=3,
                color=COLORS[method],
                label=LABELS[method],
                linewidth=1.7,
            )
        ax.set_title("Training-track " + title.lower())
        ax.set_xlabel("Collected actions (BC dashed line is a fitted reference)")
        ax.grid(alpha=0.2)
        if metric in ("completion_rate", "off_track_fraction"):
            ax.set_ylim(-2, 102)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Recorded assessments only — no held-out checkpoint selection\n" + subtitle,
        fontsize=13,
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"learning-curves.{suffix}", dpi=220)
    plt.close(fig)


def write_comparison(report, output, figures=True):
    """Write a fresh immutable artifact set; reject an edited input during analysis."""
    for run in report["runs"]:
        if any(
            _sha(Path(run["path"]) / name) != sha
            for name, sha in run["input_sha256"].items()
        ):
            raise RuntimeError(
                "Comparison input changed; rebuild from the completed runs"
            )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    _csv(output / "final-results.csv", _flat_groups(report))
    _csv(output / "learning-curves.csv", report["learning_curves"])
    _csv(
        output / "compute.csv",
        (
            {
                "method": run["method"],
                "training_seed": run["seed"],
                "collected_actions": run["status"]["step"],
                "optimizer_updates": run["status"]["updates"],
                "wall_seconds_including_evaluation": run["status"]["wall_seconds"],
                "device": run["config"]["device"],
                "utd": run["config"]["utd"],
            }
            for run in report["runs"]
        ),
    )
    (output / "comparison.md").write_text(_markdown(report))
    if figures:
        _figures(report, output)
    checks = {
        path.name: _sha(path) for path in sorted(output.iterdir()) if path.is_file()
    }
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in checks.items())
    )
    return output
