"""Plot local pace, road position and relative trajectory length for the report.

Run from repository root:
    uv run --with matplotlib --with seaborn python report/scripts/local_performance.py

Reads track-time-loss/local-metrics.json and sources.json under report/generated/.
Writes vector PDF, 150-dpi PNG, derived metrics and caption under
report/generated/aggregate-performance/. No trajectories or report sources change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from aggregate_performance import LABELS, METHODS

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "report/generated/aggregate-performance"


def trajectory_length(positions, times, rescue_active, rescue_count, finish_time):
    """3D polyline length over recorded race time; never connect rescue movements."""
    positions = np.asarray(positions, dtype=float)
    times = np.asarray(times, dtype=float)
    active = np.asarray(rescue_active, dtype=bool)
    counts = np.asarray(rescue_count, dtype=float)
    if len(times) < 2 or not np.all(np.isfinite(positions)):
        raise ValueError("Need finite positions for a completed trajectory")
    dt = np.diff(times)
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if (np.any(dt < 0) or not np.all(np.isfinite(times))
            or np.any((dt == 0) & (distances > 0))):
        raise ValueError("Trajectory timestamps must increase during movement")
    valid = ~(active[:-1] | active[1:]) & (np.diff(counts) == 0)
    # Fractionally trim the first/last recorded interval at race start/finish.
    overlap = np.maximum(0, np.minimum(times[1:], finish_time)
                         - np.maximum(times[:-1], 0))
    fraction = np.divide(overlap, dt, out=np.zeros_like(dt), where=dt > 0)
    return float(np.sum(distances * fraction * valid))


def completed_lengths(source):
    result = {}
    for track in source["tracks"]:
        track_id = track["id"]
        result[track_id] = {}
        for method in METHODS:
            entry = source["sources"][track_id][method]
            path = ROOT / entry["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Trajectory checksum changed: {path}")
            lengths = {}
            with np.load(path) as data:
                for episode in entry["episodes"]:
                    if not episode["success"]:
                        continue
                    mask = data["seed"] == episode["seed"]
                    length = trajectory_length(
                        data["world_position"][mask], data["elapsed_seconds"][mask],
                        data["rescue_active"][mask], data["rescue_count"][mask],
                        episode["lap_time"],
                    )
                    if length <= 0:
                        raise ValueError(f"Nonpositive completed trajectory: {path}")
                    lengths[str(episode["seed"])] = length
            result[track_id][method] = lengths
        expert = list(result[track_id]["expert"].values())
        if not expert:
            raise ValueError(f"No completed expert lap: {track_id}")
        result[track_id]["expert_mean_metres"] = float(np.mean(expert))
    return result


def summarize(local, source, lengths):
    if local["evaluation_seeds"] != source["seeds"]:
        raise ValueError("Local summaries and source evaluations use different seed orders")
    groups = {}
    for split in ("All tracks", "Training", "Held out"):
        tracks = [t["id"] for t in source["tracks"]
                  if split == "All tracks" or t["split"] == split]
        groups[split] = {}
        for method in METHODS:
            values = {field: [] for field in ("signed_gap", "lateral", "relative_length")}
            for track in tracks:
                data = local["tracks"][track]
                rows = data["by_seed"][method]
                if len(rows) != len(source["seeds"]):
                    raise ValueError(f"Missing seed summary: {track}/{method}")
                reference = lengths[track]["expert_mean_metres"]
                values["relative_length"].append([
                    100 * lengths[track][method][str(seed)] / reference
                    if str(seed) in lengths[track][method] else np.nan
                    for seed in source["seeds"]
                ])
                for field in ("signed_gap", "lateral"):
                    values[field].append([r[field] for r in rows])
            result = {}
            for field, rows in values.items():
                array = np.asarray(rows, dtype=float)
                seed_means = np.nanmean(array, axis=0)
                if not np.all(np.isfinite(seed_means)):
                    raise ValueError(f"Missing local values: {split}/{method}/{field}")
                result[field] = {"mean": float(np.mean(seed_means)),
                                 "sd": float(np.std(seed_means, ddof=1)),
                                 "by_seed": seed_means.tolist()}
                if field != "relative_length":
                    assert result[field] == local["groups"][split]["methods"][method][field]
            result["completed_tracks_by_seed"] = np.sum(
                np.isfinite(values["relative_length"]), axis=0
            ).tolist()
            groups[split][method] = result
    return {"evaluation_seeds": source["seeds"], "groups": groups,
            "completed_trajectory_lengths_metres": lengths}


def plot(metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    sns.set_theme(context="paper", style="ticks", palette="colorblind")
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    palette = sns.color_palette("colorblind")
    colors = ["#27323c", *(palette[i] for i in (0, 1, 2, 4, 3))]
    fig, axes = plt.subplots(1, 3, figsize=(17 / 2.54, 8.6 / 2.54), sharey=True)
    for i, (method, base) in enumerate(zip(METHODS, colors, strict=True)):
        for j, (split, marker) in enumerate(zip(
                ("Training", "Held out"), ("o", "^"), strict=True)):
            color = np.array(to_rgb(base))
            if j:
                color = .60 * color + .40
            for ax, field in zip(axes, ("signed_gap", "lateral", "relative_length"), strict=True):
                row = metrics["groups"][split][method][field]
                ax.errorbar(row["mean"], i + (-.16, .16)[j], xerr=row["sd"],
                            fmt=marker, markersize=4.5, color=color,
                            elinewidth=.9, capsize=2, capthick=.8, zorder=4)
    axes[0].set_yticks(range(len(METHODS)), LABELS)
    for tick, color in zip(axes[0].get_yticklabels(), colors, strict=True):
        tick.set_color(color)
    axes[0].set_ylabel("Policy")
    axes[0].axvline(0, color=".55", linewidth=.7)
    axes[2].axvline(100, color=".55", linewidth=.7, linestyle=(0, (3, 3)))
    for ax, title, xlabel in zip(
            axes,
            ("(a) Signed local time gap", "(b) Lateral position", "(c) Relative trajectory length"),
            ("Time gap (s / 20 m)", "Absolute offset (% half-width)", "Trajectory length (% of expert)"),
            strict=True):
        ax.set_title(title, fontsize=8.5, loc="left", pad=12)
        ax.set_xlabel(xlabel, labelpad=7)
        ax.set_ylim(5.55, -.55)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="x", color=".88", linewidth=.5)
        ax.axhline(.5, color=".65", linewidth=.6, linestyle=(0, (2, 3)))
        ax.tick_params(axis="y", length=0)
        ax.margins(x=.10)
        sns.despine(ax=ax, left=True)
    handles = [Line2D([], [], marker=m, color=c, linestyle="none", markersize=5,
                      label=label) for m, c, label in zip(
                          ("o", "^"), ("#27323c", "#7d848a"),
                          ("Training tracks", "Held-out tracks"), strict=True)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1),
               ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.13, right=.98, bottom=.18, top=.81, wspace=.32)
    for extension in ("pdf", "png"):
        fig.savefig(OUTPUT / f"local-performance.{extension}", dpi=150)
    plt.close(fig)


CAPTION = """Local driving summaries on 4 training and 4 held-out tracks, with
4 evaluation seeds per track. Dark circles denote training tracks; lighter
triangles denote held-out tracks. Markers and whiskers show mean ±1 sample SD
of 4 seed-level averages, each weighting tracks equally. Negative time gaps
indicate gains against the expert. Mean absolute lateral position is normalized
by road half-width and describes road use, not a centerline-following objective.
Relative trajectory length is 100 times completed-lap traveled distance divided
by the mean completed expert distance on that track. Distance is a 3D polyline
through recorded positions at the 0.1 s action interval; countdown, post-finish
motion and intervals touching native rescue activity or a rescue-count change
are excluded. Rescue-skipped road is not reconstructed. Only completed laps
contribute to this metric; available completed tracks are averaged equally
within each evaluation seed, so contributing tracks may differ between seeds
and policies. Expert mean length is 100% by construction. Time gap and lateral
position still use valid matched segments, including portions of failed laps;
they do not rank whole-track performance. Variability is across evaluation seeds,
not independently trained controllers.
"""


def main():
    data = ROOT / "report/generated/track-time-loss"
    local_path = data / "local-metrics.json"
    source_path = data / "sources.json"
    source = json.loads(source_path.read_text())
    metrics = summarize(json.loads(local_path.read_text()), source, completed_lengths(source))
    metrics["local_metrics_sha256"] = hashlib.sha256(local_path.read_bytes()).hexdigest()
    metrics["sources_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot(metrics)
    (OUTPUT / "local-metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (OUTPUT / "local-caption.txt").write_text(CAPTION)
    print(OUTPUT / "local-performance.pdf")


if __name__ == "__main__":
    main()
