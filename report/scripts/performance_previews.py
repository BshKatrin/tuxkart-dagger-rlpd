"""Preview seed-level metrics and expert-relative completion curves.

    uv run --with matplotlib python report/scripts/performance_previews.py

Reads the existing scorecard analysis; never runs policies or changes report assets.
Outputs exploratory PNG/SVG figures and an audit summary in a separate directory.
For the original layout with one mean/SD per training run:
    uv run --with matplotlib python report/scripts/performance_previews.py --lap-performance
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "report/generated/performance-scorecards"
OUTPUT = ROOT / "report/generated/performance-design-previews"
METHODS = ("bc", "dagger", "sac", "sac_sym", "rlpd")
LABELS = ("BC", "DAgger", "SAC", "SAC + SS", "RLPD")
SPLITS = ("Training", "Held out")
COLORS = ("#3d70a3", "#be782e", "#258b7c")
MARKERS = ("o", "s", "^")
STYLES = ("-", "--", ":")
INK = "#242a30"


def read_inputs(directory):
    groups = json.loads((directory / "seed-metrics.json").read_text())
    sources = [json.loads((directory / f"sources-s{s}.json").read_text())
               for s in range(3)]
    curves = {split: {method: [] for method in METHODS} for split in SPLITS}
    for seed, source in enumerate(sources):
        if source["tracks"] != sources[0]["tracks"] or source["seeds"] != sources[0]["seeds"]:
            raise ValueError("Evaluation tracks and seeds must match across runs")
        for split in SPLITS:
            tracks = [t["id"] for t in source["tracks"] if t["split"] == split]
            for method in METHODS:
                (row,) = [r for r in groups[split]
                          if r["method"] == method and r["seed"] == seed]
                episodes, gaps = [], []
                for track in tracks:
                    entries = source["sources"][track]
                    if entries["expert"]["episodes"] != sources[0]["sources"][track]["expert"]["episodes"]:
                        raise ValueError("Expert references must be identical")
                    expert = {e["seed"]: e for e in entries["expert"]["episodes"]}
                    records = entries[method]["episodes"]
                    if sorted(e["seed"] for e in records) != sorted(source["seeds"]):
                        raise ValueError("Missing or duplicated evaluation episodes")
                    for episode in records:
                        reference = expert[episode["seed"]]
                        if not reference["success"]:
                            raise ValueError("Expert must finish every reference lap")
                        episodes.append(episode)
                        if episode["success"]:
                            gaps.append(episode["lap_time"] - reference["lap_time"])
                attempts = len(episodes)
                exposure = sum(e["measurement_seconds"] for e in episodes)
                derived = {
                    "completion": 100 * len(gaps) / attempts,
                    "capped": np.mean([e["lap_time"] if e["success"] else 150 for e in episodes]),
                    "offroad": 100 * sum(e["off_track_seconds"] for e in episodes) / exposure,
                    "rescues": np.mean([e["rescue_count"] for e in episodes]),
                }
                if attempts != row["attempts"] or len(gaps) != row["completed"]:
                    raise ValueError("Scorecard attempt counts disagree with episode records")
                for key, value in derived.items():
                    if not np.isclose(value, row[key]):
                        raise ValueError(f"Metric mismatch: {split}/{method}/{seed}/{key}")
                curves[split][method].append({"gaps": sorted(gaps), "attempts": attempts})
    return groups, curves


def legend(*, curves=False):
    handles = [Line2D([], [], color=color, marker=None if curves else marker,
                      ls=style if curves else "none", lw=1.2, markersize=4,
                      label=f"Seed {seed}")
               for seed, (color, marker, style) in enumerate(zip(COLORS, MARKERS, STYLES))]
    handles.append(Line2D([], [], color=INK, ls="-" if curves else "none",
                          marker=None if curves else "|", markersize=10,
                          markeredgewidth=1.6, lw=1.6, label="Mean of 3 runs"))
    handles.append(Line2D([], [], color=".42", ls=(0, (3, 3)), lw=.9, label="Expert"))
    return handles


def save(fig, output, stem):
    for extension in ("svg", "png"):
        fig.savefig(output / f"{stem}.{extension}", dpi=220)
    plt.close(fig)


def dots(groups, output, *, local=False):
    if local:
        columns = [
            ("gap", "Local time gap", "s / 20 m vs. expert", (-.18, .25), [-.1, 0, .1, .2]),
            ("offset", "Lateral position", "% of road half-width", (0, 100), [0, 25, 50, 75, 100]),
            ("length", "Trajectory length", "% of expert", (88, 102), [90, 95, 100]),
        ]
    else:
        columns = [
            ("completion", "Completion", "% of attempts", (0, 105), [0, 50, 100]),
            ("capped_gap", "Capped time gap", "s vs. expert", (-12, 85), [0, 40, 80]),
            ("offroad", "Off-road time", "% of race time", (0, 85), [0, 40, 80]),
            ("rescues", "Native rescues", "per attempt", (-.25, 7.1), [0, 3, 6]),
        ]
    fig, axes = plt.subplots(2, len(columns), figsize=(17 / 2.54, 11.7 / 2.54))
    fig.subplots_adjust(left=.125, right=.98, bottom=.20, top=.82, hspace=.75, wspace=.30)
    fig.legend(handles=legend(), loc="upper center", bbox_to_anchor=(.54, .985),
               ncol=5, frameon=False, handlelength=1.4, columnspacing=1.1, fontsize=7.4)
    for r, split in enumerate(SPLITS):
        (expert,) = [v for v in groups[split] if v["method"] == "expert"]
        for c, (field, title, units, limits, ticks) in enumerate(columns):
            ax = axes[r, c]
            ax.axvline(expert[field], color=".42", lw=.85, ls=(0, (3, 3)), zorder=1)
            for i, method in enumerate(METHODS):
                rows = sorted([v for v in groups[split] if v["method"] == method],
                              key=lambda v: v["seed"])
                if [v["seed"] for v in rows] != [0, 1, 2]:
                    raise ValueError("Require all three training seeds")
                values = [v[field] for v in rows]
                if not np.all(np.isfinite(values)) or min(values) < limits[0] or max(values) > limits[1]:
                    raise ValueError(f"Values outside preview axis: {field}")
                ax.axhline(i, color=".92", lw=.6, zorder=0)
                for seed, value in enumerate(values):
                    ax.scatter(value, i + (seed - 1) * .20, s=16, marker=MARKERS[seed],
                               facecolor=COLORS[seed], edgecolor="white", lw=.4, zorder=3)
                ax.plot(np.mean(values), i, "|", color=INK, ms=10, mew=1.5, zorder=4)
            if c == 0:
                ax.set_yticks(range(5), LABELS)
                ax.text(0, 1.15, "Training tracks" if r == 0 else "Held-out tracks",
                        transform=ax.transAxes, ha="left", fontsize=9, weight="bold")
            else:
                ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8.2, pad=29)
            ax.set(xlim=limits, ylim=(4.55, -.55), xticks=ticks)
            ax.set_xlabel(units, fontsize=7.4, labelpad=5)
            ax.tick_params(axis="y", length=0, pad=5)
            for side in ("left", "top", "right"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(".65")
    note = ("Local pace and position use available matched segments; length uses completed laps only.\n"
            "Offset and length describe road use; lower values do not necessarily mean better driving.") if local else (
                "Each seed marker summarizes 4 tracks × 4 evaluation seeds (16 attempts). No error bars.\n"
                "Capped time assigns 150 s to failed laps; the gap is relative to the expert on the same split.")
    fig.text(.125, .045, note, fontsize=6.8, color=".35", linespacing=1.5)
    save(fig, output, "option-1-local" if local else "option-1-global")


def completion_curves(curves, output):
    fig, axes = plt.subplots(5, 2, figsize=(17 / 2.54, 21 / 2.54), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.98, top=.90, bottom=.12, hspace=.43, wspace=.16)
    fig.legend(handles=legend(curves=True), loc="upper center", bbox_to_anchor=(.54, .99),
               ncol=5, frameon=False, handlelength=1.8, columnspacing=1.2, fontsize=7.4)
    for c, split in enumerate(SPLITS):
        for r, method in enumerate(METHODS):
            ax = axes[r, c]
            rows = curves[split][method]
            all_gaps = [v for row in rows for v in row["gaps"]]
            if not all(-20 < value < 90 for value in all_gaps):
                raise ValueError("A completed lap falls outside the preview domain")
            x = np.unique([-20, 0, 90, *all_gaps])
            levels = []
            for seed, row in enumerate(rows):
                y = 100 * np.searchsorted(row["gaps"], x, side="right") / row["attempts"]
                if not np.all(np.diff(y) >= 0) or not np.isclose(y[-1], 100 * len(row["gaps"]) / row["attempts"]):
                    raise ValueError("Completion profile must be monotone and retain failures")
                levels.append(y)
                ax.step(x, y, where="post", color=COLORS[seed], ls=STYLES[seed], lw=1.0, alpha=.8)
            mean = np.mean(levels, axis=0)
            ax.step(x, mean, where="post", color=INK, lw=1.5, zorder=5)
            ax.step([-20, 0, 90], [0, 100, 100], where="post", color=".42", lw=.85,
                    ls=(0, (3, 3)), zorder=1)
            ax.text(.98, .09, f"{mean[-1]:.1f}% finish", transform=ax.transAxes,
                    ha="right", fontsize=7.3, color=INK)
            ax.set_title(LABELS[r], loc="left", fontsize=8, pad=3)
            ax.set(xlim=(-20, 90), ylim=(-3, 105), yticks=[0, 50, 100], xticks=[-20, 0, 30, 60, 90])
            ax.grid(axis="y", color=".90", lw=.55)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(".65")
            ax.tick_params(axis="y", length=0)
            if r == 0:
                ax.text(.5, 1.37, "Training tracks" if c == 0 else "Held-out tracks",
                        transform=ax.transAxes, ha="center", fontsize=9, weight="bold")
    fig.text(.018, .515, "Attempts completed within allowance (%)", rotation=90, va="center", fontsize=8)
    fig.text(.54, .073, "Allowed finish-time gap vs. matched expert (s)", ha="center", fontsize=8)
    fig.text(.105, .028, "At 0 s: finishes at least as fast as the expert. Plateau: completion rate.\n"
             "Each run has 16 attempts; failures stay in the denominator. Black curve: equal-weight run mean.",
             fontsize=6.8, color=".35", linespacing=1.5)
    save(fig, output, "option-3-completion-curves")


def lap_performance(directory, output):
    """Keep policy colors and split markers; each run gets its own evaluation SD."""
    from aggregate_performance import LABELS as ALL_LABELS
    from aggregate_performance import METHODS as ALL_METHODS
    from aggregate_performance import summarize
    from matplotlib.colors import to_rgb
    from matplotlib.ticker import MaxNLocator

    source_paths = [directory / f"sources-s{seed}.json" for seed in range(3)]
    sources = [json.loads(path.read_text()) for path in source_paths]
    summaries = [summarize(source) for source in sources]
    columns = (
        ("completion_pct", "completion_pct_by_evaluation_seed", "(a) Completion", "Completion rate (%)"),
        ("capped_time_mean", "capped_time_by_evaluation_seed", "(b) Capped time", "Time (s)"),
        ("offroad_pct", "offroad_pct_by_evaluation_seed", "(c) Off-road time", "Race time (%)"),
        ("rescues_per_attempt", "rescues_per_attempt_by_evaluation_seed", "(d) Native rescues", "Rescues / attempt"),
    )
    for source, summary in zip(sources, summaries, strict=True):
        for split in SPLITS:
            tracks = [t["id"] for t in source["tracks"] if t["split"] == split]
            for method in ALL_METHODS:
                row = summary["groups"][split][method]
                row["completion_pct_by_evaluation_seed"] = [
                    100 * np.mean([e["success"] for t in tracks
                                   for e in source["sources"][t][method]["episodes"]
                                   if e["seed"] == evaluation_seed])
                    for evaluation_seed in source["seeds"]
                ]
                row["plot_sd"] = {
                    mean_key: float(np.std(row[raw_key], ddof=1))
                    for mean_key, raw_key, _, _ in columns
                }
                if not np.isclose(row["completion_pct"], np.mean(row["completion_pct_by_evaluation_seed"])):
                    raise ValueError("Completion summary weights disagree")

    # Original colorblind policy palette; track splits retain their original encoding.
    colors = ("#27323c", "#0173b2", "#de8f05", "#029e73", "#cc78bc", "#d55e00")
    fig, axes = plt.subplots(1, 4, figsize=(17 / 2.54, 12.4 / 2.54), sharey=True,
                             gridspec_kw={"width_ratios": [1.1, 1.4, 1.2, 1.2]})
    extrema = [[] for _ in columns]
    for i, (method, color) in enumerate(zip(ALL_METHODS, colors, strict=True)):
        for j, (split, marker) in enumerate(zip(SPLITS, ("o", "^"), strict=True)):
            shade = np.array(to_rgb(color))
            if j:
                shade = .60 * shade + .40
            training_seeds = [0] if method == "expert" else range(3)
            for seed in training_seeds:
                # Two groups per policy; seed order is 0, 1, 2 from top to bottom.
                y = i + (-.23 if j == 0 else .23)
                if method != "expert":
                    y += (seed - 1) * .12
                row = summaries[seed]["groups"][split][method]
                for c, (ax, (mean_key, _, _, _)) in enumerate(zip(axes, columns, strict=True)):
                    value, sd = row[mean_key], row["plot_sd"][mean_key]
                    extrema[c].extend([value - sd, value + sd])
                    ax.errorbar(value, y, xerr=sd, fmt=marker, color=shade,
                                markersize=3.0, elinewidth=.65, capsize=1.35,
                                capthick=.65, markeredgewidth=.4, zorder=3)

    axes[0].set_yticks(range(len(ALL_METHODS)), ALL_LABELS)
    for tick, color in zip(axes[0].get_yticklabels(), colors, strict=True):
        tick.set_color(color)
    for c, (ax, (_, _, title, xlabel)) in enumerate(zip(axes, columns, strict=True)):
        ax.set_title(title, fontsize=8, loc="left", pad=10)
        ax.set_xlabel(xlabel, fontsize=7.4, labelpad=6)
        ax.set_ylim(5.55, -.55)
        ax.grid(axis="x", color=".88", linewidth=.5)
        ax.axhline(.5, color=".65", linewidth=.6, linestyle=(0, (2, 3)))
        ax.tick_params(axis="y", length=0, labelleft=c == 0)
        ax.tick_params(axis="x", labelsize=7)
        for side in ("left", "top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(".35")
        low, high = min(extrema[c]), max(extrema[c])
        span = max(high - low, 1)
        ax.set_xlim(low - .08 * span, high + .08 * span)
        ax.xaxis.set_major_locator(MaxNLocator(3))
    axes[0].set_xlim(-5, max(115, max(extrema[0]) + 5))
    axes[0].set_xticks([0, 50, 100])
    handles = [Line2D([], [], marker=marker, color=color, linestyle="none", markersize=4.5,
                      label=label) for marker, color, label in (
                          ("o", "#27323c", "Training tracks"),
                          ("^", "#7d848a", "Held-out tracks"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.56, .99),
               ncol=2, frameon=False, fontsize=8)
    fig.text(.55, .914, "Within each split: training seeds 0, 1, 2 from top to bottom",
             ha="center", fontsize=7.2, color=".35")
    fig.subplots_adjust(left=.135, right=.985, top=.83, bottom=.16, wspace=.32)
    fig.text(.135, .025, "Each point: one training run. Whiskers: ±1 SD across 4 evaluation-seed summaries.\n"
             "Each run uses 4 tracks per split (16 attempts). The expert is shown once per split.",
             fontsize=6.8, color=".35", linespacing=1.5)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"lap-performance-seed-whiskers.{extension}", dpi=220)
    plt.close(fig)
    caption = (
        "Lap performance on four training and four held-out tracks, evaluated with four fixed "
        "evaluation seeds per track. Policy colors follow the original figure: darker circles "
        "denote training tracks and lighter triangles denote held-out tracks. Each learned policy "
        "has three independently trained runs (training seeds 0, 1, 2, top to bottom within each "
        "split); the shared expert reference is shown once per split. Each point summarizes one "
        "training run's 16 attempts. Whiskers show ±1 sample SD across four evaluation-seed "
        "summaries within that run, not confidence intervals or between-training-run SD. Completion "
        "is reported as a percentage; each evaluation-seed rate uses four tracks. Capped time "
        "weights tracks and evaluation seeds equally and assigns 150 s to failed attempts. "
        "Off-road percentage uses pooled actual measured race time, including failures and excluding "
        "countdown; its whisker is the SD of four evaluation-seed-specific exposure-weighted "
        "percentages. Rescues are averaged per attempt, with SD across evaluation-seed means. "
        "SD whiskers are not truncated at physical bounds and can extend below zero or above "
        "100% for completion. With deterministic policies, zero SD across evaluation seeds does "
        "not imply identical outcomes across tracks.\n"
    )
    (output / "lap-performance-caption.txt").write_text(caption)
    (output / "lap-performance-metrics.json").write_text(json.dumps({
        "training_seeds": [0, 1, 2],
        "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in source_paths},
        "runs": summaries,
        "caption": caption,
    }, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--lap-performance", action="store_true",
                        help="Preview original-style lap performance with a separate mean/SD for each run")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 8,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "svg.fonttype": "none"})
    groups, curves = read_inputs(args.input)
    if args.lap_performance:
        lap_performance(args.input, args.output)
        print(args.output / "lap-performance-seed-whiskers.pdf")
        return
    dots(groups, args.output)
    dots(groups, args.output, local=True)
    completion_curves(curves, args.output)
    inputs = [args.input / "seed-metrics.json", *[args.input / f"sources-s{s}.json" for s in range(3)]]
    (args.output / "preview-metadata.json").write_text(json.dumps({
        "purpose": "Exploratory figure designs, not replacement report assets",
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "curves": curves,
        "completion_curve_definition": "Fraction of all attempts finishing at most x seconds after the track/evaluation-seed-matched expert; failed attempts never count as finishes",
        "training_seeds": [0, 1, 2],
        "evaluation_seeds": [20000, 20001, 20002, 20003],
    }, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
