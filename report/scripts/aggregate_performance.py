"""Plot retained-controller evaluations without changing the report or raw data.

From the repository root:
    uv run --with matplotlib --with seaborn python report/scripts/aggregate_performance.py

Input: report/generated/track-time-loss/sources.json.
Output: PDF, 150-dpi PNG, metrics JSON and proposed caption under
report/generated/aggregate-performance/. Failed attempts receive a 150 s score;
actual measured exposure is retained for off-road percentages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from track_time_loss import aggregate_episodes

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("expert", "bc", "dagger", "sac", "sac_sym", "rlpd")
LABELS = ("Expert", "BC", "DAgger", "SAC", "SAC + SS", "RLPD")
CAP = 150.0


def summarize(source):
    seeds = source["seeds"]
    failed = [
        e for methods in source["sources"].values()
        for entry in methods.values() for e in entry["episodes"] if not e["success"]
    ]
    recorded_horizon = failed[0]["sim_seconds"]
    groups = {}
    for split in ("All tracks", "Training", "Held out"):
        tracks = [t["id"] for t in source["tracks"]
                  if split == "All tracks" or t["split"] == split]
        groups[split] = {}
        for method in METHODS:
            records = {t: source["sources"][t][method]["episodes"] for t in tracks}
            # Existing aggregation validates seeds, finish times and actual timeouts.
            row = aggregate_episodes(records, seeds, recorded_horizon)
            by_seed = [[e for episodes in records.values() for e in episodes
                        if e["seed"] == seed] for seed in seeds]
            capped = [float(np.mean([e["lap_time"] if e["success"] else CAP
                                     for e in episodes])) for episodes in by_seed]
            expected = row["capped_time_mean"] + (
                CAP - recorded_horizon
            ) * (1 - row["completed"] / row["attempts"])
            assert np.isclose(np.mean(capped), expected)
            row.update(
                capped_time_mean=float(np.mean(capped)),
                capped_time_sd=float(np.std(capped, ddof=1)),
                capped_time_by_evaluation_seed=capped,
                offroad_pct_by_evaluation_seed=[
                    100 * sum(e["off_track_seconds"] for e in episodes)
                    / sum(e["measurement_seconds"] for e in episodes)
                    for episodes in by_seed
                ],
                rescues_per_attempt_by_evaluation_seed=[
                    float(np.mean([e["rescue_count"] for e in episodes]))
                    for episodes in by_seed
                ],
            )
            groups[split][method] = row
    return {"failure_score_seconds": CAP,
            "recorded_timeout_seconds": recorded_horizon,
            "evaluation_seeds": seeds, "groups": groups}


def plot(metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    sns.set_theme(context="paper", style="ticks", palette="colorblind")
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    from matplotlib.colors import to_rgb

    palette = sns.color_palette("colorblind")
    policy_colors = ["#27323c", *(palette[i] for i in (0, 1, 2, 4, 3))]
    fig, axes = plt.subplots(1, 4, figsize=(17 / 2.54, 8.6 / 2.54),
                             sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 2.1, 1.5, 1.5]})
    labels, time, road, rescue = axes
    panels = (time, road, rescue)
    mean_keys = ("capped_time_mean", "offroad_pct", "rescues_per_attempt")
    raw_keys = ("capped_time_by_evaluation_seed",
                "offroad_pct_by_evaluation_seed",
                "rescues_per_attempt_by_evaluation_seed")
    for i, (method, name, base) in enumerate(zip(
            METHODS, LABELS, policy_colors, strict=True)):
        labels.text(0, i, name, ha="left", va="center", fontsize=8, color=base)
        for j, (split, marker) in enumerate(zip(
                ("Training", "Held out"), ("o", "^"), strict=True)):
            color = np.array(to_rgb(base))
            if j:
                color = .60 * color + .40  # Lighter opaque shade, also visible in PDF.
            row = metrics["groups"][split][method]
            y = i + (-.16, .16)[j]
            for ax, mean_key, raw_key in zip(panels, mean_keys, raw_keys, strict=True):
                sd = float(np.std(row[raw_key], ddof=1))
                row[raw_key.removesuffix("_by_evaluation_seed") + "_sd"] = sd
                ax.errorbar(row[mean_key], y, xerr=sd, fmt=marker,
                            markersize=4.5, color=color, elinewidth=.9,
                            capsize=2, capthick=.8, zorder=4)
            labels.text(1, y, f"{row['completed']}/{row['attempts']}",
                        ha="right", va="center", fontsize=7.5, color=color)
    labels.set_xlim(0, 1.04)
    labels.set_title("Policy / finished", fontsize=8, loc="left", pad=12)
    labels.set_axis_off()
    for ax, title, xlabel in zip(
            panels,
            ("(a) Capped time", "(b) Off-road time", "(c) Native rescues"),
            ("Time (s)", "Race time (%)", "Rescues / attempt"), strict=True):
        ax.set_title(title, fontsize=8.5, loc="left", pad=12)
        ax.set_xlabel(xlabel, labelpad=7)
        ax.set_ylim(5.55, -.55)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="x", color=".88", linewidth=.5)
        ax.axhline(.5, color=".65", linewidth=.6, linestyle=(0, (2, 3)))
        ax.tick_params(axis="y", length=0, labelleft=False)
        sns.despine(ax=ax, left=True)
        ax.margins(x=.10)
    handles = [Line2D([], [], marker=m, color=c, linestyle="none", markersize=5,
                      label=label) for m, c, label in zip(
                          ("o", "^"), ("#27323c", "#7d848a"),
                          ("Training tracks", "Held-out tracks"), strict=True)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1),
               ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.025, right=.98, bottom=.18, top=.81, wspace=.27)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"aggregate-performance.{extension}", dpi=150)
    plt.close(fig)


CAPTION = """Global driving performance of the retained controllers on 4 training and
4 held-out tracks, with 4 evaluation seeds per track. Policy colors are consistent
across panels; darker circles denote training tracks and lighter triangles denote
held-out tracks. Whiskers show ±1 sample SD across 4 evaluation-seed summaries,
not confidence intervals or variability across independent training runs.
Capped time averages tracks equally within each seed, then averages seeds;
failed laps receive 150 s. Completion counts appear beside policy names.
Off-road markers show the percentage of pooled actual measured race time
(including failed attempts and excluding countdown); their whiskers show the
SD of the 4 seed-specific off-road percentages. Rescues are averaged per
attempt, with SD across seed-specific means. SD whiskers may extend below
zero for nonnegative quantities. Differences between training and held-out
performance reveal the retained controllers' generalization behavior.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path,
                        default=ROOT / "report/generated/track-time-loss/sources.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "report/generated/aggregate-performance")
    args = parser.parse_args()
    source = json.loads(args.sources.read_text())
    metrics = summarize(source)
    metrics["source_sha256"] = hashlib.sha256(args.sources.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    plot(metrics, args.output)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (args.output / "caption.txt").write_text(CAPTION)
    print(args.output)


if __name__ == "__main__":
    main()
