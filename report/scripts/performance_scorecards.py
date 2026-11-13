"""Render report-width performance figures from one row per policy, seed and split.

    uv run --with matplotlib python report/scripts/performance_scorecards.py SEED_METRICS
    uv run --with matplotlib python report/scripts/performance_scorecards.py SEED_METRICS --seed-bars

The input groups rows under "Training" and "Held out". Each group contains a
single expert reference and one row for each of seeds 0–2 per learned policy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

from aggregate_performance import LABELS, METHODS
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

OUTPUT = Path(__file__).resolve().parents[1] / "generated/performance-scorecards"
WIDTH = 17 / 2.54 * 72
INK = "#20252e"
MUTED = "#4c5662"
RULE = "#d5dbe1"
EXPERT = "#edf0f3"
SLOW = "#cf7436"
FAST = "#4890ad"


@dataclass(frozen=True)
class Metric:
    field: str
    title: str
    units: str
    precision: int
    multiplier: int = 1
    signed: bool = False
    color_scale: float = 0

    def format(self, value):
        value = round(value * self.multiplier, self.precision)
        sign = "+" if self.signed and value else ""
        return format(value or 0, f"{sign}.{self.precision}f").replace("-", "−")


GLOBAL = (
    Metric("completion", "Lap completion", "% of attempts", 0, color_scale=55),
    Metric("capped_gap", "Capped time", "s vs. expert", 1, signed=True, color_scale=60),
    Metric("offroad", "Off-road time", "% of race time", 1),
    Metric("rescues", "Rescues", "per attempt", 2),
)
LOCAL = (
    Metric(
        "gap",
        "Local time gap",
        "ms / 20 m vs. expert",
        0,
        multiplier=1000,
        signed=True,
        color_scale=120,
    ),
    Metric("offset", "Lateral offset", "% of road half-width", 1),
    Metric("length", "Trajectory length", "% of expert", 1),
)


def cell_fill(metric, average, reference):
    if not metric.color_scale:
        return "white"
    delta = average - reference
    if metric.field == "completion":
        delta = -delta
    if abs(delta) < 0.5 / (10**metric.precision * metric.multiplier):
        return "white"
    hue = SLOW if delta > 0 else FAST
    fraction = min(1, abs(delta) / metric.color_scale)
    opacity = 0.09 + 0.30 * fraction
    return tuple(1 - opacity * (1 - channel) for channel in to_rgb(hue))


def policy_rows(rows, method, attempts):
    entries = [row for row in rows if row["method"] == method]
    expected_seeds = [None] if method == "expert" else [0, 1, 2]
    if [row["seed"] for row in entries] != expected_seeds:
        raise ValueError(f"Unexpected training seeds for {method}")
    if any(row["attempts"] != attempts for row in entries):
        raise ValueError(f"Mismatched evaluation budget for {method}")
    return entries


def plot(groups, metrics, destination):
    height = 345
    fig = plt.figure(figsize=(WIDTH / 72, height / 72), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(height, 0))
    ax.set_axis_off()
    left, right, label_width = 4, WIDTH - 4, 80
    col_width = (right - label_width) / len(metrics)

    def label(x, y, content, *, size=10, color=INK, weight="normal", align="left"):
        ax.text(
            x,
            y,
            content,
            va="center",
            ha=align,
            fontsize=size,
            color=color,
            fontweight=weight,
        )

    def rule(y, width=0.6):
        ax.hlines(y, left, right, color=RULE, linewidth=width, zorder=3)

    for index, metric in enumerate(metrics):
        x = label_width + (index + 0.5) * col_width
        label(x, 9, metric.title, size=10, weight="bold", align="center")
        label(x, 23, metric.units, size=8.4, color=MUTED, align="center")
    rule(32, 0.9)

    for split_index, split in enumerate(("Training", "Held out")):
        rows = groups[split]
        (reference,) = (row for row in rows if row["method"] == "expert")
        attempts = reference["attempts"]
        top = 34 + split_index * 148
        split_title = "Training tracks" if split == "Training" else "Held-out tracks"
        label(left + 4, top + 9, split_title, size=9.5, weight="bold")
        label(
            right - 4,
            top + 9,
            f"{attempts} attempts per run",
            size=8.4,
            color=MUTED,
            align="right",
        )

        row_top = top + 18
        for method, name in zip(METHODS, LABELS, strict=True):
            entries = policy_rows(rows, method, attempts)
            is_expert = method == "expert"
            row_height = 16 if is_expert else 21.5
            y = row_top + row_height / 2
            if is_expert:
                ax.add_patch(
                    Rectangle(
                        (left, row_top),
                        right - left,
                        row_height,
                        facecolor=EXPERT,
                        edgecolor="none",
                    )
                )
            label(
                left + 4,
                y,
                name,
                size=9.7,
                weight="bold" if is_expert else "normal",
            )

            for index, metric in enumerate(metrics):
                cell_x = label_width + index * col_width
                x = cell_x + col_width / 2
                values = [row[metric.field] for row in entries]
                average = mean(values)
                low, high = metric.format(min(values)), metric.format(max(values))
                has_range = low != high
                if not is_expert:
                    ax.add_patch(
                        Rectangle(
                            (cell_x + 1, row_top + 0.7),
                            col_width - 2,
                            row_height - 1.4,
                            facecolor=cell_fill(
                                metric, average, reference[metric.field]
                            ),
                            edgecolor="none",
                        )
                    )
                label(
                    x,
                    y - 4.3 if has_range else y,
                    metric.format(average),
                    size=10,
                    weight="bold",
                    align="center",
                )
                if has_range:
                    label(
                        x,
                        y + 5.5,
                        f"{low} to {high}",
                        size=7.6,
                        color=MUTED,
                        align="center",
                    )
            row_top += row_height
            if not is_expert:
                rule(row_top, 0.4)

    note = "Mean of 3 training runs. Smaller text shows the observed range when visible at this precision."
    label(left + 4, height - 7, note, size=8, color=MUTED)

    fig.savefig(destination, format="pdf")
    plt.close(fig)


def plot_seed_completion(groups, destination, *, capped_time=False):
    """Pair training/held-out completion for each independently trained policy."""
    import numpy as np
    import seaborn as sns

    sns.set_theme(context="paper", style="ticks", palette="colorblind")
    colors = sns.color_palette("colorblind", 3)
    fig, ax = plt.subplots(figsize=(WIDTH / 72 * (.6 if capped_time else 1),
                                    2.3 if capped_time else 2.9))
    fig.subplots_adjust(left=.09, right=.985, top=.77, bottom=.19)
    positions = np.arange(len(METHODS) - 1)
    expert_rates = []
    for seed, color in enumerate(colors):
        for split_index, split in enumerate(("Training", "Held out")):
            rows = groups[split]
            (expert,) = [row for row in rows if row["method"] == "expert"]
            expert_rates.append(expert["capped"] if capped_time else
                                100 * expert["completed"] / expert["attempts"])
            entries = [policy_rows(rows, method, expert["attempts"])[seed]
                       for method in METHODS[1:]]
            rates = [row["capped"] if capped_time else
                     100 * row["completed"] / row["attempts"] for row in entries]
            if any(not np.isfinite(rate) or not 0 <= rate <= (150 if capped_time else 100)
                   or (not capped_time and not np.isclose(row["completion"], rate))
                   for row, rate in zip(entries, rates, strict=True)):
                raise ValueError("Invalid completion percentage")
            shade = np.asarray(color)
            if split_index:
                shade = .40 * shade + .60
            offset = (seed * 2 + split_index - 2.5) * .135
            split_label = "training" if split_index == 0 else "held out"
            uncertainty = [row["capped_sd"] for row in entries] if capped_time else None
            ax.bar(positions + offset, rates, width=.125, color=shade,
                   yerr=uncertainty, capsize=1.8,
                   error_kw={"elinewidth": .65, "capthick": .65, "ecolor": "#888888"},
                   edgecolor="none", label=f"Seed {seed}: {split_label}")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=RULE, lw=.5)
    if capped_time:
        upper = max(row["capped"] + row["capped_sd"]
                    for rows in groups.values() for row in rows)
        for split_index, (split, style) in enumerate((("training", (0, (4, 3))),
                                                     ("held out", (0, (1, 3))))):
            reference = expert_rates[split_index]
            ax.axhline(reference, color=MUTED, ls=style, lw=.8,
                       label=f"Expert: {split}")
        ax.set_ylabel("Mean capped time (s) ↓", fontsize=7)
        ax.set_ylim(0, max(150, 30 * np.ceil(upper / 30)))
        ax.set_yticks(np.arange(0, ax.get_ylim()[1] + 1, 30))
    else:
        if not np.allclose(expert_rates, expert_rates[0]):
            raise ValueError("Expected the same expert completion on both splits")
        ax.axhline(expert_rates[0], color=MUTED, ls=(0, (4, 3)), lw=.8)
        ax.text(.995, expert_rates[0] + 2, "Expert", ha="right", va="bottom",
                transform=ax.get_yaxis_transform(), fontsize=7, color=MUTED)
        ax.set_ylabel("Completion rate (%)", fontsize=8)
        ax.set_ylim(0, 113)
        ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Policy", fontsize=7 if capped_time else 8)
    ax.set_xticks(positions, LABELS[1:])
    ax.tick_params(axis="both", labelsize=7 if capped_time else 8, length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    handles, labels = ax.get_legend_handles_labels()
    if capped_time:
        order = sorted(range(len(labels)), key=lambda i: labels[i].startswith("Expert:"))
        handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    fig.legend(handles, labels, loc="upper center", ncol=4 if capped_time else 3,
               frameon=False, fontsize=6.5 if capped_time else 7.5,
               handlelength=1.4 if capped_time else 2,
               columnspacing=.8 if capped_time else 2.5)
    for extension in ("pdf", "png"):
        fig.savefig(destination.with_suffix(f".{extension}"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_metrics", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--seed-bars",
        action="store_true",
        help="Plot completion rates by training seed for both track splits.",
    )
    parser.add_argument("--capped-time-bars", action="store_true",
                        help="Plot mean capped time by training seed for both track splits.")
    args = parser.parse_args()
    groups = json.loads(args.seed_metrics.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    if args.capped_time_bars:
        plot_seed_completion(groups, args.output / "capped-time-by-training-seed.pdf",
                             capped_time=True)
        return
    if args.seed_bars:
        plot_seed_completion(
            groups, args.output / "completion-by-training-seed.pdf"
        )
        return
    plot(groups, GLOBAL, args.output / "global-performance-scorecard.pdf")
    plot(groups, LOCAL, args.output / "local-performance-scorecard.pdf")


if __name__ == "__main__":
    main()
