"""Plot the 4 per-seed expert observations in 3 narrow, aligned panels.

Run from the repository root:
    uv run --with seaborn --with matplotlib python report/scripts/expert_track_quality.py \
        --input runs/expert-road-quality-final/metrics.json

Writes a vector PDF and a PNG preview to report/generated/. No simulator runs are performed.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

NAMES = {
    "abyss": "Antediluvian Abyss",
    "black_forest": "Black Forest",
    "candela_city": "Candela City",
    "cocoa_temple": "Cocoa Temple",
    "cornfield_crossing": "Cornfield Crossing",
    "fortmagma": "Fort Magma",
    "gran_paradiso_island": "Gran Paradiso Island",
    "hacienda": "Hacienda",
    "lighthouse": "Around the Lighthouse",
    "mines": "Old Mine",
    "minigolf": "Minigolf",
    "olivermath": "Oliver's Math Class",
    "ravenbridge_mansion": "Ravenbridge Mansion",
    "sandtrack": "Shifting Sands",
    "scotland": "Nessie's Pond",
    "snowmountain": "Northern Resort",
    "snowtuxpeak": "Snow Peak",
    "stk_enterprise": "STK Enterprise",
    "volcano_island": "Volcan Island",
    "xr591": "XR591",
    "zengarden": "Zen Garden",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report/generated/expert-track-quality-lap-time.pdf"),
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if not data["complete"] or len(data["tracks"]) != 21:
        raise ValueError("Expected a completed 21-track evaluation")
    tracks = []
    for track in data["tracks"]:
        episodes = sorted(track["per_seed"], key=lambda episode: episode["seed"])
        if len(episodes) != 4 or len({e["seed"] for e in episodes}) != 4:
            raise ValueError("Expected 4 distinct evaluation seeds per track")
        if not all(e["success"] for e in episodes):
            raise ValueError(
                "This figure assumes complete laps for comparable exposure"
            )
        values = {}
        for key in (
            "harmful_contact_proportion",
            "off_track_fraction",
            "lap_time_seconds",
        ):
            raw = [e[key] for e in episodes]
            if all(x is None for x in raw):
                values[key] = None
            elif any(x is None for x in raw):
                raise ValueError("Mixed missing measurements within a track")
            else:
                raw = np.asarray(raw) * (1 if key == "lap_time_seconds" else 100)
                values[key] = {
                    "seeds": [episode["seed"] for episode in episodes],
                    "values": raw.tolist(),
                    "mean": float(raw.mean()),
                    "sd": float(raw.std(ddof=1)),
                    "n": len(raw),
                }
        tracks.append(
            {"track": track["track"], "name": NAMES[track["track"]], **values}
        )
    tracks.sort(key=lambda t: (-t["off_track_fraction"]["mean"], t["name"]))
    sns.set_theme(context="paper", style="ticks", palette="colorblind")
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.6,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(17 / 2.54, 12 / 2.54), sharey=True)
    fig.subplots_adjust(left=0.285, right=0.985, top=0.935, bottom=0.115, wspace=0.22)
    for ax, key, title, xlabel, color in zip(
        axes,
        ("harmful_contact_proportion", "off_track_fraction", "lap_time_seconds"),
        ("(a) Harmful contacts", "(b) Time off-track", "(c) Lap time"),
        ("Contacts / rows (%)", "Race time (%)", "Time (s)"),
        sns.color_palette("colorblind")[:3],
    ):
        for row, track in enumerate(tracks):
            value = track[key]
            if value is None:
                ax.text(0, row, "N/A", va="center", ha="left", fontsize=8, color="0.4")
            else:
                # Keep all observations on the track row; equal values overlap.
                ax.scatter(
                    value["values"],
                    np.full(4, row),
                    s=4.2**2,
                    color=color,
                    edgecolors="white",
                    linewidths=0.25,
                    zorder=3,
                )
                if key == "lap_time_seconds":
                    ax.plot(
                        value["mean"],
                        row,
                        marker="|",
                        markersize=6,
                        color="black",
                        markeredgewidth=1.1,
                        zorder=4,
                    )
                    if track["track"] == "black_forest":
                        ax.annotate(
                            f"{value['mean']:.1f} s",
                            (value["mean"], row),
                            xytext=(-7, 0),
                            textcoords="offset points",
                            fontsize=7.5,
                            ha="right",
                            va="center",
                        )
        if key == "harmful_contact_proportion":
            ax.set_xlim(-4, 104)
        else:
            maximum = max(max(t[key]["values"]) for t in tracks)
            ax.set_xlim(
                -maximum * 0.04, maximum * (1.28 if key == "lap_time_seconds" else 1.08)
            )
        ax.set_yticks(range(len(tracks)))
        ax.set_ylim(len(tracks) - 0.5, -0.5)
        ax.axvline(0, color="0.7", linewidth=0.6, zorder=0)
        ax.grid(axis="x", color="0.92", linewidth=0.5)
        ax.grid(axis="y", color="0.91", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=2, width=0.6)
        ax.set_title(title, loc="left", pad=6)
        ax.set_xlabel(xlabel, labelpad=5)
        sns.despine(ax=ax, left=True)
    axes[0].set_yticklabels([track["name"] for track in tracks])
    axes[0].set_ylabel("Track", labelpad=5)
    axes[0].set_xticks([0, 50, 100])
    axes[1].set_xticks([0, 10, 20])
    axes[2].set_xticks([0, 60, 120])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor="white")
    if args.output.suffix.lower() == ".pdf":
        fig.savefig(args.output.with_suffix(".png"), dpi=150, facecolor="white")
    plt.close(fig)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": str(args.input),
                "statistic": "4 seed observations; no offsets, identical values overlap; rows ordered by mean off-track time",
                "tracks": tracks,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
