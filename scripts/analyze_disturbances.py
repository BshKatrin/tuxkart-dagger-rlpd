"""Summarize and plot a completed kartstudy.disturbance evaluation bank.

uv run --with matplotlib python scripts/analyze_disturbances.py RUN_DIRECTORY
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from kartstudy.playback import METHODS
from kartstudy.trajectory_evolution import COLORS, digest, read_quads


def analyze(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Evaluation bank must be completed")
    rows = []
    for item in manifest["trials"]:
        for field, checksum in [
            ("result", "result_sha256"),
            ("trajectory", "trajectory_sha256"),
        ]:
            if digest(root / item[field]) != item[checksum]:
                raise ValueError("Evaluation artifact hash mismatch")
        row = json.loads((root / item["result"]).read_text())
        row["trajectory"] = str(root / item["trajectory"])
        rows.append(row)
    summary = []
    for scenario in manifest["scenarios"]:
        for method in ["expert", "scripted", "bc", "dagger", "sac", "sac_sym", "rlpd"]:
            group = [
                r
                for r in rows
                if r["scenario"]["id"] == scenario["id"] and r["controller"] == method
            ]
            if not group:
                continue
            summary.append(
                {
                    "scenario": scenario["id"],
                    "method": method,
                    "n": len(group),
                    "recovered_within_15s": sum(
                        r["recovered_within_15s"] for r in group
                    ),
                    "ever_recovered_without_prior_rescue": sum(
                        r["recovery_seconds"] is not None for r in group
                    ),
                    "completed": sum(r["success"] for r in group),
                    "completed_rescue_free": sum(
                        r["success"] and r["native_rescues_after_handover"] == 0
                        for r in group
                    ),
                    "mean_rescues": float(
                        np.mean([r["native_rescues_after_handover"] for r in group])
                    ),
                    "conditional_lap_time": float(
                        np.mean([r["lap_time"] for r in group if r["success"]])
                    )
                    if any(r["success"] for r in group)
                    else None,
                }
            )
    with (root / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    # Behavior is illustrated with the predeclared seed 0; all three seeds enter the summary.
    plot(root, manifest, rows)
    return summary


def plot(root, manifest, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from platformdirs import user_cache_dir

    for scenario in manifest["scenarios"]:
        if not scenario.get("witness") and not scenario.get("visualize"):
            continue
        group = [
            r
            for r in rows
            if r["scenario"]["id"] == scenario["id"]
            and (r["policy"] is None or r["policy"]["config"]["seed"] == 0)
        ]
        fig, axes = plt.subplots(2, 4, figsize=(13, 7))
        quads = read_quads(
            Path(user_cache_dir("PySuperTuxKart2", "bpiwowar"))
            / "1.5/data/tracks"
            / scenario["track"]
            / "quads.xml"
        )
        anchor = np.asarray(group[0]["anchor"]["position"])[[0, 2]]
        handover = np.asarray(group[0]["handover"]["position"])[[0, 2]]
        midpoint = (anchor + handover) / 2
        radius = max(35, float(np.linalg.norm(anchor - handover)) * 0.75)
        lower, upper = midpoint - radius, midpoint + radius
        for ax, row in zip(axes.flat, group):
            method = row["controller"]
            color = COLORS.get(method, "#27323c")
            with np.load(row["trajectory"]) as trace:
                pos = trace["world_position"][:, [0, 2]].astype(float)
                phase = trace["phase"]
                active = trace["rescue_active"]
                count = trace["rescue_count"]
                response = np.flatnonzero(phase == "response")
                first = response[0]
                relative_time = (
                    trace["elapsed_seconds"] - trace["elapsed_seconds"][first] + 0.1
                )
                restoration = row.get("recovery_start_seconds", 0)
                delayed = trace.get("delay_active", np.zeros(len(phase), dtype=bool))
                shown = (phase == "response") & (relative_time <= 15 + restoration)
                driven = pos.copy()
                driven[active] = np.nan
                driven[np.diff(count, prepend=0) > 0] = np.nan
                ax.add_collection(
                    PolyCollection(quads, facecolors="#e6e9ee", edgecolors="none")
                )
                ax.plot(
                    *driven[(phase == "disturbance") | delayed].T,
                    color="#7d8790",
                    ls=":",
                    lw=1.2,
                )
                ax.plot(*driven[shown & ~delayed].T, color=color, lw=1.5)
                starts = (np.diff(count, prepend=0) > 0) & shown
                if starts.any():
                    ax.plot(*pos[starts].T, ls="none", marker="X", color=color, ms=5)
            ax.plot(*handover, marker="o", color=color, mfc="white", ms=5, zorder=5)
            ax.plot(*anchor, marker="|", color="#27323c", ms=8, zorder=5)
            title = (
                "Expert"
                if method == "expert"
                else (
                    "Scripted witness → expert"
                    if method == "scripted"
                    else METHODS[method] + " · seed 0"
                )
            )
            ax.set_title(title, fontsize=10)
            recovery = row["recovery_seconds"]
            label = (
                "No independent recovery"
                if recovery is None
                else f"Recovery: {recovery:.1f} s"
            )
            if restoration:
                label += " after control restoration"
            label += (
                "\n"
                + ("Lap finished" if row["success"] else "Lap timed out")
                + f" · {row['native_rescues_after_handover']} rescues"
            )
            ax.text(
                0.5,
                -0.05,
                label,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
            )
            ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal")
            ax.set_axis_off()
        for ax in list(axes.flat)[len(group) :]:
            ax.set_visible(False)
        fig.suptitle(
            scenario["track"]
            + " · "
            + scenario.get("label", "controlled disturbance")
            + (
                " · delay + first 15 seconds after control restoration"
                if scenario.get("control_delay")
                else " · first 15 seconds after handover"
            ),
            fontsize=12,
        )
        fig.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    ls=":",
                    color="#7d8790",
                    label="Disturbance / delayed controls",
                ),
                Line2D(
                    [],
                    [],
                    marker="o",
                    ls="none",
                    mfc="white",
                    color="#27323c",
                    label="Handover",
                ),
                Line2D(
                    [],
                    [],
                    marker="|",
                    ls="none",
                    color="#27323c",
                    label="Disturbance start",
                ),
            ],
            loc="lower center",
            ncols=3,
            frameon=False,
            fontsize=9,
        )
        fig.subplots_adjust(
            top=0.89, bottom=0.12, left=0.02, right=0.98, hspace=0.4, wspace=0.12
        )
        for suffix in ["png", "pdf"]:
            fig.savefig(
                root / f"{scenario['track']}-recovery.{suffix}",
                dpi=160,
                facecolor="white",
            )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for row in analyze(args.root):
        print(
            row["scenario"],
            row["method"],
            f"recovery {row['recovered_within_15s']}/{row['n']}",
            f"completion {row['completed']}/{row['n']}",
        )


if __name__ == "__main__":
    main()
