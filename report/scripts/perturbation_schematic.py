"""Draft five compact intervention schematics, using the frozen case parameters.

uv run --with matplotlib --with seaborn python report/scripts/perturbation_schematic.py

Geometry is illustrative, not a reconstruction of measured trajectories.
No simulations are run and nothing is uploaded to Samskrift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    Polygon,
    Rectangle,
)
from matplotlib.path import Path as DrawPath
from matplotlib.transforms import Affine2D

from kartstudy.trajectory_evolution import digest

ROOT = Path(__file__).resolve().parents[2]
INK, MUTED, ROAD, FORCE = "#27333d", "#8c99a4", "#e8edf1", "#cc791b"
CASES = (
    ("left_throttle_lighthouse", "Left + throttle", "Lighthouse"),
    ("barrier_enterprise", "Stopped at barrier", "Enterprise"),
    ("wrong_direction_cornfield", "Facing backward", "Cornfield"),
    ("black_forest_alternate_branch", "Alternate branch", "Black Forest"),
    ("delay_corner_hacienda", "Delayed controls", "Hacienda"),
)


def path(start, *segments):
    vertices, codes = [start], [DrawPath.MOVETO]
    for segment in segments:
        vertices.extend(segment)
        codes.extend([DrawPath.LINETO] if len(segment) == 1 else [DrawPath.CURVE4] * 3)
    return DrawPath(vertices, codes)


def road(ax, centerline, width=18):
    from matplotlib.patches import PathPatch

    ax.add_patch(PathPatch(centerline, fill=False, color=ROAD, lw=width,
                           capstyle="round", joinstyle="round", zorder=0))


def route(ax, centerline, *, forced=False):
    ax.add_patch(FancyArrowPatch(
        path=centerline, arrowstyle="-|>", mutation_scale=6,
        color=FORCE if forced else MUTED, lw=1.65 if forced else .9,
        linestyle="-" if forced else (0, (3, 2)), zorder=3 if forced else 1,
    ))


def kart(ax, x, y, heading):
    """Top-down kart; 0 degrees points up, positive turns left."""
    transform = Affine2D().rotate_deg(heading).translate(x, y) + ax.transData
    for wheel_x in (-.068, .042):
        for wheel_y in (-.064, .026):
            ax.add_patch(Rectangle((wheel_x, wheel_y), .026, .044,
                                   color=INK, lw=0, transform=transform, zorder=5))
    ax.add_patch(FancyBboxPatch((-.047, -.082), .094, .164,
                               boxstyle="round,pad=0,rounding_size=.015",
                               color=INK, lw=0, transform=transform, zorder=6))
    ax.add_patch(Polygon([(0, .058), (-.025, -.006), (.025, -.006)],
                         closed=True, facecolor="white", edgecolor="none",
                         transform=transform, zorder=7))


def onset(ax, x, y):
    ax.add_patch(Circle((x, y), .024, facecolor="white", edgecolor=FORCE, lw=1, zorder=4))


def left_throttle(ax):
    normal = path((.58, .13), [(.58, .91)])
    road(ax, normal)
    route(ax, normal)
    route(ax, path((.58, .18), [(.58, .52), (.49, .72), (.31, .72)],
                   [(.25, .72), (.20, .64), (.13, .54)]), forced=True)
    onset(ax, .58, .18)
    kart(ax, .13, .54, 135)


def barrier(ax):
    normal = path((.44, .12), [(.44, .45), (.50, .58), (.89, .58)])
    road(ax, normal)
    route(ax, normal)
    ax.plot([.05, .48], [.80, .80], color="#697681", lw=2, zorder=2)
    for x in np.linspace(.05, .45, 7):
        ax.plot([x, x + .045], [.80, .86], color="#a4afb7", lw=.8, zorder=2)
    route(ax, path((.44, .17), [(.44, .41), (.40, .58), (.29, .70)]), forced=True)
    onset(ax, .44, .17)
    kart(ax, .29, .70, 20)


def wrong_direction(ax):
    normal = path((.50, .12), [(.50, .92)])
    road(ax, normal)
    route(ax, normal)
    route(ax, path((.50, .16), [(.54, .40), (.60, .69), (.45, .72)],
                   [(.29, .75), (.28, .52), (.29, .41)]), forced=True)
    onset(ax, .50, .16)
    kart(ax, .29, .41, 180)


def alternate_branch(ax):
    trunk = path((.50, .10), [(.50, .43)])
    left = path((.50, .43), [(.50, .63), (.30, .68), (.17, .87)])
    right = path((.50, .43), [(.50, .63), (.70, .68), (.83, .87)])
    for line in (trunk, left, right):
        road(ax, line, width=15)
    route(ax, path((.50, .14), [(.50, .43)], [(.50, .63), (.30, .68), (.17, .87)]))
    route(ax, path((.50, .17), [(.50, .43)], [(.50, .60), (.64, .65), (.75, .78)]), forced=True)
    onset(ax, .50, .17)
    kart(ax, .75, .78, -40)


def delayed_controls(ax):
    normal = path((.40, .12), [(.40, .43)], [(.40, .64), (.55, .68), (.88, .68)])
    road(ax, normal, width=19)
    route(ax, normal)
    route(ax, path((.40, .17), [(.40, .51)], [(.40, .64), (.50, .70), (.60, .75)]), forced=True)
    onset(ax, .40, .17)
    kart(ax, .60, .75, -40)
    ax.add_patch(Circle((.12, .77), .073, fill=False, edgecolor=FORCE, lw=1))
    ax.plot([.12, .12, .157], [.825, .77, .748], color=FORCE, lw=1, solid_capstyle="round")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=ROOT / "runs/five-disturbances-v1/evaluation")
    parser.add_argument("--output", type=Path, default=ROOT / "report/generated/perturbation-schematic-draft")
    args = parser.parse_args()
    manifest = json.loads((args.bank / "manifest.json").read_text())
    sources = []
    references = []
    for case, _, _ in CASES:
        entry, = (r for r in manifest["trials"] if r["id"] == case + "--expert")
        reference = args.bank / entry["result"]
        if digest(reference) != entry["result_sha256"]:
            raise ValueError("Disturbance reference changed")
        references.append(json.loads(reference.read_text()))
        sources.append(entry)
    duration = [sum(p["steps"] for p in r["scenario"]["pulses"]) * .1 for r in references]
    delay = references[-1]["scenario"]["control_delay"]
    footers = (
        f"Full left + full throttle\nfor {duration[0]:g} s",
        f"Left + throttle, {duration[1]:g} s\nNearly stopped at release",
        f"Brake → turn → brake\n{duration[2]:g} s, ≈180° reversal",
        f"Force the other route\nfor {duration[3]:g} s",
        f"{delay['steps'] * .1:g} s actuator lag\napplied for {delay['duration_steps'] * .1:g} s",
    )
    sns.set_theme(context="paper", style="ticks", palette="colorblind")
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(17 / 2.54, 2.25))
    drawings = (left_throttle, barrier, wrong_direction, alternate_branch, delayed_controls)
    for index, ((_, title, track), draw, footer) in enumerate(zip(CASES, drawings, footers, strict=True)):
        center = (index + .5) / 5
        ax = fig.add_axes([index / 5 + .006, .27, .188, .55])
        ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), aspect="equal")
        ax.set_axis_off()
        draw(ax)
        fig.text(center, .975, title, ha="center", va="top", fontsize=8.1, fontweight="bold")
        fig.text(center, .875, track, ha="center", va="top", fontsize=7, color="#6f7c87")
        fig.text(center, .235, footer, ha="center", va="top", fontsize=6.9, linespacing=1.4)
    fig.legend(handles=[
        Line2D([], [], color=MUTED, lw=.9, ls=(0, (3, 2)), label="Intended route"),
        Line2D([], [], color=FORCE, marker="o", mfc="white", ms=3, ls="none", label="Intervention onset"),
        Line2D([], [], color=FORCE, lw=1.6, label="Applied disturbance"),
        Patch(facecolor=INK, label="Kart at control restoration"),
    ], loc="lower center", bbox_to_anchor=(.5, .003), ncols=4, frameon=False,
       fontsize=6.4, handlelength=1.5, handletextpad=.5, columnspacing=1.1)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(args.output / f"five-perturbations.{extension}", dpi=150, facecolor="white")
    plt.close(fig)
    (args.output / "provenance.json").write_text(json.dumps({
        "type": "conceptual schematic; illustrative geometry, not measured trajectories",
        "case_references": sources, "script_sha256": digest(__file__),
        "duration_seconds": dict(zip((c[0] for c in CASES), duration, strict=True)),
        "control_delay": delay,
    }, indent=2))
    (args.output / "caption.txt").write_text(
        "Schematic of the 5 physical interventions. Gray dashed arrows indicate "
        "the intended route, orange shows the intervention, open circles mark its "
        "onset, and dark karts show the pose when normal control returns. "
        "The backward-facing case uses 1 s braking, 3.5 s full-left steering at "
        "50% throttle, and 0.5 s braking. The alternate route is a legal branch. "
        "Latency delays requested controls by 0.6 s for 2 s, then discards the "
        "queued commands. Geometry and paths are illustrative and not to scale; "
        "they do not depict policy-specific recovery trajectories.\n"
    )
    print(args.output / "five-perturbations.png")


if __name__ == "__main__":
    main()
