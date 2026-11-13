"""Draw the study's Cartesian geometric observations (schematic, not track data).

Run: uv run --with matplotlib python report/scripts/states_schema.py
Writes states-schema.pdf and states-schema.png under report/generated/.

Semantics checked against pystk2-gymnasium 0.10.0 envs.py and native
PySuperTuxKart2 0.7.4 DriveNode / Track bindings. In particular, center_path
is the line-to-kart offset in this wrapper, despite its field name.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
        }
    )
    ink, muted, blue, purple, green = (
        "#202e3a",
        "#5b6873",
        "#286a96",
        "#7953a3",
        "#36805c",
    )
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_axes((0.025, 0.11, 0.50, 0.78))
    ax.set(xlim=(-0.4, 7.1), ylim=(-0.2, 14.4), aspect="equal")
    ax.axis("off")
    fig.text(
        0.06, 0.955, "Geometric observations", fontsize=21, weight="bold", color=ink
    )
    fig.text(
        0.06,
        0.915,
        "Current driveline cell + 7 successor cells",
        fontsize=12,
        color=muted,
    )

    # Shared entry/exit edges make adjacent endpoint midpoints coincide.
    centers = np.array(
        [
            [2, 0.6],
            [2, 3.5],
            [2.3, 5.4],
            [3.1, 7.0],
            [4.2, 8.1],
            [5.1, 9.5],
            [5.4, 11.0],
            [5.1, 12.35],
            [4.3, 13.6],
        ]
    )
    tangents = np.gradient(centers, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    widths = np.array([2.5, 2.8, 2.5, 2.6, 2.3, 2.4, 2.2, 2.5, 2.7])
    left = centers + normals * widths[:, None] / 2
    right = centers - normals * widths[:, None] / 2
    for i in range(8):
        quad = np.array([left[i], right[i], right[i + 1], left[i + 1]])
        ax.add_patch(
            Polygon(
                quad,
                facecolor="#dcebf3" if i == 0 else ("#edf3f6" if i % 2 else "#e2edf3"),
                edgecolor="#aabcc8",
                lw=1,
            )
        )
        ax.plot(*centers[i : i + 2].T, color=blue, lw=2.2, zorder=3)
        mid = centers[i : i + 2].mean(axis=0)
        label = mid + normals[i] * (widths[i] / 2 + 0.25)
        ax.text(*label, str(i), color=muted, ha="center", va="center", fontsize=10)
    ax.scatter(*centers.T, s=28, c="white", edgecolors=blue, linewidths=1.4, zorder=4)
    for i, label, offset in [
        (0, "$S_0$", (0.15, -0.35)),
        (1, "$E_0=S_1$", (0.22, -0.12)),
        (2, "$E_1=S_2$", (0.22, -0.12)),
        (8, "$E_7$", (0.12, 0.25)),
    ]:
        ax.text(*(centers[i] + offset), label, color=blue, fontsize=11, zorder=5)

    kart = np.array([1.25, 2.0])
    projection = np.array([2.0, 2.0])
    # Top-down kart with its reference origin explicitly marked.
    ax.add_patch(
        FancyBboxPatch(
            (kart[0] - 0.25, kart[1] - 0.46),
            0.5,
            0.92,
            boxstyle="round,pad=0.03,rounding_size=0.14",
            facecolor="#c45143",
            edgecolor="#89362d",
            zorder=6,
        )
    )
    for dx in [-0.35, 0.25]:
        for dy in [-0.3, 0.18]:
            ax.add_patch(
                FancyBboxPatch(
                    (kart[0] + dx, kart[1] + dy),
                    0.1,
                    0.23,
                    boxstyle="round,pad=0.01",
                    facecolor=ink,
                    edgecolor=ink,
                    zorder=5,
                )
            )
    ax.scatter(*kart, s=15, color="white", zorder=8)
    ax.scatter(*projection, s=22, color=purple, zorder=7)
    ax.annotate(
        "",
        xy=kart,
        xytext=projection,
        arrowprops={"arrowstyle": "-|>", "color": purple, "lw": 2},
        zorder=8,
    )
    ax.text(2.2, 2.0, "$\\mathbf{c}$", color=purple, fontsize=14, va="center")
    ax.annotate(
        "",
        xy=kart + [-0.12, 1.25],
        xytext=kart,
        arrowprops={"arrowstyle": "-|>", "color": green, "lw": 2},
        zorder=8,
    )
    ax.text(0.48, 2.9, "$\\mathbf{v}$", color=green, fontsize=14)
    # Width at either end is shown separately: the observation averages them.
    for j, label in [(4, "$a_i$"), (5, "$b_i$")]:
        ax.annotate(
            "",
            xy=left[j],
            xytext=right[j],
            arrowprops={"arrowstyle": "<->", "color": purple, "lw": 1.4},
            zorder=6,
        )
        mid = centers[j]
        ax.text(
            mid[0] - 0.15,
            mid[1] + 0.25,
            label,
            fontsize=12,
            color=purple,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
            zorder=7,
        )
    ax.text(0.1, 11.1, "Driveline cells", color=muted, fontsize=11)
    ax.annotate(
        "",
        xy=left[6] + [0.3, -0.1],
        xytext=(1.1, 10.9),
        arrowprops={"arrowstyle": "-", "color": "#aabcc8", "lw": 1},
    )

    def text(x, y, s, **kwargs):
        fig.text(x, y, s, color=ink, **kwargs)

    x = 0.55
    text(x, 0.82, "Segment geometry", fontsize=14, weight="bold")
    text(
        x,
        0.775,
        "$S_i, E_i \\in \\mathbb{R}^3$   entry / exit edge midpoints",
        fontsize=11,
    )
    text(
        x,
        0.705,
        "Straight 3D centerline segment per cell.\nLengths vary; eight cells do not imply a fixed distance.",
        fontsize=11,
        linespacing=1.6,
    )
    fig.text(x, 0.652, "$w_i=(a_i+b_i)/2$", fontsize=15, color=purple)
    text(
        x,
        0.604,
        "Average entry / exit width, not an exact road outline.\nCells describe the driveline, not the visible asphalt.",
        fontsize=11,
        linespacing=1.6,
    )
    text(x, 0.521, "Kart state", fontsize=14, weight="bold")
    fig.text(
        x,
        0.476,
        "$\\mathbf{v}\\in\\mathbb{R}^3$   local velocity",
        fontsize=12,
        color=green,
    )
    fig.text(
        x,
        0.431,
        "$\\mathbf{c}\\in\\mathbb{R}^3$   centerline-to-kart offset",
        fontsize=12,
        color=purple,
    )
    text(
        x,
        0.384,
        "center_path uses the current segment's infinite line;\nthe perpendicular projection is not endpoint-clamped.",
        fontsize=10.5,
        linespacing=1.6,
    )
    fig.text(
        x,
        0.323,
        "$d_\\perp=\\|\\mathbf{c}\\|\\,\\mathrm{sign}(c_x)$",
        fontsize=14,
        color=purple,
    )
    text(x, 0.285, "center_path_distance: signed offset magnitude", fontsize=10.5)

    # Frame inset: local x left, z forward, y out of page.
    frame = fig.add_axes((0.55, 0.105, 0.14, 0.13))
    frame.set(xlim=(-1.2, 0.55), ylim=(-0.35, 1.3), aspect="equal")
    frame.axis("off")
    for target, label in [((-0.9, 0), "$x$"), ((0, 1), "$z$")]:
        frame.annotate(
            "",
            xy=target,
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": ink, "lw": 1.3},
        )
        frame.text(target[0] - 0.12, target[1] + 0.08, label, color=ink)
    frame.scatter(0, 0, s=100, facecolor="white", edgecolor=ink)
    frame.scatter(0, 0, s=8, color=ink)
    frame.text(0.15, -0.15, "$y$", color=ink)
    text(0.70, 0.205, "Kart-local coordinates", fontsize=11, weight="bold")
    text(
        0.70,
        0.155,
        "$x$ left · $y$ up · $z$ forward\nTop view; $y$ points out of the page.",
        fontsize=10.5,
        linespacing=1.6,
    )
    fig.text(
        0.06,
        0.055,
        "Schematic unbranched section · Branches may expose multiple successors · Geometry shown before normalization",
        color=muted,
        fontsize=9,
    )
    output = Path(__file__).resolve().parents[1] / "generated"
    output.mkdir(exist_ok=True)
    for suffix in ["pdf", "png"]:
        fig.savefig(output / f"states-schema.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
