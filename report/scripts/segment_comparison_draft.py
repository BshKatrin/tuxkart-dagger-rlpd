"""Prepare paired segment data and a static RLPD/Enterprise report overview.

Run from the repository root:
    uv run --with matplotlib --with seaborn python report/scripts/segment_comparison_draft.py

Reads the matched traces referenced by report/generated/track-time-loss/sources.json
and reuses the established timing, interpolation and lateral-position functions.
Writes ignored JSON, a vector PDF for the report, and a 150-dpi PNG preview
with 2 aligned full-track maps.
No simulations or Samskrift edits are performed by this script.
"""

import json
from itertools import pairwise

import numpy as np
from track_time_loss import (
    DEFAULT_OUTPUT,
    PROJECT_ROOT,
    SEEDS,
    clean_json,
    digest,
    first_crossing_values,
    first_passage_sectors,
    reference_route,
    road_lateral_positions,
    timing_gates,
)


def main():
    track = "stk_enterprise"
    method = "rlpd"
    destination = PROJECT_ROOT / "report/generated/segment-comparison-draft"
    destination.mkdir(exist_ok=True)
    sources = json.loads((DEFAULT_OUTPUT / "sources.json").read_text())["sources"][
        track
    ]
    geometry = json.loads(
        (DEFAULT_OUTPUT / "data" / track / "geometry.json").read_text()
    )
    gates, branches = timing_gates(geometry)
    lengths = np.diff(gates)
    samples, ids = [], []
    for index, (start, stop) in enumerate(pairwise(gates)):
        count = max(1, int(np.ceil((stop - start) / 2)))
        samples.extend(start + (np.arange(count) + 0.5) * (stop - start) / count)
        ids.extend([index] * count)
    samples, ids = np.asarray(samples), np.asarray(ids)
    channels = {}
    for key in ("expert", method):
        source = sources[key]
        path = PROJECT_ROOT / source["path"]
        if digest(path) != source["sha256"]:
            raise ValueError("Input trajectory changed")
        records = {record["seed"]: record for record in source["episodes"]}
        rows = []
        with np.load(path) as trace:
            for seed in SEEDS:
                mask = trace["seed"] == seed
                progress = trace["progress"][mask]
                times = first_passage_sectors(
                    progress,
                    trace["elapsed_seconds"][mask],
                    gates,
                    trace["rescue_active"][mask],
                    records[seed]["lap_time"],
                )
                values = first_crossing_values(
                    progress,
                    np.column_stack(
                        [trace["speed"][mask], trace["world_position"][mask]]
                    ),
                    samples,
                )
                q = road_lateral_positions(values[:, 1:], samples, geometry)
                speed = np.array(
                    [values[ids == i, 0].mean() for i in range(len(lengths))]
                )
                lateral = np.array(
                    [100 * np.abs(q[ids == i]).mean() for i in range(len(lengths))]
                )
                rows.append(np.column_stack([times, speed, lateral]))
        channels[key] = np.array(rows)
    valid = np.isfinite(channels[method]).all(axis=2) & np.isfinite(
        channels["expert"]
    ).all(axis=2)
    delta = (channels[method][:, :, 0] - channels["expert"][:, :, 0]) * 20 / lengths

    def estimate(values, selected):
        values = values[selected]
        return {
            "mean": float(values.mean()) if len(values) else None,
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        }

    segments = []
    for index, (start, stop) in enumerate(pairwise(gates)):
        selected = valid[:, index]
        segments.append(
            {
                "id": index,
                "start": float(start),
                "end": float(stop),
                "pairs": int(selected.sum()),
                "branch": any(
                    abs(start - a) < 0.01 and abs(stop - b) < 0.01 for a, b in branches
                ),
                "gap": estimate(delta[:, index], selected),
                "speed_gap": estimate(
                    channels[method][:, index, 1] - channels["expert"][:, index, 1],
                    selected,
                ),
                "lateral_gap": estimate(
                    channels[method][:, index, 2] - channels["expert"][:, index, 2],
                    selected,
                ),
                "speed": {
                    key: estimate(channels[key][:, index, 1], selected)
                    for key in channels
                },
                "lateral": {
                    key: estimate(channels[key][:, index, 2], selected)
                    for key in channels
                },
            }
        )
    nodes = np.asarray(geometry["nodes"])
    distances = np.asarray(geometry["distances"])
    paths = []
    for node in reference_route(geometry):
        start, stop = distances[node]
        boundaries = np.r_[start, gates[(gates > start) & (gates < stop)], stop]
        for low, high in pairwise(boundaries):
            fraction = (np.array([low, high]) - start) / (stop - start)
            points = nodes[node, 0] + fraction[:, None] * (
                nodes[node, 1] - nodes[node, 0]
            )
            paths.append(
                {
                    "id": int(
                        np.searchsorted(gates, (low + high) / 2, side="right") - 1
                    ),
                    "points": np.round(points[:, [0, 2]], 3).tolist(),
                }
            )
    result = {
        "track": "STK Enterprise",
        "method": "RLPD",
        "length": geometry["length"],
        "seeds": list(SEEDS),
        "segments": segments,
        "paths": paths,
        "context": np.round(nodes[:, :, [0, 2]], 3).tolist(),
        "default": int(np.nanargmax([segment["gap"]["mean"] for segment in segments])),
        "avg_compared": float(valid.sum(axis=1).mean()),
    }
    output = destination / "enterprise-rlpd-data.json"
    output.write_text(
        json.dumps(clean_json(result), separators=(",", ":"), allow_nan=False)
    )
    print(
        f"Prepared {len(segments)} segments; mean compared {result['avg_compared']:.2f}"
    )
    for segment in sorted(
        segments, key=lambda segment: segment["gap"]["mean"], reverse=True
    )[:3]:
        print(json.dumps(segment))
    plot_overview(result, destination)


def plot_overview(data, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    sns.set_theme(context="paper", style="ticks", palette="colorblind")
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42}
    )
    fig, axes = plt.subplots(1, 2, figsize=(17 / 2.54, 9.3 / 2.54))
    fig.subplots_adjust(left=0.025, right=0.98, top=0.80, bottom=0.23, wspace=0.12)
    fig.text(0.025, 0.97, "STK Enterprise: RLPD relative to expert", fontsize=10)
    fig.text(
        0.025,
        0.915,
        "4 paired evaluation laps; all 61 comparison segments available",
        fontsize=8,
        color="0.35",
    )
    configurations = [
        (
            "gap",
            "(a) Time gap",
            ["#15803d", "#a3a3a3", "#c62828"],
            0.5,
            ["≤ −0.5", "0", "≥ +0.5"],
            "Time gap (s / 20 m)\ngreen = faster",
        ),
        (
            "lateral_gap",
            "(b) Lateral road use",
            ["#008c95", "#a3a3a3", "#8e44ad"],
            100,
            ["≤ −100", "0", "≥ +100"],
            "Lateral difference (pp)\npurple = farther from center",
        ),
    ]
    segments = data["segments"]
    paths = {
        index: np.vstack([pieces[0][0], *[piece[1] for piece in pieces]])
        for index in range(len(segments))
        if (
            pieces := [
                np.array(path["points"])
                for path in data["paths"]
                if path["id"] == index
            ]
        )
    }
    background = np.array(data["context"])
    for ax, (key, title, colors, limit, labels, legend) in zip(axes, configurations):
        cmap = LinearSegmentedColormap.from_list(key, colors)
        norm = Normalize(-limit, limit, clip=True)
        ax.add_collection(LineCollection(background, colors="0.90", linewidths=0.6))
        for segment in segments:
            estimate = segment[key]
            if estimate["mean"] is None:
                color = "0.85"
            else:
                color = cmap(norm(estimate["mean"]))
            points = paths[segment["id"]]
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                linewidth=2.8,
                linestyle="solid",
                solid_capstyle="round",
            )
        ax.update_datalim(background.reshape(-1, 2))
        ax.autoscale_view()
        ax.margins(0.07)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
        ax.set_title(title, fontsize=9, pad=7)
        bounds = ax.get_position()
        bar_ax = fig.add_axes((bounds.x0 + 0.045, 0.17, bounds.width - 0.09, 0.023))
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=bar_ax,
            orientation="horizontal",
            extend="both",
            ticks=[-limit, 0, limit],
        )
        colorbar.ax.set_xticklabels(labels)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label(legend, fontsize=8, labelpad=3)
        colorbar.outline.set_visible(False)
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")
    output = destination / "enterprise-rlpd-overview.pdf"
    fig.savefig(output, facecolor="white")
    fig.savefig(output.with_suffix(".png"), dpi=150, facecolor="white")
    plt.close(fig)
    print(output.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
