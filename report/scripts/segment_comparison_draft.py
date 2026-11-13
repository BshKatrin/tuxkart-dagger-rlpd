"""Prepare paired segment data and a static RLPD/Enterprise report overview.

Run from the repository root:
    uv run --with matplotlib --with seaborn python report/scripts/segment_comparison_draft.py

Reads the matched traces referenced by report/generated/track-time-loss/sources.json
and reuses the established timing, interpolation and lateral-position functions.
Writes ignored JSON, a vector PDF for the report, and a 150-dpi PNG preview
with 2 aligned full-track maps.
For a 2-row, 5-policy comparison, add --all-policies. All maps use training
seed 0 and the original 4 paired evaluation seeds. Use --track hacienda to
compare Hacienda instead of STK Enterprise.
For the 3-run, 30-evaluation-seed Enterprise overview, add
--aggregate-training-seeds. This reads performance-thirty-seeds/sources-s0.json
through sources-s2.json, averages paired evaluations within each run, then
averages the 3 run means equally. Regions use solid colors; variability remains
in the data export. The expert references are shared across training runs.
No simulations or Samskrift edits are performed by this script.
"""

import argparse
import json
from itertools import pairwise

import numpy as np
from track_time_loss import (
    DEFAULT_OUTPUT,
    PROJECT_ROOT,
    clean_json,
    digest,
    first_crossing_values,
    first_passage_sectors,
    reference_route,
    road_lateral_positions,
    timing_gates,
)


def prepare(method, destination, *, render=True, track="stk_enterprise",
            source_file=None, training_seed=0):
    slug = "enterprise" if track == "stk_enterprise" else track
    track_name = "STK Enterprise" if track == "stk_enterprise" else track.replace("_", " ").title()
    destination.mkdir(parents=True, exist_ok=True)
    source_file = source_file or DEFAULT_OUTPUT / "sources.json"
    metadata = json.loads(source_file.read_text())
    seeds = metadata["seeds"]
    sources = metadata["sources"][track]
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
        if sorted(records) != sorted(seeds) or len(records) != len(source["episodes"]):
            raise ValueError("Missing or duplicate evaluation records")
        rows = []
        with np.load(path) as trace:
            for seed in seeds:
                mask = trace["seed"] == seed
                if int(mask.sum()) != records[seed]["steps"]:
                    raise ValueError("Trajectory length differs from episode record")
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
        "track": track_name,
        "track_id": track,
        "slug": slug,
        "method": method,
        "training_seed": training_seed,
        "completed": sum(e["success"] for e in sources[method]["episodes"]),
        "length": geometry["length"],
        "seeds": list(seeds),
        "expert_completed": sum(e["success"] for e in sources["expert"]["episodes"]),
        "source_metadata": {"path": str(source_file.relative_to(PROJECT_ROOT)),
                            "sha256": digest(source_file)},
        "segments": segments,
        "paths": paths,
        "context": np.round(nodes[:, :, [0, 2]], 3).tolist(),
        "default": max(range(len(segments)), key=lambda i: segments[i]["gap"]["mean"]
                       if segments[i]["gap"]["mean"] is not None else -np.inf),
        "avg_compared": float(valid.sum(axis=1).mean()),
    }
    output = destination / f"{slug}-{method}-data.json"
    output.write_text(
        json.dumps(clean_json(result), separators=(",", ":"), allow_nan=False)
    )
    print(
        f"Prepared {len(segments)} segments; mean compared {result['avg_compared']:.2f}"
    )
    for segment in sorted(
        [segment for segment in segments if segment["gap"]["mean"] is not None],
        key=lambda segment: segment["gap"]["mean"], reverse=True
    )[:3]:
        print(json.dumps(segment))
    if render:
        plot_overview(result, destination)
    return result


def aggregate_training_runs(runs):
    """Keep training runs equally weighted, even with unequal segment coverage."""
    template = runs[0]
    training_seeds = [run["training_seed"] for run in runs]
    if len(set(training_seeds)) != len(runs):
        raise ValueError("Duplicate training seeds")
    for run in runs[1:]:
        if any(run[key] != template[key] for key in
               ("track_id", "method", "seeds", "paths", "context")):
            raise ValueError("Runs must share geometry, policy method and evaluation seeds")
        if [(s["start"], s["end"]) for s in run["segments"]] != [
                (s["start"], s["end"]) for s in template["segments"]]:
            raise ValueError("Runs must share timing regions")

    def estimate(entries):
        means = [entry["mean"] for entry in entries]
        complete = all(value is not None for value in means)
        return {"mean": float(np.mean(means)) if complete else None,
                "sd": float(np.std(means, ddof=1)) if complete and len(means) > 1 else None,
                "training_means": means,
                "evaluation_sd_by_training_run": [entry["sd"] for entry in entries]}

    segments = []
    for index, original in enumerate(template["segments"]):
        entries = [run["segments"][index] for run in runs]
        segment = {key: original[key] for key in ("id", "start", "end", "branch")}
        segment.update(pairs=sum(entry["pairs"] for entry in entries),
                       pairs_by_training_run=[entry["pairs"] for entry in entries])
        for field in ("gap", "speed_gap", "lateral_gap"):
            segment[field] = estimate([entry[field] for entry in entries])
        for field in ("speed", "lateral"):
            segment[field] = {key: estimate([entry[field][key] for entry in entries])
                              for key in original[field]}
        segments.append(segment)
    result = {key: template[key] for key in
              ("track", "track_id", "slug", "method", "length", "seeds", "paths", "context")}
    result.update(training_seeds=training_seeds, segments=segments,
                  completed=sum(run["completed"] for run in runs),
                  completed_by_training_run=[run["completed"] for run in runs],
                  expert_completed=template["expert_completed"],
                  source_metadata=[run["source_metadata"] for run in runs],
                  avg_compared=float(np.mean([run["avg_compared"] for run in runs])),
                  aggregation="mean over evaluation seeds within run, then equal mean over training runs",
                  sd_unit="training-run means")
    return result


def prepare_training_average(destination):
    root = PROJECT_ROOT / "report/generated/performance-thirty-seeds"
    runs, experts = [], []
    for seed in range(3):
        source_file = root / f"sources-s{seed}.json"
        metadata = json.loads(source_file.read_text())
        if metadata["seeds"] != list(range(20000, 20030)):
            raise ValueError("Expected 30 evaluation seeds")
        experts.append(metadata["sources"]["stk_enterprise"]["expert"])
        runs.append(prepare("rlpd", destination / f"training-seed-{seed}",
                            render=False, source_file=source_file, training_seed=seed))
    if any(expert != experts[0] for expert in experts[1:]):
        raise ValueError("Expert references must be identical across training runs")
    result = aggregate_training_runs(runs)
    (destination / "enterprise-rlpd-data.json").write_text(
        json.dumps(clean_json(result), indent=2, allow_nan=False))
    plot_overview(result, destination)
    print(f"Completed {result['completed']}/90 RLPD laps; "
          f"region pair counts {min(s['pairs'] for s in result['segments'])}"
          f"--{max(s['pairs'] for s in result['segments'])}")
    return result


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
    averaged = "training_seeds" in data
    if averaged:
        fig.subplots_adjust(top=.88, bottom=.29)
    else:
        fig.text(0.025, 0.97, "STK Enterprise: RLPD relative to expert", fontsize=10)
        fig.text(0.025, 0.915,
                 "4 paired evaluation laps; all 61 comparison segments available",
                 fontsize=8, color="0.35")
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
        bar_ax = fig.add_axes((bounds.x0 + 0.045, .215 if averaged else .17,
                               bounds.width - 0.09, 0.023))
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
    print(output.resolve().relative_to(PROJECT_ROOT))


def plot_policy_grid(data, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    sns.set_theme(context="paper", style="ticks", palette="colorblind")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 5, figsize=(17 / 2.54, 8.2 / 2.54))
    fig.subplots_adjust(left=.025, right=.85, top=.86, bottom=.13, hspace=.23, wspace=.13)
    names = ("BC", "DAgger", "SAC", "SAC + SS", "RLPD")
    configurations = (
        ("gap", ["#15803d", "#a3a3a3", "#c62828"], .5,
         "Time gap\n(s / 20 m)", "faster     slower"),
        ("lateral_gap", ["#008c95", "#a3a3a3", "#8e44ad"], 100,
         "Lateral gap\n(pp half-width)", "closer     farther"),
    )
    for row, (field, colors, limit, title, direction) in enumerate(configurations):
        cmap = LinearSegmentedColormap.from_list(field, colors)
        norm = Normalize(-limit, limit, clip=True)
        for col, (record, name) in enumerate(zip(data, names, strict=True)):
            ax = axes[row, col]
            background = np.asarray(record["context"])
            ax.add_collection(LineCollection(background, colors=".90", linewidths=.45))
            segments = {segment["id"]: segment for segment in record["segments"]}
            pieces, shades, missing = [], [], []
            missing_path = []
            for piece in record["paths"]:
                segment = segments[piece["id"]]
                value = segment[field]["mean"]
                points = np.asarray(piece["points"])
                if segment["pairs"] == 4 and value is not None:
                    if missing_path:
                        missing.append(np.asarray(missing_path))
                        missing_path = []
                    pieces.append(points)
                    shades.append(cmap(norm(value)))
                else:
                    if missing_path and not np.allclose(missing_path[-1], points[0]):
                        missing.append(np.asarray(missing_path))
                        missing_path = []
                    if not missing_path:
                        missing_path.append(points[0])
                    missing_path.append(points[1])
            if missing_path:
                missing.append(np.asarray(missing_path))
            ax.add_collection(LineCollection(pieces, colors=shades, linewidths=1.75,
                                             capstyle="round"))
            ax.add_collection(LineCollection(missing, colors="black", linewidths=1.4,
                                             linestyles=(0, (2, 2.5)), capstyle="butt"))
            start = np.asarray(record["paths"][0]["points"])[0]
            ax.plot(*start, marker="_", markersize=7, color="black",
                    markeredgewidth=1.4, linestyle="none", zorder=5)
            ax.update_datalim(background.reshape(-1, 2))
            ax.autoscale_view()
            ax.margins(.07)
            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f"{name}\n{record['completed']}/4 laps finished", fontsize=7.5, pad=6)
        bounds = axes[row, -1].get_position()
        cax = fig.add_axes([.885, bounds.y0 + .025, .014, bounds.height - .05])
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                          extend="both", ticks=[-limit, 0, limit])
        bar.ax.tick_params(labelsize=6.5, length=0, pad=2)
        bar.outline.set_visible(False)
        bar.solids.set_rasterized(False)
        bar.solids.set_edgecolor("face")
        fig.text(.865, bounds.y1 + .035, title, fontsize=7, va="bottom")

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color="black", lw=1.4, ls=(0, (2, 2.5)),
               label="Fewer than 4 valid pairs"),
        Line2D([], [], color="black", lw=1.4, label="Start / finish"),
    ], loc="lower center", bbox_to_anchor=(.46, .01), ncol=2,
       frameon=False, fontsize=7, handlelength=2.0, columnspacing=2.5)
    output = destination / f"{data[0].get('slug', 'enterprise')}-all-policies-overview.pdf"
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)
    (destination / "caption.txt").write_text(
        f"{data[0]['track']} policy comparison using training seed 0 and four matched evaluation seeds. "
        "Columns show BC, DAgger, SAC, SAC with symmetric sampling, and RLPD. "
        "The top row shows mean policy-minus-expert traversal time normalized to seconds per 20 m "
        "(green faster, red slower). The bottom row shows mean difference in absolute lateral "
        "position, in percentage points of road half-width (teal closer to the center, purple farther). "
        "Lateral position is descriptive, not a quality score. Each sector requires four valid "
        "rescue-free matched traversals; otherwise it is dashed black. Gray denotes a valid value near zero. Branches are timed between common "
        "split/merge gates. Scales are shared across policies and clipped at +/-0.5 s per 20 m "
        "and +/-100 percentage points. Completion counts refer to full laps. A short horizontal black line marks start/finish; the bottom legend identifies this and the dashed missing-data segments. "
        "Maps describe evaluation variability within retained controllers, not training-seed robustness.\n")
    print(output.resolve().relative_to(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-policies", action="store_true")
    parser.add_argument("--aggregate-training-seeds", action="store_true")
    parser.add_argument("--track", default="stk_enterprise", choices=("stk_enterprise", "hacienda"))
    args = parser.parse_args()
    if args.aggregate_training_seeds:
        if args.all_policies or args.track != "stk_enterprise":
            parser.error("The training-run average currently supports RLPD/Enterprise only")
        prepare_training_average(PROJECT_ROOT / "report/generated/enterprise-rlpd-thirty-seeds")
        return
    if args.track != "stk_enterprise" and not args.all_policies:
        parser.error("--track hacienda requires --all-policies")
    if args.all_policies:
        slug = "enterprise" if args.track == "stk_enterprise" else args.track
        destination = PROJECT_ROOT / f"report/generated/{slug}-all-policies"
        data = [prepare(method, destination, render=False, track=args.track)
                for method in ("bc", "dagger", "sac", "sac_sym", "rlpd")]
        plot_policy_grid(data, destination)
    else:
        prepare("rlpd", PROJECT_ROOT / "report/generated/segment-comparison-draft")


if __name__ == "__main__":
    main()
