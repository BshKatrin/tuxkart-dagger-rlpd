"""Plot five policy close-ups with all training seeds and a shared expert.

From the repository root:
    uv run --with matplotlib --with seaborn python report/scripts/disturbance_trajectories.py

Reads the completed, hashed disturbance bank and installed native road quads.
No simulation or training is run. Writes vector PDF, 150-dpi PNG and provenance
under report/generated/disturbance-trajectories. Nothing is uploaded to Samskrift.
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
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from platformdirs import user_cache_dir

from kartstudy.trajectory_evolution import digest, read_quads

METHODS = ("bc", "dagger", "sac", "sac_sym", "rlpd")
LABELS = ("BC", "DAgger", "SAC", "SAC + SS", "RLPD")
CASES = {
    "delay_corner_hacienda": "Delayed controls (Hacienda)",
    "left_throttle_lighthouse": "Left + throttle (Lighthouse)",
}


def load_case(root, manifest, scenario_id):
    found = {}
    hashes = []
    for item in manifest["trials"]:
        if not item["id"].startswith(scenario_id + "--"):
            continue
        for field in ("result", "trajectory"):
            if digest(root / item[field]) != item[field + "_sha256"]:
                raise ValueError(f"Changed {field}: {item['id']}")
        row = json.loads((root / item["result"]).read_text())
        method = row["controller"]
        if method not in (*METHODS, "expert"):
            continue
        seed = None if row["policy"] is None else row["policy"]["config"]["seed"]
        key = method, seed
        if key in found:
            raise ValueError(f"Duplicate policy: {key}")
        with np.load(root / item["trajectory"]) as source:
            trace = {key: source[key] for key in source.files}
        start = row["handover_steps"]
        trace["time"] = trace["elapsed_seconds"] - trace["elapsed_seconds"][start - 1]
        found[key] = row, trace
        hashes.append(item)
    expected = {("expert", None)} | {(m, s) for m in METHODS for s in range(3)}
    if set(found) != expected:
        raise ValueError("Require one expert and all five methods at seeds 0, 1, 2")
    reference = found["expert", None][0]["handover"]
    if any(row["handover"] != reference for row, _ in found.values()):
        raise ValueError("Mismatched handover states")
    return found, hashes


def unaided_path(row, trace, origin, seconds):
    """Stop at the first rescue; never connect a rescue or show assisted driving."""
    start = row["handover_steps"]
    time_limit = seconds + row["recovery_start_seconds"]
    indices = np.flatnonzero(
        (trace["phase"] == "response") & (trace["time"] <= time_limit + 1e-5)
    )
    events = np.flatnonzero(
        (trace["phase"] == "response")
        & (trace["rescue_active"] | (np.diff(trace["rescue_count"], prepend=0) > 0))
    )
    event = int(events[0]) if len(events) else None
    if event is not None:
        indices = indices[indices < event]
    positions = trace["world_position"][:, [0, 2]].astype(float) - origin
    points = np.vstack([np.zeros(2), positions[indices]])
    times = np.r_[0, trace["time"][indices]]
    rescue = (
        positions[event - 1]
        if event is not None and trace["time"][event] <= time_limit + 1e-5
        else None
    )
    restoration = row["recovery_start_seconds"]
    restore_point = (
        positions[start + row["scenario"]["control_delay"]["duration_steps"] - 1]
        if restoration
        else None
    )
    return points, times, rescue, restore_point


def forced_context(row, trace, origin):
    """Disturbance onset and the shared executed forced-control path."""
    forced = np.flatnonzero(trace["phase"] == "disturbance")
    if not len(forced):
        return None
    positions = trace["world_position"][:, [0, 2]].astype(float) - origin
    onset = np.asarray(row["anchor"]["position"])[[0, 2]] - origin
    return {
        "forced": np.vstack([onset, positions[forced]]),
        "onset": onset,
    }


def plot_case(root, manifest, scenario_id, output, seconds):
    found, inputs = load_case(root, manifest, scenario_id)
    reference = found["expert", None][0]
    origin = np.asarray(reference["handover"]["position"])[[0, 2]]
    paths = {key: unaided_path(*value, origin, seconds) for key, value in found.items()}
    context = forced_context(*found["expert", None], origin)
    if context is not None:
        stop = reference["handover_steps"]
        for _, trace in found.values():
            for field in ("world_position", "action", "phase", "elapsed_seconds"):
                np.testing.assert_array_equal(
                    trace[field][:stop], found["expert", None][1][field][:stop]
                )
    # One shared crop per case, determined from the first five seconds of all
    # unaided responses. Later response segments may leave this fixed close-up.
    focus_parts = [p[t <= 5] for p, t, _, _ in paths.values()]
    if context is not None:
        focus_parts.append(context["forced"])
    focus = np.vstack(focus_parts)
    center = (focus.min(axis=0) + focus.max(axis=0)) / 2
    span = max(float(np.ptp(focus, axis=0).max()) + 16, 60)
    lower, upper = center - span / 2, center + span / 2
    # Trim empty vertical margins, retaining one common crop and equal physical
    # x/z scale for all five panels. This crops the map; it never stretches it.
    points = np.vstack([focus, *(p for p, _, _, _ in paths.values())])
    visible = points[np.all((points >= lower) & (points <= upper), axis=1)]
    lower[1] = max(lower[1], float(visible[:, 1].min()) - 5)
    upper[1] = min(upper[1], float(visible[:, 1].max()) + 5)
    quads_path = (
        Path(user_cache_dir("PySuperTuxKart2", "bpiwowar"))
        / "1.5/data/tracks"
        / reference["scenario"]["track"]
        / "quads.xml"
    )
    quads = read_quads(quads_path) - origin
    sns.set_theme(context="paper", style="ticks", palette="colorblind")
    colors = sns.color_palette("colorblind", 3)
    styles = ["-", (0, (1, 1.5)), (0, (4, 1.3, 1, 1.3))]
    ink = "#29333d"
    width, left, right, gap = 17 / 2.54, 0.008, 0.997, 0.08
    panel_width = width * (right - left) / (5 + 4 * gap)
    panel_height = panel_width * (upper[1] - lower[1]) / (upper[0] - lower[0])
    header = 0.38
    footer = 0.35
    height = panel_height + header + footer
    fig, axes = plt.subplots(1, 5, figsize=(width, height), sharex=True, sharey=True)
    for ax, method, label in zip(axes, METHODS, LABELS, strict=True):
        ax.add_collection(
            PolyCollection(quads, facecolors="#e8ecf0", edgecolors="none")
        )
        if context is not None:
            ax.plot(
                *context["forced"].T,
                color="#6c7680",
                lw=1.1,
                ls=(0, (1.1, 1.6)),
                zorder=2,
            )
        for seed, color, style in zip(range(3), colors, styles, strict=True):
            points, _, rescue, restored = paths[method, seed]
            ax.plot(*points.T, color=color, lw=1.0, ls=style, zorder=3)
            if rescue is not None:
                ax.plot(
                    *rescue,
                    marker="X",
                    ms=4.5,
                    mfc=color,
                    mec="white",
                    mew=0.4,
                    zorder=6,
                )
            if restored is not None:
                ax.plot(
                    *restored,
                    marker="D",
                    ms=3,
                    mfc="white",
                    mec=color,
                    mew=0.8,
                    zorder=5,
                )
        expert, _, rescue, restored = paths["expert", None]
        ax.plot(*expert.T, color=ink, lw=1.05, ls=(0, (5, 2.5)), zorder=4)
        if rescue is not None:
            ax.plot(*rescue, marker="X", ms=4.5, color=ink, zorder=6)
        if restored is not None:
            ax.plot(
                *restored, marker="D", ms=3, mfc="white", mec=ink, mew=0.8, zorder=5
            )
        initial = np.zeros(2) if context is None else context["onset"]
        ax.plot(*initial, marker="o", ms=3.5, mfc="white", mec=ink, mew=0.8, zorder=7)
        if context is not None:
            ax.plot(0, 0, marker="D", ms=3, mfc="white", mec=ink, mew=0.8, zorder=7)
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal")
        ax.set_title(label, fontsize=8.5, pad=5)
        ax.set_axis_off()
    handles = [
        Line2D([], [], color=c, lw=1.1, ls=s, label=f"Seed {seed}")
        for seed, c, s in zip(range(3), colors, styles, strict=True)
    ]
    handles.append(Line2D([], [], color=ink, lw=1.05, ls=(0, (5, 2.5)), label="Expert"))
    handles.append(
        Line2D(
            [],
            [],
            color=ink,
            marker="o",
            mfc="white",
            ms=3.5,
            ls="none",
            label="Initial state",
        )
    )
    if reference["scenario"].get("control_delay") or context is not None:
        handles.append(
            Line2D(
                [],
                [],
                color=ink,
                marker="D",
                mfc="white",
                ms=3,
                ls="none",
                label="Controls restored",
            )
        )
    # Use the same event key in both figures, including cases without a rescue.
    handles.append(
        Line2D([], [], color=ink, marker="X", ms=4.5, ls="none", label="Native rescue")
    )
    if len(handles) == 7:
        # Matplotlib fills legend columns first. Keep seeds 0/1/2 and expert
        # together on the first reading row, with event markers underneath.
        handles = [handles[i] for i in (0, 4, 1, 5, 2, 6, 3)]
    fig.suptitle(CASES[scenario_id], fontsize=10, y=1 - 0.015 / height)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015 / height),
        ncols=4 if len(handles) > 5 else 5,
        frameon=False,
        fontsize=7,
        handlelength=2.3,
        handletextpad=0.5,
        columnspacing=1.4,
        labelspacing=0.65,
        borderaxespad=0,
    )
    fig.subplots_adjust(
        left=left,
        right=right,
        top=1 - header / height,
        bottom=footer / height,
        wspace=gap,
    )
    name = (
        "delayed-controls"
        if reference["scenario"].get("control_delay")
        else "left-throttle"
    )
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=150, facecolor="white")
    plt.close(fig)
    return {
        "scenario": reference["scenario"],
        "initial_world_xz_m": origin.tolist(),
        "bounds_relative_m": [lower.tolist(), upper.tolist()],
        "crop_rule": "First 5 s of unaided responses define initial square bounds with 8 m padding and at least 60 m span; empty vertical margins are then trimmed to 5 m around visible responses, without expanding the original crop",
        "axes_visible": False,
        "figure_size_inches": [width, height],
        "normal_control_window_seconds": seconds,
        "trajectory_rule": "From handover through 15 s of normal control, stopping before first native rescue; X marks last pre-rescue position. Lines are clipped by common spatial bounds.",
        "forced_context": context is not None,
        "context_rule": "Complete executed forced-control segment; circle marks onset and diamond marks restoration. Pre-perturbation approach is omitted. Legend is shared with the delayed-controls figure."
        if context is not None
        else None,
        "restoration_markers": True,
        "native_quads_sha256": digest(quads_path),
        "inputs": inputs,
        "exports": {
            f"{name}.{s}": digest(output / f"{name}.{s}") for s in ("pdf", "png")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("runs/five-disturbances-v1/evaluation")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("report/generated/disturbance-trajectories")
    )
    parser.add_argument("--seconds", type=float, default=15)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Require completed evaluation bank")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = [
        plot_case(args.root, manifest, case, args.output, args.seconds)
        for case in CASES
    ]
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "source_manifest_sha256": digest(args.root / "manifest.json"),
                "renderer_sha256": digest(__file__),
                "cases": metadata,
            },
            indent=2,
        )
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
