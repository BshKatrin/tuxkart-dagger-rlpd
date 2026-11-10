"""Animate matched expert/policy responses from a saved disturbance evaluation.

uv run --with matplotlib python scripts/animate_disturbances.py \
  --root RUN/evaluation --scenario SCENARIO_ID --output runs/new-animation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from kartstudy.trajectory_evolution import COLORS, digest, read_quads

METHODS = ("expert", "bc", "dagger", "sac", "sac_sym", "rlpd")
LABELS = {
    "expert": "Expert",
    "bc": "BC",
    "dagger": "DAgger",
    "sac": "SAC",
    "sac_sym": "SAC + symmetric",
    "rlpd": "RLPD",
}


def load_trials(root, scenario_id, training_seed):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Require completed evaluation bank")
    found = {}
    for item in manifest["trials"]:
        if digest(root / item["result"]) != item["result_sha256"]:
            raise ValueError("Result hash mismatch")
        row = json.loads((root / item["result"]).read_text())
        method = row["controller"]
        if row["scenario"]["id"] != scenario_id or method not in METHODS:
            continue
        if (
            row["policy"] is not None
            and row["policy"]["config"]["seed"] != training_seed
        ):
            continue
        if method in found:
            raise ValueError("Ambiguous method/training seed")
        path = root / item["trajectory"]
        if digest(path) != item["trajectory_sha256"]:
            raise ValueError("Trajectory hash mismatch")
        with np.load(path) as saved:
            trace = {key: saved[key] for key in saved.files}
        start = np.flatnonzero(trace["phase"] == "response")[0]
        handover_time = trace["elapsed_seconds"][start - 1]
        trace["time"] = trace["elapsed_seconds"] - handover_time
        trace["position"] = trace["world_position"][:, [0, 2]].astype(float)
        trace["driven"] = trace["position"].copy()
        trace["rescue_start"] = np.diff(trace["rescue_count"], prepend=0) > 0
        trace["driven"][trace["rescue_active"] | trace["rescue_start"]] = np.nan
        found[method] = {
            "row": row,
            "trace": trace,
            "result_sha256": item["result_sha256"],
            "trajectory_sha256": item["trajectory_sha256"],
        }
    if set(found) != set(METHODS):
        raise ValueError(f"Missing methods: {set(METHODS) - set(found)}")
    reference = found["expert"]["row"]["handover"]
    for item in found.values():
        if item["row"]["handover"] != reference:
            raise ValueError("Handover states differ")
    return found


def render(trials, quads, output, view):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    reference = trials["expert"]["row"]
    anchor = np.asarray(reference["anchor"]["position"])[[0, 2]]
    handover = np.asarray(reference["handover"]["position"])[[0, 2]]
    trace0 = trials["expert"]["trace"]
    disturbance_indices = np.flatnonzero(trace0["phase"] == "disturbance")
    disturbance_start = (
        float(trace0["time"][disturbance_indices[0]] - 0.1)
        if len(disturbance_indices)
        else 0.0
    )
    restoration = max(
        t["row"].get("recovery_start_seconds", 0) for t in trials.values()
    )
    start_time = disturbance_start - 1
    end_time = (
        15 + restoration
        if view == "recovery"
        else max(float(t["trace"]["time"][-1]) for t in trials.values())
    )
    simulation_step, wall_ms = (0.2, 100) if view == "recovery" else (0.8, 100)
    if view == "recovery":
        midpoint = (anchor + handover) / 2
        radius = max(45, float(np.linalg.norm(anchor - handover)) * 0.75)
        lower, upper = midpoint - radius, midpoint + radius
    else:
        points = np.vstack(
            [quads.reshape(-1, 2), *(t["trace"]["position"] for t in trials.values())]
        )
        padding = max(5, float(np.ptp(points, axis=0).max()) * 0.03)
        lower, upper = points.min(0) - padding, points.max(0) + padding
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 8.2), dpi=100)
    artists = {}
    for ax, method in zip(axes.flat, METHODS, strict=True):
        color = COLORS.get(method, "#27323c")
        ax.add_collection(
            PolyCollection(quads, facecolors="#e6e9ee", edgecolors="none")
        )
        ax.plot(*anchor, marker="|", color="#27323c", ms=8, zorder=5)
        ax.plot(*handover, marker="o", mfc="white", mec=color, ms=5, zorder=5)
        (prefix,) = ax.plot([], [], color="#aab2bb", lw=1)
        (forced,) = ax.plot([], [], color="#7d8790", ls=":", lw=1.4)
        (driven,) = ax.plot([], [], color=color, lw=1.45)
        (rescues,) = ax.plot([], [], marker="X", ls="none", color=color, ms=5)
        (dot,) = ax.plot(
            [], [], marker="o", color=color, ms=5, mec="white", mew=0.6, zorder=6
        )
        text = ax.text(
            0.5, -0.025, "", transform=ax.transAxes, ha="center", va="top", fontsize=8
        )
        ax.set_title(LABELS[method], fontsize=11)
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal")
        ax.axis("off")
        artists[method] = (prefix, forced, driven, rescues, dot, text)
    seed = next(
        t["row"]["policy"]["config"]["seed"]
        for t in trials.values()
        if t["row"]["policy"]
    )
    title = fig.suptitle("", fontsize=12)
    fig.text(
        0.5,
        0.925,
        f"Training seed {seed} · evaluation seed {reference['scenario']['eval_seed']} · "
        + ("close-up · 2× speed" if view == "recovery" else "whole episode · 8× speed"),
        ha="center",
        fontsize=9,
    )
    fig.legend(
        handles=[
            Line2D([], [], color="#7d8790", ls=":", label="Forced disturbance"),
            Line2D(
                [],
                [],
                color="#27323c",
                marker="o",
                mfc="white",
                ls="none",
                label="Handover",
            ),
            Line2D(
                [], [], color="#27323c", marker="X", ls="none", label="Native rescue"
            ),
        ],
        loc="lower center",
        ncols=3,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.025, right=0.975, top=0.85, bottom=0.13, hspace=0.58, wspace=0.17
    )
    times = np.unique(
        np.r_[np.arange(start_time, end_time, simulation_step), 0, end_time]
    )
    images = []
    for seconds in times:
        phase_title = (
            "shared expert approach"
            if seconds < disturbance_start
            else (
                "FORCED DISTURBANCE"
                if seconds < 0
                else "DELAYED CONTROLS"
                if seconds < restoration
                else "policy control"
            )
        )
        title.set_text(
            f"{reference['scenario']['track']} · t = {seconds:+.1f} s from handover · {phase_title}"
        )
        for method, item in trials.items():
            row, trace = item["row"], item["trace"]
            prefix, forced, driven, rescues, dot, text = artists[method]
            shown = (trace["time"] >= start_time) & (trace["time"] <= seconds)
            for line, phase, positions in [
                (prefix, "prefix", trace["position"]),
                (forced, "disturbance", trace["driven"]),
                (driven, "response", trace["driven"]),
            ]:
                mask = shown & (trace["phase"] == phase)
                delayed = trace.get("delay_active", np.zeros(len(shown), dtype=bool))
                if phase == "disturbance":
                    mask |= shown & delayed
                elif phase == "response":
                    mask &= ~delayed
                line.set_data(positions[mask].T)
            mask = shown & trace["rescue_start"]
            rescues.set_data(trace["position"][mask].T)
            index = max(
                0, int(np.searchsorted(trace["time"], seconds, side="right")) - 1
            )
            pos = trace["position"][index]
            in_view = bool(np.all(pos >= lower) and np.all(pos <= upper))
            dot.set_data([pos[0]] if in_view else [], [pos[1]] if in_view else [])
            action = trace["action"][index]
            response = seconds >= 0
            finished = seconds >= trace["time"][-1]
            rescue_count = int(trace["rescue_count"][index]) - (
                row["prefix_rescues"] + row["disturbance_rescues"] if response else 0
            )
            status = "Driving" if response else phase_title
            if finished:
                status = (
                    f"Finished lap: {row['lap_time']:.1f} s"
                    if row["success"]
                    else "Timed out"
                )
            elif trace["rescue_active"][index]:
                status = "Native rescue active"
            elif trace.get("delay_active", np.zeros(len(shown), dtype=bool))[index]:
                status = "Delayed controls"
            recovery = row["recovery_seconds"]
            recovery_label = (
                f"Recovered at {recovery:.1f} s"
                if response
                and recovery is not None
                and seconds >= recovery + row.get("recovery_start_seconds", 0)
                else ""
            )
            if response and not recovery_label:
                recovery_label = "Recovery not yet observed"
            if restoration > 0:
                recovery_label = (
                    recovery_label + " after restoration"
                    if seconds >= restoration
                    else "Recovery clock starts after delay"
                )
            if not in_view:
                recovery_label += " · beyond close-up"
            controls = (
                ("Last: " if finished else "")
                + f"steer {action[0]:+.2f} · throttle {(action[1] + 1) * 50:.0f}% · brake {'ON' if action[2] > 0 else 'off'}"
            )
            text.set_text(
                f"{status} · {trace['speed'][index]:.1f} m/s · rescues {rescue_count}\n{controls}\n{recovery_label}"
            )
        fig.canvas.draw()
        image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        images.append(image.quantize(colors=128))
    path = output / f"{view}.gif"
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=[wall_ms] * (len(images) - 1) + [1500],
        loop=0,
        disposal=2,
    )
    fig.savefig(output / f"{view}-last-frame.png", dpi=120, facecolor="white")
    plt.close(fig)
    return {
        "path": path.name,
        "sha256": digest(path),
        "frames": len(images),
        "simulated_start_seconds": start_time,
        "simulated_end_seconds": end_time,
        "nominal_speed": 2 if view == "recovery" else 8,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory")
    parser.add_argument(
        "--views", nargs="+", choices=["recovery", "full"], default=["recovery", "full"]
    )
    args = parser.parse_args()
    trials = load_trials(args.root, args.scenario, args.train_seed)
    from platformdirs import user_cache_dir

    track = trials["expert"]["row"]["scenario"]["track"]
    quad_path = (
        Path(user_cache_dir("PySuperTuxKart2", "bpiwowar"))
        / "1.5/data/tracks"
        / track
        / "quads.xml"
    )
    quads = read_quads(quad_path)
    args.output.mkdir(parents=True, exist_ok=False)
    exports = []
    for view in args.views:
        exports.append(render(trials, quads, args.output, view))
        print(args.output / exports[-1]["path"], flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "scenario": trials["expert"]["row"]["scenario"],
                "training_seed": args.train_seed,
                "source_evaluation_manifest_sha256": digest(
                    args.root / "manifest.json"
                ),
                "renderer_sha256": digest(__file__),
                "native_quads_sha256": digest(quad_path),
                "inputs": {
                    method: {k: item[k] for k in ["result_sha256", "trajectory_sha256"]}
                    for method, item in trials.items()
                },
                "exports": exports,
                "note": "Animations of saved headless trajectories; no fresh simulation or training. Full-episode GIFs retain the original evaluation horizon.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
