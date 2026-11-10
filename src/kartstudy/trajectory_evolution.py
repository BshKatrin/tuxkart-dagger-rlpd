"""Evaluate saved policies and animate complete trajectories over training time.

uv run --with matplotlib python -m kartstudy.trajectory_evolution --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from .environment import ACTION_DIM, SCHEMA_VERSION, DrivingEnv
from .evaluation import evaluate
from .playback import METHODS, load_actor

COLORS = {
    "bc": "#0072B2",
    "dagger": "#E69F00",
    "sac": "#009E73",
    "sac_sym": "#CC79A7",
    "rlpd": "#D55E00",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_quads(path):
    """Read native driveline polygons, resolving earlier-corner references."""
    quads = []
    for quad in ElementTree.parse(path).getroot().findall("quad"):
        points = []
        for corner in range(4):
            value = quad.attrib[f"p{corner}"]
            if ":" in value:
                previous, index = map(int, value.split(":"))
                points.append(quads[previous][index])
            else:
                points.append(tuple(map(float, value.split())))
        quads.append(points)
    return np.asarray(quads)[:, :, [0, 2]]


def snapshot_sources(path, steps=None):
    """Discover actual saved actors; never infer or reconstruct missing history."""
    path = Path(path).resolve()
    if path.is_file():
        paths = [path]
    elif (path / "snapshots").is_dir():
        paths = sorted((path / "snapshots").glob("*/actor.pt"))
    else:
        paths = [
            path / ("actor.pt" if (path / "actor.pt").exists() else "checkpoint.pt")
        ]
    rows = []
    for checkpoint in paths:
        actor, config, step = load_actor(checkpoint)
        if config.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Incompatible observation/action schema: {checkpoint}")
        if actor.obs_dim != 84 or actor.action_dim != ACTION_DIM:
            raise ValueError(f"Incompatible policy dimensions: {checkpoint}")
        checksum = digest(checkpoint)
        manifest = checkpoint.parent / "manifest.json"
        if manifest.exists():
            recorded = json.loads(manifest.read_text())
            if "actor_file_sha256" in recorded and (
                recorded["actor_file_sha256"] != checksum
                or recorded["step"] != step
                or recorded["config"] != config
            ):
                raise ValueError(f"Snapshot provenance mismatch: {checkpoint}")
        rows.append(
            {
                "path": str(checkpoint),
                "sha256": checksum,
                "step": step,
                "config": config,
            }
        )
    if not rows:
        raise ValueError(f"No actor snapshots found: {path}")
    signature = lambda row: json.dumps(row["config"], sort_keys=True)
    if len({signature(row) for row in rows}) != 1:
        raise ValueError("Each input must contain one training run")
    if len({row["step"] for row in rows}) != len(rows):
        raise ValueError("Ambiguous snapshots at the same action count")
    rows.sort(key=lambda row: row["step"])
    if steps is not None and rows[0]["config"]["algo"] != "bc":
        rows = [row for row in rows if row["step"] in steps]
        if not rows:
            raise ValueError(f"None of the requested steps exist in {path}")
    return rows


def frame_source(run, step):
    """Only BC is a fixed reference; missing online snapshots stay missing."""
    if run["config"]["algo"] == "bc":
        return run["snapshots"][0]
    return next((row for row in run["snapshots"] if row["step"] == step), None)


def collect(inputs, output, track, seed, steps=None):
    import importlib.metadata
    import platform

    import torch
    from platformdirs import user_cache_dir

    torch.set_num_threads(1)
    runs = []
    for path in inputs:
        snapshots = snapshot_sources(path, steps)
        runs.append(
            {
                "input": str(Path(path).resolve()),
                "config": snapshots[0]["config"],
                "snapshots": snapshots,
            }
        )
    # The task horizon must match across inputs; never silently shorten failed laps.
    horizons = {run["config"]["max_steps"] for run in runs}
    if len(horizons) != 1:
        raise ValueError("Inputs must share the same evaluation horizon")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format_version": 1,
        "status": "collecting",
        "track": track,
        "evaluation_seed": seed,
        "control_interval_seconds": 0.1,
        "origin": "Fresh exploratory deterministic CPU evaluation; not official results",
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pysupertuxkart2", "pystk2-gymnasium")
        },
        "source_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in (
                "trajectory_evolution.py",
                "environment.py",
                "evaluation.py",
                "algorithms.py",
                "expert.py",
                "tracks.py",
                "native_telemetry.py",
                "native_telemetry.cpp",
            )
        },
        "runs": runs,
    }

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    save()
    env = DrivingEnv(track=track, max_steps=horizons.pop())
    try:
        manifest["expert"] = {"trajectory": "expert.npz"}
        manifest["expert"]["episodes"] = evaluate(
            env,
            lambda obs: env.expert_action(),
            [seed],
            trajectory_path=output / "expert.npz",
        )
        manifest["expert"]["trajectory_sha256"] = digest(output / "expert.npz")
        # Native assets are initialized by the first real evaluation, not an extra reset.
        quad_path = (
            Path(user_cache_dir("PySuperTuxKart2", "bpiwowar"))
            / "1.5/data/tracks"
            / track
            / "quads.xml"
        )
        quads = read_quads(quad_path)
        np.save(output / "driveline.npy", quads)
        manifest["geometry"] = {
            "path": "driveline.npy",
            "sha256": digest(output / "driveline.npy"),
            "native_path": str(quad_path),
            "native_sha256": digest(quad_path),
        }
        for index, run in enumerate(runs):
            config = run["config"]
            run["track_role"] = (
                "training"
                if track in config["tracks"]
                else "unfamiliar to learner training"
            )
            for row in run["snapshots"]:
                if digest(row["path"]) != row["sha256"]:
                    raise ValueError("Input checkpoint changed during collection")
                actor, _, _ = load_actor(row["path"])
                row["trajectory"] = f"run-{index:02d}-actions-{row['step']:09d}.npz"
                row["episodes"] = evaluate(
                    env, actor, [seed], trajectory_path=output / row["trajectory"]
                )
                row["trajectory_sha256"] = digest(output / row["trajectory"])
                episode = row["episodes"][0]
                print(
                    f"{config['algo']} seed {config['seed']}, {row['step']:,} actions: "
                    f"{'finished' if episode['success'] else 'failed'}, "
                    f"{episode['rescue_count']} rescues",
                    flush=True,
                )
                save()
        manifest["status"] = "completed"
        save()
    finally:
        env.close()
    return manifest


def read_trace(output, row):
    path = output / row["trajectory"]
    if digest(path) != row["trajectory_sha256"]:
        raise ValueError(f"Cached trajectory changed: {path}")
    with np.load(path) as trace:
        positions = trace["world_position"][:, [0, 2]].astype(float)
        active = trace["rescue_active"].astype(bool)
        starts = np.diff(trace["rescue_count"], prepend=0) > 0
    if (
        len(positions) != row["episodes"][0]["steps"]
        or not np.isfinite(positions).all()
    ):
        raise ValueError("Incomplete or nonfinite trajectory")
    driven = positions.copy()
    driven[active] = np.nan
    # No straight connection through a native teleport, including its first sample.
    driven[starts] = np.nan
    return positions, driven, positions[starts]


def render(output, duration=800, key_steps=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from PIL import Image

    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Collection is incomplete; use a fresh output directory")
    geometry = manifest["geometry"]
    if digest(output / geometry["path"]) != geometry["sha256"]:
        raise ValueError("Cached driveline changed")
    quads = np.load(output / geometry["path"])
    expert = read_trace(output, manifest["expert"])
    runs = manifest["runs"]
    steps = sorted(
        {
            row["step"]
            for run in runs
            if run["config"]["algo"] != "bc"
            for row in run["snapshots"]
        }
    ) or [0]
    traces = {
        row["trajectory"]: read_trace(output, row)
        for run in runs
        for row in run["snapshots"]
    }
    points = np.vstack(
        [quads.reshape(-1, 2), expert[0], *(v[0] for v in traces.values())]
    )
    lower, upper = points.min(0), points.max(0)
    padding = max(float((upper - lower).max()) * 0.04, 3)
    lower -= padding
    upper += padding
    ink, road = "#27323c", "#e6e9ee"

    def panel(ax, run, step):
        ax.add_collection(PolyCollection(quads, facecolors=road, edgecolors="none"))
        ax.plot(*expert[1].T, color=ink, lw=0.7, alpha=0.55, ls="--")
        ax.plot(*quads[0, :2].T, color=ink, lw=1.2)
        row = frame_source(run, step)
        config = run["config"]
        title = (
            f"{METHODS[config['algo']]} · seed {config['seed']} · {config['device']}"
        )
        if row is None:
            ax.text(
                0.5,
                0.5,
                "No snapshot at this step",
                transform=ax.transAxes,
                ha="center",
            )
            result = "Unavailable"
        else:
            positions, driven, rescues = traces[row["trajectory"]]
            color = COLORS[config["algo"]]
            ax.plot(*driven.T, color=color, lw=1.35)
            if len(rescues):
                ax.plot(
                    *rescues.T,
                    ls="none",
                    marker="X",
                    ms=5,
                    color=color,
                    mec="white",
                    mew=0.5,
                )
            episode = row["episodes"][0]
            ax.plot(
                *positions[-1],
                marker="o" if episode["timeout"] else "s",
                ms=4,
                mfc="white" if not episode["success"] else color,
                mec=color,
            )
            result = (
                f"Finished · {episode['lap_time']:.1f} s"
                if episode["success"]
                else f"{'Timeout' if episode['timeout'] else 'Failed'} · no completed lap"
            )
            result += f" · {episode['rescue_count']} rescues"
            if config["algo"] == "bc":
                result = "BC fitted reference (no online actions)\n" + result
        ax.set_title(title, fontsize=10, pad=8)
        ax.text(
            0.5,
            -0.045,
            result,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal")
        ax.set_axis_off()

    legend = [
        Line2D([], [], color=ink, ls="--", lw=0.8, label="Expert reference"),
        Line2D([], [], color=ink, marker="X", ls="none", label="Native rescue"),
        Line2D([], [], color=ink, marker="o", mfc="white", ls="none", label="Timeout"),
        Line2D([], [], color=ink, marker="s", ls="none", label="Finish"),
    ]

    def finish(fig, title):
        fig.suptitle(title, fontsize=12)
        fig.legend(
            handles=legend, loc="lower center", ncols=4, frameon=False, fontsize=8
        )
        fig.subplots_adjust(
            left=0.025, right=0.975, top=0.85, bottom=0.15, wspace=0.12, hspace=0.55
        )

    frames_dir = output / "frames"
    frames_dir.mkdir(exist_ok=True)
    images = []
    columns = min(3, len(runs))
    for step in steps:
        fig, axes = plt.subplots(
            int(np.ceil(len(runs) / columns)),
            columns,
            figsize=(4 * columns, 4.3 * np.ceil(len(runs) / columns)),
            squeeze=False,
        )
        for ax, run in zip(axes.flat, runs):
            panel(ax, run, step)
        for ax in list(axes.flat)[len(runs) :]:
            ax.set_visible(False)
        finish(
            fig,
            f"{manifest['track']} · {step:,} training actions · evaluation seed {manifest['evaluation_seed']}",
        )
        path = frames_dir / f"actions-{step:09d}.png"
        fig.savefig(path, dpi=120, facecolor="white")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
        plt.close(fig)
    images[0].save(
        output / "evolution.gif",
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )
    # A compact strip for each input, using action counts, never performance ranking.
    if key_steps is None:
        key_steps = [
            steps[i]
            for i in sorted(
                {0, min(1, len(steps) - 1), len(steps) // 2, len(steps) - 1}
            )
        ]
    if not set(key_steps).issubset(steps):
        raise ValueError("Key steps must exist on the animation timeline")
    for index, run in enumerate(runs):
        selected = [key_steps[0]] if run["config"]["algo"] == "bc" else key_steps
        fig, axes = plt.subplots(
            1, len(selected), figsize=(3.4 * len(selected), 4.2), squeeze=False
        )
        for ax, step in zip(axes.flat, selected, strict=True):
            panel(ax, run, step)
            ax.set_title(
                f"{step:,} actions"
                if run["config"]["algo"] != "bc"
                else "After fitting",
                fontsize=10,
            )
        finish(
            fig,
            f"{manifest['track']} · {METHODS[run['config']['algo']]} · training seed {run['config']['seed']} · evaluation seed {manifest['evaluation_seed']}",
        )
        for suffix in ("png", "pdf"):
            fig.savefig(
                output / f"run-{index:02d}-keyframes.{suffix}",
                dpi=160,
                facecolor="white",
            )
        plt.close(fig)
    (output / "render.json").write_text(
        json.dumps(
            {
                "steps": steps,
                "key_steps": key_steps,
                "frame_duration_ms": duration,
                "manifest_sha256": digest(output / "manifest.json"),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        help="Run folders, snapshot folders, or trusted actor/checkpoint files",
    )
    parser.add_argument("--track", help="Any installed race track ID")
    parser.add_argument("--eval-seed", type=int, default=20000)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        help="Select saved action counts; BC stays a fitted reference",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh directory for collection; existing cache for --render-only",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Redraw cached trajectories without simulation",
    )
    parser.add_argument("--frame-ms", type=int, default=800)
    parser.add_argument(
        "--key-steps", type=int, nargs="+", help="Action counts for the static strips"
    )
    args = parser.parse_args()
    if args.frame_ms < 20:
        parser.error("--frame-ms must be at least 20")
    if not args.render_only:
        if not args.input or not args.track:
            parser.error("Collection requires --input and --track")
        collect(args.input, args.output, args.track, args.eval_seed, args.steps)
    elif args.input or args.track or args.steps:
        parser.error("--render-only uses cached inputs/track/steps")
    render(args.output, args.frame_ms, args.key_steps)


if __name__ == "__main__":
    main()
