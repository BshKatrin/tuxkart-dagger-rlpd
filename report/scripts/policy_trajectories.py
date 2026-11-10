"""Preview matched policy trajectories on the native SuperTuxKart driveline.

From the repository root, after collecting the track-time-loss inputs:
    uv run --with matplotlib --with seaborn python report/scripts/policy_trajectories.py
    uv run --with matplotlib --with seaborn python report/scripts/policy_trajectories.py --track black_forest --seed 20001

Uses sources.json and the trajectory archives it references, plus the track's
installed PySuperTuxKart2 quads.xml. Replays the archived expert lap headlessly
to mark the first action of each controller recovery. Set --track, --seed or
--track-assets to change the inputs. Writes a PDF and PNG under report/generated/.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from platformdirs import user_cache_dir
from track_time_loss import DEFAULT_OUTPUT, PROJECT_ROOT, digest

from kartstudy.trajectory_evolution import read_quads

PANELS = (
    ("expert",),
    ("bc", "dagger"),
    ("sac", "sac_sym", "rlpd"),
)
LABELS = {
    "expert": "Expert",
    "bc": "BC",
    "dagger": "DAgger",
    "sac": "SAC",
    "sac_sym": "SAC + SS",
    "rlpd": "RLPD",
}


def read_lap(source, seed):
    path = PROJECT_ROOT / source["path"]
    if digest(path) != source["sha256"]:
        raise ValueError(f"Input trajectory changed: {path}")
    (episode,) = (row for row in source["episodes"] if row["seed"] == seed)
    with np.load(path) as data:
        selected = data["seed"] == seed
        positions = data["world_position"][selected][:, [0, 2]]
        rescue_active = data["rescue_active"][selected]
        rescue_starts = np.diff(data["rescue_count"][selected], prepend=0) > 0
    if len(positions) != episode["steps"] or not np.isfinite(positions).all():
        raise ValueError(f"Incomplete positions for seed {seed}: {path}")
    if rescue_starts.sum() != episode["rescue_count"]:
        raise ValueError(f"Rescue count mismatch for seed {seed}: {path}")
    driven = positions.copy()
    driven[rescue_active] = np.nan
    return positions, driven, positions[rescue_starts], episode["timeout"]


def expert_recoveries(track, seed, source):
    from kartstudy._team4.agent4 import Agent4
    from kartstudy._team4.AgentRescue import AgentRescue
    from kartstudy.environment import DrivingEnv

    with np.load(PROJECT_ROOT / source["path"]) as data:
        selected = data["seed"] == seed
        actions = data["commanded_action"][selected]
        positions = data["world_position"][selected]

    choose_recovery = AgentRescue.choose_action
    choose_action = Agent4.choose_action
    recovery_action = None
    recovering = False

    def capture_recovery(agent, obs, steer):
        nonlocal recovery_action
        active, action = choose_recovery(agent, obs, steer)
        if active:
            recovery_action = action
        return active, action

    def capture_action(agent, obs):
        nonlocal recovery_action, recovering
        recovery_action = None
        action = choose_action(agent, obs)
        # The coordinator can decline the recovery module's proposed action.
        recovering = recovery_action is not None and action is recovery_action
        return action

    events = []
    env = DrivingEnv(track=track)
    try:
        with (
            patch.object(AgentRescue, "choose_action", capture_recovery),
            patch.object(Agent4, "choose_action", capture_action),
        ):
            env.reset(seed=seed)
            was_recovering = False
            for step, (expected_action, expected_position) in enumerate(
                zip(actions, positions, strict=True), start=1
            ):
                world = env.env.unwrapped.world
                if recovering and not was_recovering:
                    x, _, z = world.karts[0].location
                    events.append((float(world.time), (float(x), float(z))))
                was_recovering = recovering
                action = env.expert_action()
                _, _, terminated, truncated, info = env.step(action)
                if (
                    not np.allclose(action, expected_action, rtol=0, atol=1e-5)
                    or not np.allclose(
                        info["world_position"], expected_position, rtol=0, atol=1e-4
                    )
                    or (terminated or truncated) != (step == len(actions))
                ):
                    raise ValueError(
                        f"Expert replay diverged from {track}, seed {seed}, step {step}"
                    )
    finally:
        env.close()
    return events


def plot(quads, laps, recoveries, output):
    sns.set_theme(context="paper", style="ticks", palette="colorblind")
    road, ink, recovery_color = "#e6e9ee", "#27323c", "#f7c548"
    palette = sns.color_palette()
    colors = dict(
        zip(LABELS, (ink, *(palette[i] for i in (0, 1, 2, 4, 3))), strict=True)
    )
    line_styles = {
        "dagger": (0, (2.7, 1.7)),
        "rlpd": (0, (3.2, 1.9)),
        "sac_sym": (0, (0.8, 1.9)),
    }
    corners = np.vstack(
        [quads.reshape(-1, 2), *(positions for positions, *_ in laps.values())]
    )
    lower, upper = corners.min(axis=0) - 3, corners.max(axis=0) + 3
    fig, axes = plt.subplots(1, 3, figsize=(17 / 2.54, 5.25 / 2.54))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.975, bottom=0.245, wspace=0.09)
    for ax, methods in zip(axes, PANELS, strict=True):
        ax.add_collection(
            PolyCollection(quads, facecolors=road, edgecolors=road, linewidths=0.2)
        )
        ax.plot(*quads[0, :2].T, color=ink, lw=0.9, zorder=3)
        lines = {}
        draw_order = ("sac", "rlpd", "sac_sym") if "sac" in methods else methods
        for method in draw_order:
            positions, driven, rescues, timeout = laps[method]
            color = colors[method]
            (line,) = ax.plot(
                *driven.T,
                color=color,
                lw=1.35 if method == "sac" else 1.05,
                ls=line_styles.get(method, "-"),
                solid_capstyle="round",
                dash_capstyle="round",
                label=LABELS[method],
            )
            lines[method] = line
            ax.plot(
                *rescues.T,
                ls="none",
                marker="X",
                ms=4.5,
                mfc=color,
                mec="white",
                mew=0.45,
                zorder=5,
            )
            if timeout:
                ax.plot(
                    *positions[-1],
                    ls="none",
                    marker="o",
                    ms=3.5,
                    mfc="white",
                    mec=color,
                    mew=0.85,
                    zorder=4,
                )
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal")
        ax.set_axis_off()
        if methods == ("expert",) and recoveries:
            ax.plot(
                *np.asarray([point for _, point in recoveries]).T,
                ls="none",
                marker="D",
                ms=4.2,
                mfc=recovery_color,
                mec=ink,
                mew=0.7,
                zorder=6,
            )

        column = ax.get_position(original=True)
        fig.legend(
            handles=[lines[method] for method in methods],
            loc="center",
            bbox_to_anchor=((column.x0 + column.x1) / 2, 0.192),
            ncols=len(lines),
            frameon=False,
            fontsize=8,
            columnspacing=0.8,
            handlelength=1.6,
            handletextpad=0.42,
        )

    handles = [
        Patch(facecolor=road, edgecolor="none", label="Driveline"),
        Line2D([], [], color=ink, marker="|", ms=7, ls="none", label="Start / finish"),
    ]
    if any(len(rescues) for _, _, rescues, _ in laps.values()):
        handles.append(
            Line2D(
                [], [], color=ink, marker="X", ms=4.5, ls="none", label="Native rescue"
            )
        )
    if recoveries:
        handles.append(
            Line2D(
                [],
                [],
                color=ink,
                marker="D",
                ms=4.2,
                ls="none",
                mfc=recovery_color,
                label="Expert recovery",
            )
        )
    if any(timeout for _, _, _, timeout in laps.values()):
        handles.append(
            Line2D(
                [],
                [],
                color=ink,
                marker="o",
                ms=3.5,
                ls="none",
                mfc="white",
                label="Timeout",
            )
        )
    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.057),
        ncols=len(handles),
        frameon=False,
        fontsize=7.5,
        handlelength=1.45,
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=150, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="zengarden")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument(
        "--track-assets",
        type=Path,
        default=Path(user_cache_dir("PySuperTuxKart2", "bpiwowar")) / "1.5/data/tracks",
    )
    args = parser.parse_args()
    metadata = json.loads((DEFAULT_OUTPUT / "sources.json").read_text())
    quads = read_quads(args.track_assets / args.track / "quads.xml")
    sources = metadata["sources"][args.track]
    laps = {method: read_lap(sources[method], args.seed) for method in LABELS}
    recoveries = expert_recoveries(args.track, args.seed, sources["expert"])
    for seconds, (x, z) in recoveries:
        print(
            f"{args.track}, seed {args.seed}: recovery at {seconds:.3f} s, x={x:.2f}, z={z:.2f}"
        )
    output = (
        DEFAULT_OUTPUT.parent / f"policy-trajectories-{args.track}-{args.seed}-grouped"
    )
    plot(quads, laps, recoveries, output)


if __name__ == "__main__":
    main()
