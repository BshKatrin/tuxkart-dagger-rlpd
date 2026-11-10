"""Collect and visualize fixed disturbances across genuine saved training actors.

Use collect with run folders; use render with one or more completed collections.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import tarfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from kartstudy.disturbance import make_env, trial
from kartstudy.playback import load_actor
from kartstudy.trajectory_evolution import COLORS, digest, read_quads, snapshot_sources

METHODS = ("bc", "dagger", "sac", "sac_sym", "rlpd")
LABELS = {
    "expert": "Expert",
    "bc": "BC",
    "dagger": "DAgger",
    "sac": "SAC",
    "sac_sym": "SAC + SS",
    "rlpd": "RLPD",
}
OUTCOMES = [
    "No timely recovery · timeout",
    "No timely recovery · finished",
    "Recovered ≤15 s · timeout",
    "Recovered ≤15 s · finished",
]
OUTCOME_COLORS = ["#d9dde2", "#E69F00", "#56B4E9", "#009E73"]


def outcome(row):
    return 2 * int(row["recovered_within_15s"]) + int(row["success"])


def collect(scenarios_path, inputs, output):
    scenarios = json.loads(scenarios_path.read_text())
    runs = [snapshot_sources(path) for path in inputs]
    keys = [(rows[0]["config"]["algo"], rows[0]["config"]["seed"]) for rows in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate method/training seed")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "scenarios": scenarios,
        "inputs": runs,
        "scenario_file_sha256": digest(scenarios_path),
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in ("torch", "numpy", "pysupertuxkart2", "pystk2-gymnasium")
        },
        "mode": "Frozen deterministic CPU actors; original 1500-action episode budget",
        "trials": [],
    }
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for path in Path("src/kartstudy").rglob("*"):
            if path.is_file() and path.suffix in (".py", ".cpp", ".yaml"):
                archive.add(path)
        archive.add(__file__, arcname="scripts/disturbance_evolution.py")
    manifest["source_archive_sha256"] = digest(output / "source.tar.gz")

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def record(stem, row, snapshot=None):
        row["snapshot"] = snapshot
        (output / f"{stem}.json").write_text(json.dumps(row, indent=2))
        manifest["trials"].append(
            {
                "id": stem,
                "result": f"{stem}.json",
                "result_sha256": digest(output / f"{stem}.json"),
                "trajectory": f"{stem}.npz",
                "trajectory_sha256": digest(output / f"{stem}.npz"),
            }
        )
        save()

    save()
    env = make_env(scenarios[0]["track"])
    try:
        for scenario in scenarios:
            reference_path = Path(scenario["reference_result"])
            if digest(reference_path) != scenario["reference_result_sha256"]:
                raise ValueError("Frozen expert reference changed")
            reference = json.loads(reference_path.read_text())
            stem = scenario["id"] + "--expert"
            expert = trial(env, scenario, "expert", save=output / stem)
            if expert["handover"] != reference["handover"]:
                raise ValueError("Expert handover differs from original selected case")
            record(stem, expert)
            for snapshots in runs:
                for snapshot in snapshots:
                    actor, config, step = load_actor(snapshot["path"])
                    if (
                        config["max_steps"] != 1500
                        or digest(snapshot["path"]) != snapshot["sha256"]
                    ):
                        raise ValueError("Incompatible horizon or changed checkpoint")
                    stem = f"{scenario['id']}--{config['algo']}-s{config['seed']}-a{step:09d}"
                    row = trial(
                        env, scenario, config["algo"], actor=actor, save=output / stem
                    )
                    if row["handover"] != expert["handover"]:
                        raise ValueError("Snapshot handover differs from frozen case")
                    record(stem, row, snapshot)
                    print(
                        stem,
                        "recover",
                        row["recovery_seconds"],
                        "finish",
                        row["success"],
                        "rescues",
                        row["native_rescues_after_handover"],
                        flush=True,
                    )
        for snapshots in runs:
            for snapshot in snapshots:
                if digest(snapshot["path"]) != snapshot["sha256"]:
                    raise ValueError("Checkpoint mutated during collection")
        manifest["status"] = "completed"
        save()
    finally:
        env.close()


def load_collections(roots):
    rows = []
    seen = set()
    scenarios = None
    for root in roots:
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError("Collections must be completed")
        if scenarios is not None and scenarios != manifest["scenarios"]:
            raise ValueError("Collections have different disturbances")
        scenarios = manifest["scenarios"]
        for item in manifest["trials"]:
            for key in ["result", "trajectory"]:
                if digest(root / item[key]) != item[key + "_sha256"]:
                    raise ValueError("Cached evidence hash mismatch")
            row = json.loads((root / item["result"]).read_text())
            snapshot = row["snapshot"]
            key = (
                row["scenario"]["id"],
                row["controller"],
                snapshot["config"]["seed"] if snapshot else None,
                snapshot["step"] if snapshot else None,
            )
            if key in seen:
                if snapshot is not None:
                    raise ValueError("Duplicate snapshot trial")
                previous = next(
                    r
                    for r in rows
                    if r["scenario"]["id"] == row["scenario"]["id"]
                    and r["snapshot"] is None
                )
                if row != {k: v for k, v in previous.items() if k != "trajectory_path"}:
                    raise ValueError("Expert replicates differ")
                continue
            seen.add(key)
            row["trajectory_path"] = str(root / item["trajectory"])
            rows.append(row)
    for scenario in scenarios:
        group = [r for r in rows if r["scenario"]["id"] == scenario["id"]]
        if any(r["handover"] != group[0]["handover"] for r in group):
            raise ValueError("Handover differs across collections")
    return scenarios, rows


def read_trace(row):
    with np.load(row["trajectory_path"]) as t:
        response = t["phase"] == "response"
        first = np.flatnonzero(response)[0]
        time = t["elapsed_seconds"] - t["elapsed_seconds"][first - 1]
        pos = t["world_position"][:, [0, 2]].astype(float)
        starts = np.diff(t["rescue_count"], prepend=0) > 0
        driven = pos.copy()
        driven[t["rescue_active"] | starts] = np.nan
        return {
            "time": time,
            "phase": t["phase"],
            "pos": pos,
            "driven": driven,
            "rescue_start": starts,
        }


def render(roots, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from PIL import Image
    from platformdirs import user_cache_dir

    scenarios, rows = load_collections(roots)
    output.mkdir(parents=True, exist_ok=False)
    seeds = sorted({r["snapshot"]["config"]["seed"] for r in rows if r["snapshot"]})
    steps = sorted(
        {
            r["snapshot"]["step"]
            for r in rows
            if r["snapshot"] and r["controller"] != "bc"
        }
    )
    summaries = []
    for row in rows:
        s = row["snapshot"]
        summaries.append(
            {
                "scenario": row["scenario"]["id"],
                "method": row["controller"],
                "training_seed": s["config"]["seed"] if s else None,
                "training_actions": s["step"] if s else None,
                "fixed_reference": s is None or row["controller"] == "bc",
                "recovery_seconds": row["recovery_seconds"],
                "recovered_within_15s": row["recovered_within_15s"],
                "completed": row["success"],
                "successful_lap_time": row["lap_time"],
                "later_native_rescues": row["native_rescues_after_handover"],
            }
        )
    (output / "results.json").write_text(json.dumps(summaries, indent=2))
    with (output / "results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for scenario in scenarios:
        case = [r for r in rows if r["scenario"]["id"] == scenario["id"]]
        expert = next(r for r in case if r["snapshot"] is None)
        lookup = {
            (r["controller"], r["snapshot"]["config"]["seed"], r["snapshot"]["step"]): r
            for r in case
            if r["snapshot"]
        }

        def get(method, seed, step, expert=expert, lookup=lookup):
            return (
                expert
                if method == "expert"
                else lookup.get((method, seed, 0 if method == "bc" else step))
            )

        # Explicit action counts; BC and expert references are held fixed, not extra trials.
        matrix = []
        labels = []
        matrix.append([outcome(expert)] * len(steps))
        labels.append("Expert · fixed")
        for seed in seeds:
            for method in METHODS:
                matrix.append(
                    [
                        outcome(get(method, seed, step))
                        if get(method, seed, step)
                        else np.nan
                        for step in steps
                    ]
                )
                labels.append(
                    f"{LABELS[method]} · seed {seed}"
                    + (" · fixed" if method == "bc" else "")
                )
        fig, ax = plt.subplots(figsize=(9.2, 5.7))
        ax.imshow(
            matrix,
            cmap=ListedColormap(OUTCOME_COLORS),
            vmin=-0.5,
            vmax=3.5,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set(
            xticks=range(len(steps)),
            xticklabels=[f"{s // 1000}" for s in steps],
            yticks=range(len(labels)),
            yticklabels=labels,
            xlabel="Training actions (thousands)",
            title=scenario["track"] + " · recovery from the same disturbed state",
        )
        ax.tick_params(length=0, labelsize=10)
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(steps)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        fig.legend(
            handles=[
                Patch(facecolor=color, label=label)
                for color, label in zip(OUTCOME_COLORS, OUTCOMES, strict=True)
            ],
            loc="lower center",
            ncols=2,
            frameon=False,
            fontsize=9,
        )
        fig.subplots_adjust(left=0.23, right=0.98, top=0.90, bottom=0.23)
        for suffix in ["png", "pdf"]:
            fig.savefig(
                output / f"{scenario['track']}-recovery-map.{suffix}",
                dpi=160,
                facecolor="white",
            )
        plt.close(fig)
        quads = read_quads(
            Path(user_cache_dir("PySuperTuxKart2", "bpiwowar"))
            / "1.5/data/tracks"
            / scenario["track"]
            / "quads.xml"
        )
        traces = {id(r): read_trace(r) for r in case}
        anchor = np.asarray(expert["anchor"]["position"])[[0, 2]]
        handover = np.asarray(expert["handover"]["position"])[[0, 2]]
        center = (anchor + handover) / 2
        radius = max(45, float(np.linalg.norm(anchor - handover)) * 0.75)
        for seed in seeds:
            fig, axes = plt.subplots(2, 3, figsize=(10.8, 8), dpi=100)
            artists = {}
            for ax, method in zip(axes.flat, ("expert", *METHODS), strict=True):
                color = COLORS.get(method, "#27323c")
                ax.add_collection(
                    PolyCollection(quads, facecolors="#e6e9ee", edgecolors="none")
                )
                trace = traces[id(expert)]
                mask = trace["phase"] == "disturbance"
                ax.plot(*trace["driven"][mask].T, ls=":", color="#7d8790", lw=1)
                ax.plot(*handover, marker="o", mfc="white", mec=color, ms=5, zorder=5)
                ax.plot(*anchor, marker="|", color="#27323c", ms=8)
                (line,) = ax.plot([], [], color=color, lw=1.4)
                (marks,) = ax.plot([], [], ls="none", marker="X", ms=5, color=color)
                text = ax.text(
                    0.5,
                    -0.035,
                    "",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
                ax.set_title(
                    LABELS[method]
                    + (
                        " · fitted reference"
                        if method == "bc"
                        else " · fixed"
                        if method == "expert"
                        else ""
                    ),
                    fontsize=10,
                )
                ax.set(
                    xlim=(center[0] - radius, center[0] + radius),
                    ylim=(center[1] - radius, center[1] + radius),
                    aspect="equal",
                )
                ax.axis("off")
                artists[method] = (line, marks, text)
            title = fig.suptitle("", fontsize=12)
            fig.text(
                0.5,
                0.925,
                f"Training seed {seed} · complete first 15 s after handover · evaluation seed {scenario['eval_seed']}",
                ha="center",
                fontsize=9,
            )
            fig.text(
                0.5,
                0.035,
                "Dotted gray: forced disturbance · ○ handover · × native rescue · outcomes refer to the full saved episode",
                ha="center",
                fontsize=8,
            )
            fig.subplots_adjust(
                left=0.025, right=0.975, top=0.85, bottom=0.14, hspace=0.55, wspace=0.15
            )
            images = []
            frame_dir = output / f"{scenario['track']}-seed{seed}-frames"
            frame_dir.mkdir()
            for step in steps:
                title.set_text(
                    f"{scenario['track']} · {step:,} training actions · fixed disturbance"
                )
                for method, (line, marks, text) in artists.items():
                    row = get(method, seed, step)
                    if row is None:
                        line.set_data([], [])
                        marks.set_data([], [])
                        text.set_text("No saved snapshot")
                        continue
                    tr = traces[id(row)]
                    mask = (tr["phase"] == "response") & (tr["time"] <= 15)
                    line.set_data(tr["driven"][mask].T)
                    marks.set_data(tr["pos"][mask & tr["rescue_start"]].T)
                    rt = row["recovery_seconds"]
                    label = (
                        "No independent recovery"
                        if rt is None
                        else f"Recovery {rt:.1f} s" + (" (late)" if rt > 15 else "")
                    )
                    label += "\n" + (
                        "Finished" + f" · {row['lap_time']:.1f} s"
                        if row["success"]
                        else "Timed out"
                    )
                    label += f" · {row['native_rescues_after_handover']} later rescues"
                    text.set_text(label)
                fig.canvas.draw()
                image = Image.fromarray(
                    np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
                )
                image.save(frame_dir / f"actions-{step:09d}.png")
                images.append(image.quantize(colors=128))
            images[0].save(
                output / f"{scenario['track']}-seed{seed}-evolution.gif",
                save_all=True,
                append_images=images[1:],
                duration=1100,
                loop=0,
                disposal=2,
            )
            plt.close(fig)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "collections": [
                    {
                        "path": str(r.resolve()),
                        "manifest_sha256": digest(r / "manifest.json"),
                    }
                    for r in roots
                ],
                "renderer_sha256": digest(__file__),
                "training_seeds": seeds,
                "training_steps": steps,
                "legend": OUTCOMES,
                "notes": "Each GIF frame changes the training checkpoint, not driving time. Complete first-15-second response paths; full-episode outcomes in labels. Fixed expert/BC references do not create additional trials. Black Forest expert failure is budget-dependent.",
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--scenarios", type=Path, required=True)
    c.add_argument("--input", type=Path, nargs="+", required=True)
    c.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("render")
    r.add_argument("--root", type=Path, nargs="+", required=True)
    r.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.scenarios, args.input, args.output)
    else:
        render(args.root, args.output)


if __name__ == "__main__":
    main()
