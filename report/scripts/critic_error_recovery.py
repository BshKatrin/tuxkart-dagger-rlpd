"""Plot per-training-seed signed critic errors colored by deterministic recovery.

uv run --with matplotlib --with seaborn python report/scripts/critic_error_recovery.py

Existing pulse/tape recovery results are reused after exact state/actor checks.
Latency is reevaluated from the critic experiment's shared post-delay state;
its original policy-specific post-delay states must not supply these colors.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from kartstudy.critic_artifacts import resolve_artifact
from kartstudy.disturbance import Recovery, signature
from kartstudy.environment import DrivingEnv
from kartstudy.playback import load_actor
from kartstudy.trajectory_evolution import digest

ROOT = Path(__file__).resolve().parents[2]
MC = ROOT / "runs/critic-mc-other-disturbances-v1"
RECOVERY = ROOT / "runs/five-disturbances-v1/evaluation"
CASES = (
    ("left_throttle_lighthouse", "Left + throttle", "Lighthouse"),
    ("barrier_enterprise", "Barrier", "Enterprise"),
    ("wrong_direction_cornfield", "Wrong direction", "Cornfield"),
    ("black_forest_alternate_branch", "Alternate branch", "Black Forest"),
    ("delay_corner_hacienda", "Delayed controls", "Hacienda"),
)
METHODS = ("sac", "sac_sym", "rlpd")
LABELS = ("SAC", "SAC+SS", "RLPD")
GREEN, RED, INK = "#00885b", "#ce3f49", "#27333d"


def read(path):
    return json.loads(Path(path).read_text())


def bank_for(case):
    if case == "left_throttle_lighthouse":
        return ROOT / "runs/critic-mc-lighthouse-v1"
    return MC / case


def aligned_delay(destination):
    """Use the exact frozen history, with normal controls for the next 15 s."""
    bank = bank_for("delay_corner_hacienda")
    manifest = read(bank / "manifest.json")
    if destination.exists():
        result = read(destination / "results.json")
        assert result["mc_manifest_sha256"] == digest(bank / "manifest.json")
        for row in result["trials"]:
            assert digest(destination / row["trace"]) == row["trace_sha256"]
        return result
    destination.mkdir(parents=True)
    state = manifest["states"]["perturbed"]
    assert digest(bank / "state-perturbed.npz") == state["tape_sha256"]
    with np.load(bank / "state-perturbed.npz") as tape:
        prefix = tape["prefix_actions"]
    env = DrivingEnv(track=state["track"], max_steps=len(prefix) + 160)
    torch.set_num_threads(1)
    trials = []
    try:
        for policy in manifest["policies"]:
            actor_path = resolve_artifact(policy["actor"], ROOT)
            assert digest(actor_path) == policy["actor_sha256"]
            actor, config, step = load_actor(actor_path)
            assert config == policy["config"] and step == policy["step"]
            obs, _ = env.reset(
                seed=manifest["scenario"]["eval_seed"], track=state["track"]
            )
            for action in prefix:
                obs, _, term, trunc, info = env.step(action)
                assert not term and not trunc
            assert signature(env, obs) == state["signature"]
            assert info["rescue_count"] == 0 and not info["rescue_active"]
            start = info["elapsed_seconds"]
            recovery = Recovery(info["progress"], info["rescue_count"])
            trace = {key: [] for key in (
                "obs", "action", "seconds", "progress", "is_on_road",
                "rescue_count", "rescue_active",
            )}
            with torch.inference_mode():
                for index in range(151):
                    trace["obs"].append(obs.copy())
                    action = actor.act(torch.as_tensor(obs).unsqueeze(0)).squeeze(0).numpy()
                    if index == 0:
                        np.testing.assert_array_equal(
                            action, np.asarray(policy["states"]["perturbed"]["first_action"], np.float32)
                        )
                    obs, _, term, trunc, info = env.step(action)
                    seconds = info["elapsed_seconds"] - start
                    trace["action"].append(action.copy())
                    trace["seconds"].append(seconds)
                    for key in ("progress", "is_on_road", "rescue_count", "rescue_active"):
                        trace[key].append(info[key])
                    if seconds <= 15:
                        recovery.update(info, seconds)
                    if seconds >= 15 or term or trunc:
                        break
            name = policy["id"] + ".npz"
            np.savez_compressed(destination / name, **trace)
            row = {
                "policy": policy["id"], "actor_sha256": policy["actor_sha256"],
                "state": state["signature"], "recovery_seconds": recovery.time,
                "recovered_within_15s": recovery.time is not None,
                "trace": name, "trace_sha256": digest(destination / name),
            }
            trials.append(row)
            print(policy["id"], "recovered" if recovery.time is not None else "no recovery", recovery.time, flush=True)
    finally:
        env.close()
    result = {
        "status": "completed", "mc_manifest_sha256": digest(bank / "manifest.json"),
        "controller": "deterministic tanh(mean), CPU",
        "state_protocol": "shared expert-generated post-delay state, actuator queue discarded",
        "recovery_definition": read(RECOVERY / "manifest.json")["recovery_definition"],
        "deadline_seconds": 15, "trials": trials,
    }
    (destination / "results.json").write_text(json.dumps(result, indent=2))
    return result


def collect(delay):
    rows = []
    metrics = read(MC / "results.json")["per_training_seed"]
    recovery_manifest = read(RECOVERY / "manifest.json")
    original_trials = {t["id"]: t for t in recovery_manifest["trials"]}
    for case, _, _ in CASES:
        bank = bank_for(case)
        manifest = read(bank / "manifest.json")
        assert manifest["status"] == "completed"
        for policy in manifest["policies"]:
            assert digest(resolve_artifact(policy["actor"], ROOT)) == policy["actor_sha256"]
            metric, = (r for r in metrics if r["scenario"] == case
                       and r["method"] == policy["method"]
                       and r["training_seed"] == policy["seed"]
                       and r["condition"] == "perturbed")
            if case == "delay_corner_hacienda":
                result, = (r for r in delay["trials"] if r["policy"] == policy["id"])
                assert result["state"] == manifest["states"]["perturbed"]["signature"]
                assert result["actor_sha256"] == policy["actor_sha256"]
                provenance = "shared post-delay state reevaluation"
            else:
                trial = original_trials[case + "--" + policy["id"]]
                path = RECOVERY / trial["result"]
                assert digest(path) == trial["result_sha256"]
                result = read(path)
                assert result["handover"] == manifest["states"]["perturbed"]["signature"]
                assert result["policy"]["sha256"] == policy["actor_sha256"]
                assert result["prefix_rescues"] == 0 and result["disturbance_rescues"] == 0
                provenance = "original deterministic recovery trial"
            assert result["recovered_within_15s"] == (
                result["recovery_seconds"] is not None and result["recovery_seconds"] <= 15
            )
            rows.append({**metric, "recovered_within_15s": result["recovered_within_15s"],
                         "recovery_seconds": result["recovery_seconds"], "recovery_source": provenance})
    assert len(rows) == 45
    for case, _, _ in CASES:
        for method in METHODS:
            assert sorted(r["training_seed"] for r in rows if r["scenario"] == case
                          and r["method"] == method) == [0, 1, 2]
    return rows


def plot(rows, output):
    sns.set_theme(context="paper", style="ticks", palette="colorblind", font="DejaVu Sans")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(5, 3, figsize=(6.1, 4.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.28, right=.985, top=.90, bottom=.20,
                        wspace=.10, hspace=.15)
    for ri, (case, title, _) in enumerate(CASES):
        for ci, method in enumerate(METHODS):
            ax = axes[ri, ci]
            ax.set_facecolor("#f3f5f6")
            ax.axvline(0, color="#a2adb5", lw=.85, zorder=1)
            selected = sorted((r for r in rows if r["scenario"] == case and r["method"] == method),
                              key=lambda r: r["training_seed"])
            for row in selected:
                ax.barh(2 - row["training_seed"], row["b"], height=.48,
                        color=GREEN if row["recovered_within_15s"] else RED,
                        edgecolor="none", zorder=2)
            ax.set(xlim=(-15, 27), ylim=(-.6, 2.6), yticks=[], xticks=[-10, 0, 10, 20])
            ax.grid(False)
            ax.tick_params(axis="x", bottom=ri == 4, labelbottom=ri == 4,
                           length=2.5, width=.6, pad=3, labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if ri == 0:
                ax.set_title(LABELS[ci], fontsize=11, fontweight="bold", pad=10)
        position = axes[ri, 0].get_position()
        fig.text(.025, (position.y0 + position.y1) / 2, title,
                 ha="left", va="center", fontsize=10.5, color=INK)
    fig.text(.6325, .11, r"Bias ($Q - G$)", ha="center", fontsize=10)
    fig.legend(handles=[
        Patch(facecolor=GREEN, label="Recovered within 15 s"),
        Patch(facecolor=RED, label="No recovery within 15 s"),
    ], loc="lower center", bbox_to_anchor=(.6325, .025), ncols=2, frameon=False,
       handlelength=1, handletextpad=.6, columnspacing=1.5, fontsize=8)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"critic-error-recovery.{extension}", dpi=150, facecolor="white")
    plt.close(fig)
    with (output / "critic-error-recovery.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "caption.txt").write_text(
        "Critic bias (Q - G) and deterministic recovery at five perturbed states. "
        "Q is the mean online critic ensemble prediction; G is the mean "
        "Monte Carlo soft return from 30 continuations, including future entropy "
        "bonuses but excluding entropy at the fixed first action. "
        "Rows identify disturbances and columns identify methods. Each bar is "
        "one disturbance/training-seed pair: five disturbances and three "
        "independently trained policies per method, 15 bars per method. "
        "The same trained policy is evaluated across all five disturbances; "
        "these are not 15 independent training runs. Seeds 0–2 appear top to "
        "bottom within each cell. All cells share the same horizontal bias scale "
        "and zero reference. "
        "Positive bias indicates overestimation and negative bias underestimation of the "
        "stochastic-continuation soft value, estimated from 30 Monte Carlo rollouts. "
        "Green bars indicate deterministic recovery within 15 s; red bars indicate no recovery "
        "within that deadline. Recovery requires 21 consecutive "
        "on-road observations spanning 2 s with monotonic progress of at least 10 m "
        "before any native rescue. All colors are paired with b at the exact same "
        "physical state and actor checkpoint. For delayed controls, this is the "
        "shared expert-generated state at control restoration, reevaluated with "
        "normal controls; these are not the original policy-specific latency outcomes. "
        "The figure shows association between stochastic-value calibration and "
        "deterministic recovery, not a causal effect.\n"
    )
    (output / "provenance.json").write_text(json.dumps({
        "script_sha256": digest(__file__),
        "mc_results_sha256": digest(MC / "results.json"),
        "original_recovery_manifest_sha256": digest(RECOVERY / "manifest.json"),
        "bars": len(rows), "training_seeds": [0, 1, 2],
        "layout": "disturbance rows, method columns; seeds 0–2 top to bottom in every cell",
        "shared_bias_limits": [-15, 27],
        "intervals": None,
        "conditions": ["perturbed"], "recovery_controller": "deterministic",
        "b_reference_controller": "fixed initial mean action, then stochastic",
        "mc_manifests_sha256": {
            case: digest(bank_for(case) / "manifest.json") for case, _, _ in CASES
        },
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "report/generated/critic-error-recovery")
    parser.add_argument("--aligned-delay", type=Path, default=ROOT / "runs/critic-recovery-figure-v1")
    args = parser.parse_args()
    delay = aligned_delay(args.aligned_delay)
    rows = collect(delay)
    plot(rows, args.output)
    print("Saved", args.output / "critic-error-recovery.png", flush=True)


if __name__ == "__main__":
    main()
