"""Portable frozen inputs and verification for the Monte Carlo release.

uv run python -m kartstudy.critic_artifacts --root .
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .algorithms import ensemble_forward
from .critic_mc import load_critic_checkpoint, soft_return
from .disturbance import Recovery
from .trajectory_evolution import digest

INPUTS = Path("runs/critic-mc-inputs-v1")
COMBINED = Path("runs/critic-mc-other-disturbances-v1")
RECOVERY = Path("runs/five-disturbances-v1/evaluation")
ALIGNED = Path("runs/critic-recovery-figure-v1")
CASES = (
    "left_throttle_lighthouse", "barrier_enterprise", "wrong_direction_cornfield",
    "black_forest_alternate_branch", "delay_corner_hacienda",
)


def read(path):
    return json.loads(Path(path).read_text())


def bank(case):
    return Path("runs/critic-mc-lighthouse-v1") if case == CASES[0] else COMBINED / case


def resolve_artifact(path, root=None):
    """Relocate original paths without rewriting the preserved run manifests."""
    root = Path.cwd() if root is None else Path(root)
    index = root / INPUTS / "manifest.json"
    path = Path(path)
    if index.exists():
        metadata = read(index)
        mapped = metadata["path_map"].get(str(path))
        if mapped is not None:
            relative = Path(mapped)
        elif path.is_absolute():
            relative = path.relative_to(metadata["original_repository_root"])
        else:
            relative = path
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError("Artifact path escapes the restored repository")
        return resolved
    return path if path.is_absolute() else root / path


def export_checkpoint(checkpoint, actor, destination):
    """Export exact evaluation weights, never replay, optimizers or resume state."""
    load_critic_checkpoint(checkpoint, actor)
    original = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    frozen = {
        "artifact_kind": "frozen-evaluation-checkpoint; not a training-resume snapshot",
        "source_checkpoint_sha256": digest(checkpoint),
        "config": original["config"], "step": original["step"],
        "actor": {k: v.clone() for k, v in original["actor"].items()},
        "learner": {"model": {k: v.clone() for k, v in original["learner"]["model"].items()}},
    }
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    torch.save(frozen, destination)
    return frozen["source_checkpoint_sha256"]


def export_inputs(root):
    """Create immutable portable inputs without modifying original run files."""
    root = Path(root).resolve()
    metadata = read(root / bank(CASES[0]) / "manifest.json")
    destination = root / INPUTS
    if destination.exists():
        raise FileExistsError(f"Frozen input export already exists: {destination}")
    destination.mkdir(parents=True)
    exported, path_map = [], {}
    for policy in metadata["policies"]:
        target = destination / policy["id"]
        target.mkdir()
        source_actor, source_checkpoint = Path(policy["actor"]), Path(policy["checkpoint"])
        if digest(source_actor) != policy["actor_sha256"] or digest(source_checkpoint) != policy["checkpoint_sha256"]:
            raise ValueError("Original actor/checkpoint digest changed")
        for name in ("actor.pt", "config.json", "status.json"):
            shutil.copyfile(source_actor.with_name(name), target / name)
        export_checkpoint(source_checkpoint, source_actor, target / "checkpoint.pt")
        entry = {"id": policy["id"], "source_checkpoint_sha256": policy["checkpoint_sha256"]}
        for name, source in (("actor", source_actor), ("checkpoint", source_checkpoint)):
            path = (target / f"{name}.pt").relative_to(root).as_posix()
            path_map[str(source)] = path
            entry[name] = path
            entry[name + "_sha256"] = digest(root / path)
        exported.append(entry)
    # Preserve the source archives used to qualify task/architecture equivalence.
    for item in read(root / bank(CASES[0]) / "source-audit.json"):
        source = Path(item["source_archive"])
        if digest(source) != item["source_archive_sha256"]:
            raise ValueError("Training source digest changed")
        target = destination / "training-sources" / (item["source_archive_sha256"] + ".tar.gz")
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        path_map[str(source)] = target.relative_to(root).as_posix()
    (destination / "policy-inputs.json").write_text(json.dumps([
        str((INPUTS / p["id"]).as_posix()) for p in exported
    ], indent=2))
    (destination / "manifest.json").write_text(json.dumps({
        "original_repository_root": str(root), "policies": exported, "path_map": path_map,
        "purpose": "Exact actor/critic/temperature evaluation weights; no replay or training-resume state",
    }, indent=2))


def prepare_release(root):
    """Return original data/support lists, reusing verified exports if present."""
    root = Path(root).resolve()
    destination = root / INPUTS
    if not destination.exists():
        export_inputs(root)
    data_roots = [bank(CASES[0]), COMBINED, RECOVERY, ALIGNED]
    files = sorted({p.relative_to(root) for folder in data_roots for p in (root / folder).rglob("*")
                    if p.is_file() and p.suffix != ".pyc" and p.name != ".DS_Store"})
    support = sorted(p.relative_to(root) for p in destination.rglob("*") if p.is_file())
    support.append(Path("runs/five-disturbances-v1/scenarios.json"))
    support.append(Path("runs/disturbance-study-20261110/policy-inputs.json"))
    figure = root / "report/generated/critic-error-recovery-report"
    support.extend(p.relative_to(root) for p in figure.iterdir()
                   if p.name.startswith("critic-error-recovery.") or p.name in ("caption.txt", "provenance.json"))
    return [("monte-carlo-rollouts.zip", files), ("monte-carlo-inputs.zip", sorted(support))]


def verify(root):
    """Recompute all 2700 returns, critic predictions and paired recovery labels."""
    root = Path(root).resolve()
    torch.set_num_threads(1)
    exports = {p["id"]: p for p in read(root / INPUTS / "manifest.json")["policies"]}
    models = {}
    for key, policy in exports.items():
        for name in ("actor", "checkpoint"):
            if digest(root / policy[name]) != policy[name + "_sha256"]:
                raise ValueError("Portable model digest mismatch")
        path = root / policy["checkpoint"]
        state = torch.load(path, weights_only=False, map_location="cpu", mmap=True)
        assert state["source_checkpoint_sha256"] == policy["source_checkpoint_sha256"]
        assert set(state["learner"]) == {"model"}
        models[key] = load_critic_checkpoint(path, root / policy["actor"])
    combined = {(r["scenario"], r["method"], r["training_seed"], r["condition"]): r
                for r in read(root / COMBINED / "results.json")["per_training_seed"]}
    assert len(combined) == 90 and len(models) == 9
    trials = groups = 0
    for case in CASES:
        directory = root / bank(case)
        manifest = read(directory / "manifest.json")
        assert manifest["status"] == "completed" and len(manifest["trials"]) == 540
        assert manifest["repetitions"] == 30 and manifest["response_horizon"] == 2500
        assert digest(directory / "source.tar.gz") == manifest["source_archive_sha256"]
        reference = resolve_artifact(manifest["reference"], root)
        assert digest(reference) == manifest["reference_sha256"]
        ref = read(reference)
        target = ref["delay_release"] if ref["scenario"].get("control_delay") else ref["handover"]
        assert manifest["states"]["perturbed"]["signature"] == target
        assert manifest["states"]["ordinary"]["signature"]["native_time"] == target["native_time"]
        for policy in manifest["policies"]:
            actor, critics, config, step, alpha = models[policy["id"]]
            export = exports[policy["id"]]
            assert config == policy["config"] and step == policy["step"] and alpha == policy["alpha"]
            assert export["actor_sha256"] == policy["actor_sha256"]
            assert export["source_checkpoint_sha256"] == policy["checkpoint_sha256"]
            assert digest(resolve_artifact(policy["actor"], root)) == policy["actor_sha256"]
            for condition in ("ordinary", "perturbed"):
                state, prediction = manifest["states"][condition], policy["states"][condition]
                tape = directory / f"state-{condition}.npz"
                assert digest(tape) == state["tape_sha256"]
                obs = np.asarray(state["signature"]["obs"], np.float32)
                action = np.asarray(prediction["first_action"], np.float32)
                with torch.inference_mode():
                    observation = torch.as_tensor(obs).unsqueeze(0)
                    np.testing.assert_array_equal(actor.act(observation).squeeze(0).numpy(), action)
                    qs = ensemble_forward(critics, observation, torch.as_tensor(action).unsqueeze(0))
                np.testing.assert_allclose(qs.squeeze(1).numpy(), prediction["critic_values"], atol=1e-5, rtol=1e-6)
                np.testing.assert_allclose(float(qs.mean()), prediction["q_mean"], atol=1e-5, rtol=1e-6)
                selected = [r for r in manifest["trials"] if r["policy"] == policy["id"] and r["condition"] == condition]
                assert sorted(r["noise_seed"] for r in selected) == manifest["action_noise_seeds"]
                returns = []
                for row in selected:
                    trace = directory / row["trace"]
                    assert digest(trace) == row["trace_sha256"]
                    with np.load(trace) as z:
                        np.testing.assert_array_equal(z["obs"][0], obs)
                        np.testing.assert_array_equal(z["action"][0], action)
                        assert z["log_prob"][0] == 0
                        value = soft_return(z["reward"], z["log_prob"], policy["gamma"], alpha)
                    assert abs(value - row["soft_return"]) < 1e-10
                    returns.append(value)
                    trials += 1
                result = combined[case, policy["method"], policy["seed"], condition]
                assert abs(prediction["q_mean"] - np.mean(returns) - result["b"]) < 1e-10
                assert abs(np.std(returns, ddof=1) / np.sqrt(30) - result["b_mc_se"]) < 1e-10
                groups += 1
    recovery_manifest = read(root / RECOVERY / "manifest.json")
    for row in recovery_manifest["trials"]:
        assert digest(root / RECOVERY / row["result"]) == row["result_sha256"]
        assert digest(root / RECOVERY / row["trajectory"]) == row["trajectory_sha256"]
    aligned = read(root / ALIGNED / "results.json")
    assert aligned["mc_manifest_sha256"] == digest(root / bank(CASES[-1]) / "manifest.json")
    assert len(aligned["trials"]) == 9
    delay_state = read(root / bank(CASES[-1]) / "manifest.json")["states"]["perturbed"]["signature"]
    for row in aligned["trials"]:
        assert row["state"] == delay_state
        assert digest(root / ALIGNED / row["trace"]) == row["trace_sha256"]
        recovery = Recovery(0, 0)
        with np.load(root / ALIGNED / row["trace"]) as z:
            np.testing.assert_array_equal(z["obs"][0], np.asarray(delay_state["obs"], np.float32))
            for i, seconds in enumerate(z["seconds"]):
                if seconds <= 15:
                    recovery.update({k: z[k][i] for k in ("progress", "is_on_road", "rescue_active", "rescue_count")}, float(seconds))
        assert recovery.time == row["recovery_seconds"]
        assert (recovery.time is not None) == row["recovered_within_15s"]
    assert trials == 2700 and groups == 90
    return {"status": "passed", "mc_rollouts": trials, "policy_state_groups": groups,
            "frozen_models": len(models), "original_recovery_trials": len(recovery_manifest["trials"]),
            "aligned_delay_trials": len(aligned["trials"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
