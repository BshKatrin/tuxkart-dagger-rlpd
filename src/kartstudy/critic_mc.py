"""Frozen-critic Monte Carlo calibration at matched disturbance states.

uv run python -m kartstudy.critic_mc --inputs POLICY_INPUTS.json \
  --reference runs/five-disturbances-v1/evaluation/left_throttle_lighthouse--expert.json \
  --output runs/new-critic-mc

No training occurs. The first action is fixed to tanh(mean); subsequent actions
are sampled. Q includes discounted future entropy, excluding the first action's
entropy. Only signed value error is summarized. Episode truncation is a numeric
integration cutoff, not a zero-valued terminal target.
"""

from __future__ import annotations

import argparse
import json
import platform
import tarfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t as student_t

from .algorithms import Critic, ensemble_forward
from .disturbance import ControlDelay, signature
from .environment import DrivingEnv
from .playback import load_actor
from .trajectory_evolution import digest


def soft_return(rewards, log_probs, gamma, alpha):
    """Finite discounted SAC Q return: entropy starts at the second action."""
    rewards, log_probs = np.asarray(rewards, float), np.asarray(log_probs, float)
    if rewards.ndim != 1 or rewards.shape != log_probs.shape or not len(rewards):
        raise ValueError("Require aligned nonempty reward/log-probability vectors")
    terms = rewards - alpha * log_probs
    terms[0] = rewards[0]
    return float(np.dot(gamma ** np.arange(len(terms)), terms))


def load_critic_checkpoint(checkpoint, portable_actor):
    """Require exact equality of full-checkpoint and evaluated actor weights."""
    actor, exported_config, exported_step = load_actor(portable_actor)
    # Trusted original experiment artifacts, verified against the release ledger.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = state["config"]
    if config != exported_config or state["step"] != exported_step:
        raise ValueError("Full checkpoint and portable actor configuration/step differ")
    if config["algo"] not in ("sac", "sac_sym", "rlpd"):
        raise ValueError("A trained SAC-family critic is required")
    if (
        config.get("schema_version") != 3
        or actor.obs_dim != 84
        or actor.action_dim != 5
    ):
        raise ValueError("Require the study observation/control schema")
    model = state["learner"]["model"]
    for name, weight in actor.state_dict().items():
        if not torch.equal(weight, state["actor"][name]) or not torch.equal(
            weight, model["actor." + name]
        ):
            raise ValueError("Critic checkpoint does not match the evaluated actor")
    n = 10 if config["algo"] == "rlpd" else 2
    critics = torch.nn.ModuleList(
        [Critic(84, 5, config["hidden"], config["algo"] == "rlpd") for _ in range(n)]
    )
    weights = {
        key.removeprefix("critics."): val
        for key, val in model.items()
        if key.startswith("critics.")
    }
    critics.load_state_dict(weights, strict=True)
    alpha = float(model["log_alpha"].exp())
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid saved entropy temperature")
    return actor.eval(), critics.eval(), config, exported_step, alpha


def freeze_states(env, reference):
    """Matched-time normal/intervened histories, measured after control restoration.

    Temporary latency uses the original expert's queued controls. This yields
    one fixed post-delay state for every evaluated policy, rather than choosing
    different post-delay states according to each policy's own requested actions.
    """
    scenario = reference["scenario"]
    states = {}
    for condition in ("ordinary", "perturbed"):
        obs, _ = env.reset(seed=scenario["eval_seed"], track=scenario["track"])
        info = None
        tape = []

        def execute(action, tape=tape):
            nonlocal obs, info
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                raise ValueError("Episode ended before the measurement state")
            tape.append(np.asarray(action).copy())

        while info is None or info["progress"] < scenario["anchor_m"]:
            execute(env.expert_action())
        anchor = signature(env, obs)
        for pulse in scenario["pulses"]:
            for _ in range(pulse["steps"]):
                action = env.expert_action()
                if condition == "perturbed":
                    for index, key in enumerate(
                        ("steer", "throttle", "brake", "nitro", "fire")
                    ):
                        if key in pulse:
                            action[index] = pulse[key]
                execute(action)
        delay_config = scenario.get("control_delay")
        if delay_config:
            delay = ControlDelay(
                tape, delay_config["steps"], delay_config["duration_steps"]
            )
            for step in range(delay_config["duration_steps"]):
                requested = env.expert_action()
                execute(
                    delay.execute(requested, step)
                    if condition == "perturbed"
                    else requested
                )
        if info["rescue_count"] or info["rescue_active"]:
            raise ValueError("Measurement state must be free of native rescue")
        states[condition] = {
            "signature": signature(env, obs),
            "anchor": anchor,
            "prefix_actions": np.asarray(tape),
            "prefix_steps": len(tape),
            "track": scenario["track"],
            "measurement_phase": "after_control_restoration",
        }
    if states["ordinary"]["anchor"] != states["perturbed"]["anchor"]:
        raise ValueError("Control and perturbation prefixes differ")
    target = (
        reference["delay_release"]
        if scenario.get("control_delay")
        else reference["handover"]
    )
    if states["perturbed"]["signature"] != target:
        raise ValueError(
            "Perturbed state differs from the original recovery experiment"
        )
    if (
        states["ordinary"]["signature"]["native_time"]
        != states["perturbed"]["signature"]["native_time"]
    ):
        raise ValueError("Ordinary and perturbed state times differ")
    return states


def continuation(
    env, state, eval_seed, actor, first_action, gamma, alpha, noise_seed, horizon
):
    obs, _ = env.reset(seed=eval_seed, track=state["track"])
    for action in state["prefix_actions"]:
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            raise ValueError("Prefix replay ended early")
    if signature(env, obs) != state["signature"]:
        raise ValueError("Monte Carlo initial state differs from the frozen state")
    rewards, logs, observations, actions = [], [], [], []
    terminal = False
    with torch.inference_mode(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(noise_seed)
        for step in range(horizon):
            observations.append(obs.copy())
            if step == 0:
                action, log_prob = first_action.copy(), 0.0
            else:
                sampled = actor(torch.as_tensor(obs).unsqueeze(0))
                action, log_prob = (
                    sampled.value.squeeze(0).numpy(),
                    float(sampled.log_prob.item()),
                )
            obs, reward, terminal, truncated, _ = env.step(action)
            actions.append(action.copy())
            rewards.append(reward)
            logs.append(log_prob)
            if terminal:
                break
            if truncated and step + 1 < horizon:
                raise ValueError("Unexpected simulator time limit within continuation")
    value = soft_return(rewards, logs, gamma, alpha)
    # A long independent continuation makes the omitted discounted tail tiny.
    # Inspect convergence without using the critic under test for bootstrapping.
    split = min(1500, len(rewards))
    early = soft_return(rewards[:split], logs[:split], gamma, alpha)
    if not terminal and abs(value - early) > 0.001:
        raise ValueError("The discounted tail has not converged to 0.001 return units")
    result = {
        "noise_seed": noise_seed,
        "soft_return": value,
        "steps": len(rewards),
        "terminal": terminal,
        "cutoff_discount": 0.0 if terminal else gamma ** len(rewards),
        "tail_after_1500_steps": value - early,
    }
    trace = {
        "obs": np.asarray(observations),
        "action": np.asarray(actions),
        "reward": np.asarray(rewards),
        "log_prob": np.asarray(logs),
    }
    if not all(np.isfinite(a).all() for a in trace.values()):
        raise ValueError("Nonfinite Monte Carlo data")
    return result, trace


def summarize(manifest, output):
    import csv

    rows = []
    for policy in manifest["policies"]:
        for condition in ("ordinary", "perturbed"):
            group = [
                r
                for r in manifest["trials"]
                if r["policy"] == policy["id"] and r["condition"] == condition
            ]
            if len(group) != manifest["repetitions"]:
                raise ValueError("Incomplete Monte Carlo group")
            values = np.asarray([r["soft_return"] for r in group])
            q = policy["states"][condition]["q_mean"]
            se = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else None
            )
            half = (
                float(student_t.ppf(0.975, len(values) - 1) * se)
                if se is not None
                else None
            )
            rows.append(
                {
                    "method": policy["method"],
                    "training_seed": policy["seed"],
                    "condition": condition,
                    "n": len(values),
                    "q_mean": q,
                    "mc_mean": float(values.mean()),
                    "b": float(q - values.mean()),
                    "b_mc_se": se,
                    "b_mc_95_halfwidth": half,
                }
            )
    means = [
        {
            "method": method,
            "condition": condition,
            "training_seeds": sorted(
                r["training_seed"]
                for r in rows
                if r["method"] == method and r["condition"] == condition
            ),
            "mean_b": float(
                np.mean(
                    [
                        r["b"]
                        for r in rows
                        if r["method"] == method and r["condition"] == condition
                    ]
                )
            ),
        }
        for method in ("sac", "sac_sym", "rlpd")
        for condition in ("ordinary", "perturbed")
    ]
    (output / "results.json").write_text(
        json.dumps(
            {"per_training_seed": rows, "across_training_seeds": means}, indent=2
        )
    )
    with (output / "signed-error.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run(inputs, reference_path, output, repetitions=30, horizon=2500):
    torch.set_num_threads(1)
    if repetitions < 2 or horizon < 2000:
        raise ValueError(
            "Require at least 2 samples and a sufficiently long integration horizon"
        )
    reference = json.loads(reference_path.read_text())
    scenario = reference["scenario"]
    if not scenario["pulses"] and not scenario.get("control_delay"):
        raise ValueError("Require an intervened reference condition")
    source_runs = [Path(p) for p in json.loads(inputs.read_text())]
    policies, models = [], {}
    for path in source_runs:
        config = json.loads((path / "config.json").read_text())
        if config["algo"] not in ("sac", "sac_sym", "rlpd"):
            continue
        if json.loads((path / "status.json").read_text())["status"] != "completed":
            raise ValueError("Require completed runs")
        actor_path, checkpoint = path / "actor.pt", path / "checkpoint.pt"
        actor, critics, config, step, alpha = load_critic_checkpoint(
            checkpoint, actor_path
        )
        key = f"{config['algo']}-s{config['seed']}"
        if key in models:
            raise ValueError("Duplicate method/training seed")
        models[key] = actor, critics
        policies.append(
            {
                "id": key,
                "method": config["algo"],
                "seed": config["seed"],
                "step": step,
                "actor": str(actor_path.resolve()),
                "actor_sha256": digest(actor_path),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": digest(checkpoint),
                "config": config,
                "gamma": config["gamma"],
                "alpha": alpha,
                "ensemble_size": len(critics),
                "states": {},
            }
        )
    if {(p["method"], p["seed"]) for p in policies} != {
        (m, s) for m in ("sac", "sac_sym", "rlpd") for s in range(3)
    }:
        raise ValueError(
            "Require all three SAC-family methods at training seeds 0, 1, 2"
        )
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "qualifying",
        "metric": "b = mean online critic Q(s, tanh(mean(s))) - Monte Carlo soft Q",
        "repetitions": repetitions,
        "response_horizon": horizon,
        "action_noise_seeds": list(range(41000, 41000 + repetitions)),
        "first_action": "deterministic tanh(mean), held fixed across repetitions",
        "continuation": "frozen stochastic actor; discounted entropy starts at the second action",
        "timeout": "numeric integration cutoff; no critic-based bootstrap or terminal-zero assumption",
        "reference": str(reference_path.resolve()),
        "reference_sha256": digest(reference_path),
        "scenario": scenario,
        "state_protocol": "Common expert prefix, then matched-duration normal or intervened expert history; measure after normal controls return",
        "delay_reference_controller": "fixed original expert"
        if scenario.get("control_delay")
        else None,
        "platform": platform.platform(),
        "versions": {
            n: version(n)
            for n in ("torch", "numpy", "pysupertuxkart2", "pystk2-gymnasium")
        },
        "policies": policies,
        "states": {},
        "trials": [],
    }
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for path in sorted(Path(__file__).parent.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".cpp", ".yaml"):
                archive.add(
                    path, arcname=str(path.relative_to(Path(__file__).parent.parent))
                )
    manifest["source_archive_sha256"] = digest(output / "source.tar.gz")

    def save():
        temporary = output / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(output / "manifest.json")

    save()
    prefix_steps = reference["handover_steps"] + scenario.get("control_delay", {}).get(
        "duration_steps", 0
    )
    env = DrivingEnv(track=scenario["track"], max_steps=prefix_steps + horizon)
    try:
        states = freeze_states(env, reference)
        for condition, state in states.items():
            np.savez_compressed(
                output / f"state-{condition}.npz",
                prefix_actions=state["prefix_actions"],
                obs=np.asarray(state["signature"]["obs"], np.float32),
            )
            manifest["states"][condition] = {
                k: v for k, v in state.items() if k != "prefix_actions"
            }
            manifest["states"][condition]["tape_sha256"] = digest(
                output / f"state-{condition}.npz"
            )
        with torch.inference_mode():
            for policy in policies:
                actor, critics = models[policy["id"]]
                for condition, state in states.items():
                    obs = torch.tensor(
                        state["signature"]["obs"], dtype=torch.float32
                    ).unsqueeze(0)
                    action = actor.act(obs)
                    qs = ensemble_forward(critics, obs, action).squeeze(1)
                    policy["states"][condition] = {
                        "first_action": action.squeeze(0).tolist(),
                        "critic_values": qs.tolist(),
                        "q_mean": float(qs.mean()),
                    }
        manifest["status"] = "running"
        save()
        for policy in policies:
            actor, _ = models[policy["id"]]
            for condition, state in states.items():
                for noise_seed in manifest["action_noise_seeds"]:
                    result, trace = continuation(
                        env,
                        state,
                        reference["scenario"]["eval_seed"],
                        actor,
                        np.asarray(
                            policy["states"][condition]["first_action"], np.float32
                        ),
                        policy["gamma"],
                        policy["alpha"],
                        noise_seed,
                        horizon,
                    )
                    key = f"{policy['id']}--{condition}--noise{noise_seed}"
                    np.savez_compressed(output / f"{key}.npz", **trace)
                    result.update(
                        {
                            "id": key,
                            "policy": policy["id"],
                            "condition": condition,
                            "trace": f"{key}.npz",
                            "trace_sha256": digest(output / f"{key}.npz"),
                        }
                    )
                    manifest["trials"].append(result)
                    save()
                group = [
                    r["soft_return"]
                    for r in manifest["trials"]
                    if r["policy"] == policy["id"] and r["condition"] == condition
                ]
                b = policy["states"][condition]["q_mean"] - np.mean(group)
                print(
                    policy["id"],
                    condition,
                    f"b={b:+.6f}",
                    f"{len(manifest['trials'])}/{len(policies) * 2 * repetitions}",
                    flush=True,
                )
        summarize(manifest, output)
        manifest["status"] = "completed"
        save()
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        save()
        raise
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=2500)
    args = parser.parse_args()
    run(args.inputs, args.reference, args.output, args.repetitions, args.horizon)


if __name__ == "__main__":
    main()
