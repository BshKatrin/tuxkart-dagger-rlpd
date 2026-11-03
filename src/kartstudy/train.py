"""Seeded experiment worker. Run through the bounded suite launcher."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import random
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from rl_mind.core import Action
from rl_mind.data import ReplayBuffer, Transitions

from .algorithms import SAC, Policy, fit_imitation, symmetric_sample
from .data import dataset_hash, load_transitions
from .environment import DrivingEnv
from .evaluation import evaluate


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.set_num_threads(1)


def run(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    seed_all(config["seed"])
    device = config["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "Metal requested but unavailable; refusing silent CPU fallback"
        )
    write_json(out / "config.json", config)
    write_json(
        out / "provenance.json",
        {
            "dataset_sha256": dataset_hash(config["dataset"]),
            "pid": os.getpid(),
            "versions": {
                p: importlib.metadata.version(p)
                for p in [
                    "torch",
                    "rl-mind",
                    "pystk2-gymnasium",
                    "pysupertuxkart2",
                    "numpy",
                ]
            },
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_dirty": subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ),
            "configuration": config,
        },
    )
    start = time.perf_counter()
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    env = DrivingEnv(track=config["track"], max_steps=config["max_steps"])
    steps = updates = expert_queries = eval_steps = episodes = 0
    algo = config["algo"]
    if algo not in ["bc", "dagger", "sac", "sac_sym", "rlpd"]:
        raise ValueError(f"Unknown algorithm: {algo}")
    offline = online = learner = None
    labels_obs = labels_actions = None
    obs, _ = env.reset(seed=config["seed"])
    actor = Policy(len(obs), hidden=config["hidden"]).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=config["lr"])
    metadata = json.loads(Path(config["dataset"]).with_suffix(".json").read_text())
    if (
        metadata["track"] != config["track"]
        or metadata["conservative"] != config["conservative"]
    ):
        raise ValueError("Dataset track/expert does not match experiment configuration")
    if not 1 <= config["demo_episodes"] <= len(metadata["train_episode_ids"]):
        raise ValueError("Demonstration episode budget overlaps validation or is empty")
    initial = load_transitions(config["dataset"], config["demo_episodes"])
    with np.load(config["dataset"]) as data:
        mask = data["episode"] < config["demo_episodes"]
        initial_labels = torch.from_numpy(data["expert_action"][mask])
        val_mask = np.isin(data["episode"], metadata["validation_episode_ids"])
        val_obs = torch.from_numpy(data["obs"][val_mask]).to(device)
        val_actions = torch.from_numpy(data["expert_action"][val_mask]).to(device)
    write_json(
        out / "resources.json",
        {
            "dataset_available_transitions": len(initial),
            "initial_transitions_used": len(initial) if algo != "sac" else 0,
            "initial_expert_labels_used": len(initial)
            if algo in ["bc", "dagger"]
            else 0,
        },
    )

    def save_state(status):
        state = {
            "config": config,
            "step": steps,
            "updates": updates,
            "expert_queries": expert_queries,
            "actor": actor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "learner": learner.checkpoint() if learner is not None else None,
            "replay": online,
            "imitation_obs": labels_obs,
            "imitation_actions": labels_actions,
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_mps": torch.mps.get_rng_state() if device == "mps" else None,
        }
        torch.save(state, out / "checkpoint.tmp")
        (out / "checkpoint.tmp").replace(out / "checkpoint.pt")
        write_json(
            out / "status.json",
            {
                "status": status,
                "step": steps,
                "updates": updates,
                "expert_queries": expert_queries,
                "evaluation_steps": eval_steps,
                "wall_seconds": time.perf_counter() - start,
                "transitions_per_second": steps
                / max(time.perf_counter() - start, 1e-9),
            },
        )

    def assessment(final=False):
        nonlocal eval_steps
        # Use CPU inference snapshot; all optimizer updates remain on requested device.
        policy = copy.deepcopy(actor).to("cpu").eval()
        rng_py, rng_np, rng_torch = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
        try:
            conditions = ["normal", "perturbed"] + (
                ["transfer_snowtuxpeak"] if final else []
            )
            for condition in conditions:
                env.env.unwrapped.default_track = (
                    "snowtuxpeak"
                    if condition == "transfer_snowtuxpeak"
                    else config["track"]
                )
                seeds = range(20000, 20000 + (10 if final else 3))
                traces = out / f"trajectories_{condition}.npz" if final else None
                records = evaluate(
                    env, policy, seeds, "cpu", condition == "perturbed", traces
                )
                eval_steps += sum(r["steps"] for r in records)
                append_json(
                    out / "evaluations.jsonl",
                    {
                        "step": steps,
                        "condition": condition,
                        "episodes": records,
                        "expert_queries": expert_queries,
                        "updates": updates,
                        "wall_seconds": time.perf_counter() - start,
                    },
                )
        finally:
            env.env.unwrapped.default_track = config["track"]
            random.setstate(rng_py)
            np.random.set_state(rng_np)
            torch.set_rng_state(rng_torch)

    try:
        if algo in ["bc", "dagger"]:
            labels_obs, labels_actions = initial.obs.clone(), initial_labels.clone()
            metrics = fit_imitation(
                actor,
                labels_obs,
                labels_actions,
                device,
                updates=config["bc_updates"],
                batch_size=config["batch_size"],
                optimizer=optimizer,
            )
            updates += config["bc_updates"]
            with torch.no_grad():
                metrics["validation_mse"] = float(
                    (actor.act(val_obs) - val_actions).square().mean().cpu()
                )
            append_json(out / "training.jsonl", dict(step=0, **metrics))
            assessment(final=algo == "bc")
            if algo == "bc":
                save_state("completed")
                return
        else:
            ensemble = 10 if algo == "rlpd" else 2
            learner = SAC(
                len(obs),
                device=device,
                ensemble=ensemble,
                layer_norm=algo == "rlpd",
                hidden=config["hidden"],
                lr=config["lr"],
                gamma=config["gamma"],
            )
            actor = learner.actor
            online = ReplayBuffer(config["steps"])
            if algo != "sac":
                offline = ReplayBuffer(len(initial))
                offline.add(initial)
            assessment()
        obs, _ = env.reset(seed=config["seed"] * 100000 + episodes)
        next_eval = config["eval_every"]
        round_index = 0
        rng = np.random.default_rng(config["seed"])
        while steps < config["steps"] and not stop:
            rollout = copy.deepcopy(actor).to("cpu").eval()
            chunk = min(
                config["dagger_round"] if algo == "dagger" else config["collect_steps"],
                config["steps"] - steps,
                next_eval - steps,
            )
            rows = {
                k: [] for k in ["obs", "action", "reward", "next_obs", "terminated"]
            }
            new_labels = []
            beta = config["beta_initial"] * config["beta_decay"] ** round_index
            for _ in range(chunk):
                with torch.inference_mode():
                    inputs = torch.from_numpy(obs).unsqueeze(0)
                    if algo == "dagger":
                        action = rollout.act(inputs).squeeze(0).numpy()
                    elif steps < config["warmup"]:
                        action = rng.uniform(-1, 1, 3).astype(np.float32)
                    else:
                        action = rollout(inputs).value.squeeze(0).numpy()
                if algo == "dagger":
                    label = env.expert_action(config["conservative"])
                    new_labels.append(label)
                    expert_queries += 1
                    if rng.random() < beta:
                        action = label
                nxt, reward, term, trunc, info = env.step(action)
                for key, value in zip(rows, [obs, action, reward, nxt, term]):
                    rows[key].append(value)
                steps += 1
                obs = nxt
                if term or trunc:
                    episodes += 1
                    append_json(
                        out / "episodes.jsonl",
                        {
                            "step": steps,
                            "success": term,
                            "sim_seconds": info["elapsed_seconds"],
                            "distance": info["distance"],
                        },
                    )
                    obs, _ = env.reset(seed=config["seed"] * 100000 + episodes)
            if algo == "dagger":
                labels_obs = torch.cat(
                    [labels_obs, torch.from_numpy(np.asarray(rows["obs"]))]
                )
                labels_actions = torch.cat(
                    [labels_actions, torch.from_numpy(np.asarray(new_labels))]
                )
                metrics = fit_imitation(
                    actor,
                    labels_obs,
                    labels_actions,
                    device,
                    updates=config["dagger_updates"],
                    batch_size=config["batch_size"],
                    optimizer=optimizer,
                )
                updates += config["dagger_updates"]
                metrics["beta"] = beta
                round_index += 1
            else:
                batch = Transitions(
                    torch.from_numpy(np.asarray(rows["obs"])),
                    Action(
                        torch.from_numpy(np.asarray(rows["action"], dtype=np.float32))
                    ),
                    torch.from_numpy(np.asarray(rows["reward"], dtype=np.float32)),
                    torch.from_numpy(np.asarray(rows["next_obs"])),
                    torch.from_numpy(np.asarray(rows["terminated"], dtype=bool)),
                )
                online.add(batch)
                metrics = {}
                if steps >= config["warmup"]:
                    for update_index in range(chunk * config["utd"]):
                        batch = symmetric_sample(online, offline, config["batch_size"])
                        metrics = learner.update(
                            batch, update_actor=(update_index + 1) % config["utd"] == 0
                        )
                        updates += 1
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"Non-finite training metrics: {metrics}")
            if algo == "dagger" or steps % 1024 == 0:
                append_json(
                    out / "training.jsonl",
                    dict(
                        step=steps,
                        updates=updates,
                        wall_seconds=time.perf_counter() - start,
                        **metrics,
                    ),
                )
            if steps >= next_eval or stop or steps == config["steps"]:
                assessment(final=steps == config["steps"])
                save_state(
                    "completed"
                    if steps == config["steps"]
                    else "stopped"
                    if stop
                    else "running"
                )
                print(
                    json.dumps(
                        dict(
                            step=steps,
                            updates=updates,
                            wall_seconds=time.perf_counter() - start,
                            **metrics,
                        )
                    ),
                    flush=True,
                )
                next_eval += config["eval_every"]
                episodes += 1
                obs, _ = env.reset(seed=config["seed"] * 100000 + episodes)
        if stop:
            save_state("stopped")
    except Exception as exc:
        write_json(
            out / "error.json",
            {"type": type(exc).__name__, "message": str(exc), "step": steps},
        )
        raise
    finally:
        env.close()


def defaults():
    return {
        "algo": "sac",
        "seed": 0,
        "device": "mps",
        "track": "lighthouse",
        "max_steps": 1500,
        "dataset": "data/expert.npz",
        "demo_episodes": 20,
        "conservative": False,
        "steps": 100000,
        "hidden": 128,
        "batch_size": 256,
        "lr": 3e-4,
        "gamma": 0.99,
        "eval_every": 10000,
        "collect_steps": 32,
        "warmup": 1000,
        "utd": 1,
        "bc_updates": 2000,
        "dagger_round": 2000,
        "dagger_updates": 500,
        "beta_initial": 0.5,
        "beta_decay": 0.5,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configuration = defaults() | json.loads(Path(args.config).read_text())
    run(configuration, args.output)
