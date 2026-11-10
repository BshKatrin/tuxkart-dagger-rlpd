"""Reproducible physical control disturbances for exploratory recovery studies.

Every trial replays an expert prefix and forced controls before handing over.
The original expert, task, policies and official evaluations are unchanged.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .environment import DrivingEnv


class Recovery:
    """Two continuous on-road seconds with >=10 m forward progress, no rescue."""

    def __init__(self, progress, rescues):
        self.rescues = rescues
        self.window = []
        self.time = None
        self.assisted = False

    def update(self, info, seconds):
        self.assisted |= info["rescue_count"] > self.rescues
        if self.time is not None:
            return
        if self.assisted or not info["is_on_road"] or info["rescue_active"]:
            self.window.clear()
            return
        self.window.append((seconds, info["progress"]))
        if len(self.window) > 21:
            self.window.pop(0)
        if (
            len(self.window) == 21
            and self.window[-1][1] - self.window[0][1] >= 10
            and all(b[1] >= a[1] for a, b in zip(self.window, self.window[1:]))
        ):
            self.time = seconds


class ControlDelay:
    """Finite actuator latency, initialized with the actual shared control history."""

    def __init__(self, history, steps, duration_steps):
        if steps < 1 or duration_steps < 1 or len(history) < steps:
            raise ValueError(
                "Delay requires positive lengths and enough command history"
            )
        self.queue = [np.asarray(action).copy() for action in history[-steps:]]
        self.duration_steps = duration_steps

    def execute(self, requested, response_index):
        if response_index >= self.duration_steps:
            return requested.copy()
        executed = self.queue.pop(0)
        self.queue.append(requested.copy())
        return executed


def native_geometry(env):
    """Read native road direction and kart pose without altering the simulator."""
    kart = env.env.unwrapped.world.karts[0]
    node = int(kart.node)
    start, end = np.asarray(env.env.unwrapped.track.path_nodes)[node]
    forward = np.asarray(kart.front) - np.asarray(kart.location)
    tangent = end - start
    cosine = np.dot(forward, tangent) / (
        np.linalg.norm(forward) * np.linalg.norm(tangent)
    )
    return {
        "node": node,
        "heading_error_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
        "speed": float(np.linalg.norm(kart.velocity)),
        "is_on_road": bool(kart.is_on_road),
    }


def signature(env, obs):
    kart = env.env.unwrapped.world.karts[0]
    return {
        "obs": np.asarray(obs).tolist(),
        "position": np.asarray(kart.location).tolist(),
        "rotation": np.asarray(kart.rotation).tolist(),
        "velocity": np.asarray(kart.velocity).tolist(),
        "native_time": float(env.env.unwrapped.world.time),
    }


def geometric_action(env, counter):
    """Independent geometry-only witness, never used as expert labels."""
    points = np.asarray(env.raw["paths_start"])
    target = points[min(2, len(points) - 1)]
    angle = float(np.arctan2(target[0], target[2]))
    speed = float(np.linalg.norm(env.raw["velocity"]))
    # Reverse briefly on a persistent low-speed block, then try forward again.
    if speed < 2 and counter % 60 >= 35:
        return np.array([-np.sign(angle), -1, 1, -1, -1], np.float32)
    return np.array(
        [np.clip(angle * 2, -1, 1), -0.3 if speed > 10 else 0.5, -1, -1, -1], np.float32
    )


def trial(env, scenario, controller="expert", actor=None, save=None, correction=None):
    """Run one full original-horizon episode from an identical disturbed prefix."""
    obs, _ = env.reset(seed=scenario["eval_seed"], track=scenario["track"])
    trace = defaultdict(list)
    done = False
    total = 0
    info = None

    def step(action, phase, requested=None, delay_active=False):
        nonlocal obs, done, total, info
        obs, _reward, term, trunc, info = env.step(action)
        done = term or trunc
        total += 1
        for key, value in {
            "phase": phase,
            "world_position": info["world_position"],
            "action": action,
            "requested_action": action if requested is None else requested,
            "delay_active": delay_active,
            "progress": info["progress"],
            "speed": info["speed"],
            "is_on_road": info["is_on_road"],
            "rescue_active": info["rescue_active"],
            "rescue_count": info["rescue_count"],
            "elapsed_seconds": info["elapsed_seconds"],
        }.items():
            trace[key].append(value)
        geometry = native_geometry(env)
        trace["native_node"].append(geometry["node"])
        trace["heading_error_degrees"].append(geometry["heading_error_degrees"])

    while not done and (info is None or info["progress"] < scenario["anchor_m"]):
        step(env.expert_action(), "prefix")
    if done:
        raise ValueError("Expert did not reach disturbance location")
    anchor = signature(env, obs)
    prefix_rescues = info["rescue_count"]
    for pulse in scenario["pulses"]:
        for _ in range(pulse["steps"]):
            action = env.expert_action()
            for key, index in [
                ("steer", 0),
                ("throttle", 1),
                ("brake", 2),
                ("nitro", 3),
                ("fire", 4),
            ]:
                if key in pulse:
                    action[index] = pulse[key]
            step(action, "disturbance")
            if done:
                raise ValueError("Episode ended during disturbance")
    handover = signature(env, obs)
    handover_geometry = native_geometry(env)
    handover_rescue_active = bool(info["rescue_active"])
    handover_steps, handover_progress = total, info["progress"]
    initial_rescues, handover_seconds = info["rescue_count"], info["elapsed_seconds"]
    recovery = Recovery(handover_progress, initial_rescues)
    delay_config = scenario.get("control_delay")
    delay = (
        ControlDelay(
            trace["action"], delay_config["steps"], delay_config["duration_steps"]
        )
        if delay_config
        else None
    )
    recovery_start_seconds = 0.0
    delay_release = None
    with torch.inference_mode():
        while not done:
            response_index = total - handover_steps
            if controller == "expert":
                action = env.expert_action()
            elif controller == "geometric":
                action = (
                    env.expert_action()
                    if recovery.time is not None
                    else geometric_action(env, total - handover_steps)
                )
            elif controller == "scripted":
                action = env.expert_action()
                offset = total - handover_steps
                for segment in correction:
                    if offset < segment["steps"]:
                        action = np.asarray(segment["action"], dtype=np.float32)
                        break
                    offset -= segment["steps"]
            else:
                action = actor.act(torch.as_tensor(obs).unsqueeze(0)).squeeze(0).numpy()
            requested = action.copy()
            active = delay is not None and response_index < delay.duration_steps
            if delay is not None:
                action = delay.execute(requested, response_index)
            step(action, "response", requested=requested, delay_active=active)
            if delay is not None and response_index + 1 == delay.duration_steps:
                recovery_start_seconds = info["elapsed_seconds"] - handover_seconds
                delay_release = signature(env, obs)
            if not active:
                recovery.update(
                    info,
                    info["elapsed_seconds"] - handover_seconds - recovery_start_seconds,
                )
    result = {
        "controller": controller,
        "correction": correction,
        "scenario": scenario,
        "anchor": anchor,
        "handover": handover,
        "handover_geometry": handover_geometry,
        "handover_steps": handover_steps,
        "handover_progress": handover_progress,
        "handover_rescue_active": handover_rescue_active,
        "disturbance_rescues": initial_rescues - prefix_rescues,
        "prefix_rescues": prefix_rescues,
        "recovery_seconds": recovery.time,
        "recovery_start_seconds": recovery_start_seconds,
        "delay_release": delay_release,
        "recovered_within_15s": recovery.time is not None and recovery.time <= 15,
        "native_rescues_after_handover": info["rescue_count"] - initial_rescues,
        "success": info["success"],
        "lap_time": info["finish_time"],
        "response_steps": total - handover_steps,
        "final_progress": info["progress"],
        "max_response_progress": max(trace["progress"][handover_steps:]),
        "timeout": not info["success"],
    }
    if save is not None:
        save = Path(save)
        np.savez_compressed(
            save.with_suffix(".npz"), **{k: np.asarray(v) for k, v in trace.items()}
        )
        save.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


def make_env(track):
    torch.set_num_threads(1)
    return DrivingEnv(track=track, max_steps=1500)


def evaluate_bank(scenarios_path, inputs, output):
    """Evaluate final actors on a frozen scenario bank; refuse output overwrites."""
    import importlib.metadata
    import platform
    import tarfile

    from .playback import load_actor
    from .trajectory_evolution import digest

    scenarios = json.loads(scenarios_path.read_text())
    identifiers = [s["id"] for s in scenarios]
    if len(identifiers) != len(set(identifiers)) or any(
        not key or not all(c.isalnum() or c in "_-" for c in key) for key in identifiers
    ):
        raise ValueError("Scenario IDs must be unique plain filename identifiers")
    policies = []
    for source in inputs:
        source = source.resolve()
        if source.is_dir():
            source = source / (
                "actor.pt" if (source / "actor.pt").exists() else "checkpoint.pt"
            )
        actor, config, step = load_actor(source)
        if (
            config.get("schema_version") != 3
            or actor.obs_dim != 84
            or actor.action_dim != 5
        ):
            raise ValueError("Require the study observation/action schema")
        policies.append(
            (
                actor,
                {
                    "path": str(source),
                    "sha256": digest(source),
                    "config": config,
                    "step": step,
                },
            )
        )
    keys = [(p[1]["config"]["algo"], p[1]["config"]["seed"]) for p in policies]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate method/training seed inputs")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "origin": "Exploratory disturbance evaluation, not official assessment",
        "scenarios": scenarios,
        "scenario_file_sha256": digest(scenarios_path),
        "policies": [p[1] for p in policies],
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pysupertuxkart2", "pystk2-gymnasium")
        },
        "recovery_definition": "21 consecutive on-road observations spanning 2 seconds, monotonically advancing at least 10 m, before any native rescue",
        "max_episode_actions": 1500,
        "action_interval_seconds": 0.1,
        "trials": [],
    }
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        package = Path(__file__).parent
        for path in sorted(package.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".cpp"):
                archive.add(path, arcname=str(path.relative_to(package.parent)))
    manifest["source_archive_sha256"] = digest(output / "source.tar.gz")

    def save_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    save_manifest()
    env = make_env(scenarios[0]["track"])
    try:
        for scenario in scenarios:
            reference = None
            controllers = [("expert", None, None, None)]
            if scenario.get("witness") is not None:
                controllers.append(("scripted", None, None, scenario["witness"]))
            controllers.extend(
                (p[1]["config"]["algo"], p[0], p[1], None) for p in policies
            )
            for controller, actor, policy, correction in controllers:
                key = (
                    controller
                    if policy is None
                    else f"{controller}-s{policy['config']['seed']}"
                )
                stem = f"{scenario['id']}--{key}"
                result = trial(
                    env,
                    scenario,
                    controller,
                    actor,
                    save=output / stem,
                    correction=correction,
                )
                state = json.dumps(result["handover"], sort_keys=True)
                if reference is None:
                    reference = state
                if state != reference:
                    raise ValueError(f"Mismatched handover state: {stem}")
                result["policy"] = policy
                result["handover_matches_expert"] = True
                result["valid_unassisted_handover"] = not (
                    result["prefix_rescues"] or result["disturbance_rescues"]
                )
                (output / f"{stem}.json").write_text(json.dumps(result, indent=2))
                manifest["trials"].append(
                    {
                        "id": stem,
                        "result": f"{stem}.json",
                        "result_sha256": digest(output / f"{stem}.json"),
                        "trajectory": f"{stem}.npz",
                        "trajectory_sha256": digest(output / f"{stem}.npz"),
                    }
                )
                save_manifest()
                print(
                    stem,
                    "finish",
                    result["success"],
                    "recovery",
                    result["recovery_seconds"],
                    "rescues",
                    result["native_rescues_after_handover"],
                    flush=True,
                )
        for _, policy in policies:
            if digest(policy["path"]) != policy["sha256"]:
                raise ValueError("Input policy changed during evaluation")
        manifest["status"] = "completed"
        save_manifest()
    finally:
        env.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        type=Path,
        required=True,
        help="Frozen JSON list of expert-prefix control disturbances",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Final run folders or trusted actor/checkpoint files",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh output directory"
    )
    args = parser.parse_args()
    evaluate_bank(args.scenarios, args.input, args.output)


if __name__ == "__main__":
    main()
