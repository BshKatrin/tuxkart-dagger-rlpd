"""Capture fresh solo laps and interpolate immutable world-pose comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .environment import DrivingEnv
from .native_telemetry import SOURCE_SHA256
from .playback import METHODS, discover_runs, load_actor
from .snapshots import actor_hash, file_hash, source_identity
from .spectator_contract import GhostConfig, GhostPose

FORMAT_VERSION = 1
COLORS = {"bc": 0.08, "dagger": 0.56, "sac": 0.0, "rlpd": 0.33}


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported manifest value: {type(value).__name__}")


def _append_pose(rows, pose, steer, rescue_active):
    previous = rows[-1]
    delta = pose["native_time"] - previous["native_time"]
    interval = delta if delta > 0 else 0.1
    displacement = np.linalg.norm(np.asarray(pose["position"]) - previous["position"])
    # Rescue timers cover animations; the distance guard catches instantaneous relocation.
    discontinuity = bool(
        rescue_active
        or previous["rescue_active"]
        or displacement
        > max(10.0, interval * max(pose["speed"], previous["speed"]) * 3)
    )
    rows.append(
        pose
        | {
            "time": previous["time"] + interval,
            "steer": float(steer),
            "rescue_active": bool(rescue_active),
            "discontinuity": discontinuity,
        }
    )


def slerp(a, b, fraction):
    """Shortest-path spherical interpolation of normalized xyzw quaternions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = np.clip(dot, -1, 1)
    if dot > 0.9995:
        result = a + fraction * (b - a)
    else:
        angle = np.arccos(dot)
        result = (
            np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b
        ) / np.sin(angle)
    return result / np.linalg.norm(result)


class ReplayComparison:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported replay format")
        if self.manifest.get("quaternion_order") != "xyzw":
            raise ValueError("Replay requires canonical xyzw orientation")
        self.track = self.manifest["track"]
        self.seed = int(self.manifest["evaluation_seed"])
        self.ghosts: list[GhostConfig] = self.manifest["ghosts"]
        self._traces = {}
        for ghost in self.ghosts:
            record = self.manifest["laps"][ghost["key"]]
            file = self.path / record["file"]
            if file_hash(file) != record["sha256"]:
                raise ValueError("Replay array checksum mismatch")
            with np.load(file, allow_pickle=False) as archive:
                if "rotation" not in archive:
                    raise ValueError(
                        "Replay requires orientation; positions alone are insufficient"
                    )
                trace = {
                    name: archive[name]
                    for name in (
                        "time",
                        "position",
                        "rotation",
                        "speed",
                        "steer",
                        "rescue_active",
                        "discontinuity",
                    )
                }
            n = len(trace["time"])
            if (
                n < 1
                or trace["time"].shape != (n,)
                or trace["time"][0] != 0
                or not np.all(np.diff(trace["time"]) > 0)
            ):
                raise ValueError(
                    "Replay times must start at zero and strictly increase"
                )
            for name, shape in (
                ("position", (n, 3)),
                ("rotation", (n, 4)),
                ("speed", (n,)),
                ("steer", (n,)),
            ):
                if trace[name].shape != shape or not np.isfinite(trace[name]).all():
                    raise ValueError(f"Invalid replay {name}")
            if not np.isfinite(trace["time"]).all() or np.any(
                np.linalg.norm(trace["rotation"], axis=1) == 0
            ):
                raise ValueError("Invalid replay time/orientation")
            for name in ("rescue_active", "discontinuity"):
                if trace[name].shape != (n,) or trace[name].dtype != np.bool_:
                    raise ValueError(f"Invalid replay {name}")
            self._traces[ghost["key"]] = trace
        self.duration = max(
            (float(t["time"][-1]) for t in self._traces.values()), default=0.0
        )

    def sample(self, time: float) -> list[GhostPose]:
        if not np.isfinite(time):
            raise ValueError("Replay time must be finite")
        poses = []
        for ghost in self.ghosts:
            trace = self._traces[ghost["key"]]
            times = trace["time"]
            t = np.clip(time, 0, times[-1])
            right = min(int(np.searchsorted(times, t, side="right")), len(times) - 1)
            left = max(0, right - 1)
            fraction = (
                float((t - times[left]) / (times[right] - times[left]))
                if right != left
                else 0.0
            )

            if trace["discontinuity"][right] and t < times[right]:
                fraction = 0.0

            def mix(name, trace=trace, left=left, right=right, fraction=fraction):
                return trace[name][left] + fraction * (
                    trace[name][right] - trace[name][left]
                )

            poses.append(
                {
                    "key": ghost["key"],
                    "position": mix("position").tolist(),
                    "rotation": slerp(
                        trace["rotation"][left], trace["rotation"][right], fraction
                    ).tolist(),
                    "speed": float(mix("speed")),
                    "steer": float(mix("steer")),
                    "visible": True,
                    "status": self.manifest["laps"][ghost["key"]]["status"]
                    if time >= times[-1]
                    else "driving",
                }
            )
        return poses


def _selections(root, track, methods, train_seed, snapshots):
    if snapshots is None:
        catalog = discover_runs(root)
        return [
            (method, catalog[(track, method, int(train_seed))], False)
            for method in methods
        ]
    if isinstance(snapshots, (str, Path)):
        base = Path(snapshots)
        if (base / "snapshots").is_dir():
            base = base / "snapshots"
        paths = (
            [base]
            if (base / "actor.pt").exists()
            else sorted(base.glob("actions-*_updates-*"))
        )
    else:
        paths = [Path(path) for path in snapshots]
    if not paths:
        raise ValueError(
            "No actor snapshots selected; historical milestones cannot be synthesized"
        )
    selected = []
    runs = set()
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text())
        method = manifest["config"]["algo"]
        if method not in methods or manifest["config"]["seed"] != int(train_seed):
            raise ValueError("Snapshot method/training seed does not match selection")
        if file_hash(path / "actor.pt") != manifest["actor_file_sha256"]:
            raise ValueError("Snapshot checksum mismatch")
        runs.add(path.resolve().parent)
        selected.append((method, path, True))
    if len(runs) != 1:
        raise ValueError("Snapshots must belong to one training run")
    return selected


def capture_comparison(
    output,
    root="runs/study-polar-four",
    track="lighthouse",
    methods=("bc", "dagger", "sac", "rlpd"),
    eval_seed=20000,
    train_seed=0,
    snapshots=None,
    max_steps=1500,
):
    """Capture sequential CPU-inference laps; never train or replace evaluation.

    ``snapshots`` accepts one run/snapshot-root directory or an iterable of
    selected snapshot directories. Returns a loaded ``ReplayComparison``.
    Failed lap poses and errors are retained with status ``stopped``.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if max_steps < 1 or not methods or any(m not in COLORS for m in methods):
        raise ValueError("Require positive max_steps and known methods")
    selections = _selections(root, track, methods, train_seed, snapshots)
    if len({str(path.resolve()) for _, path, _ in selections}) != len(selections):
        raise ValueError("Duplicate actors selected")
    import pystk2

    source = source_identity()
    manifest = {
        "format_version": FORMAT_VERSION,
        "track": track,
        "evaluation_seed": int(eval_seed),
        "training_seed": int(train_seed),
        "max_steps": int(max_steps),
        "quaternion_order": "xyzw",
        "world_axes": {"x": "right", "y": "up", "z": "forward"},
        "orientation_source": "pysupertuxkart2 0.7.4 inverse wxyz conjugated to world xyzw",
        "time_policy": "native positive deltas; countdown/duplicate times advance by control interval",
        "control_interval_seconds": 0.1,
        "discontinuity_policy": "hold previous pose across rescue-active boundaries or displacement > max(10 m, 3 * speed * interval)",
        "source": source,
        "native_binary_sha256": file_hash(pystk2.__file__),
        "native_source_sha256": SOURCE_SHA256,
        "ghosts": [],
        "laps": {},
        "complete": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    env = None
    try:
        env = DrivingEnv(track=track, max_steps=max_steps)
        for method, path, milestone in selections:
            actor, config, steps = load_actor(path)
            state_path = path / (
                "actor.pt" if (path / "actor.pt").exists() else "checkpoint.pt"
            )
            # Actor-only state contains portable counters. Legacy rolling states are trusted local data.
            state = torch.load(
                state_path,
                map_location="cpu",
                weights_only=state_path.name == "actor.pt",
                mmap=True,
            )
            status_file = path / "status.json"
            counters = (
                json.loads(status_file.read_text()) if status_file.exists() else {}
            )
            updates = state.get("updates", counters.get("updates"))
            key = f"{method}-s{train_seed}" + (
                f"-actions-{steps}-updates-{updates}" if milestone else ""
            )
            label = METHODS[method] + (
                f" · {steps} actions / {updates} updates" if milestone else ""
            )
            ghost = {
                "key": key,
                "label": label,
                "method": method,
                "kart": "tux",
                "color": COLORS[method],
            }
            manifest["ghosts"].append(ghost)
            before_hash = actor_hash(actor.state_dict())
            lap = {
                "file": f"{key}.npz",
                "method": method,
                "training_seed": int(config["seed"]),
                "evaluation_seed": int(eval_seed),
                "checkpoint_step": steps,
                "checkpoint_updates": updates,
                "dagger_round": state.get("dagger_round"),
                "actor_sha256": before_hash,
                "actor_file_sha256": file_hash(state_path),
                "actor_source": str(path.resolve()),
                "configuration": config,
                "status": "stopped",
                "lap_time": None,
                "success": False,
                "steps": 0,
            }
            provenance = path / ("manifest.json" if milestone else "provenance.json")
            lap["training_provenance"] = (
                json.loads(provenance.read_text()) if provenance.exists() else None
            )
            rows, info = [], {}
            try:
                obs, reset_info = env.reset(seed=int(eval_seed), track=track)
                initial = env.replay_pose()
                ghost["kart"] = initial["kart"]
                rows.append(
                    initial
                    | {
                        "time": 0.0,
                        "steer": 0.0,
                        "rescue_active": bool(reset_info.get("rescue_active", False)),
                        "discontinuity": False,
                    }
                )
                with torch.inference_mode():
                    for index in range(max_steps):
                        action = (
                            actor.act(
                                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                            )
                            .squeeze(0)
                            .numpy()
                        )
                        obs, _, term, trunc, info = env.step(action)
                        pose = env.replay_pose()
                        _append_pose(
                            rows,
                            pose,
                            info.get("executed_action", action)[0],
                            info.get("rescue_active", False),
                        )
                        lap["steps"] = index + 1
                        if term or trunc:
                            lap["success"] = bool(term and info.get("success", True))
                            lap["status"] = (
                                "finished"
                                if lap["success"]
                                else "timeout"
                                if trunc
                                else "stopped"
                            )
                            lap["lap_time"] = (
                                info.get("finish_time") if lap["success"] else None
                            )
                            break
                    else:
                        lap["status"] = "timeout"
            except Exception as exc:
                lap["error"] = {"type": type(exc).__name__, "message": str(exc)}
                if not rows or isinstance(exc, ValueError):
                    raise
                # Read the final available pose if a failed step moved the simulator.
                try:
                    pose = env.replay_pose()
                    if (
                        pose["native_time"] != rows[-1]["native_time"]
                        or pose["position"] != rows[-1]["position"]
                    ):
                        _append_pose(
                            rows,
                            pose,
                            rows[-1]["steer"],
                            info.get("rescue_active", False),
                        )
                except Exception as pose_error:  # noqa: BLE001 — preserve the original simulator failure
                    lap["terminal_pose_error"] = str(pose_error)
            finally:
                if actor_hash(actor.state_dict()) != before_hash:
                    raise RuntimeError("Frozen capture actor changed")
                lap["terminal_info"] = info
                manifest["laps"][key] = lap
                if rows:
                    np.savez_compressed(
                        output / lap["file"],
                        **{
                            name: np.asarray(
                                [r[name] for r in rows],
                                dtype=bool
                                if name in ("rescue_active", "discontinuity")
                                else np.float64,
                            )
                            for name in (
                                "time",
                                "native_time",
                                "position",
                                "rotation",
                                "speed",
                                "steer",
                                "rescue_active",
                                "discontinuity",
                            )
                        },
                    )
                    lap["sha256"] = file_hash(output / lap["file"])
        manifest["complete"] = True
    except Exception as exc:
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False, default=_json_value) + "\n"
        )
        if env is not None:
            env.close()
    return ReplayComparison(output)
