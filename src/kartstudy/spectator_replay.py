"""Capture fresh solo laps and interpolate immutable world-pose comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .environment import ACTION_KEYS, DrivingEnv
from .expert import Expert
from .native_telemetry import SOURCE_SHA256
from .playback import METHODS, discover_runs, load_actor
from .snapshots import actor_hash, file_hash, source_identity
from .spectator_contract import GhostConfig, GhostPose

FORMAT_VERSION = 2
TIME_POLICY = "time=max(native_time, 0); countdown never advances race time; action_time counts completed 0.1-second control steps"
COLORS = {
    "bc": 0.08,
    "dagger": 0.56,
    "sac": 0.0,
    "sac_sym": 0.78,
    "rlpd": 0.33,
    "expert": 0.15,
}


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported manifest value: {type(value).__name__}")


def _append_pose(
    rows,
    pose,
    steer,
    rescue_active,
    executed_action=None,
    terminated=False,
    truncated=False,
):
    previous = rows[-1]
    native_time = float(pose["native_time"])
    if not np.isfinite(native_time) or native_time < previous["native_time"]:
        raise ValueError("Native race timestamps must be finite and nondecreasing")
    delta = native_time - previous["native_time"]
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
            "time": max(0.0, native_time),
            "action_time": previous["action_time"]
            + (0.1 if executed_action is not None else 0.0),
            "steer": float(steer),
            "rescue_active": bool(rescue_active),
            "discontinuity": discontinuity,
            "executed_action": np.asarray(
                executed_action if executed_action is not None else [0] * 5, dtype=float
            ).tolist(),
            "action_applied": executed_action is not None,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
    )


def _render_trace(measurements):
    """Collapse only render poses, preserving full control telemetry on disk."""
    times = measurements["time"]
    # Last measurement at each positive timestamp preserves terminal/failure poses.
    last = np.flatnonzero(np.r_[times[1:] != times[:-1], True])
    indices = np.r_[0, last[times[last] > 0]]
    trace = {name: values[indices].copy() for name, values in measurements.items()}
    for destination in range(1, len(indices)):
        start, end = indices[destination - 1] + 1, indices[destination] + 1
        trace["discontinuity"][destination] = bool(
            measurements["discontinuity"][start:end].any()
        )
    return trace


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
        source_version = self.manifest.get("format_version")
        if source_version not in (1, FORMAT_VERSION):
            raise ValueError("Unsupported replay format")
        if source_version == 1:
            self.manifest["recorded_time_policy"] = self.manifest.get("time_policy")
            self.manifest["time_policy"] = (
                "time=max(native_time, 0); countdown never advances race time; action_time derived read-only (see per-lap action_time_basis)"
            )
            self.manifest["clock_conversion"] = (
                "v1 native_time used as race clock; action clock derived in memory; source files unchanged"
            )
        if self.manifest.get("quaternion_order") != "xyzw":
            raise ValueError("Replay requires canonical xyzw orientation")
        self.track = self.manifest["track"]
        self.seed = int(self.manifest["evaluation_seed"])
        self.ghosts: list[GhostConfig] = self.manifest["ghosts"]
        self._traces = {}
        self._measurements = {}
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
                names = (
                    "time",
                    "native_time",
                    "action_time",
                    "position",
                    "rotation",
                    "speed",
                    "steer",
                    "rescue_active",
                    "discontinuity",
                    "executed_action",
                    "action_applied",
                    "terminated",
                    "truncated",
                )
                optional_v1 = {
                    "action_time",
                    "executed_action",
                    "action_applied",
                    "terminated",
                    "truncated",
                }
                missing = set(names) - set(archive.files)
                if missing - (optional_v1 if source_version == 1 else set()):
                    raise ValueError("Replay is missing required measurements")
                trace = {name: archive[name] for name in names if name in archive}
            if source_version == 1:
                trace["recorded_time"] = trace["time"]
                trace["time"] = np.maximum(trace["native_time"], 0)
                # Preserve unavailable legacy controls as unavailable rather than inventing them.
                if "action_applied" in trace:
                    trace["action_time"] = np.cumsum(trace["action_applied"]) * 0.1
                    basis = (
                        "recorded action_applied at 0.1 seconds per completed control"
                    )
                else:
                    trace["action_time"] = np.arange(len(trace["time"])) * 0.1
                    basis = "measurement intervals at 0.1 seconds; legacy controls unavailable"
                record["action_time_basis"] = basis
            n = len(trace["time"])
            if (
                n < 1
                or trace["time"].shape != (n,)
                or trace["time"][0] != 0
                or not np.all(np.diff(trace["time"]) >= 0)
            ):
                raise ValueError(
                    "Recorded race times must start at zero and be nondecreasing"
                )
            for name in ("native_time", "action_time"):
                if (
                    trace[name].shape != (n,)
                    or not np.isfinite(trace[name]).all()
                    or np.any(np.diff(trace[name]) < 0)
                ):
                    raise ValueError(f"Invalid replay {name}")
            if trace["action_time"][0] != 0 or not np.array_equal(
                trace["time"], np.maximum(trace["native_time"], 0)
            ):
                raise ValueError(
                    "Replay time must follow native race time; action time starts at zero"
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
            for name in (
                "rescue_active",
                "discontinuity",
                "action_applied",
                "terminated",
                "truncated",
            ):
                if name in trace and (
                    trace[name].shape != (n,) or trace[name].dtype != np.bool_
                ):
                    raise ValueError(f"Invalid replay {name}")
            if "executed_action" in trace and (
                trace["executed_action"].shape != (n, len(ACTION_KEYS))
                or not np.isfinite(trace["executed_action"]).all()
            ):
                raise ValueError("Invalid replay executed controls")
            self._measurements[ghost["key"]] = trace
            self._traces[ghost["key"]] = _render_trace(trace)
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
            # Zero-duration laps must expose the terminal endpoint at the only
            # reachable playback time. The initial measurement remains on disk.
            if times[-1] == 0 and time >= 0:
                trace = {
                    name: values[-1:]
                    for name, values in self._measurements[ghost["key"]].items()
                }
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
        if methods is None:
            methods = tuple(
                method
                for method in COLORS
                if method == "expert" or (track, method, int(train_seed)) in catalog
            )
        if not methods:
            raise ValueError(
                "No completed spectator actors available at this training seed"
            )
        missing = [
            method
            for method in methods
            if method != "expert" and (track, method, int(train_seed)) not in catalog
        ]
        if missing:
            raise ValueError(f"No completed spectator actor for: {', '.join(missing)}")
        return [
            (
                method,
                None
                if method == "expert"
                else catalog[(track, method, int(train_seed))],
                False,
            )
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
        if (
            method not in COLORS
            or (methods is not None and method not in methods)
            or manifest["config"]["seed"] != int(train_seed)
        ):
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
    methods=None,
    eval_seed=20000,
    train_seed=0,
    snapshots=None,
    max_steps=1500,
):
    """Capture sequential CPU-inference laps; never train or replace evaluation.

    ``snapshots`` accepts one run/snapshot-root directory or an iterable of
    selected snapshot directories. Returns a loaded ``ReplayComparison``.
    Omitted methods select available completed actors at the requested training
    seed, or the methods in supplied snapshots. Explicit selections remain strict.
    Failed lap poses and errors are retained with status ``stopped``.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if max_steps < 1 or (
        methods is not None and (not methods or any(m not in COLORS for m in methods))
    ):
        raise ValueError("Require positive max_steps and known methods")
    selections = _selections(root, track, methods, train_seed, snapshots)
    if len(
        {
            str(path.resolve()) if path is not None else "expert"
            for _, path, _ in selections
        }
    ) != len(selections):
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
        "time_policy": TIME_POLICY,
        "render_duplicates": "keep the initial zero-time pose; keep the last pose at each positive race timestamp; carry discontinuities across dropped measurements",
        "control_interval_seconds": 0.1,
        "action_keys": list(ACTION_KEYS),
        "disabled_controls": {"drift": 0, "rescue": 0},
        "action_alignment": "row stores controls leading into pose; initial and failed-step terminal rows have action_applied=false",
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
            actor, state_path, updates = None, None, None
            config, state, steps = {"algo": "expert", "seed": None}, {}, None
            if method != "expert":
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
            key = (
                f"{method}-s{train_seed}-actions-{steps}-updates-{updates}"
                if milestone
                else method
            )
            label = METHODS[method]
            if milestone:
                label += (
                    f" · {updates} fitting updates"
                    if method == "bc"
                    else f" · {steps} collected actions / {updates} updates"
                )
            ghost = {
                "key": key,
                "label": label,
                "method": method,
                "kart": "tux",
                "color": COLORS[method],
            }
            manifest["ghosts"].append(ghost)
            before_hash = actor_hash(actor.state_dict()) if actor is not None else None
            lap = {
                "file": f"{key}.npz",
                "method": method,
                "training_seed": config["seed"],
                "evaluation_seed": int(eval_seed),
                "checkpoint_step": steps,
                "checkpoint_updates": updates,
                "dagger_round": state.get("dagger_round"),
                "actor_sha256": before_hash,
                "actor_file_sha256": file_hash(state_path) if state_path else None,
                "actor_source": str(path.resolve()) if path else None,
                "configuration": config,
                "status": "stopped",
                "lap_time": None,
                "success": False,
                "steps": 0,
            }
            provenance = (
                path / ("manifest.json" if milestone else "provenance.json")
                if path
                else None
            )
            lap["training_provenance"] = (
                json.loads(provenance.read_text())
                if provenance and provenance.exists()
                else None
            )
            if method == "expert":
                lap["expert_provenance"] = {
                    "name": Expert.name,
                    "source_commit": Expert.source_commit,
                    "manifest": json.loads(
                        Path(__file__)
                        .with_name("_team4")
                        .joinpath("manifest.json")
                        .read_text()
                    ),
                }
            rows, info = [], {}
            try:
                obs, reset_info = env.reset(seed=int(eval_seed), track=track)
                initial = env.replay_pose()
                if (
                    not np.isfinite(initial["native_time"])
                    or initial["native_time"] > 0
                ):
                    raise ValueError(
                        "Initial native race time must be finite and at or before zero"
                    )
                ghost["kart"] = initial["kart"]
                rows.append(
                    initial
                    | {
                        "time": 0.0,
                        "action_time": 0.0,
                        "steer": 0.0,
                        "rescue_active": bool(reset_info.get("rescue_active", False)),
                        "discontinuity": False,
                        "executed_action": [0.0] * 5,
                        "action_applied": False,
                        "terminated": False,
                        "truncated": False,
                    }
                )
                with torch.inference_mode():
                    for index in range(max_steps):
                        action = (
                            env.expert_action()
                            if actor is None
                            else (
                                actor.act(
                                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(
                                        0
                                    )
                                )
                                .squeeze(0)
                                .numpy()
                            )
                        )
                        obs, _, term, trunc, info = env.step(action)
                        pose = env.replay_pose()
                        _append_pose(
                            rows,
                            pose,
                            info.get("executed_action", action)[0],
                            info.get("rescue_active", False),
                            executed_action=info.get("executed_action", action),
                            terminated=term,
                            truncated=trunc,
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
                if actor is not None and actor_hash(actor.state_dict()) != before_hash:
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
                                if name
                                in (
                                    "rescue_active",
                                    "discontinuity",
                                    "action_applied",
                                    "terminated",
                                    "truncated",
                                )
                                else np.float64,
                            )
                            for name in (
                                "time",
                                "native_time",
                                "action_time",
                                "position",
                                "rotation",
                                "speed",
                                "steer",
                                "rescue_active",
                                "discontinuity",
                                "executed_action",
                                "action_applied",
                                "terminated",
                                "truncated",
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
