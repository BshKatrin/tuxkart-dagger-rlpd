"""Original STK scenery and visual-only replay ghosts, without simulation steps.

Use one spectator in a dedicated process. Close it before loading another track.
Quaternion inputs are xyzw world rotations, unlike WorldState's inverse wxyz.
"""

import ctypes
import math
import os
import threading

import numpy as np
import pystk2

from kartstudy.spectator_contract import (
    CameraPose,
    GhostConfig,
    GhostPose,
    TrackGeometry,
)
from kartstudy.spectator_native_build import load_adapter

_active = None


def _vector(value, size, name):
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite {size}-vector") from exc
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return np.ascontiguousarray(array)


def _scalar(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _float_pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _configs(ghosts):
    result, keys = [], set()
    for ghost in ghosts:
        try:
            config = {
                field: ghost[field] for field in ("key", "label", "method", "kart")
            }
            if any(
                not isinstance(value, str) or not value for value in config.values()
            ):
                raise ValueError("Ghost identifiers must be nonempty strings")
            if config["key"] in keys:
                raise ValueError("Duplicate ghost key")
            config["color"] = _scalar(ghost["color"], "color")
            if not 0 <= config["color"] <= 1:
                raise ValueError("Ghost color must be a hue in [0, 1]")
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed GhostConfig") from exc
        keys.add(config["key"])
        result.append(config)
    return result


class NativeSpectator:
    def __init__(
        self, track, ghosts: list[GhostConfig], width=960, height=540, seed=20000
    ):
        global _active
        if not isinstance(track, str) or not track:
            raise ValueError("Track must be a nonempty native track identifier")
        if any(
            type(size) is not int or not 64 <= size <= 4096 for size in (width, height)
        ):
            raise ValueError("Render dimensions must be integers between 64 and 4096")
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("Seed must be a nonnegative signed 32-bit integer")
        configs = _configs(ghosts)
        if _active is not None or pystk2.is_running():
            raise RuntimeError("Native spectator needs its own simulator process")
        self._native, self._fingerprint = load_adapter()
        self._owner = (os.getpid(), threading.get_ident())
        self._race = None
        self._handle = None
        self._closed = False
        self._initialized = False
        self.track, self.width, self.height, self.seed = track, width, height, seed
        self._configs = configs
        self._indices = {g["key"]: i for i, g in enumerate(configs)}
        self._camera = None
        self._visible = []
        self._renders = 0
        _active = self
        try:
            graphics = pystk2.GraphicsConfig.sd()
            graphics.display = False
            graphics.screen_width, graphics.screen_height = width, height
            graphics.motionblur = False
            pystk2.init(graphics)
            self._initialized = True
            if track not in pystk2.list_tracks(pystk2.RaceConfig.RaceMode.NORMAL_RACE):
                raise ValueError(f"Unknown native track: {track}")
            available = set(pystk2.list_karts())
            for ghost in configs:
                if ghost["kart"] not in available:
                    raise ValueError(f"Unknown native kart: {ghost['kart']}")
            config = pystk2.RaceConfig(
                track=track,
                num_kart=1,
                seed=seed,
                num_cameras=1,
                overlay=False,
                players=pystk2.VectorPlayerConfig([pystk2.PlayerConfig(kart="tux")]),
            )
            self._race = pystk2.Race(config)
            self._race.start()
            self._before = self._native.kartstudy_spectator_count()
            self._start_ticks = self._native.kartstudy_spectator_ticks()
            self._handle = self._native.kartstudy_spectator_create(width, height)
            if not self._handle:
                raise RuntimeError(self._native.kartstudy_spectator_error().decode())
            for ghost in configs:
                self._check(
                    self._native.kartstudy_spectator_add(
                        self._handle, ghost["kart"].encode(), ghost["color"]
                    )
                )
            native_track = pystk2.Track()
            native_track.update()
            self._geometry = {
                "segments": np.asarray(native_track.path_nodes).copy().tolist(),
                "widths": np.asarray(native_track.path_width).reshape(-1).tolist(),
            }
            first = np.asarray(self._geometry["segments"])[0, 0]
            self.set_camera(
                {
                    "position": (first + [0, 10, -15]).tolist(),
                    "target": first.tolist(),
                    "fov": math.radians(60),
                }
            )
            self._assert_world()
        except BaseException:
            self.close()
            raise

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Native spectator is closed")
        if self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError(
                "Native spectator must stay in its owning process/thread"
            )

    def _check(self, result):
        if result != 0:
            raise RuntimeError(self._native.kartstudy_spectator_error().decode())

    def _assert_world(self):
        if self._native.kartstudy_spectator_count() != self._before:
            raise RuntimeError("Physical kart count changed during presentation")
        if self._native.kartstudy_spectator_ticks() != self._start_ticks:
            raise RuntimeError("Simulation clock advanced during presentation")

    def set_camera(self, camera: CameraPose):
        self._ensure_open()
        try:
            position = _vector(camera["position"], 3, "camera position")
            target = _vector(camera["target"], 3, "camera target")
            fov = _scalar(camera["fov"], "camera fov")
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed CameraPose") from exc
        direction = target - position
        if np.linalg.norm(direction) < 1e-5 or np.linalg.norm(direction[[0, 2]]) < 1e-5:
            raise ValueError("Camera target must differ from position and world up")
        if not 0 < fov < math.pi:
            raise ValueError("Camera fov must be in radians between 0 and pi")
        self._check(
            self._native.kartstudy_spectator_camera(
                self._handle, _float_pointer(position), _float_pointer(target), fov
            )
        )
        self._camera = {
            "position": position.tolist(),
            "target": target.tolist(),
            "fov": fov,
        }

    def set_ghost_poses(self, poses: list[GhostPose]):
        """Replace the visible pose set atomically after validating every input."""
        self._ensure_open()
        validated, seen = [], set()
        for pose in poses:
            try:
                key = pose["key"]
                if key not in self._indices or key in seen:
                    raise ValueError(f"Unknown or duplicate ghost key: {key}")
                position = _vector(pose["position"], 3, "ghost position")
                rotation = _vector(pose["rotation"], 4, "ghost rotation")
                norm = np.linalg.norm(rotation)
                if abs(norm - 1) > 1e-3:
                    raise ValueError("Ghost xyzw quaternion must have unit length")
                rotation /= norm
                speed = _scalar(pose["speed"], "speed")
                steer = _scalar(pose["steer"], "steer")
                if not -1 <= steer <= 1:
                    raise ValueError("Steer must be in [-1, 1]")
                if type(pose["visible"]) is not bool:
                    raise ValueError("Ghost visibility must be boolean")
                if pose["status"] not in ("driving", "finished", "timeout", "stopped"):
                    raise ValueError("Unknown ghost status")
                validated.append(
                    (
                        self._indices[key],
                        position,
                        rotation,
                        speed,
                        steer,
                        pose["visible"],
                    )
                )
                seen.add(key)
            except (KeyError, TypeError) as exc:
                raise ValueError("Malformed GhostPose") from exc
        # Missing poses hide existing ghosts; seeking replaces rather than accumulates.
        zero, identity = np.zeros(3, np.float32), np.array([0, 0, 0, 1], np.float32)
        for key, index in self._indices.items():
            if key not in seen:
                validated.append((index, zero, identity, 0, 0, False))
        for index, position, rotation, speed, steer, visible in validated:
            self._check(
                self._native.kartstudy_spectator_pose(
                    self._handle,
                    index,
                    _float_pointer(position),
                    _float_pointer(rotation),
                    speed,
                    steer,
                    visible,
                )
            )
        self._visible = [pose["key"] for pose in poses if pose["visible"]]
        self._assert_world()

    def render(self):
        self._ensure_open()
        self._assert_world()
        bottom_up = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self._check(
            self._native.kartstudy_spectator_render(
                self._handle, bottom_up.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            )
        )
        self._assert_world()
        self._renders += 1
        return bottom_up[::-1].copy()

    def geometry(self) -> TrackGeometry:
        self._ensure_open()
        return {
            "segments": np.array(self._geometry["segments"]).tolist(),
            "widths": list(self._geometry["widths"]),
        }

    def diagnostics(self):
        self._ensure_open()
        return {
            **self._fingerprint,
            "track": self.track,
            "seed": self.seed,
            "physical_kart_count_before": self._before,
            "physical_kart_count_after": self._native.kartstudy_spectator_count(),
            "simulation_ticks_before": self._start_ticks,
            "simulation_ticks_after": self._native.kartstudy_spectator_ticks(),
            "ghost_count": len(self._configs),
            "visible_ghosts": list(self._visible),
            "render_count": self._renders,
            "camera": self._camera,
            "renderer": "native GL3 offscreen",
            "ghost_physics_bodies": 0,
        }

    def close(self):
        global _active
        if self._closed:
            return
        self._ensure_open()
        if self._handle:
            self._native.kartstudy_spectator_destroy(self._handle)
            self._handle = None
        if self._race:
            self._race.stop()
            self._race = None
        if self._initialized:
            pystk2.clean()
            self._initialized = False
        self._closed = True
        if _active is self:
            _active = None

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *_):
        self.close()
