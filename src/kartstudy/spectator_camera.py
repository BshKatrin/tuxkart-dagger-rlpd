"""Deterministic cameras for recorded solo-lap ghosts (Y is up)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np

from .spectator_contract import CameraPose, GhostPose, TrackGeometry
from .spectator_director import CameraRig

__all__ = [
    "CAMERA_MODES",
    "CameraKeyframe",
    "CameraRig",
    "CameraSettings",
    "camera_pose",
]

CAMERA_MODES = (
    "follow",
    "corner",
    "overview",
    "chase",
    "rail",
    "pack",
    "side",
    "cinematic",
)


@dataclass(frozen=True)
class CameraKeyframe:
    time: float
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    fov: float = 60.0

    def __post_init__(self):
        for name in ("position", "target"):
            vector = tuple(float(v) for v in getattr(self, name))
            if len(vector) != 3 or not all(math.isfinite(v) for v in vector):
                raise ValueError("Keyframes require finite 3D positions and targets")
            object.__setattr__(self, name, vector)
        if not math.isfinite(self.time) or self.time < 0:
            raise ValueError("Keyframe times must be finite and nonnegative")
        if not math.isfinite(self.fov) or not 10 <= self.fov <= 120:
            raise ValueError("Keyframe FOV must be between 10 and 120 degrees")
        direction = np.subtract(self.target, self.position)
        if np.linalg.norm(direction[[0, 2]]) < 1e-6:
            raise ValueError("Keyframes require a nonvertical viewing direction")


@dataclass(frozen=True)
class CameraSettings:
    mode: str = "follow"
    follow: str = "dagger"
    distance: float = 12.0
    height: float = 7.0
    orbit: float = 0.0  # degrees around world Y
    fov: float = 60.0  # degrees at the user boundary
    segment: int = 0
    pack_size: int = 2
    lookahead: float = 6.0
    smoothing: float = 0.5
    keyframes: tuple[CameraKeyframe, ...] = ()

    def __post_init__(self):
        if self.mode not in CAMERA_MODES:
            raise ValueError(f"Unknown camera mode: {self.mode}")
        if not all(
            math.isfinite(x)
            for x in (
                self.distance,
                self.height,
                self.orbit,
                self.fov,
                self.lookahead,
                self.smoothing,
            )
        ):
            raise ValueError("Camera values must be finite")
        if self.distance <= 0 or self.height < 0 or not 10 <= self.fov <= 120:
            raise ValueError(
                "Camera requires distance > 0, height >= 0, FOV 10–120 degrees"
            )
        if self.segment < 0:
            raise ValueError("Corner segment must be nonnegative")
        if (
            isinstance(self.pack_size, bool)
            or not isinstance(self.pack_size, int)
            or not 1 <= self.pack_size <= 32
        ):
            raise ValueError("Pack size must be an integer between 1 and 32")
        if self.lookahead < 0 or not 0 <= self.smoothing <= 5:
            raise ValueError(
                "Lookahead must be nonnegative and smoothing between 0 and 5 seconds"
            )
        try:
            frames = tuple(
                f if isinstance(f, CameraKeyframe) else CameraKeyframe(**f)
                for f in self.keyframes
            )
        except (TypeError, OverflowError) as exc:
            raise ValueError("Malformed camera keyframe") from exc
        for left, right in pairwise(frames):
            if left.time >= right.time:
                raise ValueError("Keyframe times must be strictly increasing")
            # The native camera has a fixed Y-up vector. Reject transitions that
            # interpolate through a coincident or vertical look direction.
            a = np.subtract(left.target, left.position)[[0, 2]]
            b = np.subtract(right.target, right.position)[[0, 2]] - a
            u = np.clip(-np.dot(a, b) / max(float(np.dot(b, b)), 1e-12), 0, 1)
            if np.linalg.norm(a + u * b) < 1e-6:
                raise ValueError(
                    "Keyframe transition crosses a vertical viewing direction"
                )
        object.__setattr__(self, "keyframes", frames)

    def to_dict(self):
        result = asdict(self)
        result["keyframes"] = [
            {
                "time": frame.time,
                "position": list(frame.position),
                "target": list(frame.target),
                "fov": frame.fov,
            }
            for frame in self.keyframes
        ]
        return result


def camera_pose(
    settings: CameraSettings,
    poses: list[GhostPose],
    geometry: TrackGeometry,
    aspect: float = 16 / 9,
) -> CameraPose:
    """Use the same sampled poses for GUI and video; never switch modes implicitly."""
    if not math.isfinite(aspect) or aspect <= 0:
        raise ValueError("Camera aspect ratio must be positive")
    if settings.mode in ("rail", "pack", "side", "cinematic"):
        raise ValueError("Moving cameras require CameraRig and a replay timestamp")
    segments = np.asarray(geometry["segments"], dtype=float).reshape(-1, 2, 3)
    fov = math.radians(settings.fov)
    if settings.mode == "overview":
        points = np.asarray([p["position"] for p in poses if p["visible"]], dtype=float)
        if not points.size:
            points = segments.reshape(-1, 3)
        if not points.size:
            raise ValueError("Overview requires ghosts or track geometry")
        low, high = points.min(axis=0), points.max(axis=0)
        target = (low + high) / 2
        # Conservative sphere fit in both vertical and horizontal fields of view.
        radius = max(float(np.linalg.norm(high - low)) / 2, 2.0)
        half_angle = min(fov / 2, math.atan(math.tan(fov / 2) * aspect))
        distance = max(settings.distance, radius / math.sin(half_angle) * 1.15)
        direction = np.array([0.0, 1.0, -1.0]) / math.sqrt(2)
        position = target + _orbit(direction, settings.orbit) * distance
        position[1] += settings.height
    elif settings.mode == "corner":
        if settings.segment >= len(segments):
            raise ValueError(f"Corner segment {settings.segment} is outside the track")
        segment = segments[settings.segment]
        target = segment.mean(axis=0)
        direction = segment[1] - segment[0]
        direction[1] = 0
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        if not direction.any():
            direction = np.array([0.0, 0.0, 1.0])
        side = np.array([direction[2], 0, -direction[0]])
        position = target + _orbit(side, settings.orbit) * settings.distance
        position[1] += settings.height
    else:
        selected = next((p for p in poses if p["key"] == settings.follow), None)
        if selected is None:
            raise ValueError(f"Follow target is absent: {settings.follow}")
        target = np.asarray(selected["position"], dtype=float).copy()
        x, y, z, w = selected["rotation"]
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0 or not math.isfinite(norm):
            raise ValueError("Follow target has an invalid quaternion")
        x, y, z, w = (v / norm for v in (x, y, z, w))
        forward = np.array([2 * (x * z + w * y), 0.0, 1 - 2 * (x * x + y * y)])
        forward /= max(float(np.linalg.norm(forward)), 1e-9)
        if not forward.any():
            forward = np.array([0.0, 0.0, 1.0])
        # Chase is a closer eye-level camera; knobs remain useful in both modes.
        scale = 0.45 if settings.mode == "chase" else 1.0
        position = target - _orbit(forward, settings.orbit) * settings.distance * scale
        position[1] += max(0.2, settings.height * scale)
        target[1] += 0.7
    return {"position": position.tolist(), "target": target.tolist(), "fov": fov}


def _orbit(vector, degrees):
    angle = math.radians(degrees)
    x, y, z = vector
    return np.array(
        [
            math.cos(angle) * x + math.sin(angle) * z,
            y,
            -math.sin(angle) * x + math.cos(angle) * z,
        ]
    )
