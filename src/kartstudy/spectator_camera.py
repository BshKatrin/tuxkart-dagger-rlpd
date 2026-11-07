"""Deterministic cameras for recorded solo-lap ghosts (Y is up)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .spectator_contract import CameraPose, GhostPose, TrackGeometry

CAMERA_MODES = ("follow", "corner", "overview", "chase")


@dataclass(frozen=True)
class CameraSettings:
    mode: str = "follow"
    follow: str = "dagger"
    distance: float = 12.0
    height: float = 7.0
    orbit: float = 0.0  # degrees around world Y
    fov: float = 60.0  # degrees at the user boundary
    segment: int = 0

    def __post_init__(self):
        if self.mode not in CAMERA_MODES:
            raise ValueError(f"Unknown camera mode: {self.mode}")
        if not all(
            math.isfinite(x) for x in (self.distance, self.height, self.orbit, self.fov)
        ):
            raise ValueError("Camera values must be finite")
        if self.distance <= 0 or self.height < 0 or not 10 <= self.fov <= 120:
            raise ValueError(
                "Camera requires distance > 0, height >= 0, FOV 10–120 degrees"
            )
        if self.segment < 0:
            raise ValueError("Corner segment must be nonnegative")

    def to_dict(self):
        return asdict(self)


def camera_pose(
    settings: CameraSettings,
    poses: list[GhostPose],
    geometry: TrackGeometry,
    aspect: float = 16 / 9,
) -> CameraPose:
    """Use the same sampled poses for GUI and video; never switch modes implicitly."""
    if not math.isfinite(aspect) or aspect <= 0:
        raise ValueError("Camera aspect ratio must be positive")
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
