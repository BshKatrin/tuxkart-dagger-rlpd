"""Shared presentation types; quaternion components are always x, y, z, w."""

from typing import Literal, TypedDict

ReplayStatus = Literal["driving", "finished", "timeout", "stopped"]


class GhostConfig(TypedDict):
    key: str
    label: str
    method: str
    kart: str
    color: float


class GhostPose(TypedDict):
    key: str
    position: list[float]
    rotation: list[float]
    speed: float
    steer: float
    visible: bool
    status: ReplayStatus


class CameraPose(TypedDict):
    position: list[float]
    target: list[float]
    fov: float


class TrackGeometry(TypedDict):
    segments: list[list[list[float]]]
    widths: list[float]
