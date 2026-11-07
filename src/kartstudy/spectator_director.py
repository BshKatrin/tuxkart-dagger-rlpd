"""Track-relative, seek-independent cameras for frozen replay trajectories."""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np


class TrackPath:
    def __init__(self, geometry):
        segments = np.asarray(geometry["segments"], dtype=float).reshape(-1, 2, 3)
        if not np.isfinite(segments).all():
            raise ValueError("Moving cameras require finite track geometry")
        vectors = segments[:, 1] - segments[:, 0]
        lengths = np.linalg.norm(vectors, axis=1)
        valid = lengths > 1e-6
        self.segments, self.vectors, self.lengths = (
            segments[valid],
            vectors[valid],
            lengths[valid],
        )
        if not len(self.segments):
            raise ValueError("Moving cameras require nonempty track geometry")
        self.offsets = np.r_[0.0, np.cumsum(self.lengths)]
        self.length = float(self.offsets[-1])
        self.closed = np.linalg.norm(self.segments[-1, 1] - self.segments[0, 0]) < min(
            2.0, self.length * 0.1
        )

    def point(self, progress):
        distance = (
            progress % self.length if self.closed else np.clip(progress, 0, self.length)
        )
        index = min(
            int(np.searchsorted(self.offsets, distance, side="right")) - 1,
            len(self.lengths) - 1,
        )
        alpha = (distance - self.offsets[index]) / self.lengths[index]
        return self.segments[index, 0] + alpha * self.vectors[index]

    def tangent(self, progress):
        vector = self.point(progress + 2.0) - self.point(progress - 2.0)
        vector[1] = 0
        if np.linalg.norm(vector) < 1e-6:
            distance = (
                progress % self.length
                if self.closed
                else np.clip(progress, 0, self.length)
            )
            index = min(
                int(np.searchsorted(self.offsets, distance, side="right")) - 1,
                len(self.lengths) - 1,
            )
            vector = self.vectors[index].copy()
            vector[1] = 0
        return vector / max(float(np.linalg.norm(vector)), 1e-9)

    def project(self, point, previous=None, *, discontinuity=False, allowance=10.0):
        relative = np.asarray(point) - self.segments[:, 0]
        alpha = np.clip(
            np.einsum("ij,ij->i", relative, self.vectors) / self.lengths**2, 0, 1
        )
        projected = self.segments[:, 0] + alpha[:, None] * self.vectors
        spatial = np.sum((projected - point) ** 2, axis=1)
        distance = self.offsets[:-1] + alpha * self.lengths
        if previous is None:
            result = float(distance[int(np.argmin(spatial))])
            # Starts just before the finish line are race progress near zero,
            # rather than a lap ahead of a neighbouring starting lane.
            return (
                result - self.length
                if self.closed and result > self.length - min(30, self.length * 0.1)
                else result
            )
        if self.closed:
            if discontinuity:
                distance += max(0, math.floor(previous / self.length)) * self.length
                # A rescue just before the start line belongs behind a kart
                # just after it. Never infer a new lap from a teleport.
                distance = np.where(
                    distance - previous > self.length / 2,
                    distance - self.length,
                    distance,
                )
            else:
                distance += np.round((previous - distance) / self.length) * self.length
        if discontinuity:
            return float(distance[int(np.argmin(spatial))])
        delta = distance - previous
        plausible = np.abs(delta) <= allowance
        # Continuity disambiguates nearby parallel sections and crossings. A
        # real teleport is handled separately, without manufacturing a lap.
        score = spatial + 0.02 * delta**2
        if plausible.any():
            score = np.where(plausible, score, np.inf)
        return float(distance[int(np.argmin(score))])


def _smooth(values, seconds, step):
    if seconds == 0 or len(values) < 2:
        return values.copy()
    sigma = max(seconds / step, 0.1)
    radius = min(math.ceil(3 * sigma), len(values) - 1)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    result = np.column_stack(
        [
            np.convolve(padded[:, column], kernel, mode="valid")
            for column in range(values.shape[1])
        ]
    )
    return result


def _pose(position, target, fov):
    if np.linalg.norm((target - position)[[0, 2]]) < 1e-6:
        position = position.copy()
        position[2] -= 1.0
    return {"position": position.tolist(), "target": target.tolist(), "fov": fov}


def _keyframe_pose(frames, seconds):
    times = [frame.time for frame in frames]
    right = min(int(np.searchsorted(times, seconds, side="right")), len(frames) - 1)
    left = max(0, right - 1)
    a, b = frames[left], frames[right]
    alpha = (
        np.clip((seconds - a.time) / (b.time - a.time), 0, 1) if left != right else 0.0
    )
    alpha = alpha * alpha * (3 - 2 * alpha)
    position = np.asarray(a.position) + alpha * np.subtract(b.position, a.position)
    target = np.asarray(a.target) + alpha * np.subtract(b.target, a.target)
    fov = math.radians(a.fov + alpha * (b.fov - a.fov))
    return _pose(position, target, fov)


class CameraRig:
    """Precompute on a 10 Hz replay clock, then sample without playback history."""

    def __init__(self, replay, geometry):
        self.replay, self.geometry = replay, geometry
        self.path = TrackPath(geometry)
        if not math.isfinite(replay.duration) or not 0 <= replay.duration <= 3600:
            raise ValueError("Moving cameras support replay durations up to one hour")
        count = max(1, math.ceil(replay.duration * 10) + 1)
        self.times = np.linspace(0, replay.duration, count)
        self.step = replay.duration / max(1, count - 1) or 0.1
        self.keys = [g["key"] for g in replay.ghosts]
        self.positions = np.zeros((count, len(self.keys), 3))
        self.progress = np.zeros((count, len(self.keys)))
        previous = [None] * len(self.keys)
        for i, time in enumerate(self.times):
            sample = replay.sample(float(time))
            for ghost in range(len(self.keys)):
                point = np.asarray(sample[ghost]["position"], dtype=float)
                self.positions[i, ghost] = point
                speed = abs(float(sample[ghost]["speed"]))
                jump = i > 0 and np.linalg.norm(
                    point - self.positions[i - 1, ghost]
                ) > max(10, 3 * speed * self.step)
                previous[ghost] = self.path.project(
                    point,
                    previous[ghost],
                    discontinuity=jump,
                    allowance=max(10, 3 * speed * self.step),
                )
                self.progress[i, ghost] = previous[ghost]
        self._curves = OrderedDict()

    def _leaders(self, progress, visible, pack_size):
        indices = [i for i, key in enumerate(self.keys) if key in visible]
        return sorted(indices, key=lambda i: (-progress[i], self.keys[i]))[:pack_size]

    def _moving(self, settings, progress, points, leaders, seconds, aspect):
        from .spectator_camera import _orbit

        reference = float(np.mean(progress[leaders])) if leaders else 0.0
        center = self.path.point(reference)
        tangent = self.path.tangent(reference)
        orbit, height = settings.orbit, settings.height
        if settings.mode == "cinematic" and leaders:
            phase = np.clip(seconds / max(self.replay.duration, 1), 0, 1)
            # Three smoothly joined track-relative framing keyframes.
            values = ((-25, 0), (30, 5), (-10, 1))
            segment = min(int(phase * 2), 1)
            alpha = phase * 2 - segment
            alpha = alpha * alpha * (3 - 2 * alpha)
            orbit += values[segment][0] + alpha * (
                values[segment + 1][0] - values[segment][0]
            )
            height += values[segment][1] + alpha * (
                values[segment + 1][1] - values[segment][1]
            )
        target = self.path.point(reference + settings.lookahead)
        if settings.mode == "side":
            side = np.array([tangent[2], 0, -tangent[0]])
            position = center + _orbit(side, orbit) * settings.distance
            target = np.mean(points[leaders], axis=0) if leaders else center.copy()
        elif settings.mode == "pack":
            target = (
                (points[leaders].min(axis=0) + points[leaders].max(axis=0)) / 2
                if leaders
                else center.copy()
            )
            position = target - _orbit(tangent, orbit) * settings.distance
        else:
            behind = self.path.point(reference - settings.distance) - center
            if np.linalg.norm(behind[[0, 2]]) < 1e-6:
                behind = -tangent * settings.distance
            position = center + _orbit(behind, orbit)
        position[1] += height
        target = target.copy()
        target[1] += 0.7
        return np.r_[position, target]

    def pose(self, settings, seconds, visible=None, aspect=16 / 9):
        from .spectator_camera import camera_pose

        if not math.isfinite(seconds) or not math.isfinite(aspect) or aspect <= 0:
            raise ValueError(
                "Camera time and aspect must be finite; aspect must be positive"
            )
        seconds = float(np.clip(seconds, 0, self.replay.duration))
        visible = frozenset(self.keys if visible is None else visible)
        if settings.mode not in ("rail", "pack", "side", "cinematic"):
            poses = [
                dict(p, visible=p["visible"] and p["key"] in visible)
                for p in self.replay.sample(seconds)
            ]
            return camera_pose(settings, poses, self.geometry, aspect)
        if settings.mode == "cinematic" and settings.keyframes:
            return _keyframe_pose(settings.keyframes, seconds)
        cache_key = (settings, visible, aspect)
        if cache_key not in self._curves:
            values = []
            for i, t in enumerate(self.times):
                leaders = self._leaders(self.progress[i], visible, settings.pack_size)
                values.append(
                    self._moving(
                        settings,
                        self.progress[i],
                        self.positions[i],
                        leaders,
                        float(t),
                        aspect,
                    )
                )
            self._curves[cache_key] = _smooth(
                np.asarray(values), settings.smoothing, self.step
            )
            if len(self._curves) > 8:
                self._curves.popitem(last=False)
        self._curves.move_to_end(cache_key)
        values = self._curves[cache_key]
        vector = np.array(
            [np.interp(seconds, self.times, values[:, i]) for i in range(6)]
        )
        position, target = vector[:3], vector[3:]
        fov = math.radians(settings.fov)
        if settings.mode == "pack":
            progress = np.array(
                [
                    np.interp(seconds, self.times, self.progress[:, i])
                    for i in range(len(self.keys))
                ]
            )
            leaders = self._leaders(progress, visible, settings.pack_size)
            if leaders:
                points = np.asarray(
                    [p["position"] for p in self.replay.sample(seconds)]
                )[leaders]
                radius = float(np.max(np.linalg.norm(points - target, axis=1))) + 2.0
                half_angle = min(fov / 2, math.atan(math.tan(fov / 2) * aspect))
                distance = max(
                    float(np.linalg.norm(position - target)),
                    radius / math.sin(half_angle) * 1.15,
                )
                direction = position - target
                if np.linalg.norm(direction[[0, 2]]) < 1e-6:
                    direction = np.array([0, 1, -1], dtype=float)
                position = target + direction / np.linalg.norm(direction) * distance
        return _pose(position, target, fov)
