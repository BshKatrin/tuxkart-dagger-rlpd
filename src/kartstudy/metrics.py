"""Shared track-row and banana-effect measurements."""

import numpy as np


class DrivingMetrics:
    """Time-weighted native road status and distinct rescue animations."""

    def __init__(self):
        self.seconds = self.off_track_seconds = 0.0
        self.rescue_count = 0
        self.previous_rescue_timer = None
        self.off_track_intervals = []

    def update(self, before, after, on_road, rescue_timer, finish_time=None):
        end = max(0.0, after)
        if finish_time is not None:
            end = min(end, finish_time)
        dt = max(0.0, end - max(0.0, before))
        self.seconds += dt
        self.off_track_seconds += dt * (not on_road)
        if not on_road and dt:
            self.off_track_intervals.append((max(0.0, before), end))
        if finish_time is not None:
            # Native finish detection may arrive one action after the interpolated
            # finish timestamp, so trim the preceding sampled interval too.
            self.seconds = min(self.seconds, finish_time)
            self.off_track_seconds = sum(
                max(0.0, min(stop, finish_time) - start)
                for start, stop in self.off_track_intervals
            )
        if rescue_timer is not None and (
            self.previous_rescue_timer is None
            or rescue_timer > self.previous_rescue_timer + 1e-4
        ):
            self.rescue_count += 1
        self.previous_rescue_timer = rescue_timer

    def snapshot(self):
        return {
            "measurement_seconds": self.seconds,
            "off_track_seconds": self.off_track_seconds,
            "off_track_fraction": self.off_track_seconds / self.seconds
            if self.seconds
            else 0.0,
            "rescue_count": self.rescue_count,
        }


def harmful_rows(items, track, tolerance=3.0):
    """Group bananas across the road by projected longitudinal position."""
    nodes = np.asarray(track.path_nodes, dtype=float)
    distances = np.asarray(track.path_distance, dtype=float)
    start, delta = nodes[:, 0], nodes[:, 1] - nodes[:, 0]
    entries = []
    for item in items:
        if int(item.type) == 4:
            raise ValueError("Bubblegum contact detection is not supported")
        if int(item.type) != 1:
            continue
        position = np.asarray(item.location, dtype=float)
        fraction = np.clip(
            np.sum((position - start) * delta, axis=1)
            / np.maximum(np.sum(delta * delta, axis=1), 1e-12),
            0,
            1,
        )
        nearest = np.argmin(
            np.linalg.norm(start + fraction[:, None] * delta - position, axis=1)
        )
        along = (
            distances[nearest, 0] + fraction[nearest] * np.diff(distances[nearest])[0]
        )
        entries.append(
            {
                "id": int(item.id),
                "position": position.tolist(),
                "distance": float(along % track.length),
            }
        )
    if not entries:
        return []
    entries.sort(key=lambda item: item["distance"])
    # Cut the circular track at its largest empty gap, including rows at the seam.
    positions = np.array([item["distance"] for item in entries])
    cut = int(np.argmax(np.diff(np.r_[positions, positions[0] + track.length]))) + 1
    entries = entries[cut:] + entries[:cut]
    origin = entries[0]["distance"]
    rows = []
    for item in entries:
        offset = (item["distance"] - origin) % track.length
        if not rows or offset - rows[-1]["offset"] > tolerance:
            rows.append({"offset": float(offset), "items": []})
        rows[-1]["items"].append(item)
    return rows


def banana_contact(previous, current, dt):
    """Infer a pickup from attachment onset/extension or premature bomb removal.

    PySTK exposes neither pickup events nor item availability. A banana creates
    an attachment, extends an existing anvil/parachute, or explodes a held bomb.
    Multiple contacts inside one action interval may be indistinguishable.
    """
    old_type, old_time = previous
    new_type, new_time = current
    return bool(
        (new_type in (0, 1, 2) and (new_type != old_type or new_time > old_time + 1e-4))
        or (old_type == 2 and new_type == 9 and old_time > dt + 0.02)
    )
