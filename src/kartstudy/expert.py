"""Stateless Team 4 pure-pursuit expert adapted for solo driving.

Source: osigaud/pystk_project, commit
ddf2cb6573850f890bba26d4e5f82888d9dac82a, Team 4 ("The Winners"):
src/agents/team4/{AgentPilot.py,steering.py,speed.py,configuration.yaml}.
https://github.com/osigaud/pystk_project/tree/ddf2cb6573850f890bba26d4e5f82888d9dac82a/src/agents/team4

The source repository contains no LICENSE file. This module records attribution
and implements the documented pure-pursuit equations and published parameters;
it does not assert a redistribution license for the upstream student project.

Unlike the original coordinator this expert has no opponent/item/nitro logic,
no hidden recovery timer, and never resets the environment. Thus an expert
label is a deterministic function of exactly the current raw observation,
including when DAgger visits a state absent from the demonstrations.
"""

from __future__ import annotations

import numpy as np


class Expert:
    """Team 4 driving core with its published L=2.5, gain=7, lookahead=2.

    Expects unflattened Cartesian observations in the kart reference frame:
    ``paths_start`` is (N, 3), with lateral x and forward z, and
    ``distance_down_track`` is a scalar or one-element array. Five visible path
    nodes preserve the original curvature horizon. Short arrays are supported.

    ``conservative`` caps throttle at 0.55 after the launch phase; this defines
    a deliberately slower expert for a quality/ceiling experiment. No manual
    rescue is supplied: shared action wrappers must handle controls identically
    for expert and learned policies. Completion/recovery must be benchmarked.
    """

    name = "team4-pure-pursuit-solo"
    source_commit = "ddf2cb6573850f890bba26d4e5f82888d9dac82a"

    def __init__(self, seed: int = 0, conservative: bool = False):
        self.seed = int(seed)
        self.conservative = bool(conservative)

    def reset(self) -> None:
        """Episode hook; this policy intentionally has no recurrent state."""

    @staticmethod
    def _scalar(value) -> float:
        return float(np.asarray(value).reshape(-1)[0])

    def act(self, raw_obs: dict) -> dict:
        points = np.asarray(raw_obs["paths_start"], dtype=np.float64).reshape(-1, 3)
        distance = self._scalar(raw_obs["distance_down_track"])
        steer, acceleration = 0.0, 1.0
        if len(points) > 2 and distance > 2.0:
            gx, gz = points[2, [0, 2]]
            squared_distance = gx * gx + gz * gz
            if squared_distance >= 0.01:
                steer = float(
                    np.clip(7.0 * np.arctan2(5.0 * gx, squared_distance), -1, 1)
                )
            # Upstream anti-vibration rule.
            if abs(points[2, 0]) < 0.8 and abs(steer) <= 0.05:
                steer = 0.0
            # Team 4 acceleration: clip(2.5 / sqrt(1 + |curvature|), 0, 1).
            # Unwrap segment headings to avoid artificial curvature at +/- pi.
            segment = np.diff(points[1:5], axis=0)
            if len(segment) >= 2:
                headings = np.unwrap(np.arctan2(segment[:, 2], segment[:, 0]))
                curvature = abs(float(np.mean(np.diff(headings))) * 10.0)
                acceleration = float(np.clip(2.5 / np.sqrt(1.0 + curvature), 0, 1))
            if self.conservative:
                acceleration = min(acceleration, 0.55)
        return {
            "acceleration": acceleration,
            "steer": steer,
            "brake": False,
            "drift": False,
            "nitro": False,
            "rescue": False,
            "fire": False,
        }
