import math
from dataclasses import replace

import numpy as np
import pytest

from kartstudy.spectator_camera import CameraSettings, camera_pose

GEOMETRY = {
    "segments": [[[0, 0, 0], [0, 0, 100]], [[0, 0, 100], [100, 0, 100]]],
    "widths": [8, 8],
}
POSE = {
    "key": "dagger",
    "position": [0, 0, 10],
    "rotation": [0, 0, 0, 1],
    "speed": 5,
    "steer": 0,
    "visible": True,
    "status": "driving",
}


def test_follow_and_chase_are_deterministic_and_use_normalized_rotation():
    settings = CameraSettings()
    pose = camera_pose(settings, [POSE], GEOMETRY)
    np.testing.assert_allclose(pose["position"], [0, 7, -2])
    np.testing.assert_allclose(pose["target"], [0, 0.7, 10])
    assert pose["fov"] == math.pi / 3
    rotated = dict(POSE, rotation=[0, math.sqrt(2), 0, math.sqrt(2)])
    np.testing.assert_allclose(
        camera_pose(settings, [rotated], GEOMETRY)["position"], [-12, 7, 10], atol=1e-12
    )
    orbit = camera_pose(replace(settings, orbit=90), [POSE], GEOMETRY)
    np.testing.assert_allclose(orbit["position"], [-12, 7, 10], atol=1e-12)
    chase = camera_pose(replace(settings, mode="chase"), [POSE], GEOMETRY)
    assert np.linalg.norm(np.array(chase["position"]) - POSE["position"]) < 12
    assert camera_pose(settings, [POSE], GEOMETRY) == pose
    # A hidden or terminal target stays selected; no automatic cuts.
    assert (
        camera_pose(settings, [dict(POSE, visible=False, status="finished")], GEOMETRY)
        == pose
    )


def test_corner_is_fixed_and_bounds_checked():
    settings = CameraSettings(mode="corner", segment=1)
    expected = camera_pose(settings, [POSE], GEOMETRY)
    assert (
        camera_pose(settings, [dict(POSE, position=[100, 0, 100])], GEOMETRY)
        == expected
    )
    np.testing.assert_allclose(expected["target"], [50, 0, 100])
    with pytest.raises(ValueError, match="outside"):
        camera_pose(replace(settings, segment=2), [POSE], GEOMETRY)


def test_overview_fits_visible_ghosts_in_portrait_and_uses_track_when_hidden():
    settings = CameraSettings(mode="overview")
    ghosts = [POSE, dict(POSE, key="bc", position=[200, 0, 100])]
    wide = camera_pose(settings, ghosts, GEOMETRY, 16 / 9)
    portrait = camera_pose(settings, ghosts, GEOMETRY, 0.5)
    np.testing.assert_allclose(wide["target"], [100, 0, 55])
    assert np.linalg.norm(
        np.subtract(portrait["position"], portrait["target"])
    ) > np.linalg.norm(np.subtract(wide["position"], wide["target"]))
    hidden = [dict(p, visible=False) for p in ghosts]
    np.testing.assert_allclose(
        camera_pose(settings, hidden, GEOMETRY)["target"], [50, 0, 50]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distance": 0},
        {"height": -1},
        {"fov": 121},
        {"orbit": float("nan")},
        {"segment": -1},
        {"mode": "auto"},
    ],
)
def test_bad_camera_arguments_fail(kwargs):
    with pytest.raises(ValueError):
        CameraSettings(**kwargs)
