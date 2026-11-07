import math
from dataclasses import replace

import numpy as np
import pytest

from kartstudy.spectator_camera import CameraRig, CameraSettings, camera_pose
from kartstudy.spectator_director import TrackPath

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


class StraightReplay:
    duration = 10.0
    ghosts = tuple({"key": key} for key in ("bc", "dagger", "sac"))

    def sample(self, time):
        return [
            dict(POSE, key="bc", position=[0, 0, min(time * 4, 20)], speed=30),
            dict(POSE, key="dagger", position=[2, 0, time * 8], speed=8),
            dict(POSE, key="sac", position=[-2, 0, time * 6], speed=6),
        ]


def test_rail_ranks_progress_not_speed_and_stopped_ghosts_do_not_drag_leaders():
    rig = CameraRig(StraightReplay(), GEOMETRY)
    settings = CameraSettings(mode="rail", pack_size=2, smoothing=0, lookahead=6)
    camera = rig.pose(settings, 8)
    # DAgger (64 m) and SAC (48 m) lead; BC's reported speed is higher.
    np.testing.assert_allclose(camera["position"], [0, 7, 44])
    np.testing.assert_allclose(camera["target"], [0, 0.7, 62])
    single = rig.pose(replace(settings, pack_size=1), 8)
    np.testing.assert_allclose(single["position"], [0, 7, 52])
    assert rig.pose(settings, 100) == rig.pose(settings, 10)


def test_moving_modes_are_seek_independent_and_bound_their_settings_cache():
    rig = CameraRig(StraightReplay(), GEOMETRY)
    for mode in ("rail", "pack", "side", "cinematic"):
        settings = CameraSettings(mode=mode)
        expected = rig.pose(settings, 3.75)
        for time in (9, 0, 5, 1, 10, 2):
            rig.pose(settings, time)
        assert rig.pose(settings, 3.75) == expected
        assert rig.pose(settings, 3.75, []) == rig.pose(settings, 0, [])
        assert np.isfinite(list(expected["position"]) + list(expected["target"])).all()
    for distance in range(1, 20):
        rig.pose(CameraSettings(mode="rail", distance=distance), 5)
    assert len(rig._curves) == 8


def test_side_uses_track_tangent_instead_of_kart_orientation_and_pack_fits_aspect():
    rig = CameraRig(StraightReplay(), GEOMETRY)
    side = rig.pose(CameraSettings(mode="side", smoothing=0), 8)
    np.testing.assert_allclose(side["position"], [12, 7, 56])
    np.testing.assert_allclose(side["target"], [0, 0.7, 56])
    settings = CameraSettings(mode="pack", smoothing=0.5, pack_size=3)
    landscape, portrait = [
        rig.pose(settings, 8, aspect=aspect) for aspect in (16 / 9, 0.5)
    ]
    assert np.linalg.norm(
        np.subtract(portrait["position"], portrait["target"])
    ) > np.linalg.norm(np.subtract(landscape["position"], landscape["target"]))
    points = np.asarray([p["position"] for p in rig.replay.sample(8)])
    for aspect, camera in ((16 / 9, landscape), (0.5, portrait)):
        radius = max(np.linalg.norm(points - camera["target"], axis=1)) + 2
        half_angle = min(
            camera["fov"] / 2, math.atan(math.tan(camera["fov"] / 2) * aspect)
        )
        assert radius / np.linalg.norm(
            np.subtract(camera["position"], camera["target"])
        ) < math.sin(half_angle)


def test_track_projection_unwraps_seam_but_not_rescue_and_avoids_parallel_lane():
    square = {
        "segments": [
            [[0, 0, 0], [0, 0, 10]],
            [[0, 0, 10], [1, 0, 10]],
            [[1, 0, 10], [1, 0, 0]],
            [[1, 0, 0], [0, 0, 0]],
        ]
    }
    path = TrackPath(square)
    initial = path.project([0.2, 0, 0])
    assert initial == pytest.approx(-0.2)
    crossed = path.project([0, 0, 1], initial)
    assert crossed == pytest.approx(1)
    # The nearby return lane is physically closer, but 17 m further along.
    assert path.project([0.7, 0, 2], crossed, allowance=3) == pytest.approx(2)
    before_rescue = path.project([1, 0, 1], 18)
    rescued = path.project([0, 0, 1], before_rescue, discontinuity=True)
    assert rescued == pytest.approx(1)


def test_rescue_to_before_start_line_does_not_add_a_lap():
    loop = {
        "segments": [
            [[0, 0, 0], [0, 0, 30]],
            [[0, 0, 30], [20, 0, 30]],
            [[20, 0, 30], [20, 0, 0]],
            [[20, 0, 0], [0, 0, 0]],
        ]
    }
    path = TrackPath(loop)
    assert path.length == 100
    before_start = [2, 0, 0]  # Wrapped progress 98, or race progress -2.
    assert path.project(before_start, 10, discontinuity=True) == pytest.approx(-2)
    assert path.project(before_start, -2, discontinuity=True) == pytest.approx(-2)
    assert path.project(before_start, 110, discontinuity=True) == pytest.approx(98)
    # A rescue backwards from the end stays in the current lap; it must not
    # be interpreted as driving forward across the finish line.
    assert path.project([0, 0, 1], 98, discontinuity=True) == pytest.approx(1)


def test_keyframes_are_validated_frozen_serializable_and_interpolate_with_end_holds():
    frames = [
        {"time": 2, "position": [0, 5, -10], "target": [0, 0, 0], "fov": 40},
        {"time": 6, "position": [10, 5, -10], "target": [10, 0, 0], "fov": 80},
    ]
    settings = CameraSettings(mode="cinematic", keyframes=frames)
    frames[0]["position"][0] = 999
    assert settings.keyframes[0].position == (0, 5, -10)
    assert CameraSettings(**settings.to_dict()) == settings
    assert hash(settings)
    rig = CameraRig(StraightReplay(), GEOMETRY)
    assert rig.pose(settings, 0) == rig.pose(settings, 2)
    assert rig.pose(settings, 10) == rig.pose(settings, 6)
    middle = rig.pose(settings, 4)
    np.testing.assert_allclose(middle["position"], [5, 5, -10])
    np.testing.assert_allclose(middle["target"], [5, 0, 0])
    assert middle["fov"] == pytest.approx(math.pi / 3)
    # Quarter interval uses cubic smoothstep, rather than playback-history easing.
    assert rig.pose(settings, 3)["position"][0] == pytest.approx(1.5625)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pack_size": 0},
        {"pack_size": 33},
        {"pack_size": 1.5},
        {"smoothing": -1},
        {"smoothing": 6},
        {"lookahead": float("inf")},
        {"lookahead": -1},
        {"keyframes": [{"time": -1, "position": [0, 0, -1], "target": [0, 0, 0]}]},
        {"keyframes": [{"time": 0, "position": [0, 0, 0], "target": [0, 1, 0]}]},
        {
            "keyframes": [
                {"time": 2, "position": [0, 0, -1], "target": [0, 0, 0]},
                {"time": 2, "position": [0, 0, -1], "target": [0, 0, 0]},
            ]
        },
        {
            "keyframes": [
                {"time": 0, "position": [0, 0, -1], "target": [0, 0, 0]},
                {"time": 2, "position": [0, 0, 1], "target": [0, 0, 0]},
            ]
        },
    ],
)
def test_moving_camera_bad_settings_fail_before_rendering(kwargs):
    with pytest.raises(ValueError):
        CameraSettings(**kwargs)


def test_fixed_grid_smoothing_reduces_a_rescue_camera_jump():
    class RescueReplay(StraightReplay):
        ghosts = ({"key": "dagger"},)

        def sample(self, time):
            position = time * 10 if time < 5 else 10 + (time - 5) * 10
            return [dict(POSE, position=[0, 0, position], speed=10)]

    rig = CameraRig(RescueReplay(), GEOMETRY)
    raw = CameraSettings(mode="rail", smoothing=0, pack_size=1)
    smooth = replace(raw, smoothing=0.5)
    raw_step = np.linalg.norm(
        np.subtract(rig.pose(raw, 5)["position"], rig.pose(raw, 4.9)["position"])
    )
    smooth_step = np.linalg.norm(
        np.subtract(rig.pose(smooth, 5)["position"], rig.pose(smooth, 4.9)["position"])
    )
    assert raw_step > 30 and smooth_step < 5
