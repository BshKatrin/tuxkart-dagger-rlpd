"""Native presentation acceptance runs in isolated simulator processes."""

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from kartstudy.spectator_native import NativeSpectator


@pytest.mark.parametrize(
    "kwargs",
    [
        {"track": ""},
        {"width": 0},
        {"height": 63},
        {"width": 128.5},
        {"seed": -1},
        {"ghosts": [{"key": "bad"}]},
    ],
)
def test_invalid_constructor_fails_before_starting_simulator(kwargs):
    arguments = {"track": "lighthouse", "ghosts": []} | kwargs
    with pytest.raises(ValueError):
        NativeSpectator(**arguments)


def test_binary_fingerprint_rejects_unverified_same_version(monkeypatch):
    import kartstudy.spectator_native_build as build

    build.load_adapter.cache_clear()
    monkeypatch.setattr(build.sys, "platform", "darwin")
    monkeypatch.setattr(build, "version", lambda _: "0.7.4")
    monkeypatch.setattr(build, "_sha", lambda _: "unverified")
    with pytest.raises(RuntimeError, match="fingerprint"):
        build.load_adapter()


@pytest.mark.skipif(sys.platform != "darwin", reason="Pinned native Mac renderer")
@pytest.mark.parametrize("track", ["lighthouse", "stk_enterprise", "mines"])
def test_original_track_four_ghosts_without_physics_or_clock_changes(track, tmp_path):
    proof = Path(os.environ.get("KARTSTUDY_NATIVE_PROOF_DIR", str(tmp_path))).resolve()
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), track, str(proof)],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads((proof / f"{track}-acceptance.json").read_text())
    assert (
        evidence["physical_kart_count_before"]
        == evidence["physical_kart_count_after"]
        == 1
    )
    assert evidence["race_time_before"] == evidence["race_time_after"]
    assert evidence["ghost_count"] == 4
    assert evidence["individual_visible_pixels"] == sorted(
        evidence["individual_visible_pixels"]
    )
    assert evidence["individual_visible_pixels"][0] > 50


def _native_acceptance(track, proof):
    import pystk2

    proof.mkdir(parents=True, exist_ok=True)
    configs = [
        {
            "key": f"g{i}",
            "label": f"Ghost {i}",
            "method": "fixture",
            "kart": kart,
            "color": i / 4,
        }
        for i, kart in enumerate(("tux", "sara_the_racer", "gnu", "wilber"))
    ]
    spectator = NativeSpectator(track, configs, width=960, height=540)
    try:
        geometry = spectator.geometry()
        segments = np.asarray(geometry["segments"])
        assert segments.ndim == 3 and segments.shape[1:] == (2, 3)
        assert len(geometry["widths"]) == len(segments)
        assert np.isfinite(segments).all() and min(geometry["widths"]) > 0
        p = segments[0, 0].copy()
        direction = segments[0, 1] - p
        direction[1] = 0
        direction /= np.linalg.norm(direction)
        side = np.cross([0.0, 1.0, 0.0], direction)
        # Four recorded-pose style deterministic fixtures in the original start area.
        poses = [
            {
                "key": f"g{i}",
                "position": (p + side * (i - 1.5) * 1.3 + [0, 0.25, 0]).tolist(),
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "speed": 12.0,
                "steer": 0.0,
                "visible": True,
                "status": "driving",
            }
            for i in range(4)
        ]
        camera = {
            "position": (p - direction * 10 + [0, 5, 0]).tolist(),
            "target": (p + [0, 0.5, 0]).tolist(),
            "fov": math.radians(60),
        }
        spectator.set_camera(camera)
        empty = spectator.render()
        Image.fromarray(empty).save(proof / f"{track}-empty.png")
        visible_pixels = []
        for pose in poses:
            spectator.set_ghost_poses([pose])
            single = spectator.render()
            visible_pixels.append(
                int(np.count_nonzero(np.any(single != empty, axis=-1)))
            )
        spectator.set_ghost_poses(poses)
        all_ghosts = spectator.render()
        Image.fromarray(all_ghosts).save(proof / f"{track}-four-ghosts.png")
        assert all_ghosts.shape == (540, 960, 3) and all_ghosts.dtype == np.uint8
        assert all_ghosts.std() > 10  # Textured scenery, not an empty framebuffer.
        diag = spectator.diagnostics()
        np.testing.assert_allclose(
            diag["camera"]["position"], camera["position"], atol=1e-5
        )
        np.testing.assert_allclose(
            diag["camera"]["target"], camera["target"], atol=1e-5
        )
        assert math.isclose(diag["camera"]["fov"], camera["fov"], abs_tol=1e-6)
        assert set(diag["visible_ghosts"]) == {f"g{i}" for i in range(4)}
        # Check actual native model directions against native engine pose export.
        world = pystk2.WorldState()
        rotations = [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)],
            Rotation.from_euler("xyz", [0.2, 0.7, -0.3]).as_quat().tolist(),
        ]
        orientation_errors = []
        for quaternion in rotations:
            pystk2.WorldState.set_kart_location(0, p.tolist(), quaternion, 0.0)
            world.update()
            inverse_wxyz = np.asarray(world.karts[0].rotation)
            world_xyzw = np.r_[-inverse_wxyz[1:], inverse_wxyz[0]]
            np.testing.assert_allclose(world_xyzw, quaternion, atol=1e-6)
            turned = poses[0] | {"rotation": world_xyzw.tolist()}
            spectator.set_ghost_poses([turned])
            spectator.render()
            actual = spectator.diagnostics()["ghost_transforms"]["g0"]["forward"]
            expected = Rotation.from_quat(quaternion).apply([0, 0, 1])
            np.testing.assert_allclose(actual, expected, atol=5e-6)
            orientation_errors.append(float(np.max(np.abs(actual - expected))))
        # Invalid batches leave all native poses untouched.
        before = spectator.diagnostics()["ghost_transforms"]
        invalid = poses[0] | {"rotation": [0.0, 0.0, 0.0, 0.0]}
        with pytest.raises(ValueError):
            spectator.set_ghost_poses([poses[1], invalid])
        assert spectator.diagnostics()["ghost_transforms"] == before
        for change in (
            {"position": [np.nan, 0, 0]},
            {"steer": 2},
            {"visible": 1},
            {"status": "unknown"},
            {"speed": float("inf")},
            {"key": "missing"},
        ):
            with pytest.raises(ValueError):
                spectator.set_ghost_poses([poses[0] | change])
        with pytest.raises(ValueError):
            spectator.set_ghost_poses([poses[0], poses[0]])
        for change in (
            {"fov": 180.0},
            {"target": camera["position"]},
            {"position": [0, 1]},
        ):
            with pytest.raises(ValueError):
                spectator.set_camera(camera | change)
        with pytest.raises(RuntimeError, match="own simulator"):
            NativeSpectator(track, [])
        # Seek backwards replaces transforms and hides omitted ghosts.
        spectator.set_ghost_poses(poses)
        shifted = copy.deepcopy(poses)
        for pose in shifted:
            pose["position"] = (np.asarray(pose["position"]) + direction * 5).tolist()
        spectator.set_ghost_poses(shifted)
        spectator.render()
        spectator.set_ghost_poses(poses)
        spectator.render()
        for pose in poses:
            np.testing.assert_allclose(
                spectator.diagnostics()["ghost_transforms"][pose["key"]]["position"],
                pose["position"],
                atol=1e-5,
            )
        spectator.set_ghost_poses([])
        hidden = spectator.render()
        assert spectator.diagnostics()["visible_ghosts"] == []
        # Scenery's native animations may move with wall time; ghost visibility
        # is evidenced separately by native transforms and per-model pixel changes.
        assert np.mean(np.abs(hidden.astype(float) - empty)) < 2
        spectator.set_ghost_poses(poses)
        # Force failure after RTT activation and GL readback state changes.
        spectator._native.kartstudy_spectator_test_fail_next_render(spectator._handle)
        with pytest.raises(RuntimeError, match="Injected native render failure"):
            spectator.render()
        # Finally closes this failed renderer, then the reload loop proves teardown.
        diag = spectator.diagnostics()
        diag["forced_render_failure"] = "injected after target activation"
        diag["individual_visible_pixels"] = sorted(visible_pixels)
        diag["orientation_max_errors"] = orientation_errors
        diag["geometry_segment_count"] = len(segments)
        (proof / f"{track}-acceptance.json").write_text(json.dumps(diag, indent=2))
    finally:
        spectator.close()
        spectator.close()
    with pytest.raises(RuntimeError, match="closed"):
        spectator.render()
    # Reload in the same process must free old ghost/camera/world resources.
    for reload_track in (track, "lighthouse", "stk_enterprise", "mines"):
        with NativeSpectator(reload_track, configs, width=320, height=180) as reloaded:
            assert reloaded.render().shape == (180, 320, 3)
            assert reloaded.diagnostics()["physical_kart_count_after"] == 1
    evidence_path = proof / f"{track}-acceptance.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["forced_failure_close_reload_verified"] = True
    evidence["successful_world_loads"] = 5
    evidence_path.write_text(json.dumps(evidence, indent=2))


@pytest.mark.skipif(sys.platform != "darwin", reason="Pinned native Mac renderer")
@pytest.mark.parametrize("track", ["lighthouse", "stk_enterprise", "mines"])
def test_actual_capture_pose_replacement_moves_visible_meshes(track, tmp_path):
    proof = Path(os.environ.get("KARTSTUDY_NATIVE_PROOF_DIR", str(tmp_path))).resolve()
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), track, str(proof), "actual"],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _actual_pose_acceptance(track, proof):
    fixture_path = Path(__file__).with_name("fixtures") / "spectator_actual_poses.json"
    fixture = json.loads(fixture_path.read_text())[track]
    proof.mkdir(parents=True, exist_ok=True)
    evidence = []
    with NativeSpectator(track, fixture["ghosts"]) as spectator:
        spectator.set_ghost_poses(fixture["initial_poses"])
        spectator.render()
        # Apply successively separated poses to the same native model hierarchy.
        # This catches stale children and LOD tied to the hidden loading camera.
        for case in fixture["cases"]:
            poses = [
                {
                    k: value
                    for k, value in pose.items()
                    if k not in ("source_index", "source_sha256", "native_time")
                }
                for pose in case["poses"]
            ]
            selected = poses[1]
            position = np.asarray(selected["position"])
            forward = Rotation.from_quat(selected["rotation"]).apply([0, 0, 1])
            forward[1] = 0
            forward /= np.linalg.norm(forward)
            camera = {
                "position": (position - forward * 6 + [0, 2, 0]).tolist(),
                "target": (position + [0, 0.5, 0]).tolist(),
                "fov": math.radians(70),
            }
            spectator.set_ghost_poses(poses)
            spectator.set_camera(camera)
            first_seek_frame = spectator.render()
            spectator.set_ghost_poses([])
            empty = spectator.render()
            seconds = case["race_time_requested"]
            Image.fromarray(empty).save(proof / f"{track}-actual-{seconds}-empty.png")
            pixel_counts = {}
            for label, visible in (("one", [selected]), ("four", poses)):
                spectator.set_ghost_poses(visible)
                frame = spectator.render()
                difference = np.any(frame != empty, axis=-1)
                pixel_counts[label] = int(difference.sum())
                assert pixel_counts[label] > 500, (
                    "Actual measured ghost pose is invisible"
                )
                Image.fromarray(frame).save(
                    proof / f"{track}-actual-{seconds}-{label}.png"
                )
                if label == "four":
                    np.testing.assert_array_equal(frame, first_seek_frame)
                diagnostics = spectator.diagnostics()
                for pose in visible:
                    transform = diagnostics["ghost_transforms"][pose["key"]]
                    np.testing.assert_allclose(
                        transform["model_position"], pose["position"], atol=1e-4
                    )
                    bounds = np.asarray(transform["world_bounds"])
                    assert np.all(bounds[1] > bounds[0])
                assert diagnostics["physical_kart_count_after"] == 1
                assert diagnostics["race_time_before"] == diagnostics["race_time_after"]
            # Seeking back must restore child transforms and the rendered pixels.
            spectator.set_ghost_poses([selected])
            one = spectator.render()
            spectator.set_ghost_poses([])
            spectator.render()
            spectator.set_ghost_poses([selected])
            np.testing.assert_array_equal(spectator.render(), one)
            evidence.append(
                {
                    "race_time_requested": seconds,
                    "pixel_counts": pixel_counts,
                    "source_poses": case["poses"],
                    "diagnostics": spectator.diagnostics(),
                }
            )
    (proof / f"{track}-actual-acceptance.json").write_text(
        json.dumps(evidence, indent=2)
    )


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "actual":
        _actual_pose_acceptance(sys.argv[1], Path(sys.argv[2]))
    else:
        _native_acceptance(sys.argv[1], Path(sys.argv[2]))
