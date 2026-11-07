import json
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from kartstudy import spectator
from kartstudy.spectator_camera import CameraSettings
from kartstudy.spectator_export import export, frame_times
from kartstudy.spectator_viewer import track_split

GHOSTS = [
    {"key": key, "label": key.upper(), "method": key, "kart": "tux", "color": i / 4}
    for i, key in enumerate(("bc", "dagger", "sac", "rlpd"))
]
GEOMETRY = {"segments": [[[0, 0, 0], [0, 0, 10]]], "widths": [8]}


class FixtureRenderer:
    def __init__(self, path, width=640, height=360):
        self.path, self.width, self.height = path, width, height
        self.metadata = {"geometry": GEOMETRY, "diagnostics": {"fixture": "not-native"}}
        self.calls = []
        self.closed = False

    def load(self, camera, visible, seconds=0):
        return self.seek(seconds, camera, visible)

    def seek(self, seconds, camera, visible):
        self.calls.append((seconds, camera, list(visible)))
        return {
            "time": seconds,
            "camera": camera.to_dict(),
            "frame": np.zeros((self.height, self.width, 3), dtype=np.uint8),
            "poses": [
                {
                    "key": g["key"],
                    "position": [0, 0, seconds],
                    "rotation": [0, 0, 0, 1],
                    "speed": 0 if seconds == 0.2 else 10,
                    "steer": 0,
                    "status": "finished" if seconds == 0.2 else "driving",
                    "visible": g["key"] in visible,
                }
                for g in GHOSTS
            ],
        }

    def close(self):
        self.closed = True


@pytest.fixture
def comparison(tmp_path):
    path = tmp_path / "comparison"
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"schema": "fixture"}))
    return SimpleNamespace(
        path=path,
        duration=0.2,
        track="lighthouse",
        seed=20000,
        ghosts=GHOSTS,
        manifest={"schema": "fixture", "train_seed": 0},
    )


def test_export_uniform_timing_exact_endpoint_camera_labels_and_provenance(
    comparison, tmp_path
):
    renderer = FixtureRenderer(comparison.path)
    written, options = [], {}

    def writer_factory(path, size, **kwargs):
        options.update(path=path, size=size, **kwargs)
        frame = yield
        while True:
            written.append(frame)
            frame = yield

    camera = CameraSettings(mode="corner", segment=0, fov=45)
    output = tmp_path / "movie.mp4"
    result = export(
        renderer,
        comparison,
        output,
        camera=camera,
        visible=["dagger"],
        writer_factory=writer_factory,
    )
    assert len(written) == 7
    assert [s for s, _, _ in renderer.calls] == pytest.approx(
        [i / 30 for i in range(7)]
    )
    assert all(
        c == camera and visible == ["dagger"] for _, c, visible in renderer.calls
    )
    assert options["fps"] == 30 and options["size"] == (640, 360)
    assert result["camera"] == camera.to_dict()
    assert (
        result["labels"] == GHOSTS
        and result["presentation"] == "independent solo-lap replay"
    )
    assert (
        result["track_split"] == "training track" and result["evaluation_seed"] == 20000
    )
    assert result["native_fingerprint"] == {"fixture": "not-native"}
    assert set(result["input_sha256"]) == {"manifest.json"}
    assert "spectator_camera.py" in result["source_sha256"]
    assert result["terminal_poses"][0]["status"] == "finished"
    assert result["end_seconds"] == 0.2 and not result["preview_only"]
    assert result["video_duration_seconds"] == pytest.approx(7 / 30)
    assert json.loads(output.with_suffix(".json").read_text()) == result
    assert renderer.closed
    with pytest.raises(FileExistsError, match="overwrite"):
        export(renderer, comparison, output)


def test_partial_preview_and_sidecar_overwrite_rejection(comparison, tmp_path):
    output = tmp_path / "preview.mp4"

    def writer(*args, **kwargs):
        while True:
            yield

    result = export(
        FixtureRenderer(comparison.path),
        comparison,
        output,
        start=0.1,
        end=0.15,
        fps=20,
        writer_factory=writer,
    )
    assert (
        result["preview_only"]
        and result["start_seconds"] == 0.1
        and result["end_seconds"] == 0.15
    )
    output.unlink()
    original = output.with_suffix(".json").read_bytes()
    with pytest.raises(FileExistsError):
        export(FixtureRenderer(comparison.path), comparison, output)
    assert output.with_suffix(".json").read_bytes() == original
    assert track_split("abyss") == "held-out track"


def test_export_failure_removes_only_owned_outputs(comparison, tmp_path):
    output = tmp_path / "broken.mp4"
    renderer = FixtureRenderer(comparison.path)

    def broken(*args, **kwargs):
        yield
        raise RuntimeError("encode failed")

    with pytest.raises(RuntimeError, match="encode failed"):
        export(renderer, comparison, output, writer_factory=broken)
    assert not output.exists() and not output.with_suffix(".json").exists()
    assert renderer.closed


def test_export_detects_input_edit_during_render(comparison, tmp_path):
    output = tmp_path / "changed.mp4"

    def writer(*args, **kwargs):
        yield
        (comparison.path / "manifest.json").write_text("changed")
        while True:
            yield

    with pytest.raises(RuntimeError, match="changed"):
        export(
            FixtureRenderer(comparison.path), comparison, output, writer_factory=writer
        )
    assert not output.exists() and not output.with_suffix(".json").exists()


@pytest.mark.parametrize(
    "args", [(1, 0), (1, 30, -1), (1, 30, 0, 2), (1, float("nan"))]
)
def test_export_rejects_invalid_schedule(args):
    with pytest.raises(ValueError):
        frame_times(*args)


def test_real_ffmpeg_fixture_video_dimensions_and_duration(comparison, tmp_path):
    import imageio_ffmpeg

    output = tmp_path / "fixture.mp4"
    result = export(FixtureRenderer(comparison.path), comparison, output)
    reader = imageio_ffmpeg.read_frames(str(output), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        frames = list(reader)
    finally:
        reader.close()
    assert metadata["size"] == (640, 360)
    assert metadata["fps"] == 30 and len(frames) == 7
    assert len(frames[-1]) == 640 * 360 * 3
    assert result["frame_count"] == 7 and output.stat().st_size > 1000


def install_replay_fixture(monkeypatch, comparison):
    replay = ModuleType("kartstudy.spectator_replay")
    replay.ReplayComparison = lambda path: comparison
    replay.capture_comparison = lambda output, **kwargs: {"output": output, **kwargs}
    monkeypatch.setitem(__import__("sys").modules, "kartstudy.spectator_replay", replay)


def test_cli_capture_dispatch_and_defaults(monkeypatch, comparison, tmp_path):
    install_replay_fixture(monkeypatch, comparison)
    result = spectator.main(["capture", str(tmp_path / "replay")])
    assert result["methods"] == ("bc", "dagger", "sac", "rlpd")
    assert result["eval_seed"] == 20000 and result["train_seed"] == 0
    assert result["root"] == "runs/study-polar-four" and result["track"] == "lighthouse"
    result = spectator.main(
        ["capture", "replay", "--methods", "dagger", "--snapshots", "checkpoint.pt"]
    )
    assert result["snapshots"] == "checkpoint.pt" and result["methods"] == ("dagger",)


def test_cli_view_and_export_dispatch_config(monkeypatch, comparison, tmp_path):
    from kartstudy import spectator_export, spectator_process, spectator_viewer

    install_replay_fixture(monkeypatch, comparison)
    renderers, calls = [], []

    def renderer(path, **kwargs):
        item = FixtureRenderer(path, **kwargs)
        renderers.append(item)
        return item

    monkeypatch.setattr(spectator_process, "RemoteSpectator", renderer)
    monkeypatch.setattr(
        spectator_viewer, "show", lambda r, p, c, v: calls.append((r, p, c, v))
    )
    spectator.main(
        ["view", str(comparison.path), "--visible", "dagger", "--orbit", "30"]
    )
    assert calls[0][2].follow == "dagger" and calls[0][2].orbit == 30
    assert (
        calls[0][3] == ["dagger"]
        and renderers[-1].width == 960
        and renderers[-1].height == 540
    )
    monkeypatch.setattr(
        spectator_export, "export", lambda r, p, out, **kwargs: {"fps": kwargs["fps"]}
    )
    spectator.main(["export", str(comparison.path), str(tmp_path / "movie.mp4")])
    assert renderers[-1].width == 1920 and renderers[-1].height == 1080
    assert all(r.closed for r in renderers)


def test_moving_camera_cli_and_saved_path_provenance(monkeypatch, comparison, tmp_path):
    from kartstudy import spectator_process, spectator_viewer

    install_replay_fixture(monkeypatch, comparison)
    calls = []
    monkeypatch.setattr(spectator_process, "RemoteSpectator", FixtureRenderer)
    monkeypatch.setattr(spectator_viewer, "show", lambda r, p, c, v: calls.append(c))
    for mode in ("rail", "pack", "side", "cinematic"):
        spectator.main(
            [
                "view",
                str(comparison.path),
                "--camera",
                mode,
                "--pack-size",
                "3",
                "--lookahead",
                "8",
                "--smoothing",
                "0.8",
            ]
        )
        assert calls[-1].mode == mode
        assert (calls[-1].pack_size, calls[-1].lookahead, calls[-1].smoothing) == (
            3,
            8,
            0.8,
        )
    keyframes = [
        {"time": 0, "position": [0, 7, -10], "target": [0, 0, 0], "fov": 60},
        {"time": 1, "position": [10, 7, 0], "target": [0, 0, 0], "fov": 50},
    ]
    script = tmp_path / "camera.json"
    script.write_text(json.dumps({"keyframes": keyframes}))
    spectator.main(
        [
            "view",
            str(comparison.path),
            "--camera",
            "cinematic",
            "--camera-script",
            str(script),
        ]
    )
    assert calls[-1].to_dict()["keyframes"] == keyframes

    def writer(*args, **kwargs):
        while True:
            yield

    result = export(
        FixtureRenderer(comparison.path),
        comparison,
        tmp_path / "path.mp4",
        camera=calls[-1],
        writer_factory=writer,
    )
    assert result["camera_keyframes_sha256"]
    assert result["camera"]["keyframes"] == keyframes


@pytest.mark.parametrize("data", [{}, {"keyframes": []}, {"keyframes": "bad"}])
def test_invalid_camera_script_rejected_before_native_load(tmp_path, data):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="keyframes"):
        spectator.load_camera_script(path)


def test_saved_camera_path_roundtrip_and_no_overwrite(tmp_path):
    path = tmp_path / "camera.json"
    frames = ({"time": 1, "position": [0, 7, -10], "target": [0, 0, 0], "fov": 60},)
    spectator.save_camera_script(path, frames)
    original = path.read_bytes()
    assert spectator.load_camera_script(path) == frames
    with pytest.raises(FileExistsError):
        spectator.save_camera_script(path, frames)
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        spectator.save_camera_script(tmp_path / "empty.json", ())
    assert not (tmp_path / "empty.json").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--follow", "absent"],
        ["--visible", "absent"],
        ["--width", "641"],
        ["--distance", "0"],
    ],
)
def test_cli_errors_are_clear(monkeypatch, comparison, extra, capsys):
    install_replay_fixture(monkeypatch, comparison)
    with pytest.raises(SystemExit) as error:
        spectator.main(["view", str(comparison.path), *extra])
    assert error.value.code == 2 and "error:" in capsys.readouterr().err


def test_export_can_live_inside_comparison_directory(comparison):
    def writer(*args, **kwargs):
        while True:
            yield

    output = comparison.path / "exports" / "movie.mp4"
    result = export(
        FixtureRenderer(comparison.path), comparison, output, writer_factory=writer
    )
    assert output.exists() and output.with_suffix(".json").exists()
    assert set(result["input_sha256"]) == {"manifest.json"}


@pytest.mark.parametrize("method", ["dagger", "bc"])
def test_default_follow_resolves_snapshot_key_or_first_ghost(
    monkeypatch, comparison, method
):
    from kartstudy import spectator_process, spectator_viewer

    comparison.ghosts = [
        GHOSTS[1 if method == "dagger" else 0] | {"key": f"{method}-s0-step100"}
    ]
    install_replay_fixture(monkeypatch, comparison)
    monkeypatch.setattr(spectator_process, "RemoteSpectator", FixtureRenderer)
    cameras = []
    monkeypatch.setattr(spectator_viewer, "show", lambda r, p, c, v: cameras.append(c))
    spectator.main(["view", str(comparison.path)])
    assert cameras[0].follow == comparison.ghosts[0]["key"]
    with pytest.raises(SystemExit):
        spectator.main(["view", str(comparison.path), "--follow", "dagger"])


def test_cli_malformed_replay_has_no_traceback(monkeypatch, comparison, capsys):
    install_replay_fixture(monkeypatch, comparison)

    def malformed(path):
        raise KeyError("ghosts")

    monkeypatch.setattr(
        __import__("sys").modules["kartstudy.spectator_replay"],
        "ReplayComparison",
        malformed,
    )
    with pytest.raises(SystemExit) as error:
        spectator.main(["view", str(comparison.path)])
    assert error.value.code == 2
    assert "Malformed or incompatible comparison" in capsys.readouterr().err


def test_export_rejects_stale_loaded_manifest(comparison, tmp_path):
    renderer = FixtureRenderer(comparison.path)
    original = renderer.load
    renderer.load = lambda *args: original(*args) | {"manifest": {"changed": True}}
    output = tmp_path / "stale.mp4"
    with pytest.raises(RuntimeError, match="changed before rendering"):
        export(renderer, comparison, output)
    assert not output.exists() and not output.with_suffix(".json").exists()


@pytest.mark.parametrize(
    "start,end,preview", [(0.1, 0.15, True), (0, 0.15, True), (0, 0.2, False)]
)
def test_export_frames_visibly_label_partial_range_and_retain_seeds(
    comparison, tmp_path, monkeypatch, start, end, preview
):
    from PIL import ImageDraw

    drawn = []
    original = ImageDraw.ImageDraw.text

    def text(self, xy, value, *args, **kwargs):
        drawn.append((value, self.textbbox(xy, value, font=kwargs["font"])))
        return original(self, xy, value, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", text)

    def writer(*args, **kwargs):
        while True:
            yield

    result = export(
        FixtureRenderer(comparison.path),
        comparison,
        tmp_path / "clip.mp4",
        start=start,
        end=end,
        writer_factory=writer,
    )
    labels = [bounds for value, bounds in drawn if value.startswith("PARTIAL PREVIEW")]
    assert len(labels) == (result["frame_count"] if preview else 0)
    if preview:
        expected = f"PARTIAL PREVIEW · clip {start:.2f} to {end:.2f} s of 0.20 s"
        assert sum(value == expected for value, _ in drawn) == result["frame_count"]
    seed_labels = [
        bounds
        for value, bounds in drawn
        if value.startswith("Training seed 0 · eval seed 20000")
    ]
    assert len(seed_labels) == result["frame_count"]
    assert all(
        0 <= left < right <= 640 and 0 <= top < bottom <= 360
        for left, top, right, bottom in labels + seed_labels
    )
    assert result["preview_only"] is preview
    assert result["start_seconds"] == start and result["end_seconds"] == end


@pytest.mark.parametrize(
    "selection", ["snapshot_root", "run_root", "single", "multiple"]
)
def test_cli_snapshot_directories_reach_real_selector(tmp_path, monkeypatch, selection):
    import torch

    from kartstudy import spectator_replay as replay
    from kartstudy.snapshots import export_actor_snapshot

    run = tmp_path / "run"
    actor = SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([1.0, 2.0])})
    config = {"algo": "dagger", "seed": 0}
    snapshots = [
        export_actor_snapshot(run, actor, config, steps, updates)
        for steps, updates in ((100, 10), (200, 20))
    ]
    if selection == "snapshot_root":
        arguments, expected = [run / "snapshots"], snapshots
    elif selection == "run_root":
        arguments, expected = [run], snapshots
    elif selection == "single":
        arguments, expected = snapshots[:1], snapshots[:1]
    else:
        arguments, expected = snapshots, snapshots

    def select_only(output, **kwargs):
        # Exercise the real selection boundary without starting a simulator or actor.
        return replay._selections(
            kwargs["root"],
            kwargs["track"],
            kwargs["methods"],
            kwargs["train_seed"],
            kwargs["snapshots"],
        )

    monkeypatch.setattr(replay, "capture_comparison", select_only)
    selected = spectator.main(
        [
            "capture",
            str(tmp_path / "comparison"),
            "--root",
            str(run),
            "--methods",
            "dagger",
            "--snapshots",
            *(str(path) for path in arguments),
        ]
    )
    assert selected == [("dagger", path, True) for path in expected]
