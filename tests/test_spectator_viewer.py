import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kartstudy.spectator_camera import CameraSettings
from kartstudy.spectator_process import RemoteSpectator
from kartstudy.spectator_viewer import AsyncFrames, SpectatorControls, annotated_frame

GEOMETRY = {"segments": [[[0, 0, 0], [0, 0, 100]]], "widths": [8]}
GHOSTS = [
    {"key": key, "label": key.upper(), "method": key, "kart": "tux", "color": i / 4}
    for i, key in enumerate(("bc", "dagger", "sac", "rlpd"))
]


def poses(seconds, visible):
    return [
        {
            "key": g["key"],
            "position": [i * 3, 0, min(seconds, 0.1) * 10],
            "rotation": [0, 0, 0, 1],
            "speed": 0 if seconds >= 0.1 else 10,
            "steer": 0,
            "visible": g["key"] in visible,
            "status": ("finished", "timeout", "stopped", "driving")[i]
            if seconds >= 0.1
            else "driving",
        }
        for i, g in enumerate(GHOSTS)
    ]


def fixture_worker(connection, path, width, height):
    """Transport fixture; deliberately never initializes the production simulator."""
    from kartstudy.spectator_camera import camera_pose

    while True:
        try:
            operation, args = connection.recv()
        except EOFError:
            break
        if path == "death" and operation == "frame":
            os._exit(3)
        if path == "timeout" and operation == "frame":
            time.sleep(10)
        if path == "delay":
            time.sleep(0.15)
        state = {"time": args["time"], "poses": poses(args["time"], args["visible"])}
        state["camera"] = camera_pose(
            CameraSettings(**args["camera"]), state["poses"], GEOMETRY
        )
        if operation == "load":
            state.update(
                duration=0.2,
                ghosts=GHOSTS,
                track="lighthouse",
                seed=20000,
                manifest={"fixture": True},
                geometry=GEOMETRY,
                diagnostics={"fixture_backend": True},
            )
        image = np.full((height, width, 3), round(args["time"] * 100), dtype=np.uint8)
        connection.send(("ok", state, image))
    connection.close()


def test_controls_seek_step_restart_speed_and_endpoint():
    controls = SpectatorControls(2, visible=["dagger"])
    controls.set_speed(2)
    controls.advance(0.5)
    assert controls.time == 1
    controls.toggle()
    controls.advance(1)
    assert controls.time == 1
    controls.step()
    assert controls.time == pytest.approx(1 + 1 / 30) and not controls.playing
    controls.seek(-1)
    assert controls.time == 0
    controls.seek(100)
    assert controls.time == 2
    controls.toggle()
    assert controls.time == 0 and controls.playing
    controls.advance(10)
    assert controls.time == 2 and not controls.playing
    controls.restart()
    assert controls.time == 0 and not controls.playing
    with pytest.raises(ValueError):
        controls.set_speed(0)
    with pytest.raises(ValueError):
        controls.seek(float("nan"))


def test_async_requests_never_block_and_coalesce_latest_seek_but_keep_reset():
    frames = AsyncFrames()
    gate = threading.Event()
    calls = []
    try:
        before = time.perf_counter()
        frames.submit("load", lambda: (gate.wait(2), {"loaded": True})[1])
        for i in range(20):
            frames.submit(
                "frame", lambda value=i: calls.append(value) or {"time": value}
            )
        assert time.perf_counter() - before < 0.1
        assert frames.poll() is None
        frames.submit("reset", lambda: calls.append("reset") or {"reset": True})
        frames.submit("frame", lambda: calls.append("lost") or {})
        gate.set()
        deadline = time.perf_counter() + 2
        results = []
        while len(results) < 2 and time.perf_counter() < deadline:
            result = frames.poll()
            if result is not None:
                results.append(result)
            time.sleep(0.005)
        assert results == [{"loaded": True}, {"reset": True}]
        assert calls == ["reset"]
    finally:
        gate.set()
        frames.close()


def test_remote_seek_endpoint_metadata_and_reset_preserve_presentation():
    remote = RemoteSpectator(
        "fixture", width=96, height=54, worker_target=fixture_worker
    )
    camera = CameraSettings(mode="corner")
    try:
        remote.load(camera, ["dagger", "sac"])
        state = remote.seek(0.2, camera, ["dagger", "sac"])
        assert state["time"] == 0.2 and state["frame"][0, 0, 0] == 20
        assert [p["status"] for p in state["poses"]] == [
            "finished",
            "timeout",
            "stopped",
            "driving",
        ]
        assert [p["visible"] for p in state["poses"]] == [False, True, True, False]
        pid = remote.process.pid
        reset = remote.reset()
        assert remote.process.pid != pid
        assert reset["time"] == 0.2 and reset["camera"] == state["camera"]
        assert reset["poses"] == state["poses"]
        # Seeking backward restores earlier telemetry rather than carrying terminal state.
        earlier = remote.seek(0, camera, ["dagger"])
        assert all(p["status"] == "driving" for p in earlier["poses"])
    finally:
        remote.close()
        remote.close()


@pytest.mark.parametrize(
    "path,message", [("death", "exited"), ("timeout", "timed out")]
)
def test_worker_death_and_timeout_allow_reset(path, message):
    remote = RemoteSpectator(
        path, width=96, height=54, worker_target=fixture_worker, frame_timeout=0.1
    )
    try:
        remote.load()
        child = remote.process
        before = time.perf_counter()
        with pytest.raises(RuntimeError, match=message):
            remote.seek(0.1, CameraSettings(), ["dagger"])
        assert time.perf_counter() - before < 2
        assert remote.process is None
        assert child._closed
        remote.path = "fixture"
        state = remote.reset()
        assert state["time"] == 0.1
        assert [p["key"] for p in state["poses"] if p["visible"]] == ["dagger"]
    finally:
        remote.close()


def test_annotation_keeps_terminal_and_hidden_labels(monkeypatch):
    from PIL import ImageDraw

    texts = []
    original = ImageDraw.ImageDraw.text

    def text(self, xy, value, *args, **kwargs):
        texts.append(value)
        return original(self, xy, value, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", text)
    image = annotated_frame(
        np.zeros((360, 640, 3), dtype=np.uint8),
        {
            "time": 0.2,
            "poses": poses(0.2, ["dagger"]),
            "training_seed": 0,
            "evaluation_seed": 20000,
        },
        GHOSTS,
        GEOMETRY,
        "lighthouse",
    )
    assert image.size == (640, 360)
    assert any("BC · FINISHED · hidden" in s for s in texts)
    assert any("DAGGER · TIMEOUT" in s for s in texts)
    assert any("SAC · STOPPED · hidden" in s for s in texts)
    assert any("training track" in s and "0.20 s" in s for s in texts)
    assert any("Training seed 0 · eval seed 20000" in s for s in texts)
    assert any("no shared item interactions" in s for s in texts)


def test_tk_delayed_loading_and_controls_in_fresh_process():
    probe = f"""
import runpy, sys, pytest
namespace = runpy.run_path({str(Path(__file__).resolve())!r})
try:
    namespace['_tk_probe']()
except pytest.skip.Exception:
    sys.exit(77)
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode == 77:
        pytest.skip("Tk display unavailable")
    assert result.returncode == 0, result.stdout + result.stderr


def _tk_probe():
    tk = pytest.importorskip("tkinter")
    from tkinter import ttk

    from kartstudy import spectator_viewer

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display unavailable")
    root.withdraw()
    heartbeats, calls, delays = [], [], []
    fixture = SimpleNamespace(
        duration=2,
        ghosts=GHOSTS,
        track="lighthouse",
        seed=20000,
        manifest={"train_seed": 0},
    )
    renderer = SimpleNamespace(width=640, height=360, metadata={"geometry": GEOMETRY})

    def response(seconds, camera, visible):
        started = time.perf_counter()
        time.sleep(0.12)
        delays.append((started, time.perf_counter()))
        calls.append((seconds, camera, list(visible)))
        return {
            "time": seconds,
            "poses": poses(seconds, visible),
            "frame": np.zeros((360, 640, 3), dtype=np.uint8),
        }

    renderer.load = lambda camera, visible, seconds: response(seconds, camera, visible)
    renderer.seek = response
    renderer.reset = lambda **kwargs: response(
        kwargs["time"], kwargs["camera"], kwargs["visible"]
    )
    renderer.close = lambda: None

    def widgets(parent):
        for widget in parent.winfo_children():
            yield widget
            yield from widgets(widget)

    def heartbeat():
        heartbeats.append(time.perf_counter())
        root.after(10, heartbeat)

    def interact():
        buttons = {
            w.cget("text"): w for w in widgets(root) if isinstance(w, ttk.Button)
        }
        buttons["Pause"].invoke()
        buttons["Step →"].invoke()
        for widget in widgets(root):
            if isinstance(widget, tk.Checkbutton) and widget.cget("text") == "BC":
                widget.invoke()
        buttons["Reset renderer"].invoke()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tk, "Tk", lambda: root)
        root.after(10, heartbeat)
        root.after(250, interact)
        root.after(1500, lambda: root.tk.call(root.protocol("WM_DELETE_WINDOW")))
        spectator_viewer.show(renderer, fixture, quit_after_frames=5)
    assert len(heartbeats) > 20
    assert calls
    assert all(
        sum(start < beat < end for beat in heartbeats) >= 3 for start, end in delays
    )
    assert any("bc" not in visible for _, _, visible in calls)


def test_queued_reset_survives_failed_request():
    frames = AsyncFrames()
    gate = threading.Event()

    def fail():
        gate.wait(2)
        raise RuntimeError("renderer died")

    try:
        frames.submit("frame", fail)
        frames.submit("reset", lambda: {"reset": True})
        gate.set()
        deadline = time.perf_counter() + 2
        while not frames.pending.done() and time.perf_counter() < deadline:
            time.sleep(0.005)
        with pytest.raises(RuntimeError, match="died"):
            frames.poll()
        while not frames.pending.done() and time.perf_counter() < deadline:
            time.sleep(0.005)
        assert frames.poll() == {"reset": True}
    finally:
        gate.set()
        frames.close()


def test_close_during_pending_load_is_bounded_and_cannot_respawn():
    remote = RemoteSpectator("delay", width=96, height=54, worker_target=fixture_worker)
    frames = AsyncFrames()
    try:
        frames.submit("load", remote.load)
        deadline = time.perf_counter() + 2
        while remote.process is None and time.perf_counter() < deadline:
            time.sleep(0.005)
        before = time.perf_counter()
        frames.close()
        remote.close()
        assert time.perf_counter() - before < 1.2
        assert remote.process is None and remote.connection is None
        with pytest.raises(RuntimeError, match="closed"):
            remote.reset()
        while not frames.pending.done() and time.perf_counter() < deadline:
            time.sleep(0.005)
        assert frames.pending.done()
    finally:
        frames.close()
        remote.close()


def test_production_worker_samples_and_renders_using_shared_interfaces(monkeypatch):
    import multiprocessing as mp
    from types import ModuleType

    from kartstudy.spectator_process import _render_worker

    history = []
    replay_module = ModuleType("kartstudy.spectator_replay")
    native_module = ModuleType("kartstudy.spectator_native")
    replay_module.ReplayComparison = lambda path: SimpleNamespace(
        track="lighthouse",
        ghosts=GHOSTS,
        seed=20000,
        duration=0.2,
        manifest={"train_seed": 0},
        sample=lambda seconds: poses(seconds, [g["key"] for g in GHOSTS]),
    )

    class NativeFixture:
        def __init__(self, track, ghosts, width, height, seed):
            history.append((track, ghosts, width, height, seed))
            self.width, self.height = width, height

        def geometry(self):
            return GEOMETRY

        def set_camera(self, camera):
            history.append(("camera", camera))

        def set_ghost_poses(self, ghosts):
            history.append(("poses", ghosts))

        def render(self):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def diagnostics(self):
            return {"fixture": True}

        def close(self):
            history.append("closed")

    native_module.NativeSpectator = NativeFixture
    monkeypatch.setitem(sys.modules, "kartstudy.spectator_replay", replay_module)
    monkeypatch.setitem(sys.modules, "kartstudy.spectator_native", native_module)
    parent, child = mp.Pipe()
    worker = threading.Thread(target=_render_worker, args=(child, "fixture", 96, 54))
    worker.start()
    try:
        args = {
            "time": 0.2,
            "camera": CameraSettings().to_dict(),
            "visible": ["dagger"],
        }
        parent.send(("load", args))
        assert parent.poll(2)
        kind, metadata, frame = parent.recv()
        assert kind == "ok" and metadata["time"] == 0.2 and frame.shape == (54, 96, 3)
        assert metadata["manifest"] == {"train_seed": 0}
        assert [p["visible"] for p in metadata["poses"]] == [False, True, False, False]
        assert metadata["poses"][1]["status"] == "timeout"
        parent.send(("frame", args | {"time": 0.3}))
        assert parent.poll(2)
        kind, error, frame = parent.recv()
        assert kind == "error" and "outside" in error and frame is None
        parent.send(("close", {}))
        worker.join(2)
        assert not worker.is_alive() and history[-1] == "closed"
        assert history[0] == ("lighthouse", GHOSTS, 96, 54, 20000)
    finally:
        parent.close()
        child.close()
        worker.join(2)
