"""Bounded, viewer-owned native renderer process, isolated from Tk/Cocoa."""

from __future__ import annotations

import multiprocessing as mp
import threading
import time

import numpy as np

from .spectator_camera import CameraSettings, camera_pose


def _render_worker(connection, path, width, height):
    native = None
    try:
        from .spectator_native import NativeSpectator
        from .spectator_replay import ReplayComparison

        replay = ReplayComparison(path)
        native = NativeSpectator(
            replay.track, replay.ghosts, width=width, height=height, seed=replay.seed
        )
        geometry = native.geometry()
        while True:
            operation, arguments = connection.recv()
            if operation == "close":
                break
            try:
                if operation not in ("load", "frame"):
                    raise ValueError(f"Unknown spectator operation: {operation}")
                time = float(arguments["time"])
                if not np.isfinite(time) or not 0 <= time <= replay.duration:
                    raise ValueError("Replay time is outside the comparison")
                settings = CameraSettings(**arguments["camera"])
                visible = set(arguments["visible"])
                poses = [
                    dict(p, visible=p["visible"] and p["key"] in visible)
                    for p in replay.sample(time)
                ]
                camera = camera_pose(settings, poses, geometry, width / height)
                native.set_ghost_poses(poses)
                native.set_camera(camera)
                frame = np.asarray(native.render())
                if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                    raise RuntimeError("Native spectator returned an invalid RGB frame")
                metadata = {
                    "time": time,
                    "poses": poses,
                    "camera": camera,
                    "training_seed": replay.manifest.get(
                        "training_seed",
                        replay.manifest.get("train_seed", "unspecified"),
                    ),
                    "evaluation_seed": replay.seed,
                }
                if operation == "load":
                    metadata.update(
                        duration=replay.duration,
                        ghosts=replay.ghosts,
                        track=replay.track,
                        seed=replay.seed,
                        manifest=replay.manifest,
                        geometry=geometry,
                        diagnostics=native.diagnostics(),
                    )
                connection.send(("ok", metadata, frame))
            except Exception as exc:  # noqa: BLE001 — transport explicit backend errors
                connection.send(("error", f"{type(exc).__name__}: {exc}", None))
    except EOFError:
        pass
    except Exception as exc:  # noqa: BLE001 — report load/import failures without a fallback
        try:
            connection.send(
                (
                    "error",
                    f"Spectator backend failed: {type(exc).__name__}: {exc}",
                    None,
                )
            )
        except (BrokenPipeError, OSError):
            pass
    finally:
        if native is not None:
            native.close()
        connection.close()


class RemoteSpectator:
    """Synchronous transport; GUI submits it to one background executor."""

    def __init__(
        self,
        path,
        width=960,
        height=540,
        *,
        worker_target=_render_worker,
        load_timeout=30.0,
        frame_timeout=5.0,
    ):
        self.path, self.width, self.height = str(path), width, height
        self._worker_target = worker_target
        self.load_timeout, self.frame_timeout = load_timeout, frame_timeout
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._closed = False
        self.process = self.connection = None
        self.last_request = None
        self.metadata = {}
        self._frame = None

    def _spawn(self):
        if self._closed:
            raise RuntimeError("Spectator renderer is closed")
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=self._worker_target,
            args=(child, self.path, self.width, self.height),
            daemon=True,
        )
        self.process.start()
        child.close()

    def _request(self, operation, arguments):
        self.last_request = arguments.copy()
        try:
            connection = self.connection
            if connection is None:
                raise RuntimeError("Spectator renderer is closed")
            connection.send((operation, arguments))
            timeout = self.load_timeout if operation == "load" else self.frame_timeout
            deadline = time.monotonic() + timeout
            while True:
                if self._closed:
                    raise RuntimeError("Spectator renderer is closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._stop_worker()
                    raise RuntimeError(
                        "Spectator renderer timed out; use Reset renderer to recover"
                    )
                if connection.poll(min(0.05, remaining)):
                    break
            kind, metadata, frame = connection.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as exc:
            self._stop_worker()
            raise RuntimeError(
                "Spectator renderer exited; use Reset renderer to recover"
            ) from exc
        if kind == "error":
            self._stop_worker()
            raise RuntimeError(metadata)
        self.metadata.update(metadata)
        self._frame = frame
        return metadata | {"frame": frame}

    def load(self, camera=None, visible=None, time=0.0):
        if self._closed:
            raise RuntimeError("Spectator renderer is closed")
        with self._lock:
            with self._lifecycle_lock:
                self._stop_worker()
                self._spawn()
            if visible is None:
                # Capture's default keys; CLI resolves the full list before view/export.
                visible = ("bc", "dagger", "sac", "rlpd")
            return self._request(
                "load",
                {
                    "time": time,
                    "camera": (camera or CameraSettings()).to_dict(),
                    "visible": list(visible),
                },
            )

    def seek(self, time, camera, visible):
        with self._lock:
            if self.connection is None:
                raise RuntimeError(
                    "Spectator renderer is closed; use Reset renderer to recover"
                )
            return self._request(
                "frame",
                {"time": time, "camera": camera.to_dict(), "visible": list(visible)},
            )

    def reset(self, *, time=None, camera=None, visible=None):
        request = self.last_request or {
            "time": 0.0,
            "camera": CameraSettings().to_dict(),
            "visible": ["bc", "dagger", "sac", "rlpd"],
        }
        return self.load(
            camera or CameraSettings(**request["camera"]),
            request["visible"] if visible is None else visible,
            request["time"] if time is None else time,
        )

    def frame(self):
        if self._frame is None:
            raise RuntimeError("Spectator has not rendered a frame")
        return self._frame

    def close(self):
        with self._lifecycle_lock:
            self._closed = True
            self._stop_worker()

    def _stop_worker(self):
        # Only this object's spawned child is touched; no global viewer/process search.
        with self._lifecycle_lock:
            process, connection = self.process, self.connection
            self.process = self.connection = None
            if connection is not None:
                connection.close()
            if process is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=0.5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=0.5)
                if not process.is_alive():
                    process.close()
