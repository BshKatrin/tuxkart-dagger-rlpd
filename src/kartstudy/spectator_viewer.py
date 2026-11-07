"""Presentation controls, annotations and a nonblocking Tk spectator viewer."""

from __future__ import annotations

import colorsys
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .spectator_camera import CAMERA_MODES, CameraSettings
from .tracks import HELD_OUT_TRACKS, TRAIN_TRACKS


@dataclass
class SpectatorControls:
    duration: float
    camera: CameraSettings = field(default_factory=CameraSettings)
    visible: list[str] = field(default_factory=list)
    time: float = 0.0
    playing: bool = True
    speed: float = 1.0

    def seek(self, seconds):
        if not math.isfinite(seconds):
            raise ValueError("Replay time must be finite")
        self.time = min(self.duration, max(0.0, seconds))

    def advance(self, elapsed):
        if self.playing:
            self.seek(self.time + max(0.0, elapsed) * self.speed)
            if self.time == self.duration:
                self.playing = False

    def toggle(self):
        if self.time == self.duration:
            self.time = 0.0
            self.playing = True
        else:
            self.playing = not self.playing

    def step(self, seconds=1 / 30):
        self.playing = False
        self.seek(self.time + seconds)

    def restart(self):
        self.time = 0.0

    def set_speed(self, speed):
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Playback speed must be positive and finite")
        self.speed = speed


class AsyncFrames:
    """One transport request at a time; coalesce seeks, retain requested resets."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="spectator"
        )
        self.pending = self.queued = None
        self.closed = False

    def submit(self, kind, operation):
        if self.closed:
            return
        if self.pending is not None:
            if self.queued is None or self.queued[0] != "reset" or kind == "reset":
                self.queued = (kind, operation)
        else:
            self.pending = self.executor.submit(operation)

    def poll(self):
        if self.pending is None or not self.pending.done():
            return None
        future, self.pending = self.pending, None
        try:
            result = future.result()
        except Exception:
            queued, self.queued = self.queued, None
            if queued is not None and queued[0] == "reset":
                self.submit(*queued)
            raise
        if self.queued is not None:
            kind, operation = self.queued
            self.queued = None
            self.submit(kind, operation)
        return result

    def close(self):
        self.closed = True
        self.queued = None
        self.executor.shutdown(wait=False, cancel_futures=True)


def track_split(track):
    if track in TRAIN_TRACKS:
        return "training track"
    if track in HELD_OUT_TRACKS:
        return "held-out track"
    return "track outside study split"


def training_seed(manifest):
    for name in ("train_seed", "training_seed"):
        if name in manifest:
            return manifest[name]
    capture = manifest.get("capture", {})
    if isinstance(capture, dict):
        for name in ("train_seed", "training_seed"):
            if name in capture:
                return capture[name]
    return "unspecified"


def presentation_metadata(comparison, metadata):
    return metadata | {
        "evaluation_seed": comparison.seed,
        "training_seed": training_seed(comparison.manifest),
    }


def ghost_color(ghost):
    return tuple(
        round(255 * c) for c in colorsys.hsv_to_rgb(float(ghost["color"]) % 1, 0.75, 1)
    )


def annotated_frame(frame, metadata, ghosts, geometry, track):
    """Labels and minimap stay present for hidden and frozen terminal ghosts."""
    image = Image.fromarray(np.asarray(frame)).convert("RGB")
    draw = ImageDraw.Draw(image)
    scale = max(1, image.width / 960)
    font = ImageFont.load_default(size=round(14 * scale))
    line = round(20 * scale)
    camera_labels = {
        "rail": "TRACK-RAIL DRONE",
        "pack": "FRONT PACK",
        "side": "SIDE TRACKING",
        "cinematic": "CINEMATIC CAMERA PATH",
    }
    if metadata.get("camera_mode") in camera_labels:
        label = camera_labels[metadata["camera_mode"]]
        if metadata["camera_mode"] == "pack":
            label += f" ({metadata.get('pack_size', 2)})"
        draw.rectangle((0, 0, image.width, line + 10), fill="#101820")
        draw.text((10, 5), label, font=font, fill="white")
    preview = bool(metadata.get("preview_only", False))
    strip = (len(ghosts) + 2 + int(preview)) * line + 12
    draw.rectangle((0, image.height - strip, image.width, image.height), fill="#101820")
    y = image.height - strip + 5
    draw.text(
        (10, y),
        f"Independent solo-lap replay · {track} · {track_split(track)} · race {metadata['time']:.2f} s",
        font=font,
        fill="white",
    )
    if preview:
        y += line
        draw.text(
            (10, y),
            f"PARTIAL PREVIEW · clip {metadata['start_seconds']:.2f} to {metadata['end_seconds']:.2f} s of {metadata['replay_duration_seconds']:.2f} s",
            font=font,
            fill="#ffc766",
        )
    states = {pose["key"]: pose for pose in metadata["poses"]}
    for ghost in ghosts:
        y += line
        pose = states[ghost["key"]]
        hidden = " · hidden" if not pose["visible"] else ""
        draw.text(
            (10, y),
            f"{ghost['label']} · {pose['status'].upper()}{hidden} · {pose['speed']:.1f} m/s",
            font=font,
            fill=ghost_color(ghost),
        )
    y += line
    draw.text(
        (10, y),
        f"Training seed {metadata.get('training_seed', 'unspecified')} · eval seed {metadata.get('evaluation_seed', 'unspecified')} · no shared item interactions",
        font=font,
        fill="#bec9d2",
    )
    segments = np.asarray(geometry["segments"], dtype=float).reshape(-1, 2, 3)
    if segments.size:
        size = min(round(150 * scale), image.height - strip - 20)
        if size > 20:
            left, top = image.width - size - 10, 10
            draw.rectangle((left, top, left + size, top + size), fill="#101820")
            points = segments[:, :, [0, 2]]
            low, high = points.reshape(-1, 2).min(0), points.reshape(-1, 2).max(0)
            span = np.maximum(high - low, 1)
            factor = (size - 16) / max(span)

            def map_point(position):
                x, z = (np.asarray(position)[[0, 2]] - (low + high) / 2) * factor
                return left + size / 2 + x, top + size / 2 - z

            for segment in segments:
                draw.line([map_point(p) for p in segment], fill="#6c7780", width=2)
            for ghost in ghosts:
                pose = states[ghost["key"]]
                if pose["visible"]:
                    x, y = map_point(pose["position"])
                    radius = 3 * scale
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius),
                        fill=ghost_color(ghost),
                    )
    return image


def show(renderer, comparison, camera=None, visible=None, *, quit_after_frames=None):
    import tkinter as tk
    from tkinter import ttk

    from PIL import ImageTk

    controls = SpectatorControls(
        comparison.duration,
        camera or CameraSettings(),
        list(visible if visible is not None else [g["key"] for g in comparison.ghosts]),
    )
    root = tk.Tk()
    root.title("SuperTuxKart — independent solo-lap ghost replay")
    picture = ttk.Label(root)
    picture.pack()
    toolbar = ttk.Frame(root, padding=8)
    toolbar.pack(fill="x")
    status = tk.StringVar(value="Loading native spectator…")
    ttk.Label(root, textvariable=status, padding=8).pack(anchor="w")
    frames = AsyncFrames()
    photo = None
    closed, ready, failed = False, False, False
    count = 0
    last_tick = last_request = time.perf_counter()
    scrub_update = False
    last_requested_time = None
    last_displayed = None
    marked_keyframes = []

    def request(reset=False):
        nonlocal last_requested_time
        last_requested_time = controls.time
        snapshot = (controls.time, controls.camera, list(controls.visible))
        operation = renderer.reset if reset else renderer.seek
        frames.submit(
            "reset" if reset else "frame",
            lambda: (
                operation(*snapshot)
                if not reset
                else operation(
                    time=snapshot[0], camera=snapshot[1], visible=snapshot[2]
                )
            ),
        )

    def editing(event):
        return event is not None and event.widget.winfo_class() in (
            "TCombobox",
            "TSpinbox",
            "TEntry",
            "Entry",
            "Spinbox",
        )

    def toggle(event=None):
        if editing(event):
            return
        controls.toggle()
        play.configure(text="Pause" if controls.playing else "Play")

    def step(event=None):
        if editing(event):
            return
        controls.step()
        play.configure(text="Play")
        if ready:
            request()

    def restart(event=None):
        if editing(event):
            return
        controls.restart()
        if ready:
            request()

    play = ttk.Button(toolbar, text="Pause", command=toggle)
    play.pack(side="left")
    ttk.Button(toolbar, text="Step →", command=step).pack(side="left")
    ttk.Button(toolbar, text="Restart", command=restart).pack(side="left")

    def reset():
        nonlocal failed
        failed = False
        status.set("Resetting native renderer…")
        request(reset=True)

    ttk.Button(toolbar, text="Reset renderer", command=reset).pack(side="left", padx=8)
    speed = ttk.Combobox(
        toolbar, state="readonly", values=["0.25", "0.5", "1", "2", "4"], width=5
    )
    speed.set("1")
    speed.pack(side="left")
    speed.bind(
        "<<ComboboxSelected>>", lambda event: controls.set_speed(float(speed.get()))
    )

    def scrub(value):
        if scrub_update:
            return
        controls.playing = False
        play.configure(text="Play")
        controls.seek(float(value))
        if ready:
            request()

    timeline = ttk.Scale(root, from_=0, to=comparison.duration, command=scrub)
    timeline.pack(fill="x", padx=8)
    cameras = ttk.Frame(root, padding=8)
    cameras.pack(fill="x")

    def camera_change(name, value):
        try:
            controls.camera = replace(controls.camera, **{name: value})
            if ready:
                request()
        except ValueError as exc:
            status.set(str(exc))

    mode = ttk.Combobox(cameras, state="readonly", values=CAMERA_MODES, width=10)
    mode.set(controls.camera.mode)
    mode.pack(side="left")
    mode.bind("<<ComboboxSelected>>", lambda event: camera_change("mode", mode.get()))
    labels = {g["label"]: g["key"] for g in comparison.ghosts}
    follow = ttk.Combobox(cameras, state="readonly", values=list(labels), width=17)
    follow.set(
        next(
            (label for label, key in labels.items() if key == controls.camera.follow),
            "",
        )
    )
    follow.pack(side="left", padx=5)
    follow.bind(
        "<<ComboboxSelected>>",
        lambda event: camera_change("follow", labels[follow.get()]),
    )
    for name, title, start, end, increment in [
        ("distance", "Distance", 1, 500, 1),
        ("height", "Height", 0, 200, 1),
        ("orbit", "Orbit", -180, 180, 5),
        ("fov", "FOV", 10, 120, 5),
        ("segment", "Corner segment", 0, 100000, 1),
    ]:
        ttk.Label(cameras, text=title).pack(side="left", padx=(4, 0))
        variable = tk.StringVar(value=str(getattr(controls.camera, name)))
        box = ttk.Spinbox(
            cameras,
            from_=start,
            to=end,
            increment=increment,
            textvariable=variable,
            width=5,
        )
        box.pack(side="left")

        def change(event=None, field=name, var=variable):
            try:
                value = int(var.get()) if field == "segment" else float(var.get())
                camera_change(field, value)
            except ValueError as exc:
                status.set(str(exc))

        box.configure(command=change)
        box.bind("<Return>", change)
        box.bind("<FocusOut>", change)
        if name == "segment":
            segment_box = box
    motion = ttk.Frame(root, padding=(8, 0, 8, 4))
    motion.pack(fill="x")
    for name, title, start, end, increment in [
        ("pack_size", "Front pack", 1, min(32, len(comparison.ghosts)), 1),
        ("lookahead", "Look ahead (m)", 0, 100, 1),
        ("smoothing", "Smoothing (s)", 0, 5, 0.1),
    ]:
        ttk.Label(motion, text=title).pack(side="left", padx=(4, 0))
        variable = tk.StringVar(value=str(getattr(controls.camera, name)))
        box = ttk.Spinbox(
            motion,
            from_=start,
            to=end,
            increment=increment,
            textvariable=variable,
            width=5,
        )
        box.pack(side="left")

        def motion_change(event=None, field=name, var=variable):
            try:
                value = int(var.get()) if field == "pack_size" else float(var.get())
                camera_change(field, value)
            except ValueError as exc:
                status.set(str(exc))

        box.configure(command=motion_change)
        box.bind("<Return>", motion_change)
        box.bind("<FocusOut>", motion_change)

    def load_path():
        from tkinter import filedialog

        from .spectator import load_camera_script

        path = filedialog.askopenfilename(
            parent=root, title="Load camera path", filetypes=[("Camera path", "*.json")]
        )
        if path:
            try:
                controls.camera = replace(
                    controls.camera,
                    mode="cinematic",
                    keyframes=load_camera_script(path),
                )
                mode.set("cinematic")
                if ready:
                    request()
            except (ValueError, TypeError, KeyError, OSError) as exc:
                status.set(str(exc))

    def reset_path():
        controls.camera = replace(controls.camera, mode="cinematic", keyframes=())
        mode.set("cinematic")
        if ready:
            request()

    paths = ttk.Frame(root, padding=(8, 0, 8, 4))
    paths.pack(fill="x")
    ttk.Button(paths, text="Load camera path", command=load_path).pack(
        side="left", padx=8
    )
    ttk.Button(paths, text="Use built-in path", command=reset_path).pack(side="left")

    def mark_shot():
        if last_displayed is None:
            return
        controls.playing = False
        controls.seek(last_displayed["time"])
        native_camera = last_displayed["camera"]
        shot = {
            "time": last_displayed["time"],
            "position": native_camera["position"],
            "target": native_camera["target"],
            "fov": math.degrees(native_camera["fov"]),
        }
        marked_keyframes[:] = sorted(
            [s for s in marked_keyframes if s["time"] != shot["time"]] + [shot],
            key=lambda s: s["time"],
        )
        status.set(
            f"Marked camera at race {shot['time']:.2f} s; {len(marked_keyframes)} keyframes"
        )

    def save_path():
        from tkinter import filedialog

        from .spectator import save_camera_script

        if not marked_keyframes:
            status.set("Mark at least one displayed camera pose first")
            return
        path = filedialog.asksaveasfilename(
            parent=root,
            title="Save new camera path",
            defaultextension=".json",
            filetypes=[("Camera path", "*.json")],
        )
        if path:
            try:
                save_camera_script(path, marked_keyframes)
                status.set(f"Saved {len(marked_keyframes)} camera keyframes")
            except (ValueError, TypeError, KeyError, OSError) as exc:
                status.set(str(exc))

    ttk.Button(paths, text="Mark displayed shot", command=mark_shot).pack(
        side="left", padx=8
    )
    ttk.Button(paths, text="Save camera path", command=save_path).pack(side="left")
    ghosts_bar = ttk.Frame(root, padding=8)
    ghosts_bar.pack(fill="x")
    for ghost in comparison.ghosts:
        variable = tk.BooleanVar(value=ghost["key"] in controls.visible)

        def visibility(key=ghost["key"], var=variable):
            if var.get() and key not in controls.visible:
                controls.visible.append(key)
            elif not var.get() and key in controls.visible:
                controls.visible.remove(key)
            if ready:
                request()

        tk.Checkbutton(
            ghosts_bar,
            text=ghost["label"],
            variable=variable,
            command=visibility,
            fg="#{:02x}{:02x}{:02x}".format(*ghost_color(ghost)),
        ).pack(side="left")
    ttk.Label(ghosts_bar, text="Space: pause · →: step · R: restart · Esc: close").pack(
        side="right"
    )

    def close(event=None):
        nonlocal closed
        closed = True
        root.destroy()

    def tick():
        nonlocal photo, ready, failed, count, last_tick, last_request, scrub_update
        nonlocal last_displayed
        if closed:
            return
        now = time.perf_counter()
        if ready and not failed:
            controls.advance(now - last_tick)
        last_tick = now
        try:
            metadata = frames.poll()
            if metadata is not None:
                last_displayed = metadata
                ready = True
                failed = False
                segment_box.configure(
                    to=max(0, len(renderer.metadata["geometry"]["segments"]) - 1)
                )
                image = annotated_frame(
                    metadata["frame"],
                    presentation_metadata(comparison, metadata),
                    comparison.ghosts,
                    renderer.metadata["geometry"],
                    comparison.track,
                )
                photo = ImageTk.PhotoImage(image, master=root)
                picture.configure(image=photo)
                status.set(
                    f"Race {metadata['time']:.2f} / {controls.duration:.2f} s · {track_split(comparison.track)}"
                )
                scrub_update = True
                timeline.set(metadata["time"])
                scrub_update = False
                count += 1
            if (
                ready
                and not failed
                and controls.time != last_requested_time
                and now - last_request >= 1 / 30
            ):
                request()
                last_request = now
            play.configure(text="Pause" if controls.playing else "Play")
        except (RuntimeError, ValueError, OSError) as exc:
            failed = True
            controls.playing = False
            status.set(str(exc))
            play.configure(text="Play")
        if quit_after_frames is not None and count >= quit_after_frames:
            close()
        else:
            root.after(10, tick)

    root.protocol("WM_DELETE_WINDOW", close)
    root.bind("<space>", toggle)
    root.bind("<Right>", step)
    root.bind("r", restart)
    root.bind("<Escape>", close)
    frames.submit(
        "load", lambda: renderer.load(controls.camera, controls.visible, controls.time)
    )
    root.after(10, tick)
    try:
        root.mainloop()
    finally:
        frames.close()
        renderer.close()
        if not closed:
            root.destroy()
