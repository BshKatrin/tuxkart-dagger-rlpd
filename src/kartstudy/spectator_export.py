"""Reproducible MP4 presentation of recorded, independent solo laps."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .spectator_camera import CameraSettings
from .spectator_viewer import (
    annotated_frame,
    presentation_metadata,
    track_split,
    training_seed,
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_hashes(path, exclude=()):
    path = Path(path)
    files = (
        sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    )
    excluded = {Path(p).resolve() for p in exclude}
    return {
        str(p.relative_to(path) if path.is_dir() else p.name): file_hash(p)
        for p in files
        if p.resolve() not in excluded
    }


def frame_times(duration, fps=30.0, start=0.0, end=None):
    end = duration if end is None else end
    if not all(math.isfinite(v) for v in (duration, fps, start, end)):
        raise ValueError("Export times and FPS must be finite")
    if duration < 0 or fps <= 0 or not 0 <= start <= end <= duration:
        raise ValueError(
            "Export requires positive FPS and 0 <= start <= end <= duration"
        )
    # Include the exact terminal pose; encoded duration padding is less than two frames.
    return [
        min(end, start + i / fps) for i in range(math.ceil((end - start) * fps) + 1)
    ]


def export(
    renderer,
    comparison,
    output,
    *,
    camera=None,
    visible=None,
    fps=30.0,
    start=0.0,
    end=None,
    writer_factory=None,
):
    import imageio_ffmpeg

    output = Path(output)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Spectator export output must end in .mp4")
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite MP4 or sidecar: {output}")
    times = frame_times(comparison.duration, fps, start, end)
    clip = {
        "preview_only": times[0] != 0 or times[-1] != comparison.duration,
        "start_seconds": times[0],
        "end_seconds": times[-1],
        "replay_duration_seconds": comparison.duration,
    }
    camera = camera or CameraSettings()
    visible = list(
        visible if visible is not None else [g["key"] for g in comparison.ghosts]
    )
    hashes = input_hashes(renderer.path, (output, sidecar))
    source = Path(__file__).parent
    sources = {p.name: file_hash(p) for p in sorted(source.glob("spectator*.py"))}
    output.parent.mkdir(parents=True, exist_ok=True)
    owned = []
    writer = None
    try:
        # Exclusive reservations prevent another export from winning between check and write.
        for path in (output, sidecar):
            with path.open("xb"):
                pass
            owned.append(path)
        loaded = renderer.load(camera, visible, times[0])
        for name, expected in (
            ("manifest", comparison.manifest),
            ("ghosts", comparison.ghosts),
            ("track", comparison.track),
            ("seed", comparison.seed),
            ("duration", comparison.duration),
        ):
            if name in loaded and loaded[name] != expected:
                raise RuntimeError(
                    "Comparison metadata changed before rendering; reopen the comparison"
                )
        writer = (writer_factory or imageio_ffmpeg.write_frames)(
            str(output),
            (renderer.width, renderer.height),
            fps=fps,
            codec="libx264",
            quality=5,
            macro_block_size=2,
            output_params=["-movflags", "+faststart"],
        )
        writer.send(None)
        terminal = None
        for index, seconds in enumerate(times):
            state = loaded if index == 0 else renderer.seek(seconds, camera, visible)
            writer.send(
                np.asarray(
                    annotated_frame(
                        state["frame"],
                        presentation_metadata(comparison, state) | clip,
                        comparison.ghosts,
                        renderer.metadata["geometry"],
                        comparison.track,
                    )
                )
            )
            terminal = state
        writer.close()
        writer = None
        # Detect a changed input rather than recording incorrect provenance during live edits.
        if input_hashes(renderer.path, (output, sidecar)) != hashes:
            raise RuntimeError("Replay input changed during export; export discarded")
        result = {
            "schema_version": 1,
            "presentation": "independent solo-lap replay",
            "track_effects": "omitted; no shared item interactions",
            "track": comparison.track,
            "track_split": track_split(comparison.track),
            "evaluation_seed": comparison.seed,
            "training_seed": training_seed(comparison.manifest),
            "comparison": str(Path(renderer.path).resolve()),
            "input_sha256": hashes,
            "source_sha256": sources,
            "manifest": comparison.manifest,
            "native_fingerprint": renderer.metadata["diagnostics"],
            "video": str(output.resolve()),
            "width": renderer.width,
            "height": renderer.height,
            "fps": fps,
            "frame_count": len(times),
            "video_duration_seconds": len(times) / fps,
            **clip,
            "sampling": "uniform presentation intervals plus exact terminal endpoint",
            "terminal_hold_seconds": len(times) / fps - (times[-1] - times[0]),
            "camera": camera.to_dict(),
            "visible_ghosts": visible,
            "labels": comparison.ghosts,
            "terminal_poses": terminal["poses"],
        }
        sidecar.write_text(json.dumps(result, indent=2) + "\n")
        return result
    except BaseException:
        try:
            if writer is not None:
                writer.close()
        finally:
            for path in owned:
                path.unlink(missing_ok=True)
        raise
    finally:
        renderer.close()
