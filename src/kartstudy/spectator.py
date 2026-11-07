"""Capture, view, or export independent SuperTuxKart solo-lap ghost comparisons."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .spectator_camera import CAMERA_MODES, CameraSettings

METHODS = ("bc", "dagger", "sac", "rlpd")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="Record final-policy solo laps once")
    capture.add_argument("output", type=Path)
    capture.add_argument("--root", default="runs/study-polar-four")
    capture.add_argument("--track", default="lighthouse")
    capture.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    capture.add_argument("--eval-seed", type=int, default=20000)
    capture.add_argument("--train-seed", type=int, default=0)
    capture.add_argument(
        "--snapshots",
        nargs="+",
        help="Explicit checkpoint snapshots for future comparisons",
    )
    capture.add_argument("--max-steps", type=int, default=1500)
    for name in ("view", "export"):
        command = commands.add_parser(name)
        command.add_argument("comparison", type=Path)
        if name == "export":
            command.add_argument("output", type=Path)
            command.add_argument("--fps", type=float, default=30.0)
            command.add_argument(
                "--start",
                type=float,
                default=0.0,
                help="Explicit partial-preview start (seconds)",
            )
            command.add_argument(
                "--end", type=float, help="Explicit partial-preview end (seconds)"
            )
        command.add_argument(
            "--width", type=int, default=1920 if name == "export" else 960
        )
        command.add_argument(
            "--height", type=int, default=1080 if name == "export" else 540
        )
        command.add_argument("--camera", choices=CAMERA_MODES, default="follow")
        command.add_argument(
            "--follow", help="Ghost key to follow (default: DAgger, then first ghost)"
        )
        command.add_argument("--distance", type=float, default=12.0)
        command.add_argument("--camera-height", type=float, default=7.0)
        command.add_argument(
            "--orbit", type=float, default=0.0, help="Orbit angle in degrees"
        )
        command.add_argument(
            "--fov", type=float, default=60.0, help="Vertical field of view in degrees"
        )
        command.add_argument(
            "--segment",
            type=int,
            default=0,
            help="Fixed track segment for corner camera",
        )
        command.add_argument(
            "--visible",
            nargs="*",
            help="Visible ghost keys (default: all; no keys hides all)",
        )
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.command == "capture":
            if args.max_steps < 1:
                raise ValueError("--max-steps must be positive")
            from .spectator_replay import capture_comparison

            return capture_comparison(
                args.output,
                root=args.root,
                track=args.track,
                methods=tuple(args.methods),
                eval_seed=args.eval_seed,
                train_seed=args.train_seed,
                snapshots=(
                    args.snapshots[0]
                    if args.snapshots and len(args.snapshots) == 1
                    else args.snapshots
                ),
                max_steps=args.max_steps,
            )
        if args.width < 640 or args.height < 360 or args.width % 2 or args.height % 2:
            raise ValueError("Use even dimensions of at least 640 × 360")
        camera = CameraSettings(
            args.camera,
            args.follow or "dagger",
            args.distance,
            args.camera_height,
            args.orbit,
            args.fov,
            args.segment,
        )
        from .spectator_process import RemoteSpectator
        from .spectator_replay import ReplayComparison

        comparison = ReplayComparison(args.comparison)
        keys = [ghost["key"] for ghost in comparison.ghosts]
        if not keys:
            raise ValueError("Comparison contains no ghosts")
        if args.follow is None:
            preferred = next(
                (g["key"] for g in comparison.ghosts if g["method"] == "dagger"),
                keys[0],
            )
            camera = replace(camera, follow=preferred)
        visible = keys if args.visible is None else args.visible
        if any(key not in keys for key in visible):
            raise ValueError(
                f"Unknown visible ghost; available keys: {', '.join(keys)}"
            )
        if camera.mode in ("follow", "chase") and camera.follow not in keys:
            raise ValueError(f"Unknown follow ghost; available keys: {', '.join(keys)}")
        renderer = RemoteSpectator(
            args.comparison, width=args.width, height=args.height
        )
        try:
            if args.command == "view":
                from .spectator_viewer import show

                return show(renderer, comparison, camera, visible)
            from .spectator_export import export

            result = export(
                renderer,
                comparison,
                args.output,
                camera=camera,
                visible=visible,
                fps=args.fps,
                start=args.start,
                end=args.end,
            )
            print(json.dumps(result, indent=2), flush=True)
            return result
        finally:
            renderer.close()
    except (KeyError, TypeError) as exc:
        cli.error(f"Malformed or incompatible comparison/checkpoint: {exc}")
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    main()
