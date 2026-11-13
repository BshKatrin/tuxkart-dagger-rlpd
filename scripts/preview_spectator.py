"""Render static ghosts and an external camera in the original STK scenery."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from kartstudy.spectator_native import NativeSpectator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New PNG path; existing files are refused")
    parser.add_argument("--track", default="lighthouse")
    parser.add_argument("--camera", choices=("follow", "corner", "overview"), default="follow")
    parser.add_argument("--ghosts", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    metadata = args.output.with_suffix(".json")
    if args.output.suffix.lower() != ".png":
        parser.error("Output must be a PNG")
    if args.output.exists() or metadata.exists():
        parser.error("Output or metadata already exists")

    configs = [
        {"key": f"g{i}", "label": f"Ghost {i + 1}", "method": "fixture",
         "kart": kart, "color": i / 4}
        for i, kart in enumerate(("tux", "sara_the_racer", "gnu", "wilber")[:args.ghosts])
    ]
    with NativeSpectator(args.track, configs, width=960, height=540) as spectator:
        segment = np.asarray(spectator.geometry()["segments"])[0]
        origin = segment[0]
        forward = segment[1] - origin
        forward[1] = 0
        forward /= np.linalg.norm(forward)
        side = np.cross([0, 1, 0], forward)
        yaw = math.atan2(forward[0], forward[2])
        poses = [
            {"key": config["key"],
             "position": (origin + side * (i - (args.ghosts - 1) / 2) * 1.3
                          + [0, 0.25, 0]).tolist(),
             "rotation": [0, math.sin(yaw / 2), 0, math.cos(yaw / 2)],
             "speed": 0.0, "steer": 0.0, "visible": True, "status": "stopped"}
            for i, config in enumerate(configs)
        ]
        offsets = {
            "follow": -forward * 10 + np.array([0, 5, 0]),
            "corner": side * 10 - forward * 3 + np.array([0, 5, 0]),
            "overview": -forward * 14 + np.array([0, 18, 0]),
        }
        camera = {"position": (origin + offsets[args.camera]).tolist(),
                  "target": (origin + [0, 0.5, 0]).tolist(), "fov": math.radians(60)}
        spectator.set_camera(camera)
        spectator.set_ghost_poses(poses)
        frame = spectator.render()
        diagnostics = spectator.diagnostics()

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 960, 32), fill="black")
    draw.text((12, 8), f"Static pose fixture | {args.track} | {args.camera} | "
              f"{args.ghosts} ghosts | no policy or physics steps", fill="white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        image.save(output, format="PNG")
    with metadata.open("x") as output:
        json.dump({"kind": "static pose fixture", "track": args.track,
                   "camera_mode": args.camera, "camera": camera,
                   "ghosts": configs, "poses": poses, "diagnostics": diagnostics},
                  output, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
