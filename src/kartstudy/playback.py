"""Render saved policies using the exact driving task and control interval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .algorithms import Policy
from .environment import DrivingEnv

METHODS = {
    "expert": "Expert",
    "bc": "Behavior cloning",
    "dagger": "DAgger",
    "sac": "Online SAC",
    "sac_sym": "SAC + symmetric sampling",
    "rlpd": "RLPD",
}
EXPERIMENTS = {
    "baseline": "Baseline",
    "small_data": "Small dataset",
    "conservative_expert": "Conservative expert",
}


def discover_runs(root):
    """Use actual completed-run metadata rather than assumptions about filenames."""
    catalog = {}
    for path in sorted(Path(root).glob("*/config.json")):
        config = json.loads(path.read_text())
        if config.get("algo") not in METHODS or config["algo"] == "expert":
            continue
        status_path = path.parent / "status.json"
        if not status_path.exists() or not (path.parent / "checkpoint.pt").exists():
            continue
        if json.loads(status_path.read_text()).get("status") != "completed":
            continue
        experiment = (
            "conservative_expert"
            if config.get("conservative")
            else ("small_data" if config.get("demo_episodes") == 2 else "baseline")
        )
        key = (experiment, config["algo"], int(config["seed"]))
        if key in catalog:
            raise ValueError(f"Duplicate completed run for {key}")
        catalog[key] = path.parent
    return catalog


def load_actor(path):
    # These are trusted local experiment checkpoints, containing rl_mind objects.
    state = torch.load(
        Path(path) / "checkpoint.pt", map_location="cpu", weights_only=False, mmap=True
    )
    config = state["config"]
    weights = state["actor"]
    actor = Policy(weights["encoder.0.weight"].shape[1], hidden=config["hidden"])
    actor.load_state_dict(weights)
    return actor.eval(), config, int(state["step"])


class Playback:
    def __init__(self, root="runs/study", width=960, height=540):
        torch.set_num_threads(1)
        self.root, self.width, self.height = Path(root), width, height
        self.catalog = discover_runs(root)
        self.env = None
        self.actor = None
        self.method = "expert"
        self.experiment = "baseline"
        self.train_seed = 0
        self.eval_seed = 20000
        self.track = "lighthouse"
        self.perturb = False

    def start(
        self,
        method="dagger",
        train_seed=0,
        experiment="baseline",
        track="lighthouse",
        perturb=False,
        eval_seed=20000,
        render=True,
    ):
        if method not in METHODS or experiment not in EXPERIMENTS:
            raise ValueError("Unknown policy or experiment")
        if method == "expert":
            actor, checkpoint_step = None, None
        else:
            key = (experiment, method, int(train_seed))
            if key not in self.catalog:
                raise ValueError("No completed checkpoint for this combination")
            actor, _config, checkpoint_step = load_actor(self.catalog[key])
        if self.env is None:
            self.env = DrivingEnv(
                track=track, render=render, width=self.width, height=self.height
            )
        self.actor, self.method, self.train_seed = actor, method, int(train_seed)
        self.experiment, self.track, self.perturb = experiment, track, perturb
        self.eval_seed, self.checkpoint_step = int(eval_seed), checkpoint_step
        self.env.env.unwrapped.default_track = track
        self.env.perturb = perturb
        self.obs, _ = self.env.reset(seed=self.eval_seed)
        self.done, self.steps, self.total_reward = False, 0, 0.0
        self.info = {
            "elapsed_seconds": 0.0,
            "speed": 0.0,
            "progress": 0.0,
            "lateral_error": 0.0,
            "success": False,
            "action_overridden": False,
        }
        self.action = np.array([0.0, -1.0, -1.0], np.float32)
        self.executed_action = self.action.copy()

    def restart(self):
        self.start(
            self.method,
            self.train_seed,
            self.experiment,
            self.track,
            self.perturb,
            self.eval_seed,
        )

    def step(self):
        if self.done:
            return
        with torch.inference_mode():
            if self.actor is None:
                action = self.env.expert_action(
                    self.experiment == "conservative_expert"
                )
            else:
                action = (
                    self.actor.act(torch.from_numpy(self.obs).unsqueeze(0))
                    .squeeze(0)
                    .numpy()
                )
        self.obs, reward, term, trunc, self.info = self.env.step(action)
        self.action = action.copy()
        self.executed_action = np.asarray(
            self.info["executed_action"], dtype=np.float32
        )
        self.done, self.terminated, self.truncated = term or trunc, term, trunc
        self.steps += 1
        self.total_reward += reward

    def frame(self):
        raw = self.env.env.unwrapped._stk.race.screen_capture()
        if raw is None or not raw.size:
            raise RuntimeError("The simulator returned no rendered image")
        return (
            Image.fromarray(np.asarray(raw))
            .convert("RGB")
            .resize((self.width, self.height), Image.Resampling.LANCZOS)
        )

    def summary(self):
        return {
            "method": self.method,
            "experiment": self.experiment,
            "training_seed": None if self.method == "expert" else self.train_seed,
            "evaluation_seed": self.eval_seed,
            "track": self.track,
            "perturb": self.perturb,
            "checkpoint_step": self.checkpoint_step,
            "steps": self.steps,
            "return_": self.total_reward,
            "success": bool(self.info["success"]),
            "lap_time": self.info.get("finish_time"),
            "sim_seconds": self.info["elapsed_seconds"],
            "progress": self.info["progress"],
            "timeout": bool(self.done and self.truncated),
            "control_interval_seconds": 0.1,
            "rendered": True,
        }

    def close(self):
        if self.env is not None:
            self.env.close()


def _render_worker(connection, root, width, height):
    playback = Playback(root, width, height)
    try:
        while True:
            operation, arguments = connection.recv()
            if operation == "close":
                break
            try:
                if operation == "start":
                    playback.start(**arguments)
                elif operation == "step":
                    playback.step()
                else:
                    raise ValueError(f"Unknown playback operation: {operation}")
                names = [
                    "method",
                    "train_seed",
                    "experiment",
                    "track",
                    "perturb",
                    "eval_seed",
                    "checkpoint_step",
                    "done",
                    "steps",
                    "info",
                    "action",
                    "executed_action",
                ]
                state = {name: getattr(playback, name) for name in names}
                connection.send(("ok", state, playback.frame()))
            except Exception as exc:  # noqa: BLE001 — return worker failures across the process boundary
                connection.send(("error", str(exc), None))
    except EOFError:
        pass
    finally:
        playback.close()
        connection.close()


class RemotePlayback:
    """Isolate SDL/OpenGL from Tk's Cocoa application on macOS."""

    def __init__(self, root="runs/study", width=960, height=540):
        import multiprocessing as mp

        self.root, self.width, self.height = root, width, height
        self.catalog = discover_runs(root)
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_render_worker, args=(child, root, width, height), daemon=True
        )
        self.process.start()
        child.close()

    def _request(self, operation, arguments):
        try:
            self.connection.send((operation, arguments))
            kind, state, frame = self.connection.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError) as exc:
            raise RuntimeError(
                "The renderer exited; close and reopen the viewer"
            ) from exc
        if kind == "error":
            raise RuntimeError(state)
        self.__dict__.update(state)
        self._frame = frame

    def start(self, **kwargs):
        self._request("start", kwargs)

    def step(self):
        self._request("step", {})

    def frame(self):
        return self._frame

    def restart(self):
        self.start(
            **{
                name: getattr(self, name)
                for name in [
                    "method",
                    "train_seed",
                    "experiment",
                    "track",
                    "perturb",
                    "eval_seed",
                ]
            }
        )

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send(("close", {}))
            except (BrokenPipeError, OSError):
                pass
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        self.connection.close()


def font(size=18):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def annotated_frame(playback):
    """A small telemetry strip outside the unmodified game image."""
    game = playback.frame()
    image = Image.new("RGB", (game.width, game.height + 76), "#11171b")
    image.paste(game, (0, 0))
    draw = ImageDraw.Draw(image)
    info = playback.info
    seed = (
        "" if playback.method == "expert" else f" / training seed {playback.train_seed}"
    )
    title = f"{METHODS[playback.method]}{seed}"
    state = (
        ("FINISHED" if info["success"] else "TIMEOUT")
        if playback.done
        else ("STEERING DISTURBANCE" if info["action_overridden"] else "DRIVING")
    )
    draw.text((16, game.height + 9), title, font=font(18), fill="#f1f5f7")
    a = playback.executed_action
    detail = (
        f"{state}  |  {info['elapsed_seconds']:.1f} s  |  "
        f"{info['speed']:.1f} m/s  |  {info['progress']:.0f} m  |  "
        f"steer {a[0]:+.2f}  throttle {(a[1] + 1) / 2:.0%}  brake {'ON' if a[2] > 0 else 'off'}"
    )
    draw.text((16, game.height + 40), detail, font=font(15), fill="#c1cbd1")
    return image


def record(playback, output, limit_steps=None):
    import imageio_ffmpeg

    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (playback.width, playback.height + 76),
        fps=10,
        codec="libx264",
        quality=5,
        macro_block_size=2,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        while not playback.done and (
            limit_steps is None or playback.steps < limit_steps
        ):
            playback.step()
            writer.send(np.asarray(annotated_frame(playback)))
        for _ in range(20):
            writer.send(np.asarray(annotated_frame(playback)))
    finally:
        writer.close()
    result = playback.summary() | {
        "preview_only": not playback.done,
        "video": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/study")
    parser.add_argument("--policy", choices=METHODS, default="dagger")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=20000)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="baseline")
    parser.add_argument("--track", default="lighthouse")
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument(
        "--record",
        type=Path,
        help="Record a fresh episode to MP4 instead of opening the viewer",
    )
    parser.add_argument(
        "--limit-steps",
        type=int,
        help="Cap a recording; partial videos are labeled previews",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()
    if args.width < 640 or args.height < 360 or args.width % 2 or args.height % 2:
        parser.error("Use even dimensions of at least 640 × 360")
    if args.limit_steps is not None and (not args.record or args.limit_steps < 1):
        parser.error("--limit-steps requires --record and a positive step count")
    playback_class = Playback if args.record else RemotePlayback
    playback = playback_class(args.root, args.width, args.height)
    try:
        playback.start(
            method=args.policy,
            train_seed=args.train_seed,
            experiment=args.experiment,
            track=args.track,
            perturb=args.perturb,
            eval_seed=args.eval_seed,
        )
        if args.record:
            record(playback, args.record, args.limit_steps)
        else:
            from .viewer import show

            show(playback)
    finally:
        playback.close()


if __name__ == "__main__":
    main()
