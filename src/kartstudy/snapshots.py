"""Immutable CPU actor exports at existing assessment and stop boundaries."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def actor_hash(weights):
    digest = hashlib.sha256()
    for name, value in sorted(weights.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_identity():
    root = Path(__file__).resolve().parents[2]
    return {
        "git_head": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(root)): file_hash(path)
            for path in sorted((root / "src/kartstudy").rglob("*"))
            if path.is_file() and path.suffix in (".py", ".cpp", ".json")
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "numpy",
                "pysupertuxkart2",
                "pystk2-gymnasium",
                "rl-mind",
            )
        },
    }


def export_actor_snapshot(out, actor, config, steps, updates, dagger_round=None):
    """Return a unique snapshot directory; identical boundaries deduplicate.

    Serialization consumes no Python, NumPy, Torch or device RNG and never moves
    or changes the live actor. A conflicting counter pair fails explicitly.
    """
    root = Path(out) / "snapshots"
    root.mkdir(parents=True, exist_ok=True)
    key = f"actions-{int(steps):09d}_updates-{int(updates):09d}"
    destination = root / key
    weights = {
        name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
    }
    fingerprint = actor_hash(weights)
    if destination.exists():
        previous = json.loads((destination / "manifest.json").read_text())
        if (
            previous["actor_sha256"] != fingerprint
            or previous["config"] != config
            or previous["dagger_round"] != dagger_round
            or file_hash(destination / "actor.pt") != previous["actor_file_sha256"]
        ):
            raise FileExistsError(f"Conflicting snapshot boundary: {destination}")
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=root))
    try:
        state = {
            "config": config,
            "actor": weights,
            "step": int(steps),
            "updates": int(updates),
            "dagger_round": dagger_round,
        }
        torch.save(state, temporary / "actor.pt")
        manifest = {
            "format_version": 1,
            "key": key,
            **{k: v for k, v in state.items() if k != "actor"},
            "actor_sha256": fingerprint,
            "actor_file_sha256": file_hash(temporary / "actor.pt"),
            "source": source_identity(),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination
