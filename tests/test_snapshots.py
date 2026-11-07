import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rl_mind.core import Action
from rl_mind.data import Transitions

from kartstudy import train
from kartstudy.algorithms import Policy
from kartstudy.playback import load_actor
from kartstudy.snapshots import actor_hash, export_actor_snapshot, file_hash


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0] and a[1][2:] == b[1][2:]
    np.testing.assert_array_equal(a[1][1], b[1][1])
    torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)


def test_snapshots_preserve_rng_actor_and_unique_boundaries(tmp_path):
    actor = Policy(3, hidden=8).train()
    config = {"algo": "dagger", "hidden": 8, "seed": 0}
    inputs = torch.ones(2, 3)
    expected = actor.act(inputs).detach()
    before = rng_state()
    path = export_actor_snapshot(tmp_path, actor, config, 0, 2, 0)
    assert_rng_equal(before, rng_state())
    assert actor.training
    loaded, restored, step = load_actor(path)
    assert not loaded.training and next(loaded.parameters()).device.type == "cpu"
    assert step == 0 and restored == config
    torch.testing.assert_close(loaded.act(inputs), expected, rtol=0, atol=0)
    original = {p: file_hash(p) for p in path.iterdir()}
    assert export_actor_snapshot(tmp_path, actor, config, 0, 2, 0) == path
    assert original == {p: file_hash(p) for p in original}
    assert export_actor_snapshot(tmp_path, actor, config, 0, 4, 0) != path
    with torch.no_grad():
        actor.mean.bias.add_(1)
    with pytest.raises(FileExistsError, match="Conflicting"):
        export_actor_snapshot(tmp_path, actor, config, 0, 2, 0)
    assert original == {p: file_hash(p) for p in original}
    assert not list((tmp_path / "snapshots").glob(".snapshot-*"))


@pytest.mark.parametrize(
    "algo, stopped",
    [
        ("bc", False),
        ("dagger", False),
        ("sac", False),
        ("sac_sym", False),
        ("rlpd", False),
        ("sac", True),
    ],
)
@pytest.mark.parametrize(
    "device",
    ["cpu"]
    + (["cuda"] if torch.cuda.is_available() else [])
    + (["mps"] if torch.backends.mps.is_available() else []),
)
def test_tiny_training_logging_preserves_weights_counters_and_rng(
    tmp_path, monkeypatch, algo, stopped, device
):
    """Two identical disposable mocked runs isolate the effect of export logging."""
    dataset = tmp_path / "dataset.npz"
    np.savez(
        dataset,
        expert_action=np.zeros((4, 5), np.float32),
        episode=np.arange(4),
        obs=np.zeros((4, 3), np.float32),
    )
    transitions = Transitions(
        torch.zeros(2, 3),
        Action(torch.zeros(2, 5)),
        torch.zeros(2),
        torch.zeros(2, 3),
        torch.zeros(2, dtype=torch.bool),
    )
    monkeypatch.setattr(
        train, "validate_dataset", lambda *a: {"validation_episode_ids": [2, 3]}
    )
    monkeypatch.setattr(train, "load_transitions", lambda *a: transitions)
    monkeypatch.setattr(
        train, "training_mask", lambda *a: np.array([True, True, False, False])
    )

    handlers = {}
    monkeypatch.setattr(
        train.signal,
        "signal",
        lambda number, handler: handlers.update({number: handler}),
    )

    class Env:
        def __init__(self, track, max_steps):
            self.track = track
            self.env = SimpleNamespace(unwrapped=SimpleNamespace(default_track=track))

        def reset(self, seed, track=None):
            if track:
                self.track = track
            return np.zeros(3, np.float32), {}

        def expert_action(self):
            return np.zeros(5, np.float32)

        def step(self, action):
            if stopped:
                handlers[train.signal.SIGTERM]()
            return (
                np.zeros(3, np.float32),
                0.1,
                False,
                False,
                {"distance": 0, "elapsed_seconds": 0.1},
            )

        def close(self):
            pass

    def evaluate(env, actor, seeds, *a, **kw):
        # Existing assessment restores these RNG streams after evaluation.
        random.random()
        np.random.random()
        torch.rand(1)
        if device == "cuda":
            torch.rand(1, device="cuda")
        return [{"seed": seed, "steps": 1} for seed in seeds]

    monkeypatch.setattr(train, "DrivingEnv", Env)
    monkeypatch.setattr(train, "evaluate", evaluate)
    config = train.defaults() | {
        "algo": algo,
        "device": device,
        "dataset": str(dataset),
        "tracks": ["lighthouse"],
        "steps": 4,
        "steps_per_track": 4,
        "demo_episodes": 1,
        "hidden": 8,
        "batch_size": 2,
        "bc_updates": 2,
        "dagger_updates": 2,
        "dagger_round": 2,
        "collect_steps": 2,
        "eval_every": 2,
        "warmup": 0,
    }
    baseline, logged = tmp_path / "baseline", tmp_path / "logged"
    cuda_boundaries = {}
    original_export = train.export_actor_snapshot

    def export(*args, **kwargs):
        if device == "cuda":
            cuda_boundaries.setdefault((args[3], args[4]), []).append(
                torch.cuda.get_rng_state_all()
            )
        return original_export(*args, **kwargs)

    monkeypatch.setattr(train, "export_actor_snapshot", export)
    original_update = train.SAC.update

    def immediate_metrics(self, *args, **kwargs):
        kwargs["collect_metrics"] = True
        return original_update(self, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(train, "export_actor_snapshot", lambda *a, **kw: None)
        context.setattr(train.SAC, "update", immediate_metrics)
        train.run(config, baseline)
    baseline_rng = rng_state()
    baseline_cuda = torch.cuda.get_rng_state_all() if device == "cuda" else None
    baseline_mps = torch.mps.get_rng_state().clone() if device == "mps" else None
    train.run(config, logged)
    assert_rng_equal(baseline_rng, rng_state())
    if baseline_cuda is not None:
        for expected, actual in zip(
            baseline_cuda, torch.cuda.get_rng_state_all(), strict=True
        ):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    if baseline_mps is not None:
        torch.testing.assert_close(
            baseline_mps, torch.mps.get_rng_state(), rtol=0, atol=0
        )
    a = torch.load(baseline / "checkpoint.pt", weights_only=False)
    b = torch.load(logged / "checkpoint.pt", weights_only=False)
    assert a["step"] == b["step"] and a["updates"] == b["updates"]
    assert actor_hash(a["actor"]) == actor_hash(b["actor"])

    def assert_nested_equal(expected, actual):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        elif isinstance(expected, dict):
            assert expected.keys() == actual.keys()
            for key in expected:
                assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            assert len(expected) == len(actual)
            for left, right in zip(expected, actual, strict=True):
                assert_nested_equal(left, right)
        else:
            assert expected == actual

    assert_nested_equal(a["learner"], b["learner"])
    assert_nested_equal(a["optimizer"], b["optimizer"])
    if device == "mps":
        assert_nested_equal(a["rng_mps"], b["rng_mps"])
    assert_rng_equal(
        (a["rng_python"], a["rng_numpy"], a["rng_torch"]),
        (b["rng_python"], b["rng_numpy"], b["rng_torch"]),
    )
    if device == "cuda":
        for expected, actual in zip(a["rng_cuda"], b["rng_cuda"], strict=True):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for boundaries in cuda_boundaries.values():
            for boundary in boundaries[1:]:
                for expected, actual in zip(boundaries[0], boundary, strict=True):
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for name in ("evaluations.jsonl", "training.jsonl", "episodes.jsonl"):
        if (baseline / name).exists():

            def without_time(path):
                return [
                    {k: v for k, v in json.loads(line).items() if k != "wall_seconds"}
                    for line in path.read_text().splitlines()
                ]

            assert without_time(baseline / name) == without_time(logged / name)
    snapshots = sorted((logged / "snapshots").glob("actions-*"))
    assert len(snapshots) == (1 if algo == "bc" else 2 if stopped else 3)
    assert json.loads((logged / "status.json").read_text())["status"] == (
        "stopped" if stopped else "completed"
    )
    for path in snapshots:
        actor, _, step = load_actor(path)
        metadata = json.loads((path / "manifest.json").read_text())
        assert metadata["actor_sha256"] == actor_hash(actor.state_dict())
        assert metadata["step"] == step
        assert (
            metadata["dagger_round"] is not None
            if algo == "dagger"
            else metadata["dagger_round"] is None
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_snapshot_preserves_device_rng_and_reloads_on_cpu(tmp_path):
    actor = Policy(3, hidden=8).to("mps").train()
    inputs = torch.ones(2, 3, device="mps")
    expected = actor.act(inputs).detach().cpu()
    before, device_before = rng_state(), torch.mps.get_rng_state().clone()
    path = export_actor_snapshot(
        tmp_path, actor, {"algo": "bc", "hidden": 8, "seed": 0}, 0, 2
    )
    assert_rng_equal(before, rng_state())
    torch.testing.assert_close(device_before, torch.mps.get_rng_state(), rtol=0, atol=0)
    assert actor.training and next(actor.parameters()).device.type == "mps"
    loaded, _, _ = load_actor(path)
    assert next(loaded.parameters()).device.type == "cpu" and not loaded.training
    torch.testing.assert_close(loaded.act(inputs.cpu()), expected, atol=1e-6, rtol=1e-6)
