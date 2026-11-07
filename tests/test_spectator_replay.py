import json
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

from kartstudy import spectator_replay as replay
from kartstudy.algorithms import Policy
from kartstudy.environment import SCHEMA_VERSION, DrivingEnv
from kartstudy.snapshots import export_actor_snapshot, file_hash


@pytest.fixture
def actors(tmp_path):
    root = tmp_path / "actors"
    for method in ("bc", "dagger", "sac", "rlpd"):
        path = root / method
        path.mkdir(parents=True)
        config = {
            "algo": method,
            "seed": 0,
            "hidden": 8,
            "schema_version": SCHEMA_VERSION,
        }
        (path / "config.json").write_text(json.dumps(config))
        (path / "status.json").write_text(
            json.dumps({"status": "completed", "updates": 12})
        )
        torch.save(
            {
                "config": config,
                "actor": Policy(3, hidden=8).state_dict(),
                "step": 42,
                **({"updates": 12} if method != "bc" else {}),
            },
            path / "actor.pt",
        )
    return root


class FakeEnv:
    instances: ClassVar[list] = []

    def __init__(self, track, max_steps):
        self.resets = 0
        self.closed = False
        self.instances.append(self)

    def reset(self, seed, track):
        self.n = 0
        self.resets += 1
        assert seed == 20000 and track == "lighthouse"
        return np.zeros(3, np.float32), {}

    def replay_pose(self):
        return {
            "position": [self.n, 2, 3],
            "rotation": [0, 0, 0, 1],
            "kart": "tux",
            "native_time": [-1, -1, 0, 0.1][self.n],
            "speed": self.n,
        }

    def step(self, action):
        self.n += 1
        if self.resets == 3 and self.n == 2:
            raise RuntimeError("fixture failure")
        done = self.n == 3
        return (
            np.zeros(3, np.float32),
            0,
            done and self.resets == 1,
            done and self.resets != 1,
            {
                "success": self.resets == 1,
                "finish_time": 1.2,
                "executed_action": np.zeros(5),
            },
        )

    def close(self):
        self.closed = True


def test_capture_initial_terminal_failed_laps_and_freezing(
    tmp_path, actors, monkeypatch
):
    monkeypatch.setattr(replay, "DrivingEnv", FakeEnv)
    before = {p: file_hash(p) for p in actors.rglob("actor.pt")}
    result = replay.capture_comparison(
        tmp_path / "comparison", root=actors, max_steps=3
    )
    assert result.track == "lighthouse" and result.seed == 20000
    assert result.manifest["laps"]["bc-s0"]["checkpoint_updates"] == 12
    assert len(FakeEnv.instances[-1:]) == 1 and FakeEnv.instances[-1].resets == 4
    assert FakeEnv.instances[-1].closed
    statuses = [p["status"] for p in result.sample(result.duration + 10)]
    assert statuses == ["finished", "timeout", "stopped", "timeout"]
    assert result.manifest["laps"]["sac-s0"]["error"]["message"] == "fixture failure"
    assert result.manifest["laps"]["dagger-s0"]["lap_time"] is None
    assert result.sample(-1)[0]["position"] == [0, 2, 3]
    assert result.sample(99)[0]["position"] == [3, 2, 3]
    assert before == {p: file_hash(p) for p in before}
    with np.load(result.path / "bc-s0.npz") as arrays:
        assert len(arrays["time"]) == 4 and np.all(np.diff(arrays["time"]) > 0)
        np.testing.assert_array_equal(arrays["native_time"], [-1, -1, 0, 0.1])
    with pytest.raises(FileExistsError):
        replay.capture_comparison(result.path, root=actors)


def test_world_orientation_is_conjugated_wxyz_without_simulator():
    env = DrivingEnv.__new__(DrivingEnv)
    q = np.array([np.sqrt(0.5), 0, -np.sqrt(0.5), 0])
    kart = SimpleNamespace(rotation=q, location=[1, 2, 3], kart_type="tux")
    env.env = SimpleNamespace(
        unwrapped=SimpleNamespace(world=SimpleNamespace(time=0, karts=[kart]))
    )
    env.raw = {"velocity": [3, 0, 4]}
    np.testing.assert_allclose(
        env.replay_pose()["rotation"], [0, np.sqrt(0.5), 0, np.sqrt(0.5)]
    )
    assert env.replay_pose()["speed"] == 5
    del kart.rotation
    with pytest.raises(ValueError, match="orientation"):
        env.replay_pose()


def test_slerp_shortest_path_and_midpoint():
    a = [0, 0, 0, 2]
    np.testing.assert_allclose(replay.slerp(a, [0, 0, 0, -3], 0.5), [0, 0, 0, 1])
    np.testing.assert_allclose(
        replay.slerp(a, [0, 1, 0, 0], 0.5), [0, np.sqrt(0.5), 0, np.sqrt(0.5)]
    )


def test_snapshot_milestone_keys_labels_and_interpolation(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "DrivingEnv", FakeEnv)
    config = {
        "algo": "dagger",
        "seed": 0,
        "hidden": 8,
        "schema_version": SCHEMA_VERSION,
    }
    actor = Policy(3, hidden=8)
    snapshots = [
        export_actor_snapshot(tmp_path / "run", actor, config, 0, n, dagger_round=0)
        for n in (2, 4)
    ]
    result = replay.capture_comparison(
        tmp_path / "comparison", snapshots=snapshots, methods=("dagger",), max_steps=3
    )
    assert len({g["key"] for g in result.ghosts}) == 2
    assert all(
        g["method"] == "dagger" and "DAgger" in g["label"] and "updates" in g["label"]
        for g in result.ghosts
    )
    assert result.sample(0.05)[0]["position"][0] == pytest.approx(0.5)
    path = result.path / result.manifest["laps"][result.ghosts[0]["key"]]["file"]
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files if k != "rotation"}
    np.savez_compressed(path, **arrays)
    result.manifest["laps"][result.ghosts[0]["key"]]["sha256"] = file_hash(path)
    (result.path / "manifest.json").write_text(json.dumps(result.manifest))
    with pytest.raises(ValueError, match="orientation"):
        replay.ReplayComparison(result.path)


def test_missing_native_orientation_preserves_error_and_closes(
    tmp_path, actors, monkeypatch
):
    class Missing(FakeEnv):
        def replay_pose(self):
            raise ValueError("Native replay requires orientation")

    monkeypatch.setattr(replay, "DrivingEnv", Missing)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="orientation"):
        replay.capture_comparison(output, root=actors)
    assert (
        json.loads((output / "manifest.json").read_text())["error"]["type"]
        == "ValueError"
    )
    assert Missing.instances[-1].closed


def test_rescue_discontinuity_holds_pose_until_native_sample(
    tmp_path, actors, monkeypatch
):
    class Rescue(FakeEnv):
        def replay_pose(self):
            pose = super().replay_pose()
            pose["native_time"] = 0.1 * self.n
            pose["position"] = [0 if self.n < 2 else 100 + self.n, 2, 3]
            pose["rotation"] = [0, 0, 0, 1] if self.n < 2 else [0, 1, 0, 0]
            return pose

        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            info["rescue_active"] = self.n == 2
            return obs, reward, term, trunc, info

    monkeypatch.setattr(replay, "DrivingEnv", Rescue)
    result = replay.capture_comparison(
        tmp_path / "rescue", root=actors, methods=("bc",), max_steps=3
    )
    before = result.sample(0.1)[0]
    crossing = result.sample(0.15)[0]
    assert crossing["position"] == before["position"] == [0, 2, 3]
    assert crossing["rotation"] == before["rotation"] == [0, 0, 0, 1]
    assert result.sample(0.2)[0]["position"] == [102, 2, 3]
    assert result.sample(0.25)[0]["position"] == [102, 2, 3]
    with np.load(result.path / "bc-s0.npz") as arrays:
        np.testing.assert_array_equal(
            arrays["rescue_active"], [False, False, True, False]
        )
        np.testing.assert_array_equal(
            arrays["discontinuity"], [False, False, True, True]
        )


def test_unflagged_teleport_also_disables_interpolation():
    rows = [
        {
            "native_time": 0,
            "time": 0,
            "position": [0, 0, 0],
            "speed": 1,
            "rescue_active": False,
        }
    ]
    replay._append_pose(
        rows, {"native_time": 0.1, "position": [100, 0, 0], "speed": 1}, 0, False
    )
    assert rows[-1]["discontinuity"] and not rows[-1]["rescue_active"]
