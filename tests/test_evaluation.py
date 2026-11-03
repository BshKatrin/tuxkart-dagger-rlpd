import json

import numpy as np
import pytest
import torch

from kartstudy.evaluation import evaluate, summarize_runs


class FakeEnv:
    perturb = False

    def reset(self, seed):
        self.seed = seed
        self.n = 0
        return np.zeros(2, np.float32), {}

    def step(self, action):
        self.n += 1
        ended = self.n == 2
        success = ended and self.seed == 1
        return (
            np.ones(2, np.float32),
            0.5,
            success,
            ended and not success,
            {
                "success": success,
                "elapsed_seconds": self.n * 0.1,
                "distance": self.n,
                "progress": self.n / 10,
                "speed": 5,
                "lateral_error": -0.5,
                "action_overridden": self.perturb and self.n == 1,
            },
        )


class FakeActor(torch.nn.Module):
    def act(self, obs):
        assert not self.training
        assert not torch.is_grad_enabled()
        return torch.zeros((len(obs), 3))


def test_evaluate_separates_failed_laps_and_preserves_mode(tmp_path):
    actor = FakeActor()
    env = FakeEnv()
    records = evaluate(
        env, actor, [1, 2], perturb=True, trajectory_path=tmp_path / "traces.npz"
    )
    assert actor.training
    assert not env.perturb
    assert [row["success"] for row in records] == [True, False]
    assert records[0]["lap_time"] == pytest.approx(0.2)
    assert records[1]["lap_time"] is None
    assert records[1]["timeout"]
    assert records[0]["steps"] == 2
    assert records[0]["mean_abs_lateral_error"] == 0.5
    traces = np.load(tmp_path / "traces.npz")
    assert traces["commanded_action"].shape == (4, 3)
    assert traces["action_overridden"].sum() == 2
    assert traces["terminated"].sum() == 1


def test_evaluation_restores_perturbation_on_error():
    env = FakeEnv()

    def fail(obs):
        raise RuntimeError("failure")

    with pytest.raises(RuntimeError):
        evaluate(env, fail, [1], perturb=True)
    assert not env.perturb


def test_summary_uses_training_seeds_and_latest_checkpoint(tmp_path):
    for name, seed, step, success in [
        ("old", 1, 10, False),
        ("new", 1, 20, True),
        ("other", 2, 20, False),
    ]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps({"algo": "dagger", "seed": seed})
        )
        episodes = [
            {
                "success": success,
                "lap_time": 2 if success else None,
                "return": 1,
                "mean_abs_lateral_error": 0.2,
            }
        ] * 5
        (directory / "evaluations.jsonl").write_text(
            json.dumps({"step": step, "condition": "normal", "episodes": episodes})
            + "\n"
        )
    result = summarize_runs(tmp_path, bootstrap_samples=100)[0]
    assert result["completion_rate"]["n_training_seeds"] == 2
    assert result["completion_rate"]["mean"] == 0.5
    assert result["conditional_lap_time"]["mean"] == 2
    assert result["conditional_lap_time"]["n_training_seeds"] == 1
    assert result["conditional_lap_time"]["ci95"] is None
