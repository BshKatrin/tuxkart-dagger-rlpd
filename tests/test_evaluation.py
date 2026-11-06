import json

import numpy as np
import pytest
import torch

from kartstudy.environment import SCHEMA_VERSION
from kartstudy.evaluation import evaluate, summarize_runs
from kartstudy.expert_evaluation import banana_contact, harmful_rows, summarize
from kartstudy.metrics import DrivingMetrics


def test_driving_metrics_weight_race_time_and_count_rescue_episodes():
    metrics = DrivingMetrics()
    metrics.update(-1, 0, False, None)  # Countdown does not enter the denominator.
    metrics.update(0, 0.1, True, None)
    metrics.update(0.1, 0.3, False, 1.1)
    metrics.update(0.3, 0.4, False, 1.0)  # Same rescue, not another event.
    metrics.update(0.4, 0.5, True, None)
    metrics.update(0.5, 0.6, True, 1.1)
    metrics.update(0.6, 0.8, False, 1.0, finish_time=0.7)
    assert metrics.seconds == pytest.approx(0.7)
    assert metrics.off_track_seconds == pytest.approx(0.4)
    assert metrics.snapshot()["off_track_fraction"] == pytest.approx(4 / 7)
    assert metrics.rescue_count == 2
    # A restarted timer identifies back-to-back rescues without an inactive sample.
    metrics.update(0.8, 0.9, True, 1.2)
    assert metrics.rescue_count == 3
    late_finish = DrivingMetrics()
    late_finish.update(0, 0.3, False, None)
    late_finish.update(0.3, 0.4, True, None, finish_time=0.295)
    assert late_finish.seconds == pytest.approx(0.295)
    assert late_finish.off_track_seconds == pytest.approx(0.295)


def test_banana_contacts_distinguish_expiry_from_pickups():
    assert banana_contact((9, 0), (2, 9.9), 0.1)
    assert banana_contact((0, 2), (0, 6.9), 0.1)
    assert banana_contact((2, 5), (9, 0), 0.1)
    assert not banana_contact((2, 0.05), (9, 0), 0.1)
    assert not banana_contact((0, 2), (0, 1.9), 0.1)
    assert not banana_contact((1, 0.05), (9, 0), 0.1)


def test_harmful_rows_group_across_width_and_lap_seam():
    from types import SimpleNamespace

    track = SimpleNamespace(
        length=100,
        path_nodes=np.array([[[0, 0, 0], [0, 0, 100]]]),
        path_distance=np.array([[0, 100]]),
    )
    items = [
        SimpleNamespace(id=i, type=1, location=[x, 0, z])
        for i, (x, z) in enumerate([(0, 99), (3, 1), (0, 40), (-3, 41), (0, 45)])
    ]
    rows = harmful_rows(items, track)
    assert sorted(len(row["items"]) for row in rows) == [1, 2, 2]
    assert harmful_rows([], track) == []


def test_expert_summary_keeps_failures_and_means_seed_ratios():
    rows = [
        {
            "success": True,
            "harmful_contact_proportion": 0.5,
            "average_speed_m_s": 10,
            "lap_time_seconds": 20,
        },
        {
            "success": False,
            "harmful_contact_proportion": 0.0,
            "average_speed_m_s": None,
            "lap_time_seconds": None,
        },
    ]
    result = summarize(rows)
    assert result["completed"] == 1
    assert result["harmful_contact_proportion"] == {"mean": 0.25, "n": 2}
    assert result["average_speed_m_s"] == {"mean": 10, "n": 1}


class FakeEnv:
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
    records = evaluate(env, actor, [1, 2], trajectory_path=tmp_path / "traces.npz")
    assert actor.training
    assert [row["success"] for row in records] == [True, False]
    assert records[0]["lap_time"] == pytest.approx(0.2)
    assert records[1]["lap_time"] is None
    assert records[1]["timeout"]
    assert records[0]["steps"] == 2
    assert records[0]["mean_abs_lateral_error"] == 0.5
    traces = np.load(tmp_path / "traces.npz")
    assert traces["commanded_action"].shape == (4, 3)
    assert traces["terminated"].sum() == 1


def test_evaluation_propagates_actor_error():
    env = FakeEnv()

    def fail(obs):
        raise RuntimeError("failure")

    with pytest.raises(RuntimeError):
        evaluate(env, fail, [1])


def test_summary_uses_training_seeds_and_latest_checkpoint(tmp_path):
    for name, seed, step, success in [
        ("old", 1, 10, False),
        ("new", 1, 20, True),
        ("other", 2, 20, False),
    ]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps(
                {"algo": "dagger", "seed": seed, "schema_version": SCHEMA_VERSION}
            )
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


def test_summary_separates_held_out_difficulties_and_keeps_runs_as_unit(tmp_path):
    for seed in range(2):
        directory = tmp_path / str(seed)
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "algo": "bc",
                    "seed": seed,
                    "schema_version": SCHEMA_VERSION,
                }
            )
        )
        episodes = [
            dict(
                success=track != "mines",
                lap_time=10 if track != "mines" else None,
                return_=1,
                **{"return": 1},
                mean_abs_lateral_error=0,
                track=track,
                split=split,
                difficulty=difficulty,
            )
            for track, split, difficulty in [
                ("lighthouse", "train", "simple"),
                ("abyss", "held_out", "simple"),
                ("mines", "held_out", "medium"),
            ]
        ]
        (directory / "evaluations.jsonl").write_text(
            json.dumps({"step": 0, "episodes": episodes})
        )
    summaries = {row["condition"]: row for row in summarize_runs(tmp_path)}
    assert summaries["held_out/all"]["completion_rate"]["mean"] == 0.5
    assert summaries["held_out/all"]["completion_rate"]["n_training_seeds"] == 2
    assert summaries["held_out/medium"]["conditional_lap_time"]["mean"] is None
    assert summaries["train/all"]["completion_rate"]["mean"] == 1
