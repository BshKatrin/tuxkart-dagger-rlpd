import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from kartstudy.environment import SCHEMA_VERSION
from kartstudy.evaluation import evaluate, summarize_runs
from kartstudy.expert_evaluation import banana_contact, harmful_rows, summarize
from kartstudy.metrics import DrivingMetrics


@pytest.fixture(scope="module")
def road_analysis():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "report/scripts/track_time_loss.py")
    )


def test_sector_timing_keeps_reversing_delay_and_official_finish(road_analysis):
    sectors = road_analysis["first_passage_sectors"](
        [0, 8, 4, 8, 12, 20],
        [0, 1, 2, 3, 4, 5],
        [0, 10, 20],
        [False] * 6,
        finish_time=4.9,
    )
    assert sectors == pytest.approx([3.5, 1.4])
    assert sectors.sum() == pytest.approx(4.9)


def test_trajectory_length_keeps_reversing_but_excludes_rescue_jumps(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "report/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    length = runpy.run_path(str(scripts / "local_performance.py"))["trajectory_length"]
    positions = np.zeros((7, 3))
    positions[:, 0] = [0, 0, 3, 1, 100, 103, 106]
    # Countdown repeats t=0. Reversing adds 2 m; rescue movement adds nothing.
    # The final interval contributes only its first half before official finish.
    assert length(positions, [0, 0, 1, 2, 3, 4, 5],
                  [False, False, False, False, True, False, False],
                  [0, 0, 0, 0, 1, 1, 1], 4.5) == pytest.approx(6.5)
    assert length([[0, 0, 0], [0, 3, 4]], [0, 1],
                  [False, False], [0, 0], 1) == pytest.approx(5)
    # Count changes also detect a rescue occurring between recorded snapshots.
    assert length([[0, 0, 0], [100, 0, 0]], [0, 1],
                  [False, False], [0, 1], 1) == 0


def test_sector_timing_censors_unfinished_and_rescued_sections(road_analysis):
    sectors = road_analysis["first_passage_sectors"](
        [0, 10, 20, 25],
        [0, 1, 2, 3],
        [0, 10, 20, 30],
        [False, False, True, False],
    )
    assert sectors[0] == pytest.approx(1)
    assert np.isnan(sectors[1])
    assert np.isnan(sectors[2])


def test_road_gates_compare_alternative_routes_between_shared_junctions(road_analysis):
    geometry = {
        "length": 40,
        "distances": [[0, 10], [10, 20], [20, 30], [30, 40], [10, 30]],
        "successors": [[1, 4], [2], [3], [0], [3]],
    }
    gates, branches = road_analysis["timing_gates"](geometry, spacing=5)
    assert branches == [[10, 30]]
    assert gates == pytest.approx([0, 5, 10, 30, 35, 40])


def test_report_aggregate_keeps_timeouts_and_weights_offroad_by_exposure(road_analysis):
    def episode(seed, success, time, offroad, rescues):
        return {
            "seed": seed,
            "success": success,
            "lap_time": time if success else None,
            "timeout": not success,
            "sim_seconds": time,
            "measurement_seconds": time,
            "off_track_seconds": offroad,
            "rescue_count": rescues,
        }

    records = {
        "easy": [episode(1, True, 10, 1, 0), episode(2, True, 20, 2, 0)],
        "hard": [episode(1, False, 100, 50, 1), episode(2, True, 30, 3, 1)],
    }
    result = road_analysis["aggregate_episodes"](records, [1, 2], horizon=100)
    assert result["completed"] == 3
    assert result["attempts"] == 4
    assert result["successful_lap_time_mean"] == pytest.approx(20)
    assert result["capped_time_by_evaluation_seed"] == pytest.approx([55, 25])
    assert result["capped_time_mean"] == pytest.approx(40)
    assert result["offroad_pct"] == pytest.approx(100 * 56 / 160)
    assert result["rescues_per_attempt"] == pytest.approx(0.5)
    records["hard"][0]["sim_seconds"] = 50
    with pytest.raises(ValueError, match="timeout budget"):
        road_analysis["aggregate_episodes"](records, [1, 2], horizon=100)


def test_profiles_interpolate_original_samples_after_reversing(road_analysis):
    values = road_analysis["first_crossing_values"](
        np.array([0, 8, 4, 8, 12, 20]),
        np.array([[0], [8], [2], [14], [20], [25]]),
        np.array([10, 30]),
    )
    assert values[0, 0] == pytest.approx(17)
    assert np.isnan(values[1, 0])


def test_road_lateral_position_keeps_offroad_offsets_and_ignores_height(road_analysis):
    geometry = {
        "nodes": [[[0, 0, 0], [0, 0, 100]]],
        "distances": [[0, 100]],
        "widths": [[10]],
    }
    lateral = road_analysis["road_lateral_positions"](
        np.array([[2, 0, 10], [2, 30, 20], [-6, 0, 30]]),
        np.array([10, 20, 30]),
        geometry,
    )
    assert lateral == pytest.approx([0.4, 0.4, -1.2])


def test_local_summary_cancels_signed_gaps_and_keeps_available_segments(road_analysis):
    expert = np.array([10.0, 20.0])
    speed = np.array([5.0, 30.0])
    offset = np.array([0.2, 1.0])
    lengths = np.array([20.0, 40.0])
    usable = np.array([True, True])
    partial = road_analysis["reduce_local_sectors"](
        np.array([12.0, np.nan]),
        expert,
        speed,
        offset,
        lengths,
        usable,
    )
    assert partial == pytest.approx(
        {"signed_gap": 2, "segments": 1, "speed": 5, "lateral": 20}
    )
    complete = road_analysis["reduce_local_sectors"](
        np.array([12.0, 19.0]),
        expert,
        speed,
        offset,
        lengths,
        usable,
    )
    assert complete["signed_gap"] == pytest.approx(20 / 60)
    assert complete["segments"] == 2
    assert complete["speed"] == pytest.approx(1300 / 60)
    assert complete["lateral"] == pytest.approx(100 * 44 / 60)


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


def test_reevaluation_requires_completed_unique_training_runs(tmp_path):
    discover = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                 "scripts/reevaluate.py"))["discover"]
    for name, status in (("finished", "completed"), ("interrupted", "stopped")):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({
            "algo": "rlpd", "seed": 2, "schema_version": SCHEMA_VERSION}))
        (directory / "status.json").write_text(json.dumps({"status": status}))
        (directory / "actor.pt").touch()
    jobs, excluded = discover([tmp_path])
    assert [job["name"] for job in jobs] == ["rlpd_s2"]
    assert len(excluded) == 1
    with pytest.raises(ValueError, match="Duplicate"):
        discover([tmp_path / "finished", tmp_path / "finished"])


def test_report_capped_sd_uses_seed_averages_and_retains_failures(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "report/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    metrics = runpy.run_path(str(scripts / "prepare_performance.py"))["lap_metrics"]
    def episode(seed, success, seconds):
        return {"seed": seed, "success": success, "lap_time": seconds if success else None,
                "sim_seconds": seconds, "timeout": not success,
                "measurement_seconds": seconds, "off_track_seconds": 0, "rescue_count": 0}
    row = metrics({"a": [episode(1, True, 10), episode(2, True, 30)],
                   "b": [episode(1, True, 20), episode(2, False, 148)]}, [1, 2])
    assert row["attempts"] == 4 and row["completed"] == 3
    assert row["capped_by_seed"] == [15, 90]
    assert row["capped"] == pytest.approx(52.5)
    assert row["capped_sd"] == pytest.approx(np.std([15, 90], ddof=1))


def test_segment_overview_weights_training_runs_equally(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "report/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    aggregate = runpy.run_path(str(scripts / "segment_comparison_draft.py"))[
        "aggregate_training_runs"]
    runs = []
    for seed, (mean, pairs) in enumerate(((1, 2), (9, 1), (5, 1))):
        estimate = {"mean": mean, "sd": 0}
        segment = {"id": 0, "start": 0, "end": 20, "branch": False, "pairs": pairs,
                   "gap": estimate, "speed_gap": estimate, "lateral_gap": estimate,
                   "speed": {"expert": estimate, "rlpd": estimate},
                   "lateral": {"expert": estimate, "rlpd": estimate}}
        runs.append({"training_seed": seed, "track": "Test", "track_id": "test",
                     "slug": "test", "method": "rlpd", "length": 20,
                     "seeds": [1, 2], "paths": [], "context": [], "segments": [segment],
                     "completed": pairs, "expert_completed": 2,
                     "source_metadata": {}, "avg_compared": 1})
    result = aggregate(runs)
    assert result["segments"][0]["gap"]["mean"] == 5
    assert result["segments"][0]["gap"]["sd"] == 4
    assert result["segments"][0]["pairs_by_training_run"] == [2, 1, 1]
    with pytest.raises(ValueError, match="Duplicate training seeds"):
        aggregate([runs[0], runs[0]])


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
