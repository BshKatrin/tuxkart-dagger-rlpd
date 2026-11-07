import copy
import hashlib
import json
import warnings

import pytest

from kartstudy.comparison import (
    EVALUATION_SEEDS,
    METHODS,
    build_comparison,
    lap_statistics,
    write_comparison,
)
from kartstudy.tracks import ALL_TRACKS, TRAIN_TRACKS, track_group


def episodes(tracks):
    return [
        {
            "track": track,
            "seed": seed,
            "split": track_group(track)[0],
            "difficulty": track_group(track)[1],
            "success": True,
            "lap_time": 20.0,
            "measurement_seconds": 20.0,
            "off_track_seconds": 2.0,
            "rescue_count": 1,
            "return": 10.0,
        }
        for track in tracks
        for seed in EVALUATION_SEEDS
    ]


def make_run(tmp_path, method, seed=0):
    path = tmp_path / f"{method}_s{seed}"
    path.mkdir()
    config = {
        "algo": method,
        "seed": seed,
        "schema_version": 3,
        "tracks": list(TRAIN_TRACKS),
        "steps": 400000,
        "steps_per_track": 100000,
        "bc_updates": 8000,
        "utd": 4 if method == "rlpd" else 1,
        "demo_episodes": 20,
        "max_steps": 1500,
        "eval_every": 40000,
        "device": "mps",
        "dataset": "data/demonstrations.npz",
        "hidden": 128,
    }
    steps = [0] if method == "bc" else range(0, 400001, 40000)
    rows = [
        {
            "step": step,
            "updates": 8000 if method == "bc" else step * config["utd"],
            "wall_seconds": 100.0 * (index + 1),
            "final": method == "bc" or step == 400000,
            "episodes": episodes(
                ALL_TRACKS if method == "bc" or step == 400000 else TRAIN_TRACKS
            ),
        }
        for index, step in enumerate(steps)
    ]
    status = {
        "status": "completed",
        "step": rows[-1]["step"],
        "updates": rows[-1]["updates"],
        "track_steps": {
            track: 0 if method == "bc" else 100000 for track in TRAIN_TRACKS
        },
        "wall_seconds": 1500.0,
    }
    provenance = {
        "configuration": config,
        "dataset_sha256": "a" * 64,
        "versions": {"pysupertuxkart2": "0.7.4", "torch": "2.15.0"},
        "git_head": "b" * 40,
    }
    for name, data in (
        ("config.json", config),
        ("status.json", status),
        ("provenance.json", provenance),
    ):
        (path / name).write_text(json.dumps(data))
    write_evaluations(path, rows)
    return path


def write_evaluations(path, rows):
    (path / "evaluations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def read_evaluations(path):
    return [
        json.loads(line)
        for line in (path / "evaluations.jsonl").read_text().splitlines()
    ]


def edit_json(path, name, edit):
    file = path / name
    data = json.loads(file.read_text())
    edit(data)
    file.write_text(json.dumps(data))


@pytest.fixture
def runs(tmp_path):
    return [make_run(tmp_path, method) for method in METHODS]


def group(report, method, condition="train/all"):
    return next(
        row
        for row in report["groups"]
        if (row["method"], row["condition"]) == (method, condition)
    )


def test_laps_weight_off_track_by_race_seconds_and_include_failures():
    rows = episodes(("lighthouse",))[:2]
    rows[0].update(measurement_seconds=10.0, lap_time=10.0, off_track_seconds=10.0)
    rows[1].update(
        measurement_seconds=90.0,
        success=False,
        lap_time=None,
        off_track_seconds=0.0,
        rescue_count=5,
    )
    stats = lap_statistics(rows)
    assert stats["completion_rate"] == 0.5
    assert stats["off_track_fraction"] == 0.1
    assert stats["rescues_per_lap"] == 3.0
    assert stats["conditional_lap_time_s"] == 10.0
    assert (stats["completed"], stats["evaluation_laps"]) == (1, 2)
    with pytest.raises(ValueError, match="At least one"):
        lap_statistics([])


def test_seed_zero_comparison_has_all_groups_and_no_seed_intervals(runs):
    report = build_comparison(runs)
    assert report["methods"] == list(METHODS)
    assert len(report["groups"]) == 5 * (13 + 6 + 2)
    assert len(report["learning_curves"]) == 1 + 4 * 11
    for row in report["groups"]:
        assert row["training_seeds"] == [0]
        assert row["metrics"]["completion_rate"] == {
            "mean": 1.0,
            "ci95": None,
            "n_training_seeds": 1,
        }
    assert group(report, "sac", "held_out/all")["evaluation_laps"] == 36
    bc = [point for point in report["learning_curves"] if point["method"] == "bc"]
    assert [(point["counter"], point["x"]) for point in bc] == [
        ("fitting_updates", 8000)
    ]
    assert all(point["evaluation_laps"] == 16 for point in report["learning_curves"])


def test_training_runs_have_equal_weight_not_race_duration(runs, tmp_path):
    extra = make_run(tmp_path, "sac", seed=1)
    for path, seconds, off_track, success in (
        (runs[2], 10.0, 10.0, False),
        (extra, 100.0, 0.0, True),
    ):
        rows = read_evaluations(path)
        for row in rows:
            for episode in row["episodes"]:
                episode.update(
                    measurement_seconds=seconds,
                    off_track_seconds=off_track,
                    success=success,
                    lap_time=seconds if success else None,
                )
        write_evaluations(path, rows)
    report = build_comparison([*runs, extra], bootstrap_samples=500)
    stats = group(report, "sac")
    assert stats["training_seeds"] == [0, 1]
    assert stats["metrics"]["off_track_fraction"]["mean"] == 0.5
    assert stats["metrics"]["completion_rate"] == {
        "mean": 0.5,
        "ci95": [0.0, 1.0],
        "n_training_seeds": 2,
    }
    assert stats["completed"] == 16 and stats["evaluation_laps"] == 32
    assert stats["metrics"]["conditional_lap_time_s"] == {
        "mean": 100.0,
        "ci95": None,
        "n_training_seeds": 1,
    }
    assert group(report, "bc")["training_seeds"] == [0]
    assert report == build_comparison([*runs, extra], bootstrap_samples=500)
    assert report == build_comparison([extra, *reversed(runs)], bootstrap_samples=500)


def test_distinct_dataset_paths_are_allowed_only_with_same_hash(runs):
    edit_json(
        runs[2], "config.json", lambda data: data.update(dataset="/shared/data.npz")
    )
    config = json.loads((runs[2] / "config.json").read_text())
    edit_json(
        runs[2], "provenance.json", lambda data: data.update(configuration=config)
    )
    build_comparison(runs)
    edit_json(
        runs[2], "provenance.json", lambda data: data.update(dataset_sha256="c" * 64)
    )
    with pytest.raises(ValueError, match="dataset checksums"):
        build_comparison(runs)


@pytest.mark.parametrize(
    "name,key,value,match",
    [
        ("status.json", "status", "stopped", "Completed runs"),
        ("status.json", "step", 399999, "exact action budget"),
        ("status.json", "updates", 1, "optimizer updates"),
        ("status.json", "wall_seconds", 1.0, "elapsed time"),
        ("config.json", "steps", 80000, "400,000-action"),
        ("config.json", "utd", 4, "method-specific"),
        ("config.json", "demo_episodes", 5, "20 training demos"),
        ("config.json", "seed", True, "Training seed"),
        ("provenance.json", "dataset_sha256", "z" * 64, "checksum"),
        ("provenance.json", "versions", {"torch": "other"}, "versions differ"),
    ],
)
def test_rejects_incomplete_or_mismatched_runs(runs, name, key, value, match):
    edit_json(runs[2], name, lambda data: data.update({key: value}))
    with pytest.raises(ValueError, match=match):
        build_comparison(runs)


def test_rejects_modified_configuration_and_duplicate_or_missing_methods(runs):
    with pytest.raises(ValueError, match="all five"):
        build_comparison(runs[:-1])
    with pytest.raises(ValueError, match="Duplicate method"):
        build_comparison([*runs, runs[2]])
    edit_json(runs[2], "config.json", lambda data: data.update(hidden=64))
    with pytest.raises(ValueError, match="recorded provenance"):
        build_comparison(runs)
    config = json.loads((runs[2] / "config.json").read_text())
    edit_json(
        runs[2], "provenance.json", lambda data: data.update(configuration=config)
    )
    with pytest.raises(ValueError, match="configurations do not match"):
        build_comparison(runs)


@pytest.mark.parametrize(
    "failure,match",
    [
        ("missing_final", "One final"),
        ("missing_assessment", "complete frozen schedule"),
        ("duplicate_assessment", "complete frozen schedule"),
        ("early_final", "complete frozen schedule"),
        ("non_boolean_final", "boolean final"),
        ("duplicate_episode", "track/seed pair"),
        ("wrong_seed", "track/seed pair"),
        ("wrong_split", "split/difficulty"),
        ("off_track_over_time", "bounded off-track"),
        ("zero_race_time", "positive race time"),
        ("failed_lap_time", "Failed evaluations"),
        ("zero_success_time", "positive lap time"),
        ("negative_rescues", "rescue counts"),
        ("nan_return", "finite return"),
        ("held_out_intermediate", "held-out feedback"),
        ("decreasing_updates", "must not decrease"),
    ],
)
def test_rejects_bad_assessment_grid_and_metrics(runs, failure, match):
    rows = read_evaluations(runs[2])
    final_episode = rows[-1]["episodes"][0]
    if failure == "missing_final":
        rows[-1]["final"] = False
    elif failure == "missing_assessment":
        rows.pop(1)
    elif failure == "duplicate_assessment":
        rows.insert(1, copy.deepcopy(rows[0]))
    elif failure == "early_final":
        rows[-2], rows[-1] = rows[-1], rows[-2]
    elif failure == "non_boolean_final":
        rows[-1]["final"] = "true"
    elif failure == "duplicate_episode":
        rows[-1]["episodes"][-1] = copy.deepcopy(final_episode)
    elif failure == "wrong_seed":
        final_episode["seed"] = 123
    elif failure == "wrong_split":
        final_episode["difficulty"] = "hard"
    elif failure == "off_track_over_time":
        final_episode["off_track_seconds"] = 21.0
    elif failure == "zero_race_time":
        final_episode["measurement_seconds"] = 0.0
    elif failure == "failed_lap_time":
        final_episode["success"] = False
    elif failure == "zero_success_time":
        final_episode["lap_time"] = 0.0
    elif failure == "negative_rescues":
        final_episode["rescue_count"] = -1
    elif failure == "nan_return":
        final_episode["return"] = float("nan")
    elif failure == "held_out_intermediate":
        rows[0]["episodes"].extend(episodes(ALL_TRACKS[len(TRAIN_TRACKS) :]))
    elif failure == "decreasing_updates":
        rows[2]["updates"] = 1
    write_evaluations(runs[2], rows)
    with pytest.raises(ValueError, match=match):
        build_comparison(runs)


def test_fresh_output_checksums_and_input_change_protection(runs, tmp_path):
    report = build_comparison(runs)
    output = write_comparison(report, tmp_path / "comparison", figures=False)
    assert json.loads((output / "comparison.json").read_text()) == report
    assert "BC uses 8,000 fitting updates" in (output / "comparison.md").read_text()
    assert (
        "conditional_lap_time_s_n_training_seeds"
        in (output / "final-results.csv").read_text()
    )
    assert "fitting_updates" in (output / "learning-curves.csv").read_text()
    for line in (output / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ")
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError):
        write_comparison(report, output, figures=False)
    edit_json(runs[0], "status.json", lambda data: data.update(status="stopped"))
    fresh = tmp_path / "changed"
    with pytest.raises(RuntimeError, match="input changed"):
        write_comparison(report, fresh, figures=False)
    assert not fresh.exists()


def test_plotting_no_success_bc_is_finite_and_creates_all_artifacts(runs, tmp_path):
    rows = read_evaluations(runs[0])
    for episode in rows[0]["episodes"]:
        episode.update(success=False, lap_time=None)
    write_evaluations(runs[0], rows)
    report = build_comparison(runs)
    assert group(report, "bc")["metrics"]["conditional_lap_time_s"] == {
        "mean": None,
        "ci95": None,
        "n_training_seeds": 0,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*(empty slice|invalid value).*")
        output = write_comparison(report, tmp_path / "figures")
    assert "—" in (output / "comparison.md").read_text()
    for name in ("final-results", "completion-by-track", "learning-curves"):
        for suffix in ("png", "svg"):
            assert (output / f"{name}.{suffix}").stat().st_size > 1000


def test_bootstrap_requires_positive_sample_count(runs):
    with pytest.raises(ValueError, match="positive"):
        build_comparison(runs, bootstrap_samples=0)
