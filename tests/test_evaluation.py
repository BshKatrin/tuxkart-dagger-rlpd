import copy
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

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


class ParallelActor(FakeActor):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
        self.register_buffer("calls", torch.tensor(0))

    def act(self, obs):
        assert obs.device.type == self.scale.device.type == "cpu"
        assert not self.training and not torch.is_grad_enabled()
        self.calls.add_(1)
        return self.scale.expand(len(obs), -1)


class ParallelEnv(FakeEnv):
    """Spawn fixture: no simulator, with real terminal/control/telemetry fields."""

    def __init__(self, marker=None, fail_seed=None, crash=False, hang=False):
        self.marker = marker
        self.fail_seed, self.crash, self.hang = fail_seed, crash, hang
        self.seeds = []

    def reset(self, seed):
        self.seeds.append(seed)
        # Exercise process isolation from the parent's three training RNGs.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return super().reset(seed)

    def step(self, action):
        if self.crash:
            os._exit(23)
        if self.hang:
            time.sleep(20)
        if self.seed == self.fail_seed:
            raise RuntimeError("fixture evaluation failure")
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(
            executed_action=action / 2,
            world_position=[self.n, self.seed, 3],
            track_length_m=50.0,
            finish_time=0.19 if terminated else None,
            measurement_seconds=self.n * 0.1,
            off_track_seconds=0.05,
            off_track_fraction=0.05 / (self.n * 0.1),
            rescue_count=self.seed % 3,
            rescue_active=self.n == 1,
            is_on_road=self.n != 1,
            harmful_contact_proportion=0.25,
        )
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.marker is not None:
            (self.marker / f"{os.getpid()}.json").write_text(
                json.dumps({"seeds": self.seeds, "threads": torch.get_num_threads()})
            )


class HeadlessFixtureEnv(ParallelEnv):
    def __init__(self, track="mines", max_steps=7):
        super().__init__()
        self.track, self.max_steps = track, max_steps
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(render_mode=None))

    def reset(self, seed):
        assert self.track == "mines" and self.max_steps == 7
        return super().reset(seed)


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_cpu_records_traces_order_and_parent_state(tmp_path, workers):
    seeds = [3, 1, 2, 1, 4]  # Repetitions and uneven partitions retain episode IDs.
    actor = ParallelActor()
    serial = evaluate(
        ParallelEnv(),
        copy.deepcopy(actor),
        seeds,
        trajectory_path=tmp_path / "serial.npz",
    )
    rng_python, rng_numpy, rng_torch = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    threads = torch.get_num_threads()
    state = copy.deepcopy(actor.state_dict())
    parent_env = ParallelEnv()
    markers = tmp_path / "closed"
    markers.mkdir()
    children = {process.pid for process in mp.active_children()}
    parallel = evaluate(
        parent_env,
        actor,
        iter(seeds),
        workers=workers,
        env_factory=partial(ParallelEnv, marker=markers),
        trajectory_path=tmp_path / "parallel.npz",
    )
    assert parallel == serial
    assert [row["seed"] for row in parallel] == seeds
    assert [row["success"] for row in parallel] == [False, True, False, True, False]
    assert parallel[1]["lap_time"] == 0.19
    assert parallel[1]["sim_seconds"] == 0.2
    assert parallel[1]["average_speed_m_s"] == 50.0 / 0.19
    assert parallel[2]["rescue_count"] == 2
    assert parent_env.seeds == [] and actor.training
    assert random.getstate() == rng_python
    after_numpy = np.random.get_state()
    assert after_numpy[0] == rng_numpy[0] and after_numpy[2:] == rng_numpy[2:]
    np.testing.assert_array_equal(after_numpy[1], rng_numpy[1])
    torch.testing.assert_close(torch.get_rng_state(), rng_torch, rtol=0, atol=0)
    assert torch.get_num_threads() == threads
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    with np.load(tmp_path / "serial.npz") as a, np.load(tmp_path / "parallel.npz") as b:
        assert a.files == b.files
        for key in a.files:
            assert a[key].dtype == b[key].dtype
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(b["episode"], np.repeat(range(len(seeds)), 2))
    closed = [json.loads(path.read_text()) for path in markers.glob("*.json")]
    assert len(closed) == workers and all(row["threads"] == 1 for row in closed)
    assert sorted(seed for row in closed for seed in row["seeds"]) == sorted(seeds)
    assert {process.pid for process in mp.active_children()} == children


def test_parallel_failure_closes_workers_and_leaves_no_output(tmp_path, monkeypatch):
    from kartstudy import evaluation

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    original = evaluation.tempfile.TemporaryDirectory
    monkeypatch.setattr(
        evaluation.tempfile,
        "TemporaryDirectory",
        lambda **kw: original(dir=scratch, **kw),
    )
    markers = tmp_path / "closed"
    markers.mkdir()
    children = {process.pid for process in mp.active_children()}
    output = tmp_path / "failed.npz"
    actor = ParallelActor()
    with pytest.raises(RuntimeError, match="fixture evaluation failure"):
        evaluate(
            ParallelEnv(),
            actor,
            [2, 1],
            workers=2,
            env_factory=partial(ParallelEnv, marker=markers, fail_seed=2),
            trajectory_path=output,
        )
    assert actor.training and actor.calls.item() == 0
    assert not output.exists() and not list(scratch.iterdir())
    assert list(markers.iterdir())  # The failed episode's environment was closed.
    assert {process.pid for process in mp.active_children()} == children


def test_parallel_evaluation_supports_long_tmp_without_torch_shared_memory(tmp_path):
    runtime = tmp_path / "isolated-native-evaluation-runtime" / ("x" * 90)
    runtime.mkdir(parents=True)
    assert len(str(runtime)) > 103
    script = tmp_path / "long_tmp_evaluation.py"
    tests = Path(__file__).resolve().parent
    script.write_text(
        f"""import os
import sys
sys.path.insert(0, {str(tests)!r})
import numpy as np
import torch
from test_evaluation import ParallelActor, ParallelEnv
from kartstudy.evaluation import evaluate

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    actor = ParallelActor()
    state = {{key: value.clone() for key, value in actor.state_dict().items()}}
    path = os.path.join(os.environ['TMPDIR'], 'traces.npz')
    rows = evaluate(ParallelEnv(), actor, [1, 2], workers=2,
                    env_factory=ParallelEnv, trajectory_path=path)
    assert [row['seed'] for row in rows] == [1, 2]
    assert [row['success'] for row in rows] == [True, False]
    assert actor.training and actor.calls.item() == 0
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    with np.load(path, allow_pickle=False) as traces:
        np.testing.assert_array_equal(traces['episode'], [0, 0, 1, 1])
        assert traces['commanded_action'].shape == (4, 3)
"""
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        env=os.environ | {"TMPDIR": str(runtime)},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_parallel_timeout_and_process_exit_are_bounded_and_recover(tmp_path):
    children = {process.pid for process in mp.active_children()}
    for factory, timeout, message in (
        (partial(ParallelEnv, hang=True), 0.05, "timed out"),
        (partial(ParallelEnv, crash=True), 30.0, "exited"),
    ):
        start = time.monotonic()
        with pytest.raises(RuntimeError, match=message):
            evaluate(
                ParallelEnv(),
                ParallelActor(),
                [1, 2],
                workers=2,
                env_factory=factory,
                worker_timeout=timeout,
                trajectory_path=tmp_path / "failure.npz",
            )
        assert time.monotonic() - start < 10
        assert not (tmp_path / "failure.npz").exists()
        assert {process.pid for process in mp.active_children()} == children
    records = evaluate(
        ParallelEnv(),
        ParallelActor(),
        [1, 2],
        workers=2,
        env_factory=ParallelEnv,
    )
    assert [row["success"] for row in records] == [True, False]
    assert {process.pid for process in mp.active_children()} == children


@pytest.mark.parametrize("workers", [0, 5, 1.5, True])
def test_invalid_evaluation_worker_count(workers):
    with pytest.raises(ValueError, match="integer from 1 to 4"):
        evaluate(FakeEnv(), FakeActor(), [1], workers=workers)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_parallel_timeout(timeout):
    with pytest.raises(ValueError, match="timeout must be finite and positive"):
        evaluate(
            FakeEnv(),
            FakeActor(),
            [1],
            workers=2,
            env_factory=ParallelEnv,
            worker_timeout=timeout,
        )


def test_parallel_rejects_gpu_callable_rendered_env_and_unpicklable_factory():
    from kartstudy.environment import DrivingEnv

    with pytest.raises(ValueError, match="CPU inference"):
        evaluate(FakeEnv(), FakeActor(), [1], "cuda", workers=2)
    with pytest.raises(TypeError, match="torch actor exposing act"):
        evaluate(FakeEnv(), lambda obs: obs, [1], workers=2)
    with pytest.raises(ValueError, match="headless DrivingEnv"):
        evaluate(FakeEnv(), FakeActor(), [1], workers=2)
    rendered = DrivingEnv.__new__(DrivingEnv)
    rendered.env = SimpleNamespace(unwrapped=SimpleNamespace(render_mode="human"))
    with pytest.raises(ValueError, match="headless DrivingEnv"):
        evaluate(rendered, FakeActor(), [1], workers=2)
    with pytest.raises(ValueError, match="top-level spawn-picklable"):
        evaluate(FakeEnv(), FakeActor(), [1], workers=2, env_factory=lambda: FakeEnv())
    with pytest.raises(TypeError, match="spawn-picklable callable"):
        evaluate(FakeEnv(), FakeActor(), [1], workers=2, env_factory=123)


def test_serial_default_keeps_callable_and_environment_semantics():
    env = FakeEnv()
    records = evaluate(
        env,
        lambda obs: np.zeros(3),
        [2, 1],
        workers=1,
        env_factory="unused",
        worker_timeout=0,
    )
    assert [row["seed"] for row in records] == [2, 1]
    assert env.seed == 1 and env.n == 2


def test_empty_parallel_seeds_create_serial_compatible_empty_trace(tmp_path):
    actor = ParallelActor()
    assert (
        evaluate(
            ParallelEnv(),
            actor,
            [],
            workers=4,
            env_factory=ParallelEnv,
            trajectory_path=tmp_path / "empty.npz",
        )
        == []
    )
    with np.load(tmp_path / "empty.npz") as archive:
        assert archive.files == []
    assert actor.training and actor.calls.item() == 0


def test_parallel_rejects_nested_pools(monkeypatch):
    from kartstudy import evaluation

    monkeypatch.setattr(evaluation.mp, "parent_process", lambda: object())
    with pytest.raises(RuntimeError, match="Nested parallel"):
        evaluate(FakeEnv(), FakeActor(), [1], workers=2, env_factory=ParallelEnv)


def test_default_headless_factory_copies_track_and_timeout_without_parent_reset(
    monkeypatch,
):
    from kartstudy import evaluation

    monkeypatch.setattr(evaluation, "DrivingEnv", HeadlessFixtureEnv)
    env = HeadlessFixtureEnv()
    records = evaluate(env, ParallelActor(), [1], workers=4)
    assert len(records) == 1 and records[0]["success"]
    assert env.seeds == []


def test_process_start_failure_closes_prior_workers_and_scratch(tmp_path, monkeypatch):
    from kartstudy import evaluation

    processes, pipes = [], []

    class Connection:
        def __init__(self):
            self.closed = False
            pipes.append(self)

        def close(self):
            self.closed = True

    class Process:
        def __init__(self, **kwargs):
            self.alive, self.pid, self.closed = False, None, False
            processes.append(self)

        def start(self):
            if len(processes) == 2:
                raise RuntimeError("fixture spawn failure")
            self.alive, self.pid = True, 100

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            assert timeout == 0.5

        def terminate(self):
            self.alive = False

        def close(self):
            assert not self.alive
            self.closed = True

    context = SimpleNamespace(
        Pipe=lambda **kw: (Connection(), Connection()), Process=Process
    )
    original = evaluation.tempfile.TemporaryDirectory
    monkeypatch.setattr(evaluation.mp, "get_context", lambda method: context)
    monkeypatch.setattr(
        evaluation.tempfile,
        "TemporaryDirectory",
        lambda **kw: original(dir=tmp_path, **kw),
    )
    with pytest.raises(RuntimeError, match="fixture spawn failure"):
        evaluate(FakeEnv(), FakeActor(), [1, 2], workers=2, env_factory=ParallelEnv)
    assert len(processes) == 2 and all(process.closed for process in processes)
    assert all(pipe.closed for pipe in pipes)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "workers,start_seconds,ready,receive_seconds,expected_time,started,received",
    [
        (2, 1.0, False, 0.0, 2.0, 2, 0),
        (2, 1.0, True, 0.0, 2.0, 2, 0),
        (4, 1.0, True, 0.0, 2.0, 2, 0),
        (2, 0.25, True, 2.0, 2.5, 2, 1),
        (2, 0.25, True, 1.0, 1.5, 2, 1),
    ],
    ids=[
        "startup-expiry-no-result",
        "startup-expiry-ready-result",
        "startup-expiry-stops-later-starts",
        "receive-crosses-expiry",
        "receive-at-expiry",
    ],
)
def test_parallel_shared_deadline_rejects_late_startup_and_ready_results(
    tmp_path,
    monkeypatch,
    workers,
    start_seconds,
    ready,
    receive_seconds,
    expected_time,
    started,
    received,
):
    from kartstudy import evaluation

    clock = [0.0]
    processes, connections, results_received = [], [], []

    class Connection:
        def __init__(self):
            self.closed = False
            connections.append(self)

        def poll(self):
            return ready

        def recv(self):
            clock[0] += receive_seconds
            results_received.append(self)
            return "ok", []

        def close(self):
            self.closed = True

    class Process:
        def __init__(self, **kwargs):
            self.alive, self.pid, self.exitcode = False, None, None
            self.closed = False
            processes.append(self)

        def start(self):
            clock[0] += start_seconds
            self.alive, self.pid = True, 100

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            assert timeout == 0.5

        def terminate(self):
            self.alive = False

        def close(self):
            assert not self.alive
            self.closed = True

    context = SimpleNamespace(
        Pipe=lambda **kw: (Connection(), Connection()), Process=Process
    )
    monkeypatch.setattr(evaluation.mp, "get_context", lambda method: context)
    monkeypatch.setattr(evaluation.time, "monotonic", lambda: clock[0])
    original = evaluation.tempfile.TemporaryDirectory
    monkeypatch.setattr(
        evaluation.tempfile,
        "TemporaryDirectory",
        lambda **kw: original(dir=tmp_path, **kw),
    )
    with pytest.raises(RuntimeError, match="timed out"):
        evaluate(
            FakeEnv(),
            FakeActor(),
            list(range(workers)),
            workers=workers,
            env_factory=ParallelEnv,
            worker_timeout=1.5,
        )
    assert clock[0] == expected_time and len(processes) == started
    assert len(results_received) == received
    assert all(not process.alive and process.closed for process in processes)
    assert all(connection.closed for connection in connections)
    assert not list(tmp_path.iterdir())


def test_worker_teardown_escalates_to_kill_with_bounded_joins():
    from kartstudy.evaluation import _stop_evaluation_worker

    class Process:
        def __init__(self):
            self.alive, self.closed, self.joins = True, False, []

        def join(self, timeout):
            self.joins.append(timeout)

        def is_alive(self):
            return self.alive

        def terminate(self):
            pass  # Unresponsive to polite termination.

        def kill(self):
            self.alive = False

        def close(self):
            self.closed = True

    process = Process()
    assert _stop_evaluation_worker(process)
    assert process.joins == [0.5, 0.5, 0.5]
    assert process.closed and not process.alive


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
