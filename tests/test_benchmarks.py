import json
import multiprocessing
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_throughput.py"
benchmark = runpy.run_path(str(SCRIPT))


def fake_worker(config, index, messages, barrier):
    os.setsid()
    barrier.wait(2)
    for repeat in range(2):
        for variant in ("baseline", "optimized"):
            messages.put(
                {
                    "kind": "block",
                    "worker": index,
                    "repeat": repeat,
                    "variant": variant,
                    "started_at": 100 + repeat * 10,
                    "finished_at": 102 + repeat * 10,
                    "equivalent_actions": 32,
                }
            )
    messages.put(
        {
            "kind": "done",
            "worker": index,
            "runtime": {"device": "cpu-fixture", "qualification": {"exact": True}},
        }
    )


def failed_worker(config, index, messages, barrier):
    os.setsid()
    messages.put({"kind": "error", "worker": index, "error": "fixture failure"})
    time.sleep(30)


def blocked_worker(config, index, messages, barrier):
    os.setsid()
    time.sleep(30)


def unqualified_worker(config, index, messages, barrier):
    os.setsid()
    messages.put(
        {"kind": "done", "worker": index, "runtime": {"device": "cpu-fixture"}}
    )


@pytest.mark.parametrize(
    "difference",
    ["checkpoint", "optimizer", "gradient", "python_rng", "cpu_rng", "device_rng"],
)
def test_full_final_qualification_rejects_any_state_difference(difference):
    import copy
    from types import SimpleNamespace

    import torch

    state = {
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {0: {"exp_avg": torch.tensor([0.5])}}},
        "updates": 32,
    }
    changed = copy.deepcopy(state)
    gradient = torch.tensor([0.25])
    changed_gradient = gradient.clone()
    rng = {
        "python": (3, (1, 2), None),
        "torch_cpu": torch.tensor([1, 2], dtype=torch.uint8),
        "torch_device": [torch.tensor([3, 4], dtype=torch.uint8)],
    }
    changed_rng = copy.deepcopy(rng)
    learners = {
        "baseline": SimpleNamespace(
            checkpoint=lambda: state,
            named_parameters=lambda: [("weight", SimpleNamespace(grad=gradient))],
        ),
        "optimized": SimpleNamespace(
            checkpoint=lambda: changed,
            named_parameters=lambda: [
                ("weight", SimpleNamespace(grad=changed_gradient))
            ],
        ),
    }
    states = {"baseline": rng, "optimized": changed_rng}
    result = benchmark["qualify_learners"](learners, states)
    assert result["exact"] and len(result["snapshot_sha256"]) == 64
    if difference == "checkpoint":
        changed["model"]["weight"][0] += 1
    elif difference == "optimizer":
        changed["optimizer"]["state"][0]["exp_avg"][0] += 1
    elif difference == "gradient":
        changed_gradient[0] += 1
    elif difference == "python_rng":
        changed_rng["python"] = (3, (1, 3), None)
    elif difference == "cpu_rng":
        changed_rng["torch_cpu"][0] += 1
    else:
        changed_rng["torch_device"][0][0] += 1
    with pytest.raises(RuntimeError, match="qualification differs exactly"):
        benchmark["qualify_learners"](learners, states)


def test_ready_without_final_qualification_is_not_acceptance():
    with pytest.raises(RuntimeError, match="exact final state qualification"):
        benchmark["run_workers"](
            {"workers": 1, "timeout": 5, "background": []}, target=unqualified_worker
        )


@pytest.fixture
def options(tmp_path):
    baseline, batches = tmp_path / "baseline.py", tmp_path / "batches.npz"
    baseline.write_text("# immutable baseline\n")
    batches.write_bytes(b"frozen batches")
    batches.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "rlpd",
                "phase": "cached-training-update",
                "sha256": benchmark["digest"](batches),
                "sources": [
                    {
                        "kind": "SACsym proxy plus training-only offline",
                        "sha256": "a" * 64,
                    }
                ],
            }
        )
    )
    return benchmark["parser"]().parse_args(
        [
            "--baseline",
            str(baseline),
            "--batches",
            str(batches),
            "--device",
            "cuda",
            "--method",
            "rlpd",
            "--label",
            "background-rlpd",
            "--output",
            str(tmp_path / "out"),
        ]
    )


def test_parent_does_not_import_torch():
    code = "import runpy,sys; runpy.run_path(sys.argv[1]); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code, str(SCRIPT)], check=True, timeout=5)


@pytest.mark.parametrize("device", ["mps", "cuda"])
@pytest.mark.parametrize("optimization", ["metrics", "foreach", "both"])
def test_benchmark_variants_isolate_metrics_and_target_updates(device, optimization):
    from types import SimpleNamespace

    baseline_loop, optimized_loop, grouped = object(), object(), object()
    baseline = SimpleNamespace(soft_update=baseline_loop)
    optimized = SimpleNamespace(soft_update=optimized_loop, foreach_soft_update=grouped)
    settings = benchmark["configure_variant"](optimized, baseline, device, optimization)
    selected = (
        optimized.foreach_soft_update if device == "cuda" else optimized.soft_update
    )
    if optimization == "metrics":
        assert selected is (baseline_loop if device == "cuda" else optimized_loop)
    else:
        assert selected is grouped
    assert baseline.soft_update is baseline_loop
    assert settings == benchmark["variant_settings"](device, optimization)
    assert settings["target_update"]["optimized"] == (
        "loop" if optimization == "metrics" else "grouped-foreach"
    )
    assert settings["metrics_reporting"]["optimized"] == (
        "each-update" if optimization == "foreach" else "32-action-chunk-last"
    )


@pytest.mark.parametrize(
    "selector,expected", [("cpu", "loop"), ("mps", "loop"), ("cuda", "foreach")]
)
def test_production_default_dispatch_uses_device_selector_with_only_cpu_tensors(
    monkeypatch, selector, expected
):
    from types import SimpleNamespace

    import torch
    from rl_mind.core import Action

    from kartstudy import algorithms

    learner = algorithms.SAC(2, hidden=8, device="cpu")
    learner.device = torch.device(selector)
    batch = SimpleNamespace(
        obs=torch.zeros(4, 2),
        action=Action(torch.zeros(4, 5)),
        next_obs=torch.zeros(4, 2),
        reward=torch.zeros(4),
        terminated=torch.zeros(4, dtype=torch.bool),
    )
    # Test only the dispatch selector; never transfer the CPU fixture to a GPU.
    batch._map = lambda transform: batch
    calls = []
    loop = algorithms.soft_update
    monkeypatch.setattr(
        algorithms, "soft_update", lambda *args: calls.append("loop") or loop(*args)
    )
    monkeypatch.setattr(
        algorithms,
        "foreach_soft_update",
        lambda *args: calls.append("foreach") or loop(*args),
    )
    learner.update(batch)
    assert calls == [expected]
    assert all(parameter.device.type == "cpu" for parameter in learner.parameters())


def test_provenance_and_budget_validation(options):
    assert benchmark["validate"](options)["method"] == "rlpd"
    assert options.repeats == 5
    options.block_actions = 33
    with pytest.raises(ValueError, match="multiples of 32"):
        benchmark["validate"](options)
    options.block_actions = 32
    options.batches.write_bytes(b"changed batches")
    with pytest.raises(ValueError, match="provenance"):
        benchmark["validate"](options)


def test_profile_worker_limit_and_sac_repeat_minimum(options):
    options.profile, options.workers = True, 2
    with pytest.raises(ValueError, match="exactly one"):
        benchmark["validate"](options)
    options.profile, options.workers, options.method, options.repeats = (
        False,
        1,
        "sac",
        5,
    )
    with pytest.raises(ValueError, match="10–16"):
        benchmark["validate"](options)


def test_fixed_batches_require_normalized_controls_and_finite_float32():
    np = pytest.importorskip("numpy")
    arrays = {
        "obs": np.zeros((2, 256, 84), dtype=np.float32),
        "next_obs": np.zeros((2, 256, 84), dtype=np.float32),
        "actions": np.zeros((2, 256, 5), dtype=np.float32),
        "reward": np.zeros((2, 256), dtype=np.float32),
        "terminated": np.zeros((2, 256), dtype=bool),
    }
    benchmark["validate_arrays"](arrays)
    arrays["actions"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="normalized training replay"):
        benchmark["validate_arrays"](arrays)
    arrays["actions"][0, 0, 0] = 0
    arrays["obs"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite float32"):
        benchmark["validate_arrays"](arrays)


def test_spawned_cpu_fixtures_complete_and_aggregate():
    blocks, background, runtime = benchmark["run_workers"](
        {"workers": 2, "timeout": 5, "background": []}, target=fake_worker
    )
    result = benchmark["aggregate"](blocks, 2, 2)
    assert result["rates"]["baseline"] == [32, 32] and result["median_speedup"] == 1
    assert len(runtime) == 2 and background
    with pytest.raises(ValueError, match="Missing or duplicated"):
        benchmark["aggregate"](blocks[:-1], 2, 2)


@pytest.mark.parametrize(
    "target,timeout,message",
    [
        (failed_worker, 5, "fixture failure"),
        (blocked_worker, 0.2, "overall timeout"),
    ],
)
def test_failure_or_timeout_reaps_owned_workers(target, timeout, message):
    previous = {process.pid for process in multiprocessing.active_children()}
    start = time.monotonic()
    with pytest.raises((RuntimeError, TimeoutError), match=message):
        benchmark["run_workers"](
            {"workers": 1, "timeout": timeout, "background": []}, target=target
        )
    assert time.monotonic() - start < 6
    assert {process.pid for process in multiprocessing.active_children()} == previous


def test_background_evaluation_crossings_are_compact(tmp_path):
    job = tmp_path / "shared_rlpd_s1"
    job.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"jobs": [{"name": job.name, "state": "running", "output": str(job)}]}
        )
    )
    (job / "status.json").write_text(
        json.dumps(
            {
                "step": 40000,
                "phase": "training",
                "updates": 160000,
                "evaluation_steps": 5000,
            },
            indent=2,
        )
    )
    evaluation = job / "evaluations.jsonl"
    evaluation.write_text(json.dumps({"step": 40000, "episodes": ["omitted"]}) + "\n")
    first = benchmark["background_state"]([manifest])
    serialized = benchmark["background_state"]([str(manifest)])
    assert serialized[0]["manifest"] == first[0]["manifest"]
    assert serialized[0]["jobs"][0]["status"] == first[0]["jobs"][0]["status"]
    assert first[0]["jobs"][0]["status"]["step"] == 40000
    assert first[0]["jobs"][0]["status"]["updates"] == 160000
    evaluation.write_text(json.dumps({"step": 80000, "episodes": ["omitted"]}) + "\n")
    second = benchmark["background_state"]([manifest])
    assert "episodes" not in first[0]["jobs"][0]["last_evaluation"]
    crossings = benchmark["evaluation_crossings"](
        [{"at": 1, "background": first}, {"at": 2, "background": second}]
    )
    assert len(crossings) == 1 and crossings[0]["at"] == 2


def test_queue_estimate_uses_the_same_six_measured_jobs():
    jobs = [
        {"name": f"{method}{seed}", "method": method, "seed": seed, "seconds": seconds}
        for seed in (1, 2)
        for method, seconds in (("sac", 1), ("sac_sym", 2), ("rlpd", 4))
    ]
    result = benchmark["queue_replay"](jobs)
    assert result["phase"] == "shape-limited-queue-estimate"
    assert result["strategies"]["existing"]["makespan_seconds"] == 9
    assert result["strategies"]["long-first"]["makespan_seconds"] == 7
    for strategy in result["strategies"].values():
        assert {row["name"]: row["seconds"] for row in strategy["jobs"]} == {
            row["name"]: row["seconds"] for row in jobs
        }
    with pytest.raises(ValueError, match="six"):
        benchmark["queue_replay"](jobs[:-1])


def test_cli_refuses_overwrite_before_spawning(options):
    options.output.mkdir()
    sentinel = options.output / "retained.json"
    sentinel.write_text("original")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(options.baseline),
            "--batches",
            str(options.batches),
            "--device",
            "cuda",
            "--method",
            "rlpd",
            "--label",
            "test",
            "--output",
            str(options.output),
        ],
        check=False,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 2 and sentinel.read_text() == "original"
