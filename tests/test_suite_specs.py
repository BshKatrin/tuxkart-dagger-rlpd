import runpy
from pathlib import Path

import pytest

specs = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/run_suite.py")
)["experiment_specs"]


def test_selected_replications_keep_existing_budgets_and_update_schedules():
    jobs = specs("/data/demonstrations.npz", ("sac", "sac_sym", "rlpd"), (1, 2))
    assert len(jobs) == 6
    assert [name for name, _ in jobs] == [
        f"shared_{m}_s{s}" for s in (1, 2) for m in ("sac", "sac_sym", "rlpd")
    ]
    for _, config in jobs:
        assert config["steps"] == 400000 and config["steps_per_track"] == 100000
        assert config["dataset"] == "/data/demonstrations.npz"
        assert config["utd"] == (4 if config["algo"] == "rlpd" else 1)
        assert config["eval_every"] == 40000 and config["demo_episodes"] == 20


def test_cuda_suite_changes_only_the_explicit_device():
    metal = specs(methods=("sac", "sac_sym", "rlpd"), seeds=(1,))
    cuda = specs(methods=("sac", "sac_sym", "rlpd"), seeds=(1,), device="cuda")
    assert cuda == [(name, config | {"device": "cuda"}) for name, config in metal]
    with pytest.raises(ValueError, match="no automatic CPU fallback"):
        specs(device="cpu")


@pytest.mark.parametrize(
    "methods,seeds",
    [
        ((), (1,)),
        (("sac", "sac"), (1,)),
        (("bad",), (1,)),
        (("sac",), (1, 1)),
        (("sac",), (-1,)),
    ],
)
def test_ambiguous_or_invalid_replication_selection_rejected(methods, seeds):
    with pytest.raises(ValueError):
        specs(methods=methods, seeds=seeds)


def test_long_first_changes_only_queue_order_for_the_same_six_jobs():
    existing = specs(methods=("sac", "sac_sym", "rlpd"), seeds=(1, 2))
    priority = specs(
        methods=("sac", "sac_sym", "rlpd"), seeds=(1, 2), queue_order="long-first"
    )
    assert [name for name, _ in priority] == [
        f"shared_{method}_s{seed}"
        for method in ("rlpd", "sac_sym", "sac")
        for seed in (1, 2)
    ]
    assert dict(priority) == dict(existing)
    assert specs(queue_order="existing") == specs()
    with pytest.raises(ValueError, match="Queue order"):
        specs(queue_order="random")


def test_long_first_ties_use_seed_and_preserve_remaining_method_order():
    jobs = specs(
        methods=("dagger", "bc", "sac"), seeds=(2, 1), queue_order="long-first"
    )
    assert [name for name, _ in jobs] == [
        "shared_sac_s1",
        "shared_sac_s2",
        "shared_dagger_s1",
        "shared_bc_s1",
        "shared_dagger_s2",
        "shared_bc_s2",
    ]
