import json

import pytest
import torch

from kartstudy.algorithms import Policy
from kartstudy.playback import discover_runs, load_actor


def test_playback_catalog_requires_completed_checkpoint_and_unique_training_run(
    tmp_path,
):
    config = {"algo": "dagger", "seed": 2, "demo_episodes": 20, "conservative": False}
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(config))
    (run / "status.json").write_text(json.dumps({"status": "running"}))
    (run / "checkpoint.pt").touch()
    assert discover_runs(tmp_path) == {}
    (run / "status.json").write_text(json.dumps({"status": "completed"}))
    assert discover_runs(tmp_path) == {("baseline", "dagger", 2): run}
    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    for name in ["config.json", "status.json", "checkpoint.pt"]:
        (duplicate / name).write_bytes((run / name).read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        discover_runs(tmp_path)


def test_loaded_checkpoint_preserves_deterministic_actions_and_architecture(tmp_path):
    actor = Policy(7, hidden=16).eval()
    config = {"hidden": 16, "algo": "dagger"}
    torch.save(
        {"actor": actor.state_dict(), "config": config, "step": 50000},
        tmp_path / "checkpoint.pt",
    )
    loaded, restored, step = load_actor(tmp_path)
    obs = torch.randn(10, 7)
    torch.testing.assert_close(loaded.act(obs), actor.act(obs), atol=0, rtol=0)
    assert restored == config and step == 50000
    assert not loaded.training
    assert next(loaded.parameters()).device.type == "cpu"
