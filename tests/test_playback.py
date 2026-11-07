import json

import pytest
import torch

from kartstudy.algorithms import Policy
from kartstudy.environment import SCHEMA_VERSION
from kartstudy.playback import discover_runs, load_actor
from kartstudy.tracks import ALL_TRACKS


def test_playback_catalog_requires_completed_checkpoint_and_unique_training_run(
    tmp_path,
):
    config = {
        "algo": "dagger",
        "seed": 2,
        "demo_episodes": 20,
        "tracks": ["stk_enterprise"],
        "schema_version": SCHEMA_VERSION,
    }
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(config))
    (run / "status.json").write_text(json.dumps({"status": "running"}))
    (run / "checkpoint.pt").touch()
    assert discover_runs(tmp_path) == {}
    (run / "status.json").write_text(json.dumps({"status": "completed"}))
    assert discover_runs(tmp_path) == {
        (track, "dagger", 2): run for track in ALL_TRACKS
    }
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
    # A portable actor-only file must work without replay/optimizer objects.
    torch.save(
        {"actor": actor.state_dict(), "config": config, "step": 50000},
        tmp_path / "actor.pt",
    )
    (tmp_path / "checkpoint.pt").unlink()
    lightweight, _, _ = load_actor(tmp_path)
    torch.testing.assert_close(lightweight.act(obs), actor.act(obs), atol=0, rtol=0)


def test_selected_policies_hide_extra_runs_without_removing_checkpoints(tmp_path):
    for seed in (0, 1):
        run = tmp_path / f"shared_dagger_s{seed}"
        run.mkdir()
        (run / "config.json").write_text(
            json.dumps(
                {
                    "algo": "dagger",
                    "seed": seed,
                    "schema_version": SCHEMA_VERSION,
                }
            )
        )
        (run / "status.json").write_text(json.dumps({"status": "completed"}))
        (run / "checkpoint.pt").touch()
    (tmp_path / "selected-policies.json").write_text(
        json.dumps({"dagger": "shared_dagger_s0"})
    )
    catalog = discover_runs(tmp_path)
    assert {key[2] for key in catalog} == {0}
    assert (tmp_path / "shared_dagger_s1" / "checkpoint.pt").exists()


def test_remote_timeout_is_bounded_and_closes_stalled_worker():
    from types import SimpleNamespace

    from kartstudy.playback import RemotePlayback

    playback = RemotePlayback.__new__(RemotePlayback)
    closed = []
    playback.connection = SimpleNamespace(
        send=lambda request: None, poll=lambda timeout: False
    )
    playback.close = lambda: closed.append(True)
    with pytest.raises(RuntimeError, match="Reset simulator"):
        playback._request("start", {})
    assert closed == [True]
