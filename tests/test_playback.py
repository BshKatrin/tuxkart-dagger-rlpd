import json

import pytest
import torch

from kartstudy.algorithms import Policy
from kartstudy.environment import SCHEMA_VERSION
from kartstudy.playback import discover_runs, load_actor
from kartstudy.tracks import ALL_TRACKS


def test_critic_diagnostic_rejects_a_different_actor_at_the_same_seed(tmp_path):
    from kartstudy.algorithms import SAC
    from kartstudy.critic_artifacts import export_checkpoint
    from kartstudy.critic_mc import load_critic_checkpoint

    config = {"algo": "sac", "seed": 0, "hidden": 8, "schema_version": 3, "gamma": 0.99}
    learner = SAC(84, device="cpu", hidden=8)
    checkpoint = tmp_path / "checkpoint.pt"
    portable = tmp_path / "actor.pt"
    actor_weights = {
        name: value.clone() for name, value in learner.actor.state_dict().items()
    }
    torch.save(
        {
            "config": config,
            "step": 400000,
            "actor": actor_weights,
            "learner": {"model": learner.state_dict(), "optimizers": {}},
            "replay": torch.arange(1000),
        },
        checkpoint,
    )
    torch.save({"config": config, "step": 400000, "actor": actor_weights}, portable)
    actor, critics, _, _, alpha = load_critic_checkpoint(checkpoint, portable)
    torch.testing.assert_close(actor.mean.bias, learner.actor.mean.bias)
    assert len(critics) == 2 and alpha == pytest.approx(0.1)
    frozen = tmp_path / "evaluation.pt"
    export_checkpoint(checkpoint, portable, frozen)
    exported = torch.load(frozen, weights_only=False)
    assert "replay" not in exported and set(exported["learner"]) == {"model"}
    restored_actor, restored_critics, _, _, restored_alpha = load_critic_checkpoint(frozen, portable)
    torch.testing.assert_close(restored_actor.mean.bias, actor.mean.bias, atol=0, rtol=0)
    for before, after in zip(critics.parameters(), restored_critics.parameters(), strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert restored_alpha == alpha
    with pytest.raises(FileExistsError):
        export_checkpoint(checkpoint, portable, frozen)
    actor_weights["mean.bias"][0] += 1
    torch.save({"config": config, "step": 400000, "actor": actor_weights}, portable)
    with pytest.raises(ValueError, match="does not match"):
        load_critic_checkpoint(checkpoint, portable)


def test_critic_release_resolves_restored_paths_before_existing_originals(tmp_path):
    from kartstudy.critic_artifacts import INPUTS, resolve_artifact

    original = tmp_path / "original"
    original.mkdir()
    actor = original / "actor.pt"
    actor.write_bytes(b"original")
    restored = tmp_path / "restored"
    index = restored / INPUTS / "manifest.json"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({
        "original_repository_root": str(original),
        "path_map": {str(actor): "runs/frozen/actor.pt"},
    }))
    assert resolve_artifact(actor, restored) == restored / "runs/frozen/actor.pt"
    assert resolve_artifact(original / "runs/ref.json", restored) == restored / "runs/ref.json"
    assert resolve_artifact("runs/new.json", restored) == restored / "runs/new.json"
    with pytest.raises(ValueError, match="escapes"):
        resolve_artifact("../outside.pt", restored)


def test_playback_offers_nonstudy_tracks_with_learned_policies(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pystk2

    from kartstudy import playback, viewer

    tracks = ["lighthouse", "zengarden", "snowtuxpeak"]
    assert "zengarden" not in ALL_TRACKS
    monkeypatch.setattr(pystk2, "list_tracks", lambda mode: tracks)
    player = playback.Playback(tmp_path)
    player.catalog = {("lighthouse", "dagger", 0): tmp_path}
    actor = object()
    player.actors[tmp_path] = (actor, {}, 100)
    resets = []

    def reset(**kwargs):
        resets.append(kwargs)
        return None, {}

    player.env = SimpleNamespace(reset=reset)
    player.start(method="dagger", track="zengarden", render=False)
    assert set(dict(viewer.choices(player, "track"))) == set(tracks)
    assert "dagger" in dict(viewer.choices(player, "method"))
    assert viewer.choices(player, "train_seed") == [(0, "Training seed 0")]
    viewer.select(player, "track", "snowtuxpeak")
    assert player.method == "dagger"
    assert player.actor is actor
    assert resets[-1]["track"] == "snowtuxpeak"


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
    exact, _, _ = load_actor(tmp_path / "checkpoint.pt")
    torch.testing.assert_close(exact.act(obs), actor.act(obs), atol=0, rtol=0)
    # A portable actor-only file must work without replay/optimizer objects.
    torch.save(
        {"actor": actor.state_dict(), "config": config, "step": 50000},
        tmp_path / "actor.pt",
    )
    (tmp_path / "checkpoint.pt").unlink()
    lightweight, _, _ = load_actor(tmp_path)
    torch.testing.assert_close(lightweight.act(obs), actor.act(obs), atol=0, rtol=0)
    exact, _, _ = load_actor(tmp_path / "actor.pt")
    torch.testing.assert_close(exact.act(obs), actor.act(obs), atol=0, rtol=0)


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
        (run / "actor.pt").touch()
    (tmp_path / "selected-policies.json").write_text(
        json.dumps({"dagger": "shared_dagger_s0"})
    )
    catalog = discover_runs(tmp_path)
    assert {key[2] for key in catalog} == {0}
    assert (tmp_path / "shared_dagger_s1" / "actor.pt").exists()


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


def test_viewer_steps_do_not_toggle_controls_or_status(monkeypatch):
    import os
    import subprocess
    import sys
    from types import SimpleNamespace

    # Native simulator tests initialize Cocoa/SDL. Tk must use a fresh process.
    if os.environ.get("KARTSTUDY_TK_TEST_CHILD") != "1":
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_viewer_steps_do_not_toggle_controls_or_status",
            ],
            env=os.environ | {"KARTSTUDY_TK_TEST_CHILD": "1"},
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return

    tk = pytest.importorskip("tkinter")
    from tkinter import ttk

    from PIL import Image

    from kartstudy import viewer

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display unavailable")
    root.withdraw()
    monkeypatch.setattr(tk, "Tk", lambda: root)
    updates, messages = [], []
    original_configure = ttk.Combobox.configure
    original_set = tk.StringVar.set

    def configure(self, *args, **kwargs):
        updates.append(kwargs.copy())
        return original_configure(self, *args, **kwargs)

    def set_text(self, value):
        messages.append(value)
        return original_set(self, value)

    monkeypatch.setattr(ttk.Combobox, "configure", configure)
    monkeypatch.setattr(tk.StringVar, "set", set_text)
    monkeypatch.setattr(
        viewer, "annotated_frame", lambda playback: Image.new("RGB", (640, 360))
    )
    playback = SimpleNamespace(
        method="dagger",
        track="lighthouse",
        train_seed=0,
        eval_seed=20000,
        checkpoint_step=400000,
        tracks=["lighthouse"],
        catalog={("lighthouse", "dagger", 0): None},
        done=False,
        steps=0,
        close=lambda: None,
    )

    def step():
        playback.steps += 1

    playback.step = step
    viewer.show(playback, quit_after_steps=3)
    assert playback.steps >= 3
    assert not any("state" in update for update in updates)
    assert sum("values" in update for update in updates) == 2  # Initial selectors only.
    assert len(messages) == 1  # Stable status throughout playback.


def test_selection_can_register_a_separate_ablation(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    run = tmp_path / "ablation" / "sac_sym_s0"
    run.mkdir(parents=True)
    (run / "config.json").write_text(
        json.dumps({"algo": "sac_sym", "seed": 0, "schema_version": SCHEMA_VERSION})
    )
    (run / "status.json").write_text(json.dumps({"status": "completed"}))
    (run / "actor.pt").touch()
    (root / "selected-policies.json").write_text(
        json.dumps({"sac_sym": "../ablation/sac_sym_s0"})
    )
    catalog = discover_runs(root)
    assert catalog[("lighthouse", "sac_sym", 0)] == run


def test_evolution_uses_real_snapshot_steps_and_checks_provenance(tmp_path):
    from kartstudy.trajectory_evolution import digest, frame_source, snapshot_sources

    config = {"hidden": 16, "algo": "sac", "seed": 1, "schema_version": SCHEMA_VERSION}
    actor = Policy(84, hidden=16)
    for step in [80000, 0, 40000]:
        directory = tmp_path / "snapshots" / str(step)
        directory.mkdir(parents=True)
        path = directory / "actor.pt"
        torch.save({"actor": actor.state_dict(), "config": config, "step": step}, path)
        (directory / "manifest.json").write_text(
            json.dumps(
                {"actor_file_sha256": digest(path), "step": step, "config": config}
            )
        )
    rows = snapshot_sources(tmp_path)
    assert [r["step"] for r in rows] == [0, 40000, 80000]
    assert [r["step"] for r in snapshot_sources(tmp_path, [40000])] == [40000]
    run = {"config": config, "snapshots": rows}
    assert frame_source(run, 60000) is None
    assert frame_source(run, 40000)["step"] == 40000
    run["config"] = {**config, "algo": "bc"}
    assert frame_source(run, 60000) is rows[0]
    path = tmp_path / "snapshots/40000/actor.pt"
    torch.save({"actor": actor.state_dict(), "config": config, "step": 60000}, path)
    with pytest.raises(ValueError, match="provenance mismatch"):
        snapshot_sources(tmp_path)


def test_evolution_breaks_rescue_teleports_and_rejects_changed_cache(tmp_path):
    import numpy as np

    from kartstudy.trajectory_evolution import digest, read_trace

    path = tmp_path / "trace.npz"
    np.savez(
        path,
        world_position=[[0, 0, 0], [1, 0, 1], [30, 0, 30], [31, 0, 31]],
        rescue_active=[False, True, False, False],
        rescue_count=[0, 1, 1, 1],
    )
    row = {
        "trajectory": path.name,
        "trajectory_sha256": digest(path),
        "episodes": [{"steps": 4}],
    }
    positions, driven, rescues = read_trace(tmp_path, row)
    assert np.isnan(driven[1]).all()
    np.testing.assert_array_equal(rescues, positions[[1]])
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Cached trajectory changed"):
        read_trace(tmp_path, row)
