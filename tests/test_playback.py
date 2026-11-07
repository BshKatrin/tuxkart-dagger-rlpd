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


def test_viewer_steps_do_not_toggle_controls_or_status():
    import subprocess
    import sys
    from pathlib import Path

    # Native simulator tests initialize SDL/Cocoa; Tk needs a fresh process.
    probe = f"""
import runpy
import sys
import pytest
namespace = runpy.run_path({str(Path(__file__).resolve())!r})
try:
    with pytest.MonkeyPatch.context() as monkeypatch:
        namespace['_assert_viewer_controls'](monkeypatch)
except pytest.skip.Exception:
    sys.exit(77)
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode == 77:
        pytest.skip("Tk display unavailable")
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_viewer_controls(monkeypatch):
    from types import SimpleNamespace

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
