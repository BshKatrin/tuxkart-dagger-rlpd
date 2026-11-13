import numpy as np

from kartstudy.environment import (
    DrivingEnv,
    decode_action,
    item_features,
    normalize_action,
)
from kartstudy.expert import Expert


def test_original_expert_recovers_and_resets_episode_state():
    expert = Expert()
    observation = {
        "paths_start": [np.array([0.0, 0.0, z]) for z in range(0, 80, 10)],
        "paths_end": [np.array([0.0, 0.0, z]) for z in range(10, 90, 10)],
        "distance_down_track": np.array([20.0]),
        "velocity": np.zeros(3),
        "energy": np.zeros(1),
        "paths_width": [np.array([10.0])],
        "center_path_distance": np.zeros(1),
        "items_position": [],
        "items_type": [],
        "karts_position": [],
    }
    expert.reset(observation)
    first = expert.act(observation)
    actions = [expert.act(observation) for _ in range(35)]
    assert any(action["brake"] and action["acceleration"] == 0 for action in actions)
    expert.reset(observation)
    assert expert.act(observation) == first


def test_original_sources_match_pinned_manifest():
    import hashlib
    import json
    from pathlib import Path

    import kartstudy._team4

    root = Path(kartstudy._team4.__file__).parent
    manifest = json.loads((root / "manifest.json").read_text())
    for name, hashes in manifest.items():
        assert (
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            == hashes["local_sha256"]
        )


def test_five_controls_preserve_active_expert_controls_and_disable_drift_rescue():
    physical = {
        "steer": 0.25,
        "acceleration": 0.55,
        "brake": False,
        "drift": True,
        "nitro": True,
        "rescue": True,
        "fire": True,
    }
    decoded = decode_action(normalize_action(physical))
    assert np.isclose(decoded["steer"][0], 0.25)
    assert np.isclose(decoded["acceleration"][0], 0.55)
    assert not decoded["brake"]
    assert all(decoded[key] for key in ("nitro", "fire"))
    assert decoded["drift"] == decoded["rescue"] == 0
    assert len(normalize_action(physical)) == 5
    assert not any(
        decode_action(np.zeros(5))[key] for key in ("brake", "nitro", "fire")
    )


def test_real_solo_lap_reward_and_cached_stateful_expert():
    env = DrivingEnv(track="zengarden")
    try:
        obs, _ = env.reset(seed=20000)
        assert obs.shape == (84,)
        assert env.expert_action().shape == (5,)
        preserved = obs.copy()
        total = 0.0
        for _ in range(1500):
            np.testing.assert_array_equal(env.expert_action(), env.expert_action())
            nxt, reward, done, trunc, info = env.step(env.expert_action())
            total += reward
            if done or trunc:
                break
        np.testing.assert_array_equal(obs, preserved)
        assert done and not trunc
        assert np.isclose(total, info["distance"] / 10 - 0.1 * env.steps + 10)
        assert info["finish_time"] > 0
        assert np.isclose(info["measurement_seconds"], info["finish_time"])
        assert 0 <= info["off_track_fraction"] <= 1
        assert np.isfinite(nxt).all()
    finally:
        env.close()


def test_native_rescue_counter_counts_one_animation_not_active_frames(monkeypatch):
    # Inject one native rescue to test telemetry; it remains unavailable to policies.
    import kartstudy.environment as module

    def injected_rescue(action):
        physical = decode_action(action)
        physical["rescue"] = int(step == 60)
        return physical

    monkeypatch.setattr(module, "decode_action", injected_rescue)
    env = DrivingEnv()
    try:
        env.reset(seed=0)
        active_steps = 0
        for step in range(100):
            action = env.expert_action()
            _, _, _, _, info = env.step(action)
            active_steps += info["rescue_active"]
        assert active_steps > 1
        assert info["rescue_count"] == 1
    finally:
        env.close()


def test_polar_hazards_filter_sort_by_forward_distance_and_pad_absence():
    raw = {
        "items_position": [
            [2, 0, 0.1],
            [0, 0, 0.2],
            [0, 0, 0.3],
            [0, 0, 0],
            [2.1, 0, 0],
            [0, 0, -1],
            [0, 0, 18],
        ],
        "items_type": [1, 4, 1, 0, 1, 1, 4],
        "center_path_distance": [0],
        "paths_width": [[4]],
    }
    slots = item_features(raw).reshape(2, 3)
    np.testing.assert_allclose(
        slots[0], [np.hypot(2, 0.1) / np.sqrt(293), np.arctan2(2, 0.1) / (np.pi / 2), 1]
    )
    np.testing.assert_allclose(slots[1], [0.2 / np.sqrt(293), 0, 1])
    # With the road shifted, the first hazard falls outside its edge.
    raw["center_path_distance"] = [1]
    assert item_features(raw)[1] == 0
    raw["items_position"], raw["items_type"] = [[0, 0, 0]], [1]
    np.testing.assert_array_equal(item_features(raw), [0, 0, 1, 0, 0, 0])
    raw["items_position"], raw["items_type"] = [], []
    np.testing.assert_array_equal(item_features(raw), np.zeros(6))


def test_real_track_switch_preserves_one_simulator_and_resets_expert():
    env = DrivingEnv()
    try:
        first, _ = env.reset(seed=0)
        native = env.env.unwrapped
        second, _ = env.reset(seed=0, track="minigolf")
        assert env.env.unwrapped is native
        assert env.track == "minigolf"
        assert native.config.track == "minigolf"
        assert first.shape == second.shape == (84,)
        assert not np.array_equal(first, second)
    finally:
        env.close()


def test_disturbance_recovery_requires_sustained_progress_and_separates_native_rescue():
    from kartstudy.disturbance import Recovery

    def sample(progress, on_road=True, rescues=0):
        return {
            "progress": progress,
            "is_on_road": on_road,
            "rescue_active": False,
            "rescue_count": rescues,
        }

    recovery = Recovery(100, 0)
    for i in range(20):
        recovery.update(sample(100 + i), i * 0.1)
    assert recovery.time is None
    recovery.update(sample(120), 2.0)
    assert recovery.time == 2.0
    # A native rescue must disqualify subsequent driving as independent recovery.
    assisted = Recovery(100, 0)
    assisted.update(sample(100, rescues=1), 0.1)
    for i in range(30):
        assisted.update(sample(200 + i, rescues=1), 0.2 + i * 0.1)
    assert assisted.time is None and assisted.assisted
    # A slow/stationary kart and interrupted road contact do not qualify.
    stalled = Recovery(100, 0)
    for i in range(30):
        stalled.update(sample(100), i * 0.1)
    assert stalled.time is None
    stalled.update(sample(100, on_road=False), 3.0)
    assert not stalled.window


def test_real_disturbance_replays_identical_state_and_preserves_expert_history():
    from kartstudy.critic_mc import freeze_states
    from kartstudy.disturbance import trial

    env = DrivingEnv(track="lighthouse")
    scenario = {
        "track": "lighthouse",
        "eval_seed": 20000,
        "anchor_m": 2,
        "pulses": [
            {
                "steps": 30,
                "steer": -1,
                "throttle": 1,
                "brake": -1,
                "nitro": -1,
                "fire": -1,
            }
        ],
    }
    try:
        expert = trial(env, scenario)
        witness = trial(
            env,
            scenario,
            "scripted",
            correction=[{"steps": 15, "action": [0, -1, 1, -1, -1]}],
        )
        states = freeze_states(env, expert)
        assert states["perturbed"]["signature"] == expert["handover"]
        assert states["ordinary"]["signature"] != states["perturbed"]["signature"]
        assert (
            states["ordinary"]["prefix_steps"]
            == states["perturbed"]["prefix_steps"]
            == 57
        )
        assert (
            states["ordinary"]["signature"]["native_time"]
            == states["perturbed"]["signature"]["native_time"]
        )
    finally:
        env.close()
    assert expert["handover"] == witness["handover"]
    assert expert["anchor"] == witness["anchor"]
    assert expert["response_steps"] + expert["handover_steps"] == 1500
    assert not expert["success"] and expert["recovery_seconds"] is None
    assert expert["native_rescues_after_handover"] == 0
    assert witness["success"] and witness["recovered_within_15s"]
    assert witness["native_rescues_after_handover"] == 0


def test_temporary_control_delay_uses_shared_history_and_restores_current_control():
    from kartstudy.disturbance import ControlDelay

    history = [np.array([v, 1, -1, -1, -1], np.float32) for v in [0.1, 0.2, 0.3]]
    delay = ControlDelay(history, steps=2, duration_steps=3)
    requests = [np.array([v, 1, -1, -1, -1], np.float32) for v in [0.4, 0.5, 0.6, 0.7]]
    executed = [delay.execute(action, index) for index, action in enumerate(requests)]
    np.testing.assert_allclose([a[0] for a in executed], [0.2, 0.3, 0.4, 0.7])
    np.testing.assert_allclose([a[0] for a in history], [0.1, 0.2, 0.3])
    np.testing.assert_allclose([a[0] for a in requests], [0.4, 0.5, 0.6, 0.7])


def test_real_delayed_controls_record_requests_execution_and_restoration(tmp_path):
    from kartstudy.algorithms import Policy
    from kartstudy.critic_mc import continuation, freeze_states
    from kartstudy.disturbance import trial

    env = DrivingEnv(track="hacienda")
    scenario = {
        "track": "hacienda",
        "eval_seed": 20000,
        "anchor_m": 100,
        "pulses": [],
        "control_delay": {"steps": 3, "duration_steps": 6},
    }
    try:
        result = trial(env, scenario, save=tmp_path / "delay")
        states = freeze_states(env, result)
        assert states["perturbed"]["signature"] == result["delay_release"]
        assert states["perturbed"]["signature"] != result["handover"]
        assert (
            states["ordinary"]["prefix_steps"]
            == states["perturbed"]["prefix_steps"]
            == result["handover_steps"] + 6
        )
        assert (
            states["ordinary"]["signature"]["native_time"]
            == states["perturbed"]["signature"]["native_time"]
        )
        actor = Policy(84, hidden=8).eval()
        _, trace = continuation(
            env,
            states["perturbed"],
            20000,
            actor,
            np.array([0, -1, -1, -1, -1], np.float32),
            gamma=0.99,
            alpha=0.01,
            noise_seed=41000,
            horizon=2,
        )
        np.testing.assert_array_equal(
            trace["obs"][0], np.asarray(result["delay_release"]["obs"], np.float32)
        )
    finally:
        env.close()
    with np.load(tmp_path / "delay.npz") as trace:
        start = np.flatnonzero(trace["phase"] == "response")[0]
        np.testing.assert_array_equal(
            trace["action"][start : start + 6],
            trace["requested_action"][start - 3 : start + 3],
        )
        np.testing.assert_array_equal(
            trace["action"][start + 6 :], trace["requested_action"][start + 6 :]
        )
        assert trace["delay_active"].sum() == 6
    assert abs(result["recovery_start_seconds"] - 0.6) < 1e-4
    assert result["delay_release"] is not None
    assert result["recovery_seconds"] is None or result["recovery_seconds"] >= 2
