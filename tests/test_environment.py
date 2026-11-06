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
