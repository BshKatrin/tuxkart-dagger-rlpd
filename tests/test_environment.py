import numpy as np

from kartstudy.environment import DrivingEnv, decode_action, normalize_action


def test_control_round_trip_preserves_expert_capabilities():
    physical = {"steer": 0.25, "acceleration": 0.55, "brake": False}
    decoded = decode_action(normalize_action(physical))
    assert np.isclose(decoded["steer"][0], 0.25)
    assert np.isclose(decoded["acceleration"][0], 0.55)
    assert not decoded["brake"]


def test_real_solo_lap_reward_and_progress_triggered_intervention():
    env = DrivingEnv(perturb=True)
    try:
        obs, _ = env.reset(seed=20000)
        preserved = obs.copy()
        total, overrides = 0.0, []
        for _ in range(1500):
            previous_progress = env.previous_distance
            nxt, reward, done, trunc, info = env.step(env.expert_action())
            total += reward
            if info["action_overridden"]:
                overrides.append(previous_progress)
            if done or trunc:
                break
        np.testing.assert_array_equal(obs, preserved)
        assert done and not trunc
        assert len(overrides) == 5
        assert overrides[0] >= env.perturb_distance
        assert np.isclose(total, info["distance"] / 10 - 0.1 * env.steps + 10)
        assert info["finish_time"] > 0
        assert np.isfinite(nxt).all()
    finally:
        env.close()
