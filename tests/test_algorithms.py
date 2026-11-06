import copy
import math

import pytest
import torch
from rl_mind.core import Action
from rl_mind.data import ReplayBuffer, Transitions

from kartstudy.algorithms import (
    SAC,
    Critic,
    Policy,
    bellman_target,
    ensemble_forward,
    fit_imitation,
    symmetric_sample,
)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
)
@pytest.mark.parametrize("layer_norm", [False, True])
def test_batched_ensemble_matches_outputs_and_parameter_action_gradients(
    device, layer_norm
):
    torch.manual_seed(31)
    critics = torch.nn.ModuleList([Critic(7, 3, 16, layer_norm) for _ in range(4)]).to(
        device
    )
    cloned = copy.deepcopy(critics)
    obs = torch.randn(13, 7, device=device)
    action = torch.randn(13, 3, device=device, requires_grad=True)
    cloned_action = action.detach().clone().requires_grad_(True)
    looped = torch.stack([critic(obs, action) for critic in critics])
    batched = ensemble_forward(cloned, obs, cloned_action)
    torch.testing.assert_close(batched, looped, atol=2e-6, rtol=2e-5)
    gradient = torch.randn_like(looped)
    looped.backward(gradient)
    batched.backward(gradient)
    torch.testing.assert_close(action.grad, cloned_action.grad, atol=2e-6, rtol=2e-5)
    for a, b in zip(critics.parameters(), cloned.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)


def transitions(n=16, marker=0.0):
    return Transitions(
        obs=torch.randn(n, 7) + marker,
        action=Action(torch.rand(n, 3) * 2 - 1),
        reward=torch.randn(n),
        next_obs=torch.randn(n, 7),
        terminated=torch.arange(n) % 2 == 0,
    )


def test_terminal_target_and_timeout_bootstrap():
    target = bellman_target(
        torch.tensor([1.0, 1.0]),
        torch.tensor([True, False]),
        torch.tensor([10.0, 10.0]),
        torch.tensor([-2.0, -2.0]),
        gamma=0.9,
        alpha=0.5,
    )
    torch.testing.assert_close(target, torch.tensor([1.0, 10.9]))


def test_symmetric_sampling_preserves_prior_fraction_after_online_growth():
    prior, online = ReplayBuffer(100), ReplayBuffer(1000)
    for buffer, count, value in [(prior, 10, -1.0), (online, 1000, 1.0)]:
        batch = transitions(count)
        buffer.add(
            Transitions(
                torch.full_like(batch.obs, value),
                batch.action,
                batch.reward,
                batch.next_obs,
                batch.terminated,
            )
        )
    batch = symmetric_sample(online, prior, 64)
    assert (batch.obs[:, 0] == -1).sum() == 32
    assert (batch.obs[:, 0] == 1).sum() == 32


@pytest.mark.parametrize(
    "device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
)
@pytest.mark.parametrize("ensemble,layer_norm", [(2, False), (10, True)])
def test_sac_updates_are_finite_and_targets_have_no_gradients(
    device, ensemble, layer_norm
):
    torch.manual_seed(0)
    learner = SAC(
        7,
        action_dim=3,
        hidden=16,
        device=device,
        ensemble=ensemble,
        layer_norm=layer_norm,
    )
    old = copy.deepcopy(learner.actor.state_dict())
    metrics = learner.update(transitions())
    assert all(math.isfinite(x) for x in metrics.values())
    assert any(
        not torch.equal(value, old[key])
        for key, value in learner.actor.state_dict().items()
    )
    assert all(p.grad is None for p in learner.target_critics.parameters())
    assert all(p.grad is None for p in learner.critics.parameters())
    clone = SAC(
        7,
        action_dim=3,
        hidden=16,
        device=device,
        ensemble=ensemble,
        layer_norm=layer_norm,
    )
    clone.load_checkpoint(learner.checkpoint())
    assert clone.updates == learner.updates
    for a, b in zip(learner.parameters(), clone.parameters()):
        torch.testing.assert_close(a, b)


def test_saturated_policy_density_stays_finite_and_imitation_learns():
    torch.manual_seed(4)
    actor = Policy(7, action_dim=3, hidden=32)
    obs = torch.randn(128, 7)
    target = torch.stack((obs[:, 0], -obs[:, 1], obs[:, 2]), -1).tanh()
    with torch.no_grad():
        initial = (actor.act(obs) - target).square().mean()
    fit_imitation(actor, obs, target, "cpu", updates=300, batch_size=64)
    assert (actor.act(obs) - target).square().mean() < initial / 4
    with torch.no_grad():
        actor.mean.bias.fill_(100)
    sample = actor(obs)
    assert torch.isfinite(sample.log_prob).all()
    sample.log_prob.mean().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters()
    )


@pytest.mark.parametrize("algo", ["bc", "dagger", "sac", "rlpd"])
def test_shared_training_enforces_track_quotas_and_excludes_validation_and_evaluation(
    tmp_path, monkeypatch, algo
):
    import json
    from types import SimpleNamespace

    import numpy as np

    from kartstudy import train
    from kartstudy.data import dataset_hash, load_transitions
    from kartstudy.environment import SCHEMA_VERSION
    from kartstudy.expert import Expert
    from kartstudy.tracks import HELD_OUT_TRACKS, track_group

    tracks = ["lighthouse", "stk_enterprise"]
    path = tmp_path / "demonstrations.npz"
    # Each track has one training episode and two validation episodes.
    observations = np.zeros((6, 84), np.float32)
    observations[[1, 2, 4, 5]] = 99  # A marker that must never reach fitting/replay.
    actions = np.zeros((6, 5), np.float32)
    np.savez(
        path,
        obs=observations,
        next_obs=observations,
        action=actions,
        expert_action=actions,
        reward=np.zeros(6, np.float32),
        terminated=np.ones(6, bool),
        episode=np.arange(6),
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "tracks": tracks,
        "expert": Expert.name,
        "source_commit": Expert.source_commit,
        "control_noise": False,
        "max_steps": 3,
        "sha256": dataset_hash(path),
        "train_episode_ids": [0, 3],
        "validation_episode_ids": [1, 2, 4, 5],
        "episodes": [
            {"episode": i, "track": tracks[i // 3], "success": True} for i in range(6)
        ],
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    assert len(load_transitions(path)) == 2
    assert not (load_transitions(path, 1).obs == 99).any()

    class Env:
        def __init__(self, track, max_steps):
            self.track = track
            self.env = SimpleNamespace(unwrapped=SimpleNamespace(default_track=track))

        def reset(self, seed, track=None):
            if track is not None:
                self.track = track
            self.n = 0
            return np.full(84, tracks.index(self.track), np.float32), {}

        def expert_action(self):
            return np.zeros(5, np.float32)

        def step(self, action):
            self.n += 1
            # Timeouts must bootstrap; the next track must not replace next_obs.
            return (
                np.full(84, tracks.index(self.track), np.float32),
                0.1,
                False,
                self.n == 3,
                {"elapsed_seconds": 0.1 * self.n, "distance": self.n},
            )

        def close(self):
            pass

    def evaluation(env, policy, seeds, device, trajectory_path=None):
        assert not policy.training
        # Deliberately change the simulator's state, like real evaluation does.
        assert env.track in tracks + list(HELD_OUT_TRACKS)
        return [{"seed": int(seed), "success": False, "steps": 1} for seed in seeds]

    original_fit = train.fit_imitation

    def fitting(actor, obs, actions, *args, **kwargs):
        assert not (obs == 99).any()
        return original_fit(actor, obs, actions, *args, **kwargs)

    monkeypatch.setattr(train, "DrivingEnv", Env)
    monkeypatch.setattr(train, "evaluate", evaluation)
    monkeypatch.setattr(train, "fit_imitation", fitting)
    config = train.defaults() | {
        "algo": algo,
        "device": "cpu",
        "tracks": tracks,
        "dataset": str(path),
        "demo_episodes": 1,
        "steps": 10,
        "steps_per_track": 5,
        "max_steps": 3,
        "hidden": 8,
        "batch_size": 4,
        "bc_updates": 2,
        "dagger_updates": 2,
        "dagger_round": 4,
        "collect_steps": 4,
        "eval_every": 6,
        "warmup": 0,
        "utd": 4 if algo == "rlpd" else 1,
    }
    out = tmp_path / "run"
    train.run(config, out)
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["track_steps"] == dict.fromkeys(tracks, 0 if algo == "bc" else 5)
    state = torch.load(out / "checkpoint.pt", weights_only=False)
    if algo in ("sac", "rlpd"):
        replay = state["replay"]._storage
        assert set(replay.obs[:, 0].tolist()) == {0, 1}
        assert torch.equal(replay.obs, replay.next_obs)
        assert not replay.terminated.any()
    evaluations = [
        json.loads(line)
        for line in (out / "evaluations.jsonl").read_text().splitlines()
    ]
    for row in evaluations:
        for episode in row["episodes"]:
            assert (episode["split"], episode["difficulty"]) == track_group(
                episode["track"]
            )
        if not row["final"]:
            assert {ep["track"] for ep in row["episodes"]} == set(tracks)
    assert {ep["track"] for ep in evaluations[-1]["episodes"]} == set(tracks) | set(
        HELD_OUT_TRACKS
    )
