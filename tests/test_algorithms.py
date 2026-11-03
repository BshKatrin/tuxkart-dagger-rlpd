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
    learner = SAC(7, hidden=16, device=device, ensemble=ensemble, layer_norm=layer_norm)
    old = copy.deepcopy(learner.actor.state_dict())
    metrics = learner.update(transitions())
    assert all(math.isfinite(x) for x in metrics.values())
    assert any(
        not torch.equal(value, old[key])
        for key, value in learner.actor.state_dict().items()
    )
    assert all(p.grad is None for p in learner.target_critics.parameters())
    assert all(p.grad is None for p in learner.critics.parameters())
    clone = SAC(7, hidden=16, device=device, ensemble=ensemble, layer_norm=layer_norm)
    clone.load_checkpoint(learner.checkpoint())
    assert clone.updates == learner.updates
    for a, b in zip(learner.parameters(), clone.parameters()):
        torch.testing.assert_close(a, b)


def test_saturated_policy_density_stays_finite_and_imitation_learns():
    torch.manual_seed(4)
    actor = Policy(7, hidden=32)
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
