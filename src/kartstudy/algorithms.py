"""Small SAC/RLPD and imitation learners built on ``rl_mind`` containers.

Actions and their densities live in [-1, 1]^action_dim. Physical control
conversion belongs to the environment wrapper. SAC uses a learned entropy
temperature, the mean ensemble Q for the actor (as in the RLPD reference),
and the minimum of two randomly selected target critics. Full RLPD's high
update-to-data ratio is controlled by the caller, not hidden in this module.
"""

from __future__ import annotations

import copy
import math
import random

import torch
from rl_mind.core import Actor, StochasticAction
from rl_mind.data import ReplayBuffer, Transitions
from rl_mind.nn import build_mlp, soft_update
from torch import Tensor, nn
from torch.nn import functional as F


class Policy(Actor[StochasticAction]):
    """Identical actor capacity for imitation learning and reward learning."""

    def __init__(self, obs_dim: int, action_dim: int = 5, hidden: int = 128):
        super().__init__()
        self.obs_dim, self.action_dim, self.hidden = obs_dim, action_dim, hidden
        self.encoder = build_mlp([obs_dim, hidden, hidden], output_activation=nn.ReLU())
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        nn.init.constant_(self.log_std.bias, -0.5)

    def act(self, obs: Tensor) -> Tensor:
        return self.mean(self.encoder(obs)).tanh()

    def forward(self, obs: Tensor) -> StochasticAction:
        features = self.encoder(obs)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-5.0, 2.0)
        noise = torch.randn_like(mean)
        pre_tanh = mean + log_std.exp() * noise
        action = pre_tanh.tanh()
        # Stable tanh Jacobian, including when tanh saturates at +/-1.
        log_density = -0.5 * (noise.square() + math.log(2.0 * math.pi)) - log_std
        log_jacobian = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        return StochasticAction(action, (log_density - log_jacobian).sum(-1))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int, layer_norm: bool):
        super().__init__()
        layers: list[nn.Module] = []
        width = obs_dim + action_dim
        for _ in range(2):
            layers.append(nn.Linear(width, hidden))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            width = hidden
        layers.append(nn.Linear(hidden, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        return self.network(torch.cat((obs, action), dim=-1)).squeeze(-1)


def ensemble_forward(critics: nn.ModuleList, obs: Tensor, action: Tensor) -> Tensor:
    """Evaluate all critics with batched kernels, preserving individual weights.

    Stacking parameters retains autograd connections to the original modules,
    so optimizers and checkpoints keep the same format. LayerNorm acts only on
    each critic's feature dimension; it never mixes examples or ensemble members.
    This removes repeated Metal dispatches without changing the network equations.
    """
    # For a twin critic the stacking overhead offsets the dispatch savings.
    if len(critics) <= 2:
        return torch.stack([critic(obs, action) for critic in critics])
    x = torch.cat((obs, action), dim=-1).unsqueeze(0).expand(len(critics), -1, -1)
    for index, layer in enumerate(critics[0].network):
        members = [critic.network[index] for critic in critics]
        if isinstance(layer, nn.Linear):
            weight = torch.stack([member.weight for member in members])
            bias = torch.stack([member.bias for member in members])
            x = torch.bmm(x, weight.transpose(1, 2)) + bias.unsqueeze(1)
        elif isinstance(layer, nn.LayerNorm):
            x = F.layer_norm(x, layer.normalized_shape, eps=layer.eps)
            weight = torch.stack([member.weight for member in members])
            bias = torch.stack([member.bias for member in members])
            x = x * weight.unsqueeze(1) + bias.unsqueeze(1)
        elif isinstance(layer, nn.ReLU):
            x = x.relu()
        else:
            raise TypeError(f"Unsupported ensemble layer: {type(layer).__name__}")
    return x.squeeze(-1)


def bellman_target(
    reward: Tensor,
    terminated: Tensor,
    next_q: Tensor,
    next_log_prob: Tensor,
    gamma: float,
    alpha: Tensor | float,
) -> Tensor:
    """Only real terminal states suppress bootstrapping; timeouts do not."""
    return reward + gamma * (~terminated.bool()).float() * (
        next_q - alpha * next_log_prob
    )


def symmetric_sample(
    online: ReplayBuffer,
    offline: ReplayBuffer | None,
    batch_size: int,
    offline_ratio: float = 0.5,
) -> Transitions:
    """Separate buffers preserve the requested prior-data fraction over time."""
    if not 0 <= offline_ratio <= 1:
        raise ValueError("offline_ratio must lie in [0, 1]")
    if offline is None or offline_ratio == 0:
        return online.sample(batch_size)
    n_offline = round(batch_size * offline_ratio)
    if n_offline == 0:
        return online.sample(batch_size)
    if n_offline == batch_size:
        return offline.sample(batch_size)
    return Transitions.cat(
        [
            offline.sample(n_offline),
            online.sample(batch_size - n_offline),
        ]
    )


def foreach_soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Grouped target updates with the same multiply-then-add equation.

    SAC selects this on CUDA, qualified with the committed dependency lock.
    CPU and Metal retain ``rl_mind.soft_update``; unsupported kernels raise.
    """
    sources, targets = list(source.parameters()), list(target.parameters())
    if len(sources) != len(targets):
        raise ValueError("Source and target parameter counts must match")
    if not targets:
        return
    with torch.no_grad():
        torch._foreach_mul_(targets, 1.0 - tau)
        torch._foreach_add_(targets, sources, alpha=tau)


class SAC(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 5,
        device: str = "mps",
        ensemble: int = 2,
        layer_norm: bool = False,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.1,
        hidden: int = 128,
        target_entropy: float | None = None,
        learn_temperature: bool = True,
    ):
        super().__init__()
        if ensemble < 2:
            raise ValueError("At least two critics are required")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.device = torch.device(device)
        self.gamma, self.tau = gamma, tau
        self.target_entropy = (
            -action_dim / 2 if target_entropy is None else target_entropy
        )
        self.learn_temperature = learn_temperature
        self.actor = Policy(obs_dim, action_dim, hidden)
        self.critics = nn.ModuleList(
            [Critic(obs_dim, action_dim, hidden, layer_norm) for _ in range(ensemble)]
        )
        self.target_critics = copy.deepcopy(self.critics).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha)))
        self.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.updates = 0

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def update(
        self,
        batch: Transitions,
        update_actor: bool = True,
        *,
        collect_metrics: bool = True,
    ) -> dict[str, float]:
        """Update weights normally; optionally defer CPU scalar reporting.

        ``collect_metrics=False`` returns an empty dictionary without reporting
        reductions or device-to-host synchronization. It retains update cadence,
        target updates and RNG behavior. The default preserves the existing API.
        """
        batch = batch._map(lambda tensor: tensor.to(self.device))
        with torch.no_grad():
            next_action = self.actor(batch.next_obs)
            selected = random.sample(range(len(self.target_critics)), 2)
            next_q = torch.stack(
                [
                    self.target_critics[i](batch.next_obs, next_action.value)
                    for i in selected
                ]
            ).amin(0)
            target = bellman_target(
                batch.reward,
                batch.terminated,
                next_q,
                next_action.log_prob,
                self.gamma,
                self.alpha,
            )
        qs = ensemble_forward(self.critics, batch.obs, batch.action.value)
        critic_loss = (qs - target.unsqueeze(0)).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        metrics = (
            {"critic_loss": critic_loss.detach(), "q": qs.detach().mean()}
            if collect_metrics
            else {}
        )

        if update_actor:
            # Freeze Q weights, retaining gradients through Q with respect to action.
            self.critics.requires_grad_(False)
            try:
                sampled = self.actor(batch.obs)
                q = ensemble_forward(self.critics, batch.obs, sampled.value).mean(0)
                actor_loss = (self.alpha.detach() * sampled.log_prob - q).mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optimizer.step()
            finally:
                self.critics.requires_grad_(True)
            if self.learn_temperature:
                alpha_loss = -(
                    self.log_alpha * (sampled.log_prob.detach() + self.target_entropy)
                ).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                if collect_metrics:
                    metrics["alpha_loss"] = alpha_loss.detach()
            if collect_metrics:
                metrics.update(
                    actor_loss=actor_loss.detach(),
                    entropy=-sampled.log_prob.detach().mean(),
                )

        target_update = (
            foreach_soft_update if self.device.type == "cuda" else soft_update
        )
        target_update(self.critics, self.target_critics, self.tau)
        self.updates += 1
        if not collect_metrics:
            return {}
        metrics["alpha"] = self.alpha.detach()
        # One GPU synchronization for all scalars rather than one per metric.
        return dict(zip(metrics, torch.stack(list(metrics.values())).cpu().tolist()))

    def checkpoint(self) -> dict:
        """Model and optimizer state; the harness saves experiment/RNG state."""
        return {
            "model": self.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_checkpoint(self, state: dict) -> None:
        self.load_state_dict(state["model"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.updates = state["updates"]


def fit_imitation(
    actor: Policy,
    obs: Tensor,
    actions: Tensor,
    device: str | torch.device,
    updates: int = 1000,
    batch_size: int = 256,
    optimizer: torch.optim.Optimizer | None = None,
    lr: float = 3e-4,
) -> dict[str, float]:
    """MSE over the same normalized controls used by SAC.

    Pass a persistent optimizer to retain its state across DAgger rounds.
    Samples with replacement from the entire aggregated dataset.
    """
    if len(obs) == 0 or len(obs) != len(actions):
        raise ValueError(
            "Imitation observations and actions must be nonempty and aligned"
        )
    actor.to(device)
    optimizer = optimizer or torch.optim.Adam(actor.parameters(), lr=lr)
    last_loss = None
    for _ in range(updates):
        idx = torch.randint(len(obs), (batch_size,), device=obs.device)
        prediction = actor.act(obs[idx].to(device))
        loss = F.mse_loss(prediction, actions[idx].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = loss.detach()
    return {
        "imitation_loss": float(last_loss.cpu())
        if last_loss is not None
        else float("nan")
    }
