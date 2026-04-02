import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Minimal actor-critic policy conditioned on state + latent."""

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = state_dim + latent_dim
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def _forward(self, state: torch.Tensor, latent: torch.Tensor):
        x = torch.cat([state, latent], dim=-1)
        h = self.base(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value

    def act(self, state: torch.Tensor, latent: torch.Tensor, deterministic: bool = False):
        logits, value = self._forward(state, latent)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return action, log_prob, entropy, value

    def evaluate_actions(
        self, state: torch.Tensor, latent: torch.Tensor, action: torch.Tensor
    ):
        if action.dim() > 1:
            action = action.squeeze(-1)
        action = action.long()

        logits, value = self._forward(state, latent)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().mean()
        return value, log_prob, entropy

    def get_value(self, state: torch.Tensor, latent: torch.Tensor):
        _, value = self._forward(state, latent)
        return value
