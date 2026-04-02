import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardDecoder(nn.Module):
    """Predicts reward from latent + next_state + action."""

    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        action_dim: int,
        action_embed_dim: int = 16,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, action_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + state_dim + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, latent: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        if action.dim() > 1:
            action = action.squeeze(-1)
        action = action.long()
        ha = F.relu(self.action_embed(action))
        x = torch.cat([latent, next_state, ha], dim=-1)
        return self.net(x)


class StateDecoder(nn.Module):
    """Predicts next_state from latent + prev_state + action."""

    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        action_dim: int,
        action_embed_dim: int = 16,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, action_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + state_dim + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self, latent: torch.Tensor, prev_state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        if action.dim() > 1:
            action = action.squeeze(-1)
        action = action.long()
        ha = F.relu(self.action_embed(action))
        x = torch.cat([latent, prev_state, ha], dim=-1)
        return self.net(x)
