import torch
import torch.nn as nn
import torch.nn.functional as F


class VariBadEncoder(nn.Module):
    """
    Minimal VariBAD-style recurrent encoder.
    Input each step: (next_state, action, reward)
    Output each step: latent sample + posterior params.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        state_embed_dim: int = 32,
        action_embed_dim: int = 16,
        reward_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.state_encoder = nn.Linear(state_dim, state_embed_dim)
        self.action_encoder = nn.Embedding(action_dim, action_embed_dim)
        self.reward_encoder = nn.Linear(1, reward_embed_dim)

        gru_input_dim = state_embed_dim + action_embed_dim + reward_embed_dim
        self.gru = nn.GRUCell(gru_input_dim, hidden_dim)

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def prior(self, batch_size: int, device: torch.device):
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        sample = self.reparameterize(mu, logvar)
        return sample, mu, logvar, hidden

    def step(
        self,
        next_state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        hidden: torch.Tensor,
    ):
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        if action.dim() > 1:
            action = action.squeeze(-1)
        action = action.long()

        hs = F.relu(self.state_encoder(next_state))
        ha = F.relu(self.action_encoder(action))
        hr = F.relu(self.reward_encoder(reward))
        gru_in = torch.cat([hs, ha, hr], dim=-1)

        hidden = self.gru(gru_in, hidden)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        sample = self.reparameterize(mu, logvar)
        return sample, mu, logvar, hidden

    def encode_sequence(
        self,
        next_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ):
        """
        next_states: [T, B, state_dim]
        actions: [T, B]
        rewards: [T, B, 1]
        Returns latent tensors with length T+1 (including prior).
        """
        t_steps, batch_size, _ = next_states.shape
        device = next_states.device

        sample, mu, logvar, hidden = self.prior(batch_size=batch_size, device=device)
        samples = [sample]
        means = [mu]
        logvars = [logvar]
        hiddens = [hidden]

        for t in range(t_steps):
            sample, mu, logvar, hidden = self.step(
                next_state=next_states[t],
                action=actions[t],
                reward=rewards[t],
                hidden=hidden,
            )
            samples.append(sample)
            means.append(mu)
            logvars.append(logvar)
            hiddens.append(hidden)

        return (
            torch.stack(samples, dim=0),
            torch.stack(means, dim=0),
            torch.stack(logvars, dim=0),
            torch.stack(hiddens, dim=0),
        )
