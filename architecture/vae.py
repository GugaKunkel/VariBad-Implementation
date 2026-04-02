# from typing import Dict

import torch
# import torch.nn.functional as F

from .decoder import RewardDecoder, StateDecoder
from .encoder import VariBadEncoder


class VariBadVAE:
    """VAE wrapper: encoder + reward/state decoders."""
    def __init__(
        self,
        encoder: VariBadEncoder,
        reward_decoder: RewardDecoder,
        state_decoder: StateDecoder,
        lr: float = 1e-3,
        kl_weight: float = 0.1,
        state_loss_coef: float = 0.5,
    ) -> None:
        self.encoder = encoder
        self.reward_decoder = reward_decoder
        self.state_decoder = state_decoder
        self.kl_weight = kl_weight
        self.state_loss_coef = state_loss_coef

        params = (
            list(self.encoder.parameters())
            + list(self.reward_decoder.parameters())
            + list(self.state_decoder.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=lr)

#     def update(
#         self,
#         prev_states: torch.Tensor,  # [T, state_dim]
#         next_states: torch.Tensor,  # [T, state_dim]
#         actions: torch.Tensor,  # [T]
#         rewards: torch.Tensor,  # [T, 1]
#     ) -> Dict[str, float]:
#         t_steps = prev_states.shape[0]
#         actions = actions.view(t_steps)
#         rewards = rewards.view(t_steps, 1)

#         next_states_seq = next_states.unsqueeze(1)  # [T, 1, state_dim]
#         actions_seq = actions.view(t_steps, 1)  # [T, 1]
#         rewards_seq = rewards.view(t_steps, 1, 1)  # [T, 1, 1]

#         latent_samples, latent_means, latent_logvars, _ = self.encoder.encode_sequence(
#             next_states=next_states_seq,
#             actions=actions_seq,
#             rewards=rewards_seq,
#         )
#         latents = latent_samples[1:, 0, :]  # [T, latent_dim]
#         means = latent_means[1:, 0, :]  # [T, latent_dim]
#         logvars = latent_logvars[1:, 0, :]  # [T, latent_dim]

#         reward_pred = self.reward_decoder(latents, next_states, actions)
#         reward_loss = F.mse_loss(reward_pred, rewards)

#         state_pred = self.state_decoder(latents, prev_states, actions)
#         state_loss = F.mse_loss(state_pred, next_states)

#         kl_loss = 0.5 * torch.mean(
#             torch.sum(torch.exp(logvars) + means.pow(2) - 1.0 - logvars, dim=-1)
#         )

#         total_loss = reward_loss + self.state_loss_coef * state_loss + self.kl_weight * kl_loss

#         self.optimizer.zero_grad()
#         total_loss.backward()
#         self.optimizer.step()

#         return {
#             "vae_loss": float(total_loss.item()),
#             "reward_loss": float(reward_loss.item()),
#             "state_loss": float(state_loss.item()),
#             "kl_loss": float(kl_loss.item()),
#         }
