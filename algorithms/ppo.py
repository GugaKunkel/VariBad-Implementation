from typing import Dict

import torch
import torch.nn.functional as F


class PPO:
    def __init__(
        self,
        policy,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        epochs: int = 4,
        minibatch_size: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> None:
        self.policy = policy
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        states = batch["states"]
        latents = batch["latents"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_steps = states.shape[0]
        p_losses = []
        v_losses = []
        entropies = []
        totals = []

        for _ in range(self.epochs):
            indices = torch.randperm(num_steps, device=states.device)
            for start in range(0, num_steps, self.minibatch_size):
                mb_idx = indices[start : start + self.minibatch_size]

                values, log_probs, entropy = self.policy.evaluate_actions(
                    state=states[mb_idx],
                    latent=latents[mb_idx],
                    action=actions[mb_idx],
                )

                ratio = torch.exp(log_probs - old_log_probs[mb_idx])
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * advantages[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns[mb_idx])
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                p_losses.append(policy_loss.detach())
                v_losses.append(value_loss.detach())
                entropies.append(entropy.detach())
                totals.append(total_loss.detach())

        return {
            "policy_loss": float(torch.stack(p_losses).mean().item()),
            "value_loss": float(torch.stack(v_losses).mean().item()),
            "entropy": float(torch.stack(entropies).mean().item()),
            "total_loss": float(torch.stack(totals).mean().item()),
        }
