# from typing import Dict

import torch
import torch.nn.functional as F


class A2C:
    def __init__(
        self,
        policy,
        lr: float = 3e-4,
#         value_coef: float = 0.5,
#         entropy_coef: float = 0.01,
#         max_grad_norm: float = 0.5,
    ) -> None:
        self.policy = policy
        # self.value_coef = value_coef
#         self.entropy_coef = entropy_coef
#         self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

#     def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
#         values, log_probs, entropy = self.policy.evaluate_actions(
#             state=batch["states"],
#             latent=batch["latents"],
#             action=batch["actions"],
#         )

#         advantages = batch["advantages"]
#         returns = batch["returns"]

#         policy_loss = -(advantages.detach() * log_probs).mean()
#         value_loss = F.mse_loss(values, returns)
#         total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

#         self.optimizer.zero_grad()
#         total_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
#         self.optimizer.step()

#         return {
#             "policy_loss": float(policy_loss.item()),
#             "value_loss": float(value_loss.item()),
#             "entropy": float(entropy.item()),
#             "total_loss": float(total_loss.item()),
#         }
