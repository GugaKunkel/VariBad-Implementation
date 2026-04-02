import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from algorithms import A2C, PPO
from architecture import ActorCritic, RewardDecoder, StateDecoder, VariBadEncoder, VariBadVAE
from gridworld import GridWorld


@dataclass
class Config:
    algo: str = "ppo"
    seed: int = 0
    updates: int = 400
    rollout_len: int = 64

    grid_size: int = 5
    episode_len: int = 15
    step_penalty: float = -0.1
    goal_reward: float = 1.0

    latent_dim: int = 8
    encoder_hidden_dim: int = 64
    policy_hidden_dim: int = 128

    policy_lr: float = 3e-4
    vae_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95

    ppo_clip_ratio: float = 0.2
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    kl_weight: float = 0.1
    state_loss_coef: float = 0.5

    log_every: int = 10
    device: str = "cpu"


class MetaLearner:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        self._set_seed(cfg.seed)

        self.env = GridWorld(
            size=cfg.grid_size,
            max_steps=cfg.episode_len,
            step_penalty=cfg.step_penalty,
            goal_reward=cfg.goal_reward,
        )
        self.env.seed(cfg.seed)

        state_dim = self.env.observation_dim
        action_dim = self.env.action_space_n

        self.encoder = VariBadEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.encoder_hidden_dim,
        ).to(self.device)

        reward_decoder = RewardDecoder(
            latent_dim=cfg.latent_dim,
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(self.device)
        state_decoder = StateDecoder(
            latent_dim=cfg.latent_dim,
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(self.device)
        self.vae = VariBadVAE(
            encoder=self.encoder,
            reward_decoder=reward_decoder,
            state_decoder=state_decoder,
            lr=cfg.vae_lr,
            kl_weight=cfg.kl_weight,
            state_loss_coef=cfg.state_loss_coef,
        )

        self.policy = ActorCritic(
            state_dim=state_dim,
            latent_dim=cfg.latent_dim,
            action_dim=action_dim,
            hidden_dim=cfg.policy_hidden_dim,
        ).to(self.device)

        if cfg.algo == "ppo":
            self.algo = PPO(
                policy=self.policy,
                lr=cfg.policy_lr,
                clip_ratio=cfg.ppo_clip_ratio,
                epochs=cfg.ppo_epochs,
                minibatch_size=cfg.ppo_minibatch_size,
                value_coef=cfg.value_coef,
                entropy_coef=cfg.entropy_coef,
                max_grad_norm=cfg.max_grad_norm,
            )
        elif cfg.algo == "a2c":
            self.algo = A2C(
                policy=self.policy,
                lr=cfg.policy_lr,
                value_coef=cfg.value_coef,
                entropy_coef=cfg.entropy_coef,
                max_grad_norm=cfg.max_grad_norm,
            )
        else:
            raise ValueError(f"Unsupported algo: {cfg.algo}")

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _compute_returns_advantages(
        self,
        rewards: torch.Tensor,  # [T, 1]
        dones: torch.Tensor,  # [T, 1]
        values: torch.Tensor,  # [T, 1]
        next_value: torch.Tensor,  # [1, 1]
    ):
        t_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(1, device=self.device)

        for t in reversed(range(t_steps)):
            if t == t_steps - 1:
                next_values = next_value
            else:
                next_values = values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_values * non_terminal - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        return returns, advantages

    def _collect_rollout(self):
        state = self.env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            _, latent_mean, _, hidden = self.encoder.prior(batch_size=1, device=self.device)

        states = []
        latents = []
        actions = []
        old_log_probs = []
        values = []
        rewards = []
        dones = []

        prev_states_vae = []
        next_states_vae = []
        actions_vae = []
        rewards_vae = []

        rollout_return = 0.0
        rollout_success = 0

        for _ in range(self.cfg.rollout_len):
            latent_for_policy = latent_mean.detach()
            with torch.no_grad():
                action, log_prob, _, value = self.policy.act(
                    state=state_tensor,
                    latent=latent_for_policy,
                    deterministic=False,
                )

            action_i = int(action.item())
            next_state, reward, done, info = self.env.step(action_i)

            next_state_tensor = torch.tensor(
                next_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action_tensor = torch.tensor([action_i], dtype=torch.long, device=self.device)
            reward_scalar = torch.tensor([reward], dtype=torch.float32, device=self.device)
            reward_tensor = reward_scalar.view(1, 1)

            states.append(state_tensor.squeeze(0))
            latents.append(latent_for_policy.squeeze(0))
            actions.append(action_tensor.squeeze(0))
            old_log_probs.append(log_prob.squeeze(0))
            values.append(value.squeeze(0))
            rewards.append(reward_scalar)
            dones.append(torch.tensor([float(done)], dtype=torch.float32, device=self.device))

            prev_states_vae.append(state_tensor.squeeze(0))
            next_states_vae.append(next_state_tensor.squeeze(0))
            actions_vae.append(action_tensor.squeeze(0))
            rewards_vae.append(reward_scalar)

            with torch.no_grad():
                _, latent_mean, _, hidden = self.encoder.step(
                    next_state=next_state_tensor,
                    action=action_tensor,
                    reward=reward_tensor,
                    hidden=hidden,
                )

            rollout_return += reward
            if info.get("reached_goal", False):
                rollout_success += 1

            if done:
                state = self.env.reset()
            else:
                state = next_state
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        with torch.no_grad():
            next_value = self.policy.get_value(
                state=state_tensor,
                latent=latent_mean.detach(),
            )

        states = torch.stack(states, dim=0)
        latents = torch.stack(latents, dim=0)
        actions = torch.stack(actions, dim=0)
        old_log_probs = torch.stack(old_log_probs, dim=0)
        values = torch.stack(values, dim=0)
        rewards = torch.stack(rewards, dim=0)
        dones = torch.stack(dones, dim=0)

        returns, advantages = self._compute_returns_advantages(
            rewards=rewards,
            dones=dones,
            values=values,
            next_value=next_value.squeeze(0),
        )

        rl_batch = {
            "states": states,
            "latents": latents,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "returns": returns.detach(),
            "advantages": advantages.detach(),
        }

        vae_batch = {
            "prev_states": torch.stack(prev_states_vae, dim=0).detach(),
            "next_states": torch.stack(next_states_vae, dim=0).detach(),
            "actions": torch.stack(actions_vae, dim=0).detach(),
            "rewards": torch.stack(rewards_vae, dim=0).detach(),
        }

        return rl_batch, vae_batch, rollout_return, rollout_success

    def train(self) -> None:
        returns = []
        successes = []

        for update_idx in range(1, self.cfg.updates + 1):
            goal = self.env.reset_task()
            rl_batch, vae_batch, rollout_return, rollout_success = self._collect_rollout()

            policy_stats = self.algo.update(rl_batch)
            vae_stats = self.vae.update(**vae_batch)

            returns.append(rollout_return)
            successes.append(rollout_success)

            if update_idx % self.cfg.log_every == 0 or update_idx == 1:
                avg_return = float(np.mean(returns[-self.cfg.log_every :]))
                avg_success = float(np.mean(successes[-self.cfg.log_every :]))
                print(
                    f"[update {update_idx:04d}] "
                    f"goal={goal} "
                    f"return={avg_return:.3f} "
                    f"success={avg_success:.2f} "
                    f"policy_loss={policy_stats['policy_loss']:.4f} "
                    f"value_loss={policy_stats['value_loss']:.4f} "
                    f"vae_loss={vae_stats['vae_loss']:.4f}"
                )
