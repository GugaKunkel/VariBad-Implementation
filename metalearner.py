import random
from typing import Dict

import numpy as np
import torch

from algorithms import A2C, PPO
from architecture import ActorCritic, RewardDecoder, StateDecoder, VariBadEncoder, VariBadVAE
from gridworld import GridWorld
from configs.VariBad_config import Config


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
        action_dim = self.env.action_dim

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
            use_prev_state=cfg.reward_decoder_use_prev_state,
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

    #TODO: Might get moved to a utils file when DQN is implemented
    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)
    
    #TODO: Might get moved to a utils file when DQN is implemented
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
        values: torch.Tensor,  # [T, 1]
        next_value: torch.Tensor,  # [1, 1]
        masks: torch.Tensor,  # [T, 1]
        bad_masks: torch.Tensor,  # [T, 1]
    ):
        t_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(1, device=self.device)

        for t in reversed(range(t_steps)):
            next_values = next_value if t == t_steps - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.gamma * next_values * masks[t] - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * masks[t] * gae
            gae = gae * bad_masks[t]
            advantages[t] = gae

        returns = advantages + values
        returns = returns * bad_masks + (1.0 - bad_masks) * values
        return returns, advantages

    def _collect_rollout(self):
        state = self.env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            latent_sample, _, _, hidden = self.encoder.prior(batch_size=1, device=self.device)

        states = []
        latents = []
        actions = []
        old_log_probs = []
        values = []
        rewards = []
        dones = []
        masks = []
        bad_masks = []

        prev_states_vae = []
        next_states_vae = []
        actions_vae = []
        rewards_vae = []
        dones_vae = []

        rollout_return = 0.0
        rollout_success = 0

        for _ in range(self.cfg.rollout_len):
            latent_for_policy = latent_sample.detach()
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
            done_tensor = torch.tensor(
                [float(done)],
                dtype=torch.float32,
                device=self.device,
            ).view(1, 1)
            mask = torch.tensor(
                [0.0 if done else 1.0],
                dtype=torch.float32,
                device=self.device,
            )
            bad_mask = torch.tensor(
                [0.0 if info.get("bad_transition", False) else 1.0],
                dtype=torch.float32,
                device=self.device,
            )

            states.append(state_tensor.squeeze(0))
            latents.append(latent_for_policy.squeeze(0))
            actions.append(action_tensor.squeeze(0))
            old_log_probs.append(log_prob.squeeze(0))
            values.append(value.squeeze(0))
            rewards.append(reward_scalar)
            dones.append(done_tensor.squeeze(0))
            masks.append(mask)
            bad_masks.append(bad_mask)

            prev_states_vae.append(state_tensor.squeeze(0))
            next_states_vae.append(next_state_tensor.squeeze(0))
            actions_vae.append(action_tensor.squeeze(0))
            rewards_vae.append(reward_scalar)
            dones_vae.append(done_tensor.squeeze(0))

            with torch.no_grad():
                latent_sample, _, _, hidden = self.encoder.step(
                    next_state=next_state_tensor,
                    action=action_tensor,
                    reward=reward_tensor,
                    hidden=hidden,
                    done=done_tensor,
                    reset_on_done=self.cfg.reset_encoder_on_done,
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
                latent=latent_sample.detach(),
            )

        states = torch.stack(states, dim=0)
        latents = torch.stack(latents, dim=0)
        actions = torch.stack(actions, dim=0)
        old_log_probs = torch.stack(old_log_probs, dim=0)
        values = torch.stack(values, dim=0)
        rewards = torch.stack(rewards, dim=0)
        dones = torch.stack(dones, dim=0)
        masks = torch.stack(masks, dim=0).view(-1, 1)
        bad_masks = torch.stack(bad_masks, dim=0).view(-1, 1)

        returns, advantages = self._compute_returns_advantages(
            rewards=rewards,
            values=values,
            next_value=next_value.squeeze(0),
            masks=masks,
            bad_masks=bad_masks,
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
            "dones": torch.stack(dones_vae, dim=0).detach(),
            "reset_on_done": self.cfg.reset_encoder_on_done,
        }

        return rl_batch, vae_batch, rollout_return, rollout_success

    def train(self) -> None:
        returns = []
        successes = []

        for update_idx in range(1, self.cfg.updates + 1):
            goal = self.env.reset_task()
            rl_batch, vae_batch, rollout_return, rollout_success = self._collect_rollout()

            policy_stats = self.algo.update(rl_batch)
            vae_stats: Dict[str, float] = {}
            for _ in range(self.cfg.vae_updates_per_rollout):
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
