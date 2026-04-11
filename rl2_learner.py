import datetime
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from environments.parallel_envs import make_vec_envs
from environments.navigation.gridworld import plot_bb
from utils import helpers as utl


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RL2Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        in_dim = obs_dim + action_dim + 2  # obs, prev_action(one-hot), prev_reward, prev_done
        self.input_fc = nn.Linear(in_dim, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

        for name, p in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(p, 0)
            elif "weight" in name:
                nn.init.orthogonal_(p)

    def zero_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def _build_input(self, obs, prev_action, prev_reward, prev_done):
        prev_a_oh = F.one_hot(prev_action.squeeze(-1), num_classes=self.action_dim).float()
        return torch.cat([obs, prev_a_oh, prev_reward, prev_done], dim=-1)

    def step(self, obs, prev_action, prev_reward, prev_done, hidden_state, deterministic=False):
        x = self._build_input(obs, prev_action, prev_reward, prev_done)
        x = torch.tanh(self.input_fc(x)).unsqueeze(0)
        out, next_hidden = self.gru(x, hidden_state)
        feat = out.squeeze(0)

        logits = self.actor(feat)
        value = self.critic(feat)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1, keepdim=True) if deterministic else dist.sample().unsqueeze(-1)
        log_prob = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return value, action, log_prob, entropy, next_hidden


class RL2Learner:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.total_frames = int(args.num_frames)
        self.log_interval = args.log_interval
        self.vis_interval = args.vis_interval

        timestamp = datetime.datetime.now().strftime("%H_%M_%S__%d_%m")
        project_root = os.path.dirname(os.path.abspath(__file__))
        base_log_dir = args.results_log_dir if getattr(args, "results_log_dir", None) else os.path.join(project_root, "logs")
        algo_name = getattr(args, "algo_name", "rl2").lower()
        self.run_dir = os.path.join(base_log_dir, algo_name, f"{args.env_name}__{self.seed}__{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(self.run_dir)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
        )

        utl.seed(args.seed)
        self._print_varibad_alignment_report()

        self.envs = make_vec_envs(
            env_name=args.env_name,
            seed=args.seed,
            num_processes=args.num_processes,
            gamma=args.policy_gamma,
            device=device,
            episodes_per_task=args.max_rollouts_per_task,
            normalise_rew=True,
            ret_rms=None,
            tasks=None,
        )

        self.obs_dim = self.envs.observation_space.shape[0]
        self.action_dim = self.envs.action_space.n

        self.policy = RL2Policy(self.obs_dim, self.action_dim, args.rl2_hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr_policy, eps=args.policy_eps)

        self.frames = 0
        self.hidden_state = self.policy.zero_hidden(args.num_processes)
        self._rollout_count = 0

    def _print_varibad_alignment_report(self):
        expected = {
            "max_rollouts_per_task": 4,
            "num_processes": 16,
            "policy_num_steps": 60,
            "lr_policy": 7e-4,
            "policy_gamma": 0.95,
            "policy_tau": 0.95,
            "policy_value_loss_coef": 0.5,
            "policy_entropy_coef": 0.01,
            "policy_max_grad_norm": 0.5,
            "reset_task_on_episode": False,
        }
        mismatches = []
        for k, v in expected.items():
            if hasattr(self.args, k) and getattr(self.args, k) != v:
                mismatches.append((k, getattr(self.args, k), v))

        if mismatches:
            print("[RL2] VariBAD-config alignment check: some key defaults differ:")
            for key, got, want in mismatches:
                print(f"[RL2]   {key}: current={got} | VariBAD-RL2-default={want}")
        else:
            print("[RL2] VariBAD-config alignment check: key RL2 defaults match.")

        print(
            "[RL2] Note: this implementation is a standalone recurrent actor-critic; "
            "VariBAD's original RL2 uses its latent-encoder + PPO training stack."
        )

    def _format_obs_for_policy(self, obs_t, prev_done):
        # Train env may include done-info in obs (via VariBadWrapper); eval env may not.
        if obs_t.shape[-1] == self.obs_dim:
            return obs_t
        if obs_t.shape[-1] < self.obs_dim:
            pad = self.obs_dim - obs_t.shape[-1]
            if pad == 1:
                return torch.cat([obs_t, prev_done], dim=-1)
            zeros = torch.zeros((obs_t.shape[0], pad), dtype=obs_t.dtype, device=obs_t.device)
            return torch.cat([obs_t, zeros], dim=-1)
        return obs_t[..., :self.obs_dim]

    def _make_eval_env(self):
        env = gym.make(self.args.env_name)
        return _ResetTaskOnReset(
            env,
            episodes_per_task=self.args.max_rollouts_per_task,
            reset_every_episode=self.args.reset_task_on_episode,
        )

    def _visualise_trajectories(self, global_step, num_episodes=3):
        eval_env = self._make_eval_env()
        eval_hidden = self.policy.zero_hidden(1)

        prev_action = torch.zeros((1, 1), dtype=torch.long, device=device)
        prev_reward = torch.zeros((1, 1), dtype=torch.float32, device=device)
        prev_done = torch.zeros((1, 1), dtype=torch.float32, device=device)

        episode_all_obs = []
        episode_goals = []
        episodes_completed = 0

        for ep in range(num_episodes):
            obs, _ = eval_env.reset(seed=self.seed + global_step + ep)
            obs_t = torch.tensor(obs.copy(), dtype=torch.float32, device=device).unsqueeze(0)
            done = False
            observations = [obs_t.detach().cpu()]

            while not done:
                obs_in = self._format_obs_for_policy(obs_t, prev_done)
                with torch.no_grad():
                    _, action, _, _, eval_hidden = self.policy.step(
                        obs_in, prev_action, prev_reward, prev_done, eval_hidden, deterministic=True
                    )
                next_obs, reward, terminated, truncated, _ = eval_env.step(int(action.item()))
                done = bool(terminated or truncated)

                prev_action = action
                prev_reward = torch.tensor([[reward]], dtype=torch.float32, device=device)
                prev_done = torch.tensor([[1.0 if done else 0.0]], dtype=torch.float32, device=device)

                obs_t = torch.tensor(next_obs.copy(), dtype=torch.float32, device=device).unsqueeze(0)
                observations.append(obs_t.detach().cpu())

            goal = eval_env.unwrapped.get_task() if hasattr(eval_env.unwrapped, "get_task") else np.array([0.0, 0.0])
            episode_all_obs.append(observations)
            episode_goals.append(goal)

            episodes_completed += 1
            if self.args.reset_task_on_episode or (episodes_completed % self.args.max_rollouts_per_task == 0):
                eval_hidden = self.policy.zero_hidden(1)
                prev_action.zero_(); prev_reward.zero_(); prev_done.zero_()

        class _VisArgs:
            pass

        vis_args = _VisArgs()
        vis_args.max_rollouts_per_task = num_episodes

        plot_bb(
            env=eval_env.unwrapped,
            args=vis_args,
            episode_all_obs=episode_all_obs,
            episode_goals=episode_goals,
            reward_decoder=None,
            episode_latent_means=None,
            episode_latent_logvars=None,
            image_folder=self.run_dir,
            iter_idx=global_step,
            episode_beliefs=None,
        )
        eval_env.close()
        print(f"[RL2] Saved trajectory plot: {self.run_dir}/{global_step}_behaviour")

    def train(self):
        obs = self.envs.reset()

        prev_action = torch.zeros((self.args.num_processes, 1), dtype=torch.long, device=device)
        prev_reward = torch.zeros((self.args.num_processes, 1), dtype=torch.float32, device=device)
        prev_done = torch.zeros((self.args.num_processes, 1), dtype=torch.float32, device=device)

        start_time = time.time()
        last_total_loss = None

        while self.frames < self.total_frames:
            self.hidden_state = self.hidden_state.detach()

            values = []
            log_probs = []
            entropies = []
            rewards = []
            dones = []

            for _ in range(self.args.policy_num_steps):
                value, action, log_prob, entropy, self.hidden_state = self.policy.step(
                    obs, prev_action, prev_reward, prev_done, self.hidden_state, deterministic=False
                )

                [next_obs, _], (rew_raw, rew_norm), terminated, truncated, infos = utl.env_step(self.envs, action, self.args)
                done = torch.as_tensor(np.logical_or(terminated, truncated), device=device, dtype=torch.float32).view(-1, 1)

                values.append(value)
                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards.append(rew_norm)
                dones.append(done)

                prev_action = action.detach()
                prev_reward = rew_norm.detach()
                prev_done = done.detach()
                obs = next_obs

                # Reset hidden state only at BAMDP end.
                self.hidden_state = self.hidden_state * (1.0 - done).view(1, -1, 1)
                self.frames += self.args.num_processes

                if self.frames >= self.total_frames:
                    break

            with torch.no_grad():
                next_value, _, _, _, _ = self.policy.step(
                    obs, prev_action, prev_reward, prev_done, self.hidden_state, deterministic=True
                )

            # GAE on sequential rollout
            T = len(rewards)
            returns = [None] * T
            advantages = [None] * T
            gae = torch.zeros_like(next_value)
            next_v = next_value

            for t in reversed(range(T)):
                mask = 1.0 - dones[t]
                delta = rewards[t] + self.args.policy_gamma * next_v * mask - values[t]
                gae = delta + self.args.policy_gamma * self.args.policy_tau * mask * gae
                advantages[t] = gae
                returns[t] = gae + values[t]
                next_v = values[t]

            values_t = torch.stack(values, dim=0)
            log_probs_t = torch.stack(log_probs, dim=0)
            entropies_t = torch.stack(entropies, dim=0)
            adv_t = torch.stack(advantages, dim=0)
            ret_t = torch.stack(returns, dim=0)

            policy_loss = -(adv_t.detach() * log_probs_t).mean()
            value_loss = 0.5 * (ret_t.detach() - values_t).pow(2).mean()
            entropy_bonus = entropies_t.mean()

            total_loss = (
                policy_loss
                + self.args.policy_value_loss_coef * value_loss
                - self.args.policy_entropy_coef * entropy_bonus
            )
            last_total_loss = float(total_loss.item())

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.policy_max_grad_norm)
            self.optimizer.step()

            self._rollout_count += 1
            if self._rollout_count % max(1, self.args.log_interval // max(1, self.args.policy_num_steps)) == 0:
                sps = int(self.frames / max(time.time() - start_time, 1e-8))
                print(
                    f"[RL2] Frames {self.frames}/{self.total_frames} | "
                    f"loss={last_total_loss:.5f} | SPS={sps}"
                )
                self.writer.add_scalar("losses/total", total_loss.item(), self.frames)
                self.writer.add_scalar("losses/policy", policy_loss.item(), self.frames)
                self.writer.add_scalar("losses/value", value_loss.item(), self.frames)
                self.writer.add_scalar("losses/entropy", entropy_bonus.item(), self.frames)
                self.writer.add_scalar("charts/sps", sps, self.frames)

            if self._rollout_count % max(1, self.args.vis_interval // max(1, self.args.policy_num_steps)) == 0:
                self._visualise_trajectories(self.frames, num_episodes=3)

        self.envs.close()
        self.writer.close()


class _ResetTaskOnReset(gym.Wrapper):
    def __init__(self, env, episodes_per_task=1, reset_every_episode=False):
        super().__init__(env)
        self.episodes_per_task = max(1, int(episodes_per_task))
        self.reset_every_episode = reset_every_episode
        self._episodes_since_task_reset = 0

    def reset(self, **kwargs):
        if hasattr(self.env.unwrapped, "reset_task"):
            should_reset = self.reset_every_episode or (self._episodes_since_task_reset == 0)
            if should_reset:
                self.env.unwrapped.reset_task()
            self._episodes_since_task_reset = (self._episodes_since_task_reset + 1) % self.episodes_per_task
        return self.env.reset(**kwargs)
