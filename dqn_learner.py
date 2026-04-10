import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import environments
from environments.navigation.gridworld import plot_bb
from utils import helpers as utl
from utils.cleanrl_buffers import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNLearner:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.total_timesteps = int(args.num_frames)
        self.gamma = args.gamma
        self.tau = args.tau
        self.target_network_frequency = args.target_network_update_freq
        self.batch_size = args.batch_size
        self.start_e = args.start_e
        self.end_e = args.end_e
        self.exploration_fraction = args.exploration_fraction
        self.learning_starts = args.learning_starts
        self.train_frequency = args.train_frequency
        self.log_interval = args.log_interval
        self.vis_interval = args.vis_interval

        self.run_dir = f"runs/{args.env_name}__{self.seed}__{int(time.time())}"
        self.writer = SummaryWriter(self.run_dir)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
        )
        utl.seed(args.seed)
        
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_name, self.seed + i, args.reset_task_on_episode) for i in range(args.num_processes)]
        )
        
        self.q_network = QNetwork(self.envs).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.lr)
        self.target_network = QNetwork(self.envs).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.rb = ReplayBuffer(
            args.size_buffer,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            device,
            n_envs=args.num_processes,
            handle_timeout_termination=False,
        )
    
    def _linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        if duration <= 0:
            return end_e
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)
    
    def _maybe_log_episode(self, infos, global_step):
        if not isinstance(infos, dict) or "final_info" not in infos:
            return
        for info in infos["final_info"]:
            if info and "episode" in info:
                self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

    def _greedy_actions(self, obs):
        with torch.no_grad():
            q_values = self.q_network(torch.as_tensor(obs, dtype=torch.float32, device=device))
            return torch.argmax(q_values, dim=1).cpu().numpy()

    def _visualise_trajectories(self, global_step, num_episodes=3):
        eval_env = make_env(self.args.env_name, self.seed + global_step, self.args.reset_task_on_episode)()
        episode_all_obs = []
        episode_goals = []
        for ep in range(num_episodes):
            obs, _ = eval_env.reset(seed=self.seed + global_step + ep)
            done = False
            observations = [torch.from_numpy(obs).float().unsqueeze(0)]

            while not done:
                action = self._greedy_actions(obs[None, ...])[0]
                next_obs, reward, terminated, truncated, _ = eval_env.step(action)
                observations.append(torch.from_numpy(next_obs).float().unsqueeze(0))
                obs = next_obs
                done = bool(terminated or truncated)

            goal = eval_env.unwrapped.get_task() if hasattr(eval_env.unwrapped, "get_task") else np.array([0.0, 0.0])
            episode_all_obs.append(observations)
            episode_goals.append(goal)

        class VisArgs:
            pass

        vis_args = VisArgs()
        vis_args.max_rollouts_per_task = num_episodes

        # Reuse the exact GridWorld behaviour plot routine used by MetaLearner.
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
        print(f"[DQN] Saved trajectory plot: {self.run_dir}/{global_step}_behaviour")
    
    def train(self):
        obs, _ = self.envs.reset(seed=self.seed)
        start_time = time.time()
        last_loss = None
        
        for global_step in range(self.total_timesteps):
            epsilon = self._linear_schedule(
                self.start_e,
                self.end_e,
                int(self.exploration_fraction * self.total_timesteps),
                global_step,
            )
            if random.random() < epsilon:
                actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                actions = self._greedy_actions(obs)
            
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
            self._maybe_log_episode(infos, global_step)
            
            real_next_obs = next_obs.copy()
            if isinstance(infos, dict) and "final_observation" in infos:
                for idx, trunc in enumerate(truncations):
                    if trunc:
                        real_next_obs[idx] = infos["final_observation"][idx]
            
            rb_infos = infos if isinstance(infos, list) else [{} for _ in range(self.envs.num_envs)]
            self.rb.add(obs, real_next_obs, actions, rewards, terminations, rb_infos)
            obs = next_obs
            
            if global_step >= self.learning_starts and self.rb.size() >= self.batch_size:
                if global_step % self.train_frequency == 0:
                    data = self.rb.sample(self.batch_size)
                    with torch.no_grad():
                        target_max, _ = self.target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + self.gamma * target_max * (1 - data.dones.flatten())
                    
                    old_val = self.q_network(data.observations).gather(1, data.actions.long()).squeeze()
                    loss = F.mse_loss(td_target, old_val)
                    last_loss = float(loss.item())
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    if global_step % 100 == 0:
                        sps = int(global_step / max(time.time() - start_time, 1e-8))
                        self.writer.add_scalar("losses/td_loss", loss.item(), global_step)
                        self.writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                        self.writer.add_scalar("charts/SPS", sps, global_step)
                
                if global_step % self.target_network_frequency == 0:
                    for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                        target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)

            if (global_step + 1) % self.log_interval == 0:
                sps = int((global_step + 1) / max(time.time() - start_time, 1e-8))
                loss_str = f"{last_loss:.5f}" if last_loss is not None else "n/a"
                print(
                    f"[DQN] Step {global_step + 1}/{self.total_timesteps} | "
                    f"eps={epsilon:.3f} | buffer={self.rb.size()} | loss={loss_str} | SPS={sps}"
                )

            if (global_step + 1) % self.vis_interval == 0:
                self._visualise_trajectories(global_step + 1, num_episodes=3)
        
        self.envs.close()
        self.writer.close()

class ResetTaskOnReset(gym.Wrapper):
    """Resample task on reset for envs that implement `reset_task()`."""
    
    def __init__(self, env, enabled=True):
        super().__init__(env)
        self.enabled = enabled
    
    def reset(self, **kwargs):
        if self.enabled and hasattr(self.env.unwrapped, "reset_task"):
            self.env.unwrapped.reset_task()
        return self.env.reset(**kwargs)


def make_env(env_id, seed, reset_task_on_episode):
    def thunk():
        env = gym.make(env_id)
        env = ResetTaskOnReset(env, enabled=reset_task_on_episode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

# TODO: Might need to adjust this to improve learning performance (e.g. add more layers, or use a different architecture)
class QNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        n_actions = envs.single_action_space.n
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_actions),
        )

    def forward(self, x):
        x = x.float().view(x.shape[0], -1)
        return self.network(x)
