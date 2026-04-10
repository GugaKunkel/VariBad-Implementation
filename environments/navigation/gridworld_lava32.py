import itertools
import math
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.utils import seeding
from matplotlib.patches import Rectangle
from torch.nn import functional as F

from utils import helpers as utl


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GridNaviLava32(gym.Env):
    """
    Larger GridNavi variant with fixed walls and lava tiles.
    - Grid size defaults to 16x16
    - Entering lava ends the current MDP episode (terminated=True)
    - Lava is rendered in orange to avoid confusion with red belief overlays
    """

    def __init__(self, num_cells=16, num_steps=15):
        super(GridNaviLava32, self).__init__()

        self.seed()
        self.num_cells = num_cells
        self.num_states = num_cells ** 2
        self.num_tasks = self.num_states

        self._max_episode_steps = num_steps
        self.step_count = 0

        self.observation_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.full(2, self.num_cells - 1, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)  # noop, up, right, down, left
        self.task_dim = 2
        self.belief_dim = self.num_states

        self.starting_state = (0.0, 0.0)

        self.walls, self.lava = self._build_obstacles(self.num_cells)

        self.possible_goals = list(itertools.product(range(num_cells), repeat=2))
        start_block = {(0, 0), (0, 1), (1, 0), (1, 1)}
        forbidden = set(self.walls) | set(self.lava) | start_block
        self.possible_goals = [p for p in self.possible_goals if p not in forbidden]

        if len(self.possible_goals) == 0:
            raise ValueError("No valid goal cells available after placing walls/lava.")

        self._env_state = np.array(self.starting_state, dtype=np.float32)
        self._goal = self.reset_task()
        self._belief_state = self._reset_belief()

    @staticmethod
    def _rect_cells(x0, x1, y0, y1):
        cells = set()
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                cells.add((x, y))
        return cells

    def _build_obstacles(self, n):
        walls = set()
        lava = set()

        if n < 8:
            return walls, lava

        # Walls with gaps to keep regions connected.
        x1 = n // 4
        x2 = n // 2
        y1 = n // 2
        y2 = (3 * n) // 4

        for y in range(0, n - 4):
            if y != n // 3:
                walls.add((x1, y))

        for y in range(3, n):
            if y != (2 * n) // 3:
                walls.add((x2, y))

        for x in range(2, n - 2):
            if x not in {n // 3, (3 * n) // 4}:
                walls.add((x, y1))

        for x in range(0, n - 6):
            if x != n // 5:
                walls.add((x, y2))

        # Lava pools (orange in visualisation)
        lava |= self._rect_cells(n // 6, n // 6 + 2, n // 6, n // 6 + 2)
        lava |= self._rect_cells((2 * n) // 3, (2 * n) // 3 + 2, n // 5, n // 5 + 2)
        lava |= self._rect_cells(n // 3, n // 3 + 2, (4 * n) // 5, n - 2)

        # Never block start tile.
        walls.discard((0, 0))
        lava.discard((0, 0))
        return walls, lava

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        seed32 = int(seed) % (2 ** 32)
        random.seed(seed32)
        np.random.seed(seed32)
        return [seed]

    def reset_task(self, task=None):
        if task is None:
            self._goal = np.array(random.choice(self.possible_goals))
        else:
            task_t = tuple(np.array(task).astype(int).tolist())
            if task_t in self.walls or task_t in self.lava:
                raise ValueError(f"Invalid task {task_t}: cannot place goal on wall/lava")
            self._goal = np.array(task_t)
        self._reset_belief()
        return self._goal

    def _reset_belief(self):
        self._belief_state = np.zeros((self.num_cells ** 2), dtype=np.float32)
        p = 1.0 / len(self.possible_goals)
        for pg in self.possible_goals:
            idx = self.task_to_id(np.array(pg))
            self._belief_state[idx] = p
        return self._belief_state

    def update_belief(self, state, action):
        on_goal = state[0] == self._goal[0] and state[1] == self._goal[1]

        if action == 5 or on_goal:
            possible_goals = self.possible_goals.copy()
            if tuple(self._goal) in possible_goals:
                possible_goals.remove(tuple(self._goal))
            wrong_hint = possible_goals[random.choice(range(len(possible_goals)))]
            self._belief_state *= 0
            self._belief_state[self.task_to_id(self._goal)] = 0.5
            self._belief_state[self.task_to_id(wrong_hint)] = 0.5
        else:
            self._belief_state[self.task_to_id(state)] = 0
            self._belief_state = np.ceil(self._belief_state)
            s = float(np.sum(self._belief_state))
            if s > 0:
                self._belief_state /= s
            else:
                self._reset_belief()

        assert (1 - float(np.sum(self._belief_state))) < 1e-4
        return self._belief_state

    def get_task(self):
        return self._goal.copy()

    def get_belief(self):
        return self._belief_state.copy()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.step_count = 0
        self._env_state = np.array(self.starting_state, dtype=np.float32)
        return self._env_state.copy(), {}

    def state_transition(self, action):
        next_state = self._env_state.copy()

        if action == 1:  # up
            next_state[1] = min([next_state[1] + 1, self.num_cells - 1])
        elif action == 2:  # right
            next_state[0] = min([next_state[0] + 1, self.num_cells - 1])
        elif action == 3:  # down
            next_state[1] = max([next_state[1] - 1, 0])
        elif action == 4:  # left
            next_state[0] = max([next_state[0] - 1, 0])

        proposed = (int(next_state[0]), int(next_state[1]))
        if proposed not in self.walls:
            self._env_state = next_state

        return self._env_state.astype(np.float32, copy=False)

    def step(self, action):
        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action[0]
        assert self.action_space.contains(action)

        terminated = False
        truncated = False

        state = self.state_transition(action)

        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            terminated = True

        cell = (int(self._env_state[0]), int(self._env_state[1]))
        on_goal = self._env_state[0] == self._goal[0] and self._env_state[1] == self._goal[1]
        on_lava = cell in self.lava

        if on_lava:
            reward = -1.0
            terminated = True
        elif on_goal:
            reward = 1.0
        else:
            reward = -0.1

        self.update_belief(self._env_state, action)

        task = self.get_task()
        task_id = self.task_to_id(task)
        info = {
            'task': task,
            'task_id': task_id,
            'belief': self.get_belief(),
            'on_lava': on_lava,
        }
        return state.astype(np.float32, copy=False), reward, terminated, truncated, info

    def task_to_id(self, goals):
        if isinstance(goals, (list, tuple)):
            goals = np.array(goals)
        if isinstance(goals, np.ndarray):
            goals = torch.from_numpy(goals)
        mat = torch.arange(0, self.num_cells ** 2, device=goals.device).long().reshape((self.num_cells, self.num_cells))
        goals = goals.long()

        if goals.dim() == 1:
            goals = goals.unsqueeze(0)

        goal_shape = goals.shape
        if len(goal_shape) > 2:
            goals = goals.reshape(-1, goals.shape[-1])

        classes = mat[goals[:, 0], goals[:, 1]]
        classes = classes.reshape(goal_shape[:-1])
        return classes

    @staticmethod
    def visualise_behaviour(env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            reward_decoder=None,
                            image_folder=None,
                            **kwargs):
        num_episodes = args.max_rollouts_per_task

        episode_all_obs = [[] for _ in range(num_episodes)]
        episode_prev_obs = [[] for _ in range(num_episodes)]
        episode_next_obs = [[] for _ in range(num_episodes)]
        episode_actions = [[] for _ in range(num_episodes)]
        episode_rewards = [[] for _ in range(num_episodes)]

        episode_returns = []
        episode_lengths = []

        episode_goals = []
        if args.pass_belief_to_policy and (encoder is None):
            episode_beliefs = [[] for _ in range(num_episodes)]
        else:
            episode_beliefs = None

        if encoder is not None:
            episode_latent_samples = [[] for _ in range(num_episodes)]
            episode_latent_means = [[] for _ in range(num_episodes)]
            episode_latent_logvars = [[] for _ in range(num_episodes)]
        else:
            episode_latent_samples = episode_latent_means = episode_latent_logvars = None

        curr_latent_sample = curr_latent_mean = curr_latent_logvar = None

        env.unwrapped.reset_task()
        state, belief = utl.reset_env(env, args)
        start_obs = state.clone()

        for episode_idx in range(args.max_rollouts_per_task):
            curr_goal = env.get_task()
            curr_rollout_rew = []

            if encoder is not None:
                if episode_idx == 0:
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                    curr_latent_sample = curr_latent_sample[0].to(device)
                    curr_latent_mean = curr_latent_mean[0].to(device)
                    curr_latent_logvar = curr_latent_logvar[0].to(device)

                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            episode_all_obs[episode_idx].append(start_obs.clone())
            if args.pass_belief_to_policy and (encoder is None):
                episode_beliefs[episode_idx].append(belief)

            for step_idx in range(1, env._max_episode_steps + 1):
                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs.clone())
                else:
                    episode_prev_obs[episode_idx].append(state.clone())

                _, action = utl.select_action(
                    policy=policy,
                    state=state.view(-1),
                    belief=belief,
                    deterministic=True,
                    latent_sample=curr_latent_sample.view(-1) if (curr_latent_sample is not None) else None,
                    latent_mean=curr_latent_mean.view(-1) if (curr_latent_mean is not None) else None,
                    latent_logvar=curr_latent_logvar.view(-1) if (curr_latent_logvar is not None) else None,
                )

                [state, belief], (rew_raw, rew_normalised), terminated, truncated, infos = utl.env_step(env, action, args)
                done = np.logical_or(terminated, truncated)

                if encoder is not None:
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                        action.float().to(device),
                        state,
                        rew_raw.reshape((1, 1)).float().to(device),
                        hidden_state,
                        return_prior=False,
                    )

                    episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                episode_all_obs[episode_idx].append(state.clone())
                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew_raw.clone())
                episode_actions[episode_idx].append(action.clone())

                curr_rollout_rew.append(rew_raw.clone())

                if args.pass_belief_to_policy and (encoder is None):
                    episode_beliefs[episode_idx].append(belief)

                if infos[0]['done_mdp'] and not done:
                    start_obs = infos[0]['start_state']
                    start_obs = torch.from_numpy(start_obs).float().reshape((1, -1)).to(device)
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)
            episode_goals.append(curr_goal)

        if encoder is not None:
            episode_latent_means = [torch.stack(e) for e in episode_latent_means]
            episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.cat(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        rew_pred_means, rew_pred_vars = plot_bb(
            env,
            args,
            episode_all_obs,
            episode_goals,
            reward_decoder,
            episode_latent_means,
            episode_latent_logvars,
            image_folder,
            iter_idx,
            episode_beliefs,
        )

        if reward_decoder:
            plot_rew_reconstruction(env, rew_pred_means, rew_pred_vars, image_folder, iter_idx)

        return (
            episode_latent_means,
            episode_latent_logvars,
            episode_prev_obs,
            episode_next_obs,
            episode_actions,
            episode_rewards,
            episode_returns,
        )


def _base_env(env):
    if hasattr(env, 'venv'):
        e = env.venv.unwrapped.envs[0]
    else:
        e = env
    if hasattr(e, 'unwrapped'):
        e = e.unwrapped
    return e


def plot_rew_reconstruction(env, rew_pred_means, rew_pred_vars, image_folder, iter_idx):
    num_rollouts = len(rew_pred_means)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    test_rew_mus = torch.cat(rew_pred_means).cpu().detach().numpy()
    plt.plot(range(test_rew_mus.shape[0]), test_rew_mus, '.-', alpha=0.5)
    plt.plot(range(test_rew_mus.shape[0]), test_rew_mus.mean(axis=1), 'k.-')
    for tj in np.cumsum([0, *[env._max_episode_steps for _ in range(num_rollouts)]]):
        span = test_rew_mus.max() - test_rew_mus.min()
        plt.plot([tj + 0.5, tj + 0.5], [test_rew_mus.min() - 0.05 * span, test_rew_mus.max() + 0.05 * span], 'k--', alpha=0.5)
    plt.title('output - mean')

    plt.subplot(1, 3, 2)
    test_rew_vars = torch.cat(rew_pred_vars).cpu().detach().numpy()
    plt.plot(range(test_rew_vars.shape[0]), test_rew_vars, '.-', alpha=0.5)
    plt.plot(range(test_rew_vars.shape[0]), test_rew_vars.mean(axis=1), 'k.-')
    for tj in np.cumsum([0, *[env._max_episode_steps for _ in range(num_rollouts)]]):
        span = test_rew_vars.max() - test_rew_vars.min()
        plt.plot([tj + 0.5, tj + 0.5], [test_rew_vars.min() - 0.05 * span, test_rew_vars.max() + 0.05 * span], 'k--', alpha=0.5)
    plt.title('output - variance')

    plt.subplot(1, 3, 3)
    p = np.clip(test_rew_vars, 1e-12, 1.0)
    rew_pred_entropy = -(p * np.log(p)).sum(axis=1)
    plt.plot(range(len(test_rew_vars)), rew_pred_entropy, 'r.-')
    for tj in np.cumsum([0, *[env._max_episode_steps for _ in range(num_rollouts)]]):
        span = rew_pred_entropy.max() - rew_pred_entropy.min()
        plt.plot([tj + 0.5, tj + 0.5], [rew_pred_entropy.min() - 0.05 * span, rew_pred_entropy.max() + 0.05 * span], 'k--', alpha=0.5)
    plt.title('Reward prediction entropy')

    plt.tight_layout()
    if image_folder is not None:
        plt.savefig('{}/{}_rew_decoder'.format(image_folder, iter_idx))
        plt.close()
    else:
        plt.show()


def plot_bb(env, args, episode_all_obs, episode_goals, reward_decoder,
            episode_latent_means, episode_latent_logvars, image_folder, iter_idx, episode_beliefs):
    plt.figure(figsize=(1.5 * env._max_episode_steps, 1.5 * args.max_rollouts_per_task))

    num_episodes = len(episode_all_obs)
    num_steps = len(episode_all_obs[0])

    rew_pred_means = [[] for _ in range(num_episodes)]
    rew_pred_vars = [[] for _ in range(num_episodes)]

    for episode_idx in range(num_episodes):
        for step_idx in range(num_steps):
            curr_obs = episode_all_obs[episode_idx][:step_idx + 1]
            curr_goal = episode_goals[episode_idx]

            if episode_latent_means is not None:
                curr_means = episode_latent_means[episode_idx][:step_idx + 1]
                curr_logvars = episode_latent_logvars[episode_idx][:step_idx + 1]

            plt.subplot(
                args.max_rollouts_per_task,
                math.ceil(env._max_episode_steps) + 1,
                1 + episode_idx * (1 + math.ceil(env._max_episode_steps)) + step_idx,
            )

            plot_behaviour(env, curr_obs, curr_goal)

            if episode_latent_means is not None:
                rm, rv = compute_beliefs(env, args, reward_decoder, curr_means[-1], curr_logvars[-1])
                rew_pred_means[episode_idx].append(rm)
                rew_pred_vars[episode_idx].append(rv)
                plot_belief(env, rm)
            elif episode_beliefs is not None:
                curr_beliefs = episode_beliefs[episode_idx][step_idx]
                plot_belief(env, curr_beliefs)
            else:
                rew_pred_means = rew_pred_vars = None

            if episode_idx == 0:
                plt.title('t = {}'.format(step_idx))

            if step_idx == 0:
                plt.ylabel('Episode {}'.format(episode_idx + 1))

    if episode_latent_means is not None:
        rew_pred_means = [torch.stack(r) for r in rew_pred_means]
        rew_pred_vars = [torch.stack(r) for r in rew_pred_vars]

    plt.tight_layout()
    if image_folder is not None:
        plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
        plt.close()
    else:
        plt.show()

    return rew_pred_means, rew_pred_vars


def plot_behaviour(env, observations, goal):
    base = _base_env(env)
    num_cells = int(env.observation_space.high[0] + 1)

    # base grid
    for i in range(num_cells):
        for j in range(num_cells):
            rec = Rectangle((i, j), 1, 1, facecolor='none', alpha=0.3, edgecolor='k', linewidth=0.5)
            plt.gca().add_patch(rec)

    # walls in gray
    for (x, y) in base.walls:
        rec = Rectangle((x, y), 1, 1, facecolor='dimgray', alpha=0.8, edgecolor='k', linewidth=0.5)
        plt.gca().add_patch(rec)

    # lava in orange (not red so belief overlay remains readable)
    for (x, y) in base.lava:
        rec = Rectangle((x, y), 1, 1, facecolor='orange', alpha=0.8, edgecolor='k', linewidth=0.5)
        plt.gca().add_patch(rec)

    if isinstance(observations, (tuple, list)):
        observations = torch.cat(observations)
    observations = observations.cpu().numpy() + 0.5
    goal = np.array(goal) + 0.5

    plt.plot(observations[:, 0], observations[:, 1], 'b-')
    plt.plot(observations[-1, 0], observations[-1, 1], 'b.')
    plt.plot(goal[0], goal[1], 'kx')

    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, num_cells])
    plt.ylim([0, num_cells])


def compute_beliefs(env, args, reward_decoder, latent_mean, latent_logvar):
    samples = torch.distributions.Normal(
        latent_mean.view(-1),
        torch.exp(0.5 * latent_logvar.view(-1))
    ).rsample((100,))

    rew_pred = reward_decoder(samples)
    if args.rew_pred_type == 'categorical':
        rew_pred = F.softmax(rew_pred, dim=-1)
    elif args.rew_pred_type == 'bernoulli':
        rew_pred = torch.sigmoid(rew_pred)

    rew_pred_means = torch.mean(rew_pred, dim=0)
    rew_pred_vars = torch.var(rew_pred, dim=0)
    return rew_pred_means, rew_pred_vars


def plot_belief(env, beliefs):
    num_cells = int(env.observation_space.high[0] + 1)
    base = _base_env(env)

    alphas = []
    for i in range(num_cells):
        for j in range(num_cells):
            idx = base.task_to_id(torch.tensor([[i, j]]))
            alpha = beliefs[idx]
            alphas.append(alpha.item())
    alphas = np.array(alphas)
    alphas[alphas < 0] = 0
    alphas[alphas > 1] = 1

    count = 0
    for i in range(num_cells):
        for j in range(num_cells):
            cell = (i, j)
            if cell in base.walls:
                rec = Rectangle((i, j), 1, 1, facecolor='dimgray', alpha=0.8, edgecolor='k', linewidth=0.5)
            elif cell in base.lava:
                rec = Rectangle((i, j), 1, 1, facecolor='orange', alpha=0.8, edgecolor='k', linewidth=0.5)
            else:
                rec = Rectangle((i, j), 1, 1, facecolor='r', alpha=alphas[count], edgecolor='k', linewidth=0.5)
            plt.gca().add_patch(rec)
            count += 1
