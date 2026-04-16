import itertools
import math
import os
import random
from torch.nn import functional as F

import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.utils import seeding

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GridNavi(gym.Env):
    def __init__(self, num_cells=5, num_steps=15, lava_positions=None, lava_penalty=-1.0):
        super(GridNavi, self).__init__()
        self.seed()
        self.num_cells = num_cells
        self.num_states = num_cells ** 2
        self.num_tasks = self.num_states
        self.lava_positions = set(tuple(map(int, p)) for p in (lava_positions or []))
        self.lava_penalty = float(lava_penalty)
        
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
        
        # possible starting states
        self.starting_state = (0.0, 0.0)
        
        # goals can be anywhere except on possible starting states and immediately around it
        self.possible_goals = list(itertools.product(range(num_cells), repeat=2))
        self.possible_goals.remove((0, 0))
        self.possible_goals.remove((0, 1))
        self.possible_goals.remove((1, 1))
        self.possible_goals.remove((1, 0))
        self.possible_goals = [g for g in self.possible_goals if g not in self.lava_positions]
        if len(self.possible_goals) == 0:
            raise ValueError("No valid goals remain after lava filtering.")
        
        # reset the environment state
        self._env_state = np.array(self.starting_state, dtype=np.float32)
        # reset the goal
        self._goal = self.reset_task()
        # reset the belief
        self._belief_state = self._reset_belief()
    
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
            task_tuple = tuple(map(int, task))
            if task_tuple not in self.possible_goals:
                raise ValueError(f"Task {task_tuple} is not a valid goal for this environment.")
            self._goal = np.array(task_tuple)
        self._reset_belief()
        return self._goal
    
    def _reset_belief(self):
        self._belief_state = np.zeros((self.num_cells ** 2))
        for pg in self.possible_goals:
            idx = self.task_to_id(np.array(pg))
            self._belief_state[idx] = 1.0 / len(self.possible_goals)
        return self._belief_state
    
    def update_belief(self, state, action):
        on_goal = state[0] == self._goal[0] and state[1] == self._goal[1]
        
        # hint
        if action == 5 or on_goal:
            possible_goals = self.possible_goals.copy()
            possible_goals.remove(tuple(self._goal))
            wrong_hint = possible_goals[random.choice(range(len(possible_goals)))]
            self._belief_state *= 0
            self._belief_state[self.task_to_id(self._goal)] = 0.5
            self._belief_state[self.task_to_id(wrong_hint)] = 0.5
        else:
            self._belief_state[self.task_to_id(state)] = 0
            self._belief_state = np.ceil(self._belief_state)
            self._belief_state /= sum(self._belief_state)
        
        assert (1-sum(self._belief_state)) < 1e-4
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
        """
        Moving the agent between states
        """
        if action == 1:  # up
            self._env_state[1] = min([self._env_state[1] + 1, self.num_cells - 1])
        elif action == 2:  # right
            self._env_state[0] = min([self._env_state[0] + 1, self.num_cells - 1])
        elif action == 3:  # down
            self._env_state[1] = max([self._env_state[1] - 1, 0])
        elif action == 4:  # left
            self._env_state[0] = max([self._env_state[0] - 1, 0])
        return self._env_state.astype(np.float32, copy=False)
    
    def step(self, action):
        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action[0]
        assert self.action_space.contains(action)
        
        terminated = False
        truncated = False
        
        # perform state transition
        state = self.state_transition(action)
        
        # check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            terminated = True
        
        # compute reward
        if self._env_state[0] == self._goal[0] and self._env_state[1] == self._goal[1]:
            reward = 1.0
        elif tuple(map(int, self._env_state.tolist())) in self.lava_positions:
            reward = self.lava_penalty
        else:
            reward = -0.1
        
        # update ground-truth belief
        self.update_belief(self._env_state, action)
        task = self.get_task()
        task_id = self.task_to_id(task)
        info = {'task': task,
                'task_id': task_id,
                'belief': self.get_belief(),
                'in_lava': tuple(map(int, self._env_state.tolist())) in self.lava_positions}
        return state.astype(np.float32, copy=False), reward, terminated, truncated, info
    
    def task_to_id(self, goals):
        if isinstance(goals, list) or isinstance(goals, tuple):
            goals = np.array(goals)
        if isinstance(goals, np.ndarray):
            goals = torch.from_numpy(goals)
        mat = torch.arange(
            0, self.num_cells ** 2, device=goals.device
        ).long().reshape((self.num_cells, self.num_cells))
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
                            **kwargs
                            ):
        """
        Visualises the behaviour of the policy, together with the latent state and belief.
        The environment passed to this method should be a SubProcVec or DummyVecEnv, not the raw env!
        """

        num_episodes = args.max_rollouts_per_task

        # --- initialise things we want to keep track of ---

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
            # keep track of latent spaces
            episode_latent_samples = [[] for _ in range(num_episodes)]
            episode_latent_means = [[] for _ in range(num_episodes)]
            episode_latent_logvars = [[] for _ in range(num_episodes)]
        else:
            episode_latent_samples = episode_latent_means = episode_latent_logvars = None

        curr_latent_sample = curr_latent_mean = curr_latent_logvar = None

        # --- roll out policy ---

        env.unwrapped.reset_task()
        state, belief = utl.reset_env(env, args)
        start_obs = state.clone()

        for episode_idx in range(args.max_rollouts_per_task):

            curr_goal = env.get_task()
            curr_rollout_rew = []
            curr_rollout_goal = []

            if encoder is not None:

                if episode_idx == 0:
                    # reset to prior
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

                # act
                _, action = utl.select_action(args=args, policy=policy,
                                                 state=state.view(-1),
                                                 belief=belief,
                                                 deterministic=True,
                                                 latent_sample=curr_latent_sample.view(-1) if (curr_latent_sample is not None) else None,
                                                 latent_mean=curr_latent_mean.view(-1) if (curr_latent_mean is not None) else None,
                                                 latent_logvar=curr_latent_logvar.view(-1) if (curr_latent_logvar is not None) else None,
                                                 )

                # observe reward and next obs
                [state, belief], (rew_raw, rew_normalised), terminated, truncated, infos = utl.env_step(env, action, args)
                done = np.logical_or(terminated, truncated)
                if encoder is not None:
                    # update task embedding
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                        action.float().to(device),
                        state,
                        rew_raw.reshape((1, 1)).float().to(device),
                        hidden_state,
                        return_prior=False)

                    episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                episode_all_obs[episode_idx].append(state.clone())
                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew_raw.clone())
                episode_actions[episode_idx].append(action.clone())

                curr_rollout_rew.append(rew_raw.clone())
                curr_rollout_goal.append(env.get_task().copy())

                if args.pass_belief_to_policy and (encoder is None):
                    episode_beliefs[episode_idx].append(belief)

                if infos[0]['done_mdp'] and not done:
                    start_obs = infos[0]['start_state']
                    start_obs = torch.from_numpy(start_obs).float().reshape((1, -1)).to(device)
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)
            episode_goals.append(curr_goal)

        # clean up

        if encoder is not None:
            episode_latent_means = [torch.stack(e) for e in episode_latent_means]
            episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.cat(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        # plot behaviour & visualise belief in env

        rew_pred_means, rew_pred_vars = plot_bb(env, args, episode_all_obs, episode_goals, reward_decoder,
                                                episode_latent_means, episode_latent_logvars,
                                                image_folder, iter_idx, episode_beliefs)

        if reward_decoder:
            plot_rew_reconstruction(env, rew_pred_means, rew_pred_vars, image_folder, iter_idx)

        return episode_latent_means, episode_latent_logvars, \
               episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
               episode_returns


def plot_rew_reconstruction(env,
                            rew_pred_means,
                            rew_pred_vars,
                            image_folder,
                            iter_idx,
                            ):
    """
    Note that env might need to be a wrapped env!
    """

    num_rollouts = len(rew_pred_means)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    test_rew_mus = torch.cat(rew_pred_means).cpu().detach().numpy()
    plt.plot(range(test_rew_mus.shape[0]), test_rew_mus, '.-', alpha=0.5)
    plt.plot(range(test_rew_mus.shape[0]), test_rew_mus.mean(axis=1), 'k.-')
    for tj in np.cumsum([0, *[env._max_episode_steps for _ in range(num_rollouts)]]):
        span = test_rew_mus.max() - test_rew_mus.min()
        plt.plot([tj + 0.5, tj + 0.5], [test_rew_mus.min() - 0.05 * span, test_rew_mus.max() + 0.05 * span], 'k--',
                 alpha=0.5)
    plt.title('output - mean')

    plt.subplot(1, 3, 2)
    test_rew_vars = torch.cat(rew_pred_vars).cpu().detach().numpy()
    plt.plot(range(test_rew_vars.shape[0]), test_rew_vars, '.-', alpha=0.5)
    plt.plot(range(test_rew_vars.shape[0]), test_rew_vars.mean(axis=1), 'k.-')
    for tj in np.cumsum([0, *[env._max_episode_steps for _ in range(num_rollouts)]]):
        span = test_rew_vars.max() - test_rew_vars.min()
        plt.plot([tj + 0.5, tj + 0.5], [test_rew_vars.min() - 0.05 * span, test_rew_vars.max() + 0.05 * span],
                 'k--', alpha=0.5)
    plt.title('output - variance')

    plt.subplot(1, 3, 3)
    p = np.clip(test_rew_vars, 1e-12, 1.0)
    rew_pred_entropy = -(p * np.log(p)).sum(axis=1)
    plt.plot(range(len(test_rew_vars)), rew_pred_entropy, 'r.-')
    for tj in np.cumsum([0, *[env._max_episode_steps for _ in range(num_rollouts)]]):
        span = rew_pred_entropy.max() - rew_pred_entropy.min()
        plt.plot([tj + 0.5, tj + 0.5], [rew_pred_entropy.min() - 0.05 * span, rew_pred_entropy.max() + 0.05 * span],
                 'k--', alpha=0.5)
    plt.title('Reward prediction entropy')

    plt.tight_layout()
    if image_folder is not None:
        plt.savefig('{}/{}_rew_decoder'.format(image_folder, iter_idx))
        plt.close()
    else:
        plt.show()


def plot_bb(env, args, episode_all_obs, episode_goals, reward_decoder,
            episode_latent_means, episode_latent_logvars, image_folder, iter_idx, episode_beliefs):
    """
    Plot behaviour and belief.
    """
    num_cells = int(env.observation_space.high[0] + 1)
    num_episodes = len(episode_all_obs)
    num_steps = len(episode_all_obs[0])
    # Wrap behaviour panels after t=10 so long horizons remain readable.
    wrap_after_t = 10
    cols_per_row = min(num_steps, wrap_after_t + 1)
    rows_per_episode = int(math.ceil(num_steps / cols_per_row))
    total_rows = num_episodes * rows_per_episode

    # Slightly larger plots for dense grids (e.g., 16x16 lava map).
    subplot_scale = max(1.55, num_cells / 10.0)
    plt.figure(figsize=(subplot_scale * cols_per_row, subplot_scale * total_rows))

    rew_pred_means = [[] for _ in range(num_episodes)]
    rew_pred_vars = [[] for _ in range(num_episodes)]

    # loop through the experiences
    for episode_idx in range(num_episodes):
        for step_idx in range(num_steps):

            curr_obs = episode_all_obs[episode_idx][:step_idx + 1]
            curr_goal = episode_goals[episode_idx]

            if episode_latent_means is not None:
                curr_means = episode_latent_means[episode_idx][:step_idx + 1]
                curr_logvars = episode_latent_logvars[episode_idx][:step_idx + 1]

            # choose correct subplot
            local_row = step_idx // cols_per_row
            local_col = step_idx % cols_per_row
            subplot_row = episode_idx * rows_per_episode + local_row
            subplot_idx = subplot_row * cols_per_row + local_col + 1
            plt.subplot(total_rows, cols_per_row, subplot_idx)

            # plot the behaviour
            plot_behaviour(env, curr_obs, curr_goal)

            if reward_decoder is not None:
                # visualise belief in env
                rm, rv = compute_beliefs(args,
                                         reward_decoder,
                                         curr_means[-1],
                                         curr_logvars[-1])
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

            if local_col == 0:
                plt.ylabel('Episode {}'.format(episode_idx + 1))

    if reward_decoder is not None:
        rew_pred_means = [torch.stack(r) for r in rew_pred_means]
        rew_pred_vars = [torch.stack(r) for r in rew_pred_vars]

    # save figure that shows policy behaviour
    plt.tight_layout()
    save_dpi = 220 if num_cells >= 12 else 140
    if image_folder is not None:
        os.makedirs(image_folder, exist_ok=True)
        plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx), dpi=save_dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return rew_pred_means, rew_pred_vars


def plot_behaviour(env, observations, goal):
    num_cells = int(env.observation_space.high[0] + 1)
    lava_positions = getattr(env, "lava_positions", set())

    # draw grid
    for i in range(num_cells):
        for j in range(num_cells):
            pos_i = i
            pos_j = j
            cell_is_lava = (i, j) in lava_positions
            face_color = 'orange' if cell_is_lava else 'none'
            face_alpha = 0.55 if cell_is_lava else 0.5
            rec = Rectangle((pos_i, pos_j), 1, 1, facecolor=face_color, alpha=face_alpha, edgecolor='k')
            plt.gca().add_patch(rec)

    # shift obs and goal by half a stepsize
    if isinstance(observations, tuple) or isinstance(observations, list):
        observations = torch.cat(observations)
    observations = observations.cpu().numpy() + 0.5
    goal = np.array(goal) + 0.5

    # visualise behaviour, current position, goal
    plt.plot(observations[:, 0], observations[:, 1], 'b-')
    plt.plot(observations[-1, 0], observations[-1, 1], 'b.')
    plt.plot(goal[0], goal[1], 'kx')

    # make it look nice
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, num_cells])
    plt.ylim([0, num_cells])


def compute_beliefs(args, reward_decoder, latent_mean, latent_logvar):
    # take several samples from the latent distribution
    samples = torch.distributions.Normal(
        latent_mean.view(-1),
        torch.exp(0.5 * latent_logvar.view(-1))
    ).rsample((100,))

    # compute reward predictions for those
    rew_pred = reward_decoder(samples)
    if args.rew_pred_type == 'categorical':
        rew_pred = F.softmax(rew_pred, dim=-1)
    elif args.rew_pred_type == 'bernoulli':
        rew_pred = torch.sigmoid(rew_pred)
    rew_pred_means = torch.mean(rew_pred, dim=0)  # .reshape((1, -1))
    rew_pred_vars = torch.var(rew_pred, dim=0)  # .reshape((1, -1))
    return rew_pred_means, rew_pred_vars


def plot_belief(env, beliefs):
    """
    Plot the belief by taking 100 samples from the latent space and plotting the average predicted reward per cell.
    """

    num_cells = int(env.observation_space.high[0] + 1)
    unwrapped_env = env.venv.unwrapped.envs[0]
    lava_positions = getattr(unwrapped_env.unwrapped, "lava_positions", set())

    # draw probabilities for each grid cell
    alphas = []
    for i in range(num_cells):
        for j in range(num_cells):
            pos_i = i
            pos_j = j
            idx = unwrapped_env.unwrapped.task_to_id(torch.tensor([[pos_i, pos_j]]))
            alpha = beliefs[idx]
            if (pos_i, pos_j) in lava_positions:
                alphas.append(0.0)
            else:
                alphas.append(alpha.item())
    alphas = np.array(alphas)
    # cut off values (this only happens if we don't use sigmoid/softmax)
    alphas[alphas < 0] = 0
    alphas[alphas > 1] = 1
    # alphas = (np.array(alphas)-min(alphas)) / (max(alphas) - min(alphas))
    count = 0
    for i in range(num_cells):
        for j in range(num_cells):
            pos_i = i
            pos_j = j
            rec = Rectangle((pos_i, pos_j), 1, 1, facecolor='r', alpha=alphas[count],
                            edgecolor='k')
            plt.gca().add_patch(rec)
            count += 1

    # draw lava again on top so it remains visible under belief overlays
    for (x, y) in lava_positions:
        rec = Rectangle((x, y), 1, 1, facecolor='orange', alpha=0.8, edgecolor='k')
        plt.gca().add_patch(rec)


def _build_lava_layout_11():
    """
    A fixed 11x11 lava layout with open borders and interior barriers.
    """
    lava = set()

    # horizontal lava bands (with holes)
    for x in range(2, 9):
        if x not in {4, 7}:
            lava.add((x, 3))
    for x in range(2, 9):
        if x not in {3, 6}:
            lava.add((x, 6))
    for x in range(2, 9):
        if x not in {5}:
            lava.add((x, 8))

    # vertical lava bands (with holes)
    for y in range(2, 9):
        if y not in {3, 7}:
            lava.add((4, y))
    for y in range(2, 9):
        if y not in {5}:
            lava.add((7, y))

    # keep the start region free
    lava.discard((0, 0))
    lava.discard((0, 1))
    lava.discard((1, 0))
    lava.discard((1, 1))
    # keep an easy corridor on the left/top borders
    for y in range(0, 11):
        lava.discard((0, y))
    for x in range(0, 11):
        lava.discard((x, 10))
    return lava


class GridNaviLava11(GridNavi):
    def __init__(self, num_cells=11, num_steps=30, lava_penalty=-1.0):
        if num_cells != 11:
            raise ValueError("GridNaviLava11 is designed for num_cells=11.")
        super().__init__(
            num_cells=num_cells,
            num_steps=num_steps,
            lava_positions=_build_lava_layout_11(),
            lava_penalty=lava_penalty,
        )


# Backward-compatible alias: existing runs/commands might still reference Lava16.
class GridNaviLava16(GridNaviLava11):
    def __init__(self, num_cells=11, num_steps=30, lava_penalty=-1.0):
        super().__init__(num_cells=num_cells, num_steps=num_steps, lava_penalty=lava_penalty)
