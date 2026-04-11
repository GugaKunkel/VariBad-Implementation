import argparse
import datetime
import json
import os
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import environments  # noqa: F401 - ensures env registration
from dqn_learner import QNetwork
from environments.parallel_envs import make_vec_envs
from environments.navigation.gridworld import plot_bb
from utils import helpers as utl


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _latest_run_dir(logs_root: str, algo_name: str) -> Optional[str]:
    algo_dir = os.path.join(logs_root, algo_name)
    if not os.path.isdir(algo_dir):
        return None
    run_dirs = [
        os.path.join(algo_dir, d)
        for d in os.listdir(algo_dir)
        if os.path.isdir(os.path.join(algo_dir, d))
    ]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return run_dirs[0]


def _load_run_config(run_dir: str) -> SimpleNamespace:
    cfg_path = os.path.join(run_dir, "config.json")
    cfg = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    defaults = {
        "add_nonlinearity_to_latent": False,
        "pass_belief_to_policy": False,
        "max_rollouts_per_task": 3,
    }
    defaults.update(cfg)
    return SimpleNamespace(**defaults)


def _parse_tasks(task_string: str) -> List[Tuple[int, int]]:
    tasks = []
    for chunk in task_string.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        xy = chunk.split(",")
        if len(xy) != 2:
            raise ValueError(f"Invalid task '{chunk}'. Expected format x,y.")
        tasks.append((int(xy[0]), int(xy[1])))
    return tasks


def _select_tasks(env_name: str, num_tasks: int, task_string: Optional[str], seed: int) -> List[Tuple[int, int]]:
    env = gym.make(env_name)
    try:
        possible_goals = [tuple(map(int, g)) for g in env.unwrapped.possible_goals]
    finally:
        env.close()

    if task_string:
        tasks = _parse_tasks(task_string)
        invalid = [t for t in tasks if t not in possible_goals]
        if invalid:
            raise ValueError(f"These tasks are invalid for {env_name}: {invalid}")
        return tasks

    if num_tasks > len(possible_goals):
        raise ValueError(f"Requested {num_tasks} tasks but env only has {len(possible_goals)} possible goals.")
    rng = random.Random(seed)
    return rng.sample(possible_goals, num_tasks)


def _build_latent_for_policy(args, latent_mean, latent_logvar):
    if getattr(args, "add_nonlinearity_to_latent", False):
        latent_mean = torch.relu(latent_mean)
        latent_logvar = torch.relu(latent_logvar)
    latent = torch.cat((latent_mean, latent_logvar), dim=-1)
    if latent.dim() > 1 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    return latent


class _PlotArgs:
    pass


@dataclass
class MetaModel:
    name: str
    run_dir: str
    args: SimpleNamespace
    policy: torch.nn.Module
    encoder: torch.nn.Module
    reward_decoder: Optional[torch.nn.Module]


@dataclass
class DQNModel:
    name: str
    run_dir: str
    q_network: torch.nn.Module


def _load_meta_model(name: str, run_dir: Optional[str], logs_root: str) -> Optional[MetaModel]:
    if run_dir is None:
        run_dir = _latest_run_dir(logs_root, name.lower())
    if run_dir is None:
        print(f"[{name}] No run directory found under logs.")
        return None

    model_dir = os.path.join(run_dir, "models")
    policy_path = os.path.join(model_dir, "policy.pt")
    encoder_path = os.path.join(model_dir, "encoder.pt")
    reward_decoder_path = os.path.join(model_dir, "reward_decoder.pt")
    if not (os.path.isfile(policy_path) and os.path.isfile(encoder_path)):
        print(f"[{name}] Missing policy/encoder checkpoint in: {model_dir}")
        return None

    policy = _safe_torch_load(policy_path).to(device).eval()
    encoder = _safe_torch_load(encoder_path).to(device).eval()
    reward_decoder = None
    if os.path.isfile(reward_decoder_path):
        maybe_decoder = _safe_torch_load(reward_decoder_path)
        if maybe_decoder is not None:
            reward_decoder = maybe_decoder.to(device).eval()
    args = _load_run_config(run_dir)
    print(f"[{name}] Loaded run: {run_dir}")
    return MetaModel(
        name=name,
        run_dir=run_dir,
        args=args,
        policy=policy,
        encoder=encoder,
        reward_decoder=reward_decoder,
    )


def _load_dqn_model(run_dir: Optional[str], dqn_checkpoint: Optional[str], logs_root: str, env_name: str) -> Optional[DQNModel]:
    if dqn_checkpoint is None:
        if run_dir is None:
            run_dir = _latest_run_dir(logs_root, "dqn")
        if run_dir is not None:
            dqn_checkpoint = os.path.join(run_dir, "models", "latest.pt")

    if dqn_checkpoint is None or not os.path.isfile(dqn_checkpoint):
        print("[DQN] No checkpoint found; skipping DQN evaluation.")
        return None

    checkpoint = _safe_torch_load(dqn_checkpoint)
    env = gym.make(env_name)
    try:
        space_holder = SimpleNamespace(
            single_observation_space=env.observation_space,
            single_action_space=env.action_space,
        )
    finally:
        env.close()
    q_network = QNetwork(space_holder).to(device)
    q_network.load_state_dict(checkpoint["q_network"])
    q_network.eval()

    run_dir = run_dir or os.path.dirname(os.path.dirname(dqn_checkpoint))
    print(f"[DQN] Loaded checkpoint: {dqn_checkpoint}")
    return DQNModel(name="DQN", run_dir=run_dir, q_network=q_network)


def _plot_behaviour_for_task(
    env,
    episode_all_obs,
    episode_goals,
    episodes_per_task: int,
    image_folder: str,
    filename_prefix: str,
    rew_pred_type: str = "bernoulli",
    reward_decoder=None,
    episode_latent_means=None,
    episode_latent_logvars=None,
    episode_beliefs=None,
):
    plot_env = env
    if not hasattr(plot_env, "_max_episode_steps") and hasattr(plot_env, "unwrapped"):
        plot_env = plot_env.unwrapped

    plot_args = _PlotArgs()
    plot_args.max_rollouts_per_task = episodes_per_task
    plot_args.rew_pred_type = rew_pred_type
    plot_bb(
        env=plot_env,
        args=plot_args,
        episode_all_obs=episode_all_obs,
        episode_goals=episode_goals,
        reward_decoder=reward_decoder,
        episode_latent_means=episode_latent_means,
        episode_latent_logvars=episode_latent_logvars,
        image_folder=image_folder,
        iter_idx=filename_prefix,
        episode_beliefs=episode_beliefs,
    )
    print(f"Saved behaviour plot: {os.path.join(image_folder, filename_prefix)}_behaviour")


def _evaluate_meta_model(
    model: MetaModel,
    env_name: str,
    tasks: Sequence[Tuple[int, int]],
    episodes_per_task: int,
    seed: int,
    output_root: str,
) -> np.ndarray:
    model_dir = os.path.join(output_root, model.name.lower())
    os.makedirs(model_dir, exist_ok=True)

    returns = np.zeros((len(tasks), episodes_per_task), dtype=np.float32)

    eval_args = SimpleNamespace(
        env_name=env_name,
        seed=seed,
        num_processes=1,
        policy_gamma=getattr(model.args, "policy_gamma", 0.95),
        pass_belief_to_policy=bool(getattr(model.args, "pass_belief_to_policy", False)),
        add_nonlinearity_to_latent=bool(getattr(model.args, "add_nonlinearity_to_latent", False)),
        add_done_info=bool(getattr(model.args, "max_rollouts_per_task", 1) > 1),
        max_rollouts_per_task=episodes_per_task,
    )

    for task_idx, task in enumerate(tasks):
        show_vari_bad_beliefs = model.name.lower() == "varibad" and model.reward_decoder is not None
        envs = make_vec_envs(
            env_name=env_name,
            seed=seed + task_idx * 1000,
            num_processes=1,
            gamma=eval_args.policy_gamma,
            device=device,
            episodes_per_task=episodes_per_task,
            normalise_rew=True,
            ret_rms=None,
            tasks=[task],
            rank_offset=10_000 + task_idx * 10,
            add_done_info=eval_args.add_done_info,
        )
        try:
            state, belief = utl.reset_env(envs, eval_args)
            latent_sample, latent_mean, latent_logvar, hidden_state = model.encoder.prior(1)
            latent_sample = latent_sample[0].to(device)
            latent_mean = latent_mean[0].to(device)
            latent_logvar = latent_logvar[0].to(device)

            episode_all_obs = []
            episode_goals = []
            if show_vari_bad_beliefs:
                episode_latent_means = [[] for _ in range(episodes_per_task)]
                episode_latent_logvars = [[] for _ in range(episodes_per_task)]
            else:
                episode_latent_means = None
                episode_latent_logvars = None
            max_steps = envs._max_episode_steps

            for ep in range(episodes_per_task):
                ep_return = 0.0
                traj_obs = [state.clone().detach().cpu()]
                if show_vari_bad_beliefs:
                    episode_latent_means[ep].append(latent_mean[0].clone())
                    episode_latent_logvars[ep].append(latent_logvar[0].clone())

                for _ in range(max_steps):
                    with torch.no_grad():
                        _, action = utl.select_action(
                            args=eval_args,
                            policy=model.policy,
                            state=state.view(-1),
                            belief=belief,
                            deterministic=True,
                            latent_sample=latent_sample.view(-1),
                            latent_mean=latent_mean.view(-1),
                            latent_logvar=latent_logvar.view(-1),
                        )

                    [state, belief], (rew_raw, _), terminated, truncated, infos = utl.env_step(envs, action, eval_args)
                    done_task = bool(np.logical_or(terminated, truncated)[0])
                    done_mdp = bool(infos[0]["done_mdp"])
                    ep_return += float(rew_raw.view(-1)[0].item())

                    # For Meta-RL adaptation we keep hidden state across MDP episodes inside a task.
                    with torch.no_grad():
                        latent_sample, latent_mean, latent_logvar, hidden_state = model.encoder(
                            actions=action.float().to(device),
                            states=state,
                            rewards=rew_raw.reshape((1, 1)).float().to(device),
                            hidden_state=hidden_state,
                            return_prior=False,
                        )
                    if show_vari_bad_beliefs:
                        episode_latent_means[ep].append(latent_mean[0].clone())
                        episode_latent_logvars[ep].append(latent_logvar[0].clone())
                    traj_obs.append(state.clone().detach().cpu())

                    if done_mdp:
                        if not done_task:
                            start_state = infos[0]["start_state"]
                            state = torch.from_numpy(start_state).float().reshape((1, -1)).to(device)
                        break

                returns[task_idx, ep] = ep_return
                episode_all_obs.append(traj_obs)
                episode_goals.append(np.array(task, dtype=np.int64))

            if show_vari_bad_beliefs:
                episode_latent_means = [torch.stack(e) for e in episode_latent_means]
                episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]
            prefix = f"task_{task_idx}_{task[0]}_{task[1]}"
            _plot_behaviour_for_task(
                env=envs,
                episode_all_obs=episode_all_obs,
                episode_goals=episode_goals,
                episodes_per_task=episodes_per_task,
                image_folder=model_dir,
                filename_prefix=prefix,
                rew_pred_type=getattr(model.args, "rew_pred_type", "bernoulli"),
                reward_decoder=model.reward_decoder if show_vari_bad_beliefs else None,
                episode_latent_means=episode_latent_means,
                episode_latent_logvars=episode_latent_logvars,
            )
        finally:
            envs.close()
    return returns


def _evaluate_dqn_model(
    model: DQNModel,
    env_name: str,
    tasks: Sequence[Tuple[int, int]],
    episodes_per_task: int,
    seed: int,
    output_root: str,
) -> np.ndarray:
    model_dir = os.path.join(output_root, model.name.lower())
    os.makedirs(model_dir, exist_ok=True)

    returns = np.zeros((len(tasks), episodes_per_task), dtype=np.float32)
    env = gym.make(env_name)
    try:
        for task_idx, task in enumerate(tasks):
            env.unwrapped.reset_task(task)
            episode_all_obs = []
            episode_goals = []
            for ep in range(episodes_per_task):
                obs, _ = env.reset(seed=seed + task_idx * 100 + ep)
                done = False
                ep_return = 0.0
                traj_obs = [torch.tensor(obs, dtype=torch.float32).unsqueeze(0)]

                while not done:
                    with torch.no_grad():
                        q_vals = model.q_network(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                        action = int(torch.argmax(q_vals, dim=1).item())
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated)
                    ep_return += float(reward)
                    obs = next_obs
                    traj_obs.append(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))

                returns[task_idx, ep] = ep_return
                episode_all_obs.append(traj_obs)
                episode_goals.append(np.array(task, dtype=np.int64))

            prefix = f"task_{task_idx}_{task[0]}_{task[1]}"
            _plot_behaviour_for_task(
                env=env,
                episode_all_obs=episode_all_obs,
                episode_goals=episode_goals,
                episodes_per_task=episodes_per_task,
                image_folder=model_dir,
                filename_prefix=prefix,
                rew_pred_type="bernoulli",
                reward_decoder=None,
                episode_latent_means=None,
                episode_latent_logvars=None,
            )
    finally:
        env.close()
    return returns


def _save_combined_chart(
    output_root: str,
    episodes_per_task: int,
    model_returns: List[Tuple[str, np.ndarray]],
):
    plt.figure(figsize=(7, 4))
    x = np.arange(1, episodes_per_task + 1)
    for model_name, returns in model_returns:
        mean_per_episode = returns.mean(axis=0)
        plt.plot(x, mean_per_episode, marker="o", label=model_name)
    plt.xlabel("Episode Within Task")
    plt.ylabel("Average Return (across tasks)")
    plt.title("Average Return Across Episodes")
    plt.xticks(x)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    chart_path = os.path.join(output_root, "combined_avg_returns.png")
    plt.savefig(chart_path, dpi=180)
    plt.close()
    print(f"Saved combined returns chart: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved VariBAD / RL2 / DQN models on fixed GridNavi tasks.")
    parser.add_argument("--env_name", type=str, default="GridNavi-v0")
    parser.add_argument("--logs_root", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--num_tasks", type=int, default=4)
    parser.add_argument("--tasks", type=str, default=None, help="Specific tasks as 'x1,y1;x2,y2;...'")
    parser.add_argument("--episodes_per_task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=73)

    parser.add_argument("--varibad_run", type=str, default=None, help="Path to a specific varibad run dir.")
    parser.add_argument("--rl2_run", type=str, default=None, help="Path to a specific rl2 run dir.")
    parser.add_argument("--dqn_run", type=str, default=None, help="Path to a specific dqn run dir.")
    parser.add_argument("--dqn_checkpoint", type=str, default=None, help="Path to a specific DQN checkpoint (latest.pt).")

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%H_%M_%S__%d_%m")
    output_root = args.output_dir or os.path.join(args.logs_root, "model_tests", timestamp)
    os.makedirs(output_root, exist_ok=True)

    tasks = _select_tasks(args.env_name, args.num_tasks, args.tasks, args.seed)
    print(f"Using tasks: {tasks}")
    print(f"Writing outputs under: {output_root}")

    model_returns: List[Tuple[str, np.ndarray]] = []

    varibad_model = _load_meta_model("varibad", args.varibad_run, args.logs_root)
    if varibad_model is not None:
        returns = _evaluate_meta_model(
            model=varibad_model,
            env_name=args.env_name,
            tasks=tasks,
            episodes_per_task=args.episodes_per_task,
            seed=args.seed,
            output_root=output_root,
        )
        model_returns.append(("VariBAD", returns))
        print(f"[VariBAD] Mean returns by episode: {returns.mean(axis=0).tolist()}")

    rl2_model = _load_meta_model("rl2", args.rl2_run, args.logs_root)
    if rl2_model is not None:
        returns = _evaluate_meta_model(
            model=rl2_model,
            env_name=args.env_name,
            tasks=tasks,
            episodes_per_task=args.episodes_per_task,
            seed=args.seed,
            output_root=output_root,
        )
        model_returns.append(("RL2", returns))
        print(f"[RL2] Mean returns by episode: {returns.mean(axis=0).tolist()}")

    dqn_model = _load_dqn_model(args.dqn_run, args.dqn_checkpoint, args.logs_root, args.env_name)
    if dqn_model is not None:
        returns = _evaluate_dqn_model(
            model=dqn_model,
            env_name=args.env_name,
            tasks=tasks,
            episodes_per_task=args.episodes_per_task,
            seed=args.seed,
            output_root=output_root,
        )
        model_returns.append(("DQN", returns))
        print(f"[DQN] Mean returns by episode: {returns.mean(axis=0).tolist()}")

    if not model_returns:
        print("No models were loaded. Nothing to evaluate.")
        return

    _save_combined_chart(output_root, args.episodes_per_task, model_returns)
    print("Evaluation finished.")


if __name__ == "__main__":
    main()
