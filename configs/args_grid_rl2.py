import argparse
from utils.helpers import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # --- GENERAL ---
    parser.add_argument('--num_frames', type=int, default=2e7, help='number of frames to train')
    parser.add_argument('--max_rollouts_per_task', type=int, default=4, help='number of MDP episodes for adaptation')
    parser.add_argument('--exp_label', default='rl2', help='label (typically name of method)')
    parser.add_argument('--env_name', default='GridNavi-v0', help='environment to train on')

    # --- RL2 policy / optimiser ---
    parser.add_argument('--lr_policy', type=float, default=7e-4, help='learning rate')
    parser.add_argument('--policy_eps', type=float, default=1e-8, help='optimizer epsilon')
    parser.add_argument('--num_processes', type=int, default=16, help='number of parallel environments')
    parser.add_argument('--policy_num_steps', type=int, default=60, help='rollout horizon before each update')
    parser.add_argument('--policy_gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--policy_tau', type=float, default=0.95, help='GAE parameter')
    parser.add_argument('--policy_value_loss_coef', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument('--policy_entropy_coef', type=float, default=0.01, help='entropy bonus coefficient')
    parser.add_argument('--policy_max_grad_norm', type=float, default=0.5, help='max grad norm')
    parser.add_argument('--rl2_hidden_size', type=int, default=128, help='GRU hidden size for RL2 policy')
    parser.add_argument('--pass_belief_to_policy', type=boolean_argument, default=False, help='kept for compatibility with shared env_step helper')

    # If False, keep task fixed across max_rollouts_per_task episodes (fair vs VariBad)
    parser.add_argument('--reset_task_on_episode', type=boolean_argument, default=False)

    # --- logging / misc ---
    parser.add_argument('--log_interval', type=int, default=20000, help='log interval in frames')
    parser.add_argument('--vis_interval', type=int, default=20000, help='visualisation interval in frames')
    parser.add_argument('--save_interval', type=int, default=1000, help='unused placeholder for compatibility')
    parser.add_argument('--eval_interval', type=int, default=500, help='unused placeholder for compatibility')
    parser.add_argument('--seed', type=int, default=73)

    return parser.parse_args(rest_args)
