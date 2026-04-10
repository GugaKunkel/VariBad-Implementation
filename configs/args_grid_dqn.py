import argparse
from utils.helpers import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # --- GENERAL ---

    parser.add_argument('--num_frames', type=int, default=2e7, help='number of frames to train')
    parser.add_argument('--max_rollouts_per_task', type=int, default=4, help='number of MDP episodes for adaptation')
    parser.add_argument('--exp_label', default='dqn', help='label (typically name of method)')
    parser.add_argument('--env_name', default='GridNavi-v0', help='environment to train on')

    # other hyperparameters
    parser.add_argument('--lr_policy', type=float, default=0.0025, help='learning rate (default: 0025)')
    parser.add_argument('--num_processes', type=int, default=16, help='how many training CPU processes / parallel environments to use (default: 16)')
    parser.add_argument('--size_buffer', type=int, default=100000, help='how many trajectories (!) to keep in buffer')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--tau', type=float, default=1.0, help='target network update rate')
    parser.add_argument('--target_network_update_freq', type=int, default=500, help='how many steps to update the target network')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for updates')
    parser.add_argument('--start_e', type=float, default=1.0, help='starting value of epsilon for exploration')
    parser.add_argument('--end_e', type=float, default=0.05, help='final value of epsilon for exploration')
    parser.add_argument('--exploration_fraction', type=float, default=0.5, help='fraction of total training period over which the exploration rate is annealed')
    parser.add_argument('--learning_starts', type=int, default=10000, help='number of steps before learning starts')
    parser.add_argument('--train_frequency', type=int, default=10, help='number of steps between each training update')
    parser.add_argument('--reset_task_on_episode', type=boolean_argument, default=True, help='if True, call env.unwrapped.reset_task() on every env reset when available (needed for GridNavi task randomization)')
    parser.add_argument('--seed', type=int, default=73)

    # --- OTHERS ---

    # logging, saving, evaluation
    parser.add_argument('--log_interval', type=int, default=500, help='log interval, one log per n updates')
    parser.add_argument('--save_interval', type=int, default=1000, help='save interval, one save per n updates')
    parser.add_argument('--save_intermediate_models', type=boolean_argument, default=False, help='save all models')
    parser.add_argument('--eval_interval', type=int, default=500, help='eval interval, one eval per n updates')
    parser.add_argument('--vis_interval', type=int, default=500, help='visualisation interval, one eval per n updates')

    return parser.parse_args(rest_args)