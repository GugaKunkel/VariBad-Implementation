"""
Main script to start experiments
"""
import argparse
from configs import args_grid_varibad, args_grid_dqn, args_grid_rl2, args_grid_lg_varibad
from metalearner import MetaLearner
from dqn_learner import DQNLearner
# from rl2_learner import RL2Learner

def main(experiment: str, rest_args=None):
    experiment = experiment.lower()
    if experiment == "varibad":
        args = args_grid_varibad.get_args(rest_args)
        args.algo_name = "varibad"
        learner = MetaLearner(args)
        learner.train()
    elif experiment == "varibad_lg":
        args = args_grid_lg_varibad.get_args(rest_args)
        args.algo_name = "varibad"
        learner = MetaLearner(args)
        learner.train()
    elif experiment == "dqn":
        args = args_grid_dqn.get_args(rest_args)
        args.algo_name = "dqn"
        learner = DQNLearner(args)
        learner.train()
    elif experiment == "rl2":
        args = args_grid_rl2.get_args(rest_args)
        args.algo_name = "rl2"
        # clean up arguments
        if args.disable_decoder:
            args.decode_reward = False
        learner = MetaLearner(args)
        learner.train()
        # learner = RL2Learner(args)
        # learner.train()
    else:
        raise ValueError(f"Unknown experiment '{experiment}'. Use 'varibad', 'varibad_lg', 'dqn', or 'rl2'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["varibad", "varibad_lg", "dqn", "rl2"], help="Which experiment to run")
    parsed_args, rest_args = parser.parse_known_args()
    main(parsed_args.experiment, rest_args)
