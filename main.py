"""
Main script to start experiments
"""
import argparse
from configs import args_grid_varibad, args_grid_dqn, args_grid_recurrent_dqn
from metalearner import MetaLearner
from dqn_learner import DQNLearner
from recurrent_dqn_learner import RecurrentDQNLearner

def main(experiment: str, rest_args=None):
    experiment = experiment.lower()
    if experiment == "varibad":
        args = args_grid_varibad.get_args(rest_args)
        learner = MetaLearner(args)
        learner.train()
    elif experiment == "dqn":
        args = args_grid_dqn.get_args(rest_args)
        learner = DQNLearner(args)
        learner.train()
    elif experiment in ["recurrent_dqn", "rdqn"]:
        args = args_grid_recurrent_dqn.get_args(rest_args)
        learner = RecurrentDQNLearner(args)
        learner.train()
    else:
        raise ValueError(f"Unknown experiment '{experiment}'. Use 'varibad', 'dqn', or 'recurrent_dqn'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["varibad", "dqn", "recurrent_dqn", "rdqn"], help="Which experiment to run")
    parsed_args, rest_args = parser.parse_known_args()
    main(parsed_args.experiment, rest_args)
