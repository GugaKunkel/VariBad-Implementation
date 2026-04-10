"""
Main script to start experiments
"""
import argparse
from configs import args_grid_varibad, args_grid_dqn
from metalearner import MetaLearner
from dqn_learner import DQNLearner

def main(experiment: str):
    experiment = experiment.lower()
    if experiment == "varibad":
        args = args_grid_varibad.get_args(None)
        learner = MetaLearner(args)
        learner.train()
    elif experiment == "dqn":
        args = args_grid_dqn.get_args(None)
        learner = DQNLearner(args)
        learner.train()
    else:
        raise ValueError(f"Unknown experiment '{experiment}'. Use 'varibad' or 'dqn'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["varibad", "dqn"], help="Which experiment to run")
    parsed_args = parser.parse_args()
    main(parsed_args.experiment)