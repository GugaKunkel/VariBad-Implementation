"""
Main script to start experiments
"""
from configs import args_grid_varibad
from metalearner import MetaLearner

def main():
    # TODO: Build args for grid_dqn
    args = args_grid_varibad.get_args(None)
    learner = MetaLearner(args)
    learner.train()

if __name__ == '__main__':
    main()
