import argparse

from metalearner import Config, MetaLearner


def parse_args():
    parser = argparse.ArgumentParser(description="Bare-minimum VariBAD-style MVP on GridWorld")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c"])
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--rollout-len", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--episode-len", type=int, default=15)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        algo=args.algo,
        updates=args.updates,
        rollout_len=args.rollout_len,
        grid_size=args.grid_size,
        episode_len=args.episode_len,
        latent_dim=args.latent_dim,
        seed=args.seed,
        device=args.device,
        policy_lr=args.policy_lr,
        vae_lr=args.vae_lr,
    )

    learner = MetaLearner(cfg)
    learner.train()


if __name__ == "__main__":
    main()
