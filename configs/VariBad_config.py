from dataclasses import dataclass
#TODO: If you change args that relate to the VeriBad implementation then you will need to change this config
@dataclass
class Config:
    algo: str = "ppo"
    seed: int = 0
    updates: int = 400
    rollout_len: int = 64

    grid_size: int = 5
    episode_len: int = 15
    step_penalty: float = -0.1
    goal_reward: float = 1.0

    latent_dim: int = 8
    encoder_hidden_dim: int = 64
    policy_hidden_dim: int = 128

    policy_lr: float = 3e-4
    vae_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95

    ppo_clip_ratio: float = 0.2
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    kl_weight: float = 0.1
    state_loss_coef: float = 0.5

    log_every: int = 10
    device: str = "cpu"