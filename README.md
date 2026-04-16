# VariBAD Implementation

Compact implementation of meta-RL experiments for GridNavi, including VariBAD, RL2-style training, and DQN baselines.

## Quick Start

1. Create and activate a Python environment (3.9+ recommended).
2. Install dependencies used by your setup (core libs include `torch`, `gymnasium`, `numpy`, `matplotlib`, `seaborn`, and `tensorboard`).

Run an experiment:

```bash
python main.py varibad
python main.py rl2
python main.py dqn
python main.py varibad_lg
```

## Repo Layout

- `main.py` - experiment entrypoint
- `metalearner.py` - main meta-learning training loop
- `dqn_learner.py` - DQN baseline learner
- `configs/` - experiment argument presets
- `environments/` - GridNavi and environment utilities
- `algorithms/` - PPO and rollout storage
- `trained_models/` - saved checkpoints
