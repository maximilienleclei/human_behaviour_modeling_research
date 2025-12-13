# HuggingFace Control Tasks

Loaders for pre-trained agent trajectories from HuggingFace datasets.

## Purpose

Provides access to RL agent behavior data from NathanGavenski's HuggingFace repositories. These datasets contain observation-action pairs from trained agents, useful for behavior cloning and imitation learning experiments.

## Contents

### loaders.py
- `load_cartpole_data()` - CartPole-v1 dataset loader
- `load_lunarlander_data()` - LunarLander-v2 dataset loader

## Datasets

Data is downloaded from HuggingFace hub:
- NathanGavenski/CartPole-v1 - Trained agent playing CartPole
- NathanGavenski/LunarLander-v2 - Trained agent playing LunarLander

All loaders return 90/10 train/test splits with shuffling. No temporal or session information is included (unlike simexp human behavioral data).
