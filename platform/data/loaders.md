# Data Loading Functions

Functions for loading behavioral data from HuggingFace datasets (CartPole, LunarLander) and local JSON files (human data).

## What it's for

Provides unified interface for loading training data from different sources. Handles downloading HuggingFace datasets, reading local JSON files, computing temporal features, and returning PyTorch tensors ready for training.

## What it contains

### HuggingFace Loaders
- `load_cartpole_data()` - Loads CartPole-v1 dataset from HuggingFace, returns train/test splits
- `load_lunarlander_data()` - Loads LunarLander-v2 dataset from HuggingFace, returns train/test splits

### Human Data Loader
- `load_human_data()` - Loads human behavioral data from JSON files with optional continual learning features

## Key Details

The HuggingFace loaders download RL agent trajectories and perform 90/10 train/test splits with shuffling. The human data loader reads JSON files containing episode observations, actions, timestamps, seeds, and returns. When `use_cl_features=True`, it computes and normalizes session/run IDs (using preprocessing.py functions) and appends them to observations as additional input features. All loaders return tensors with consistent shapes that work across the platform. Human data files are specified in platform/config.py's ENV_CONFIGS dictionary.
