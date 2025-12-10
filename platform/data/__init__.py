"""Data loading and preprocessing for behavior modeling datasets.

This module handles loading data from HuggingFace datasets and human behavioral
data files, as well as preprocessing including continual learning features and
episode-based datasets.

Functions:
    load_cartpole_data: Load CartPole-v1 from HuggingFace
    load_lunarlander_data: Load LunarLander-v2 from HuggingFace
    load_human_data: Load human behavioral data from JSON files
    compute_session_run_ids: Compute session and run IDs from timestamps
    normalize_session_run_features: Normalize session/run IDs for CL features

Classes:
    EpisodeDataset: PyTorch Dataset for episode-based training
"""

__all__ = [
    "load_cartpole_data",
    "load_lunarlander_data",
    "load_human_data",
    "compute_session_run_ids",
    "normalize_session_run_features",
    "EpisodeDataset",
    "episode_collate_fn",
]
