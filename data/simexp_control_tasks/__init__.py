"""SimExp control tasks human behavioral data.

Loaders, configurations, and preprocessing for human gameplay episodes
collected through SimExp experiments.

Functions:
    load_human_data: Load human behavioral data from JSON files
    get_data_file: Get data filename for environment/subject
    compute_session_run_ids: Compute session/run IDs from timestamps
    normalize_session_run_features: Normalize session/run IDs to [-1,1]

Constants:
    ENV_CONFIGS: Environment configuration dictionary

Classes:
    EpisodeDataset: PyTorch Dataset for episode-based training
"""

from data.simexp_control_tasks.environments import ENV_CONFIGS, get_data_file
from data.simexp_control_tasks.loaders import load_human_data
from data.simexp_control_tasks.preprocessing import (
    EpisodeDataset,
    compute_session_run_ids,
    episode_collate_fn,
    normalize_session_run_features,
)

__all__ = [
    "load_human_data",
    "ENV_CONFIGS",
    "get_data_file",
    "compute_session_run_ids",
    "normalize_session_run_features",
    "EpisodeDataset",
    "episode_collate_fn",
]
