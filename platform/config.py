"""Configuration schemas for experiments.

Consolidates all hyperparameters:
- Model configs (hidden_size, architecture params)
- Optimizer configs (lr, population_size, sigma, etc.)
- Training configs (batch_size, max_time, etc.)
- Data configs (split ratios, CL features, etc.)
- Environment configs and paths
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch


# Device configuration (will be set via set_device() based on --gpu argument)
DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_device(gpu_index: int | None = None) -> None:
    """Set the global DEVICE variable.

    Args:
        gpu_index: GPU index to use, or None for CPU
    """
    global DEVICE
    if gpu_index is None:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(f"cuda:{gpu_index}")


# Directory paths (relative to project root)
PLATFORM_DIR: Path = Path(__file__).parent
PROJECT_ROOT: Path = PLATFORM_DIR.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_size: int = 50
    # Add more model-specific params as needed


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    # SGD params
    learning_rate: float = 1e-3

    # GA params
    population_size: int = 50
    adaptive_sigma_init: float = 1e-3
    adaptive_sigma_noise: float = 1e-2


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    max_optim_time: int = 36000  # seconds (10 hours)
    loss_eval_interval_seconds: int = 60
    ckpt_and_behav_eval_interval_seconds: int = 300


@dataclass
class DataConfig:
    """Data configuration."""
    holdout_pct: float = 0.1  # test split
    val_pct: float = 0.1  # validation split
    use_cl_info: bool = False  # continual learning features
    num_eval_episodes: int = 100


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Experiment metadata
    experiment_number: int
    dataset: str  # cartpole, mountaincar, acrobot, lunarlander
    method: str  # SGD_reservoir, adaptive_ga_trainable, etc.
    subject: str = "sub01"
    seed: int = 42

    # Sub-configs
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()


# Environment configurations
ENV_CONFIGS: dict[str, dict] = {
    "cartpole": {
        "data_files": {
            "sub01": "sub01_data_cartpole.json",
            "sub02": "sub02_data_cartpole.json",
        },
        "obs_dim": 4,
        "action_dim": 2,
        "name": "CartPole",
    },
    "mountaincar": {
        "data_files": {
            "sub01": "sub01_data_mountaincar.json",
            "sub02": "sub02_data_mountaincar.json",
        },
        "obs_dim": 2,
        "action_dim": 3,
        "name": "MountainCar",
    },
    "acrobot": {
        "data_files": {
            "sub01": "sub01_data_acrobot.json",
            "sub02": "sub02_data_acrobot.json",
        },
        "obs_dim": 6,
        "action_dim": 3,
        "name": "Acrobot",
    },
    "lunarlander": {
        "data_files": {
            "sub01": "sub01_data_lunarlander.json",
            "sub02": "sub02_data_lunarlander.json",
        },
        "obs_dim": 8,
        "action_dim": 4,
        "name": "LunarLander",
    },
}


def get_data_file(env_name: str, subject: str) -> str:
    """Get data filename for environment and subject.

    Args:
        env_name: Environment name
        subject: Subject identifier (sub01, sub02)

    Returns:
        Data filename

    Raises:
        ValueError: If environment or subject not found
    """
    if env_name not in ENV_CONFIGS:
        raise ValueError(f"Unknown environment: {env_name}")

    env_config: dict = ENV_CONFIGS[env_name]

    if subject not in env_config["data_files"]:
        available_subjects: list[str] = list(env_config["data_files"].keys())
        raise ValueError(
            f"No data for subject '{subject}' in environment '{env_name}'. "
            f"Available: {available_subjects}"
        )

    return env_config["data_files"][subject]
