"""Shared configuration module for experiments.

Centralizes all configuration schemas:
- device: DEVICE and set_device()
- environments: ENV_CONFIGS and environment utilities
- paths: Project directory paths
- state: State persistence configuration
- experiments: Experiment configuration dataclasses
"""

from config.device import DEVICE, set_device
from config.environments import ENV_CONFIGS, get_data_file
from config.experiments import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from config.paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from config.state import StatePersistenceConfig

__all__ = [
    # Device
    "DEVICE",
    "set_device",
    # Environments
    "ENV_CONFIGS",
    "get_data_file",
    # Paths
    "DATA_DIR",
    "PROJECT_ROOT",
    "RESULTS_DIR",
    # State
    "StatePersistenceConfig",
    # Experiments
    "ModelConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "DataConfig",
    "ExperimentConfig",
]
