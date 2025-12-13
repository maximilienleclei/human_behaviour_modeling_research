"""Experiment configuration dataclasses.

Defines structured configurations for models, optimizers, training, and data.
Used by both dl/ and ne/ modules.
"""

from dataclasses import dataclass, field


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
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
