# Experiment Configuration

Centralized configuration schemas consolidating all experiment hyperparameters, paths, and environment specifications.

## What it's for

Provides a single source of truth for all configurable aspects of experiments. Used by the runner (runner.py) and all optimizer/data modules to access hyperparameters and environment settings.

## What it contains

### Configuration Classes
- `ModelConfig` - Model architecture parameters (hidden_size, etc.)
- `OptimizerConfig` - Optimizer hyperparameters (learning_rate, population_size, sigma values)
- `TrainingConfig` - Training loop parameters (batch_size, max_time, evaluation intervals)
- `DataConfig` - Data split ratios and continual learning flags
- `ExperimentConfig` - Top-level config aggregating all sub-configs plus experiment metadata

### Environment Specifications
- `ENV_CONFIGS` - Dictionary mapping environment names to observation/action dimensions and data files
- `get_data_file()` - Helper to retrieve data filename for specific environment and subject

### Global State
- `DEVICE` - Global torch device (CPU or CUDA)
- `set_device()` - Function to configure device based on GPU index
- Path constants: `PROJECT_ROOT`, `DATA_DIR`, `RESULTS_DIR`

## Key Details

All experiment parameters are consolidated here to avoid scattered magic numbers throughout the codebase. The dataclass-based structure provides type safety and clear documentation of available options. Environment configs define the properties of each task (CartPole, MountainCar, Acrobot, LunarLander) including observation/action dimensions and subject data mappings. The global DEVICE variable is set once at experiment startup and used throughout the platform for tensor operations.
