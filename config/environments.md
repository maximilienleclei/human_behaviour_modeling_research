# Environment Configurations

Environment-specific settings for control tasks (CartPole, MountainCar, Acrobot, LunarLander).

## Purpose
Centralizes environment metadata used by both data loaders and evaluation code.

## Contents
- `ENV_CONFIGS` - Dictionary mapping environment names to configuration dicts containing:
  - `data_files` - Subject-specific data filenames
  - `obs_dim` - Observation space dimensionality
  - `action_dim` - Action space dimensionality
  - `name` - Human-readable environment name
- `get_data_file()` - Utility to get data filename for specific environment/subject combination
