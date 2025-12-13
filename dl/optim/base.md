# Optimizer Shared Utilities

Common helper functions used by all optimizer implementations (SGD, evolutionary methods) for data handling and checkpoint management.

## What it's for

Provides shared functionality to avoid code duplication across different optimizer types. Used by both gradient-based (sgd.py) and gradient-free (evolution.py, genetic.py) optimizers.

## What it contains

### Data Processing
- `create_episode_list()` - Converts flat observation/action tensors into structured episode dictionaries using boundary indices

### Checkpoint Management
- `save_checkpoint()` - Saves model and optimizer state to disk
- `load_checkpoint()` - Loads checkpoint from disk, returns None if not found

## Key Details

These utilities handle the common operations that all optimizers need: converting between flat and episodic data representations (required for recurrent model training), and managing training checkpoints for resumption. The episode list format is used throughout the platform layer for consistent data handling.
