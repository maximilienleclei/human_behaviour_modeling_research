# DL Module

Deep learning module for supervised imitation learning from human behavioral data.

## Purpose
Contains all deep learning code migrated from claude_repo/platform/, organized into clear subdirectories.

## Structure
- `models/` - Neural network architectures (MLP, recurrent, dynamic)
- `optim/` - Training optimizers (SGD, evolution strategies, genetic algorithms)
- `eval/` - Evaluation and comparison utilities
- `data/` - Data loaders and preprocessing for behavioral datasets

## Usage
This module uses shared utilities from config/ and eval/ for device management, paths, and metrics.
