# Experiment Configuration

Structured configuration dataclasses for defining experiments.

## Purpose
Provides type-safe configuration objects that combine all hyperparameters for an experiment (model, optimizer, training, data).

## Contents
- `ModelConfig` - Model architecture parameters (hidden_size, etc.)
- `OptimizerConfig` - Optimizer hyperparameters (learning_rate, population_size, sigma, etc.)
- `TrainingConfig` - Training settings (batch_size, max_time, eval intervals)
- `DataConfig` - Data split ratios and preprocessing flags
- `ExperimentConfig` - Top-level config combining all sub-configs plus experiment metadata
