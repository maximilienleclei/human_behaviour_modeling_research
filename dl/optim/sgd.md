# SGD Optimizer

Gradient-based optimization using stochastic gradient descent with backpropagation.

## What it's for

Provides standard supervised learning approach for training neural networks to match human behavior. Uses cross-entropy loss between model predictions and human actions as training objective.

## What it contains

### Optimizer Function
- `optimize_sgd()` - Main optimization loop with periodic evaluation, checkpointing, and optional database logging

## Key Details

The optimizer handles both feedforward and recurrent models, with special support for episode-based training required by recurrent architectures (using EpisodeDataset from platform/data/preprocessing.py). Training uses batched gradient descent with configurable batch size and learning rate. The optimization loop runs for a specified time budget (default 10 hours) with periodic evaluations at configurable intervals (loss/F1 every 60s, checkpoints/behavioral comparison every 300s). Supports optional database logging via experiments/tracking/logger.py for experiment tracking. For recurrent models, creates episode batches that preserve temporal structure and properly handle hidden state resets. Checkpoints include model state, optimizer state, and iteration count for resumption. Used by platform/runner.py when method includes "SGD" in the name.
