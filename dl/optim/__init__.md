# Deep Learning Optimizers Directory

Gradient-based optimization methods for training behavior models.

## Overview

This directory implements deep learning optimization algorithms used to fit neural network models to human behavioral data using gradient descent and backpropagation.

## Files

### Shared Utilities
- **base.py** - Common functions used across optimizers
  - `create_episode_list()` - Converts flat data to episode structure
  - `save_checkpoint()` / `load_checkpoint()` - Checkpoint management

### Gradient-Based Optimization
- **sgd.py** - Stochastic gradient descent with backpropagation
  - `optimize_sgd()` - Main SGD training loop
  - Supports both feedforward and recurrent models
  - Episode-based batching for recurrent architectures
  - Periodic evaluation and checkpointing

## Key Concepts

**Gradient Descent**: Optimization via backpropagation, computing gradients of loss with respect to parameters and updating in direction that reduces loss.

**Episode Batching**: For recurrent models, episodes are batched together with padding to handle variable lengths while preserving temporal structure.

**Checkpointing**: Model state, optimizer state, and training history saved periodically for resumption.

## Usage

The SGD optimizer is used for supervised learning where gradient information is available. For evolutionary optimization (GA, ES, CMA-ES), see the `ne/optim/` module which provides unified gradient-free optimization algorithms.
