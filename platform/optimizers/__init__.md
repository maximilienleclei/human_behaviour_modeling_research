# Platform Optimizers Directory

Optimization methods for training behavior models, including gradient-based (SGD) and gradient-free (GA, ES, CMA-ES) approaches.

## Overview

This directory implements all optimization algorithms used to fit models to human behavioral data. Optimizers handle both feedforward and recurrent architectures with efficient GPU-parallel computation for evolutionary methods.

## Files

### Shared Utilities
- **base.py** - Common functions used across all optimizers
  - `create_episode_list()` - Converts flat data to episode structure
  - `save_checkpoint()` / `load_checkpoint()` - Checkpoint management

### Gradient-Based Optimization
- **sgd.py** - Stochastic gradient descent with backpropagation
  - `optimize_sgd()` - Main SGD training loop
  - Supports both feedforward and recurrent models
  - Episode-based batching for recurrent architectures

### Evolutionary Optimization - Feedforward
- **evolution.py** - GA/ES/CMA-ES for feedforward networks
  - `BatchedPopulation` - GPU-batched population for Simple GA/ES
  - `CMAESPopulation` - Diagonal CMA-ES population
  - `optimize_ga_feedforward()` - Simple Genetic Algorithm
  - `optimize_es_feedforward()` - Evolution Strategy
  - `optimize_cmaes_feedforward()` - CMA-ES

### Evolutionary Optimization - Recurrent
- **genetic.py** - GA/ES/CMA-ES for recurrent networks
  - `BatchedRecurrentPopulation` - GPU-batched recurrent population
  - `CMAESRecurrentPopulation` - Diagonal CMA-ES for recurrent nets
  - `optimize_ga()` - Simple GA for recurrent models
  - `optimize_es_recurrent()` - ES for recurrent models
  - `optimize_cmaes_recurrent()` - CMA-ES for recurrent models

## Key Concepts

**Gradient-Free Methods**: Evolutionary algorithms that work by mutating population parameters and selecting based on fitness, no backpropagation required.

**Batched Populations**: All networks in population stored as batched tensors for parallel GPU evaluation, dramatically faster than sequential evaluation.

**Adaptive Sigma**: Per-parameter mutation strengths that co-evolve with the parameters themselves.

## Usage

Optimizers are selected automatically by platform/runner.py based on method name patterns (SGD vs GA vs ES).
