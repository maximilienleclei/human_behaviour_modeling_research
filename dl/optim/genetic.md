# Genetic Algorithm Optimizers for Recurrent Networks

Evolutionary optimization methods (GA, ES, CMA-ES) for recurrent neural networks with GPU-parallel population evaluation.

## What it's for

Provides gradient-free optimization for recurrent models using evolution strategies. Supports both reservoir (frozen W_hh) and trainable (rank-1 W_hh) recurrent architectures with batched GPU computation for efficiency.

## What it contains

### Population Classes
- `BatchedRecurrentPopulation` - GPU-batched population for Simple GA and Simple ES
- `CMAESRecurrentPopulation` - Specialized population for CMA-ES with full covariance

### Optimization Functions
- `optimize_ga()` - Simple Genetic Algorithm with hard selection (top K parents)
- `optimize_es_recurrent()` - Evolution Strategy with soft selection (weighted by fitness rank)
- `optimize_cmaes_recurrent()` - Covariance Matrix Adaptation Evolution Strategy (diagonal approximation)

### Key Features
- Batched forward passes across entire population (parallel GPU evaluation)
- Adaptive mutation strength (sigma) for each parameter
- Episode-based evaluation for recurrent models (preserves hidden state structure)
- Periodic behavioral comparison and checkpointing

## Key Details

BatchedRecurrentPopulation stores all network parameters as batched tensors [pop_size, ...] for parallel evaluation. For reservoir models, W_hh is frozen (initialized once); for trainable models, uses rank-1 factorization (u, v vectors). Adaptive sigma mutation: each parameter has its own mutation strength that co-evolves with the parameters. Simple GA selects top fraction (e.g., top 25%) as parents and mutates them. Simple ES uses rank-weighted selection where all individuals contribute weighted by fitness rank. CMA-ES maintains diagonal covariance (not full matrix due to memory constraints) and adapts search distribution. Episode evaluation creates episode lists (platform/optimizers/base.py) to properly handle recurrent hidden state resets between episodes. Supports periodic behavioral comparison (platform/evaluation/comparison.py) and database logging (experiments/tracking/logger.py). Used by platform/runner.py when method includes "ga" or "es" with recurrent models.
