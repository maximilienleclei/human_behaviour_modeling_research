# Evolution-Based Optimizers for Feedforward Networks

Evolutionary optimization methods (GA, ES, CMA-ES) for feedforward MLP architectures with GPU-parallel population evaluation.

## What it's for

Provides gradient-free optimization for feedforward models using evolution strategies. Implements Simple GA, Simple ES, and diagonal CMA-ES with efficient batched GPU computation across populations.

## What it contains

### Population Classes
- `BatchedPopulation` - GPU-batched population for Simple GA and Simple ES
- `CMAESPopulation` - Specialized population for CMA-ES with diagonal covariance

### Optimization Functions
- `optimize_ga_feedforward()` - Simple Genetic Algorithm with hard selection
- `optimize_es_feedforward()` - Evolution Strategy with soft selection (rank-based weighting)
- `optimize_cmaes_feedforward()` - Diagonal Covariance Matrix Adaptation ES
- `_optimize_neuroevolution()` - Shared optimization loop for GA/ES

### Key Features
- Parallel fitness evaluation across entire population on GPU
- Adaptive per-parameter mutation strength (sigma co-evolution)
- Batched computation: all networks evaluated simultaneously
- Configurable selection pressure (parent fraction, rank weighting)

## Key Details

BatchedPopulation stores all MLP parameters (fc1_weight, fc1_bias, fc2_weight, fc2_bias) as batched tensors [pop_size, ...]. Forward pass broadcasts input across population dimension for parallel evaluation. Adaptive sigma: each weight has its own mutation strength updated based on selection pressure. Simple GA: selects top K% as parents (hard cutoff), mutates with Gaussian noise. Simple ES: all individuals contribute weighted by fitness rank, softer selection pressure. CMA-ES: maintains diagonal covariance matrix (full matrix too memory-intensive), adapts mean and step-size. Fitness is negative cross-entropy loss (maximization). Supports optional behavioral comparison evaluation and database logging. The _optimize_neuroevolution() function provides shared loop for GA/ES variants. Used by platform/runner.py for feedforward models with "ga" or "es" methods. Dynamic networks are handled separately by platform/optimizers/genetic.py since they need special topology handling.
