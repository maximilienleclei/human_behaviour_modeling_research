# Human Behaviour Modeling Research

Research codebase for modeling human behavior in control tasks using both gradient-based deep learning and gradient-free neuroevolution methods.

## Repository Structure

### Core Modules

#### config/
Shared configuration utilities used across both DL and NE methods.
- **device.py** - Global PyTorch device management (CPU/GPU)
- **paths.py** - Project-wide directory paths (DATA_DIR, RESULTS_DIR)
- **state.py** - State persistence configuration for continual learning
- **experiments.py** - Experiment configuration dataclasses (model, optimizer, training, data)

#### data/
Behavioral data loading and preprocessing.
- **hf_control_tasks/** - HuggingFace dataset loaders for pre-trained agent trajectories
- **simexp_control_tasks/** - Human behavioral data from SimExp experiments
  - Raw JSON data files in `source/` subdirectory
  - Environment configurations, loaders, preprocessing utilities
  - Session/run temporal segmentation for continual learning features

#### dl/
Deep learning module for gradient-based training (SGD with backpropagation).
- **models/** - Neural network architectures (feedforward, recurrent, dynamic)
- **optim/** - SGD optimizer with episode batching for recurrent models

#### ne/
Neuroevolution module for gradient-free evolutionary optimization.
- **net/** - Batched network implementations (feedforward, recurrent, dynamic topology)
- **eval/** - Evaluation layer with fitness functions and training orchestration
- **pop/** - Population adapter layer between networks and optimizers
- **optim/** - Evolutionary algorithms (GA, ES, CMA-ES with unified base)

#### metrics/
Shared evaluation metrics used by both DL and NE.
- **metrics.py** - Cross-entropy loss and macro F1 score
- **comparison.py** - Behavioral comparison (model vs human returns on matched episodes)

### Supporting Directories

#### results/
Output directory for model checkpoints, training logs, and plots (created automatically).

#### exca/
External experiment orchestration library (dependency).

#### old/
Archived code from previous experiments (not actively maintained).

## Research Focus

This codebase supports research on:
1. **Supervised imitation learning** - Training models to predict human actions from observations
2. **Continual learning** - Models that adapt over sessions/runs using temporal features
3. **Comparison of optimization methods** - Gradient-based (SGD) vs gradient-free (GA, ES, CMA-ES)
4. **Recurrent architectures** - Echo state networks (frozen reservoir) vs trainable recurrent weights
5. **Dynamic topologies** - Evolving network architectures through mutation

## Key Design Principles

**Modularity**: Shared utilities (config, data, metrics) used by both dl/ and ne/ modules to avoid duplication.

**Batched Computation**: Neuroevolution uses batched tensors [num_nets, ...] for GPU-parallel population evaluation.

**Separation of Concerns**: Clean layer separation in ne/ module (eval → pop → optim → net).

**Dual Optimization**: Both gradient-based (dl/optim/sgd.py) and evolutionary (ne/optim/*) methods supported.

**Documentation Standard**: Every .py file has corresponding .md documentation at ~1/10th length. Every directory has __init__.md overview.

## Getting Started

1. Data is in `data/simexp_control_tasks/source/` (human behavioral episodes) and loaded via `data.simexp_control_tasks.loaders.load_human_data()`

2. For deep learning: Use `dl.optim.sgd.optimize_sgd()` with models from `dl.models`

3. For neuroevolution: Use high-level functions from `ne.eval` (e.g., `train_supervised()`, `train_environment()`)

4. Metrics are computed via `metrics.metrics` (loss/F1) and `metrics.comparison` (behavioral similarity)

5. All experiments use configuration from `config.experiments.ExperimentConfig`

## File Documentation Convention

- **Individual files**: Each .py file has a .md file describing its contents (~1/10th character length)
- **Directories**: Each directory has __init__.md overview of all files and subdirectories within it
- This enables quick navigation and understanding of codebase structure without reading full implementations
