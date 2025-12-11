# Platform Directory

Unified platform for training neural networks to model human behavior across reinforcement learning tasks.

## Overview

The platform provides a complete ecosystem for behavior modeling experiments: data loading, model architectures, optimization algorithms, evaluation metrics, and experiment orchestration. Supports both gradient-based (SGD) and gradient-free (evolutionary) optimization methods.

## Directory Structure

### models/
Neural network architectures for behavior prediction:
- Feedforward networks (MLP)
- Recurrent networks (reservoir and trainable)
- Dynamic networks with evolving topology

### optimizers/
Training algorithms:
- SGD with backpropagation
- Simple GA and ES for feedforward networks
- Simple GA and ES for recurrent networks
- Diagonal CMA-ES variants
- Shared utilities (checkpointing, episode handling)

### data/
Data loading and preprocessing:
- HuggingFace dataset loaders (CartPole, LunarLander)
- Human behavioral data loaders (JSON format)
- Session/run computation (continual learning features)
- Episode dataset creation for recurrent models

### evaluation/
Performance measurement:
- Standard metrics (cross-entropy, F1 score)
- Behavioral comparison (return matching on same episodes)

### Core Files
- **runner.py** - Main experiment orchestrator, dispatches to appropriate optimizer
- **config.py** - Configuration schemas and environment specifications

## Design Philosophy

**Unified Interface**: All models implement common methods (`forward()`, `get_probs()`), all optimizers follow similar patterns, enabling easy experimentation across methods.

**Optimizer Agnostic**: Most models work with both SGD and evolutionary methods (except dynamic networks which require evolution due to topology mutations).

**GPU Efficiency**: Evolutionary methods use batched populations for parallel evaluation, dramatically faster than sequential computation.

**Flexible Data**: Supports both standard RL datasets and human behavioral data with optional continual learning features.

## Usage

The platform is typically invoked via runner.py:
```bash
python platform/runner.py --dataset cartpole --method SGD_reservoir --seed 42
```

Method name determines model and optimizer automatically (e.g., "SGD_reservoir" → RecurrentMLPReservoir + SGD).

## Integration

The platform integrates with experiments/tracking/ for database logging and experiments/cli/ for SLURM job management.
