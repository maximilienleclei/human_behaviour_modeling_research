# Unified Experimentation Platform

A reusable framework for training neural networks (feedforward and recurrent) using gradient-based (SGD) and evolutionary (GA) methods for human behavior modeling.

## Directory Structure

```
platform/
├── models/          # Neural network architectures
├── optimizers/      # Training algorithms (SGD, GA)
├── data/           # Dataset loading and preprocessing
├── evaluation/     # Metrics and behavioral comparison
├── config.py       # Configuration and constants
└── runner.py       # Main experiment runner
```

## Quick Start

### Basic Usage

```bash
# Run SGD on CartPole with frozen recurrent weights
python -m platform.runner \
  --dataset cartpole \
  --method SGD_reservoir \
  --model reservoir \
  --optimizer sgd \
  --subject sub01 \
  --max-time 600

# Run GA on LunarLander with trainable recurrent weights  
python -m platform.runner \
  --dataset lunarlander \
  --method GA_trainable \
  --model trainable \
  --optimizer ga \
  --subject sub01 \
  --use-cl-info \
  --max-time 3600
```

### Available Options

**Datasets:**
- `cartpole`, `mountaincar`, `acrobot`, `lunarlander` - Human behavioral data
- `HF:CartPole-v1`, `HF:LunarLander-v2` - HuggingFace datasets

**Models:**
- `reservoir` - Recurrent MLP with frozen reservoir (452 params)
- `trainable` - Recurrent MLP with trainable rank-1 weights (552 params)

**Optimizers:**
- `sgd` - Stochastic gradient descent with backpropagation
- `ga` - Genetic algorithm with mutation-based evolution

**Key Arguments:**
- `--use-cl-info` - Include session/run features for continual learning
- `--subject` - Subject identifier (sub01, sub02)
- `--seed` - Random seed for reproducibility
- `--gpu` - GPU index to use
- `--hidden-size` - Hidden layer dimension (default: 50)
- `--max-time` - Maximum optimization time in seconds
- `--batch-size` - Batch size for SGD (default: 32)
- `--population-size` - Population size for GA (default: 50)

## Architecture

### Models (`platform/models/`)

**Feedforward:**
- `MLP` - Two-layer feedforward network with tanh activation

**Recurrent:**
- `RecurrentMLPReservoir` - Frozen recurrent weights (echo state network)
- `RecurrentMLPTrainable` - Trainable recurrent weights (rank-1 factorization)

**Dynamic (GA-exclusive):**
- `DynamicNetPopulation` - Evolving network topology

### Optimizers (`platform/optimizers/`)

**SGD (`optimize_sgd`):**
- Gradient-based optimization
- Episode-based training with DataLoader
- Checkpoint and resumption support
- Periodic evaluation

**GA (`optimize_ga`):**
- Mutation-based population evolution
- GPU-parallelized BatchedRecurrentPopulation
- Adaptive sigma mutations
- Simple GA selection (top 50% survive)

### Data (`platform/data/`)

**Loaders:**
- `load_cartpole_data()` - HuggingFace CartPole-v1
- `load_lunarlander_data()` - HuggingFace LunarLander-v2
- `load_human_data()` - Human behavioral data from JSON files

**Preprocessing:**
- `compute_session_run_ids()` - Session/run detection from timestamps
- `normalize_session_run_features()` - Normalize CL features to [-1, 1]
- `EpisodeDataset` - PyTorch Dataset for episode-based training
- `episode_collate_fn()` - Collate function with padding

### Evaluation (`platform/evaluation/`)

**Metrics:**
- `compute_cross_entropy()` - Cross-entropy loss
- `compute_macro_f1()` - Macro F1 score with sampling

**Comparison:**
- `evaluate_progression_recurrent()` - Compare model vs human on environment rollouts

## Configuration

Platform configuration is centralized in `platform/config.py`:

```python
from platform.config import DEVICE, set_device, ENV_CONFIGS, RESULTS_DIR

# Set GPU
set_device(0)  # Use GPU 0

# Access environment configs
env = ENV_CONFIGS["cartpole"]
obs_dim = env["obs_dim"]  # 4
action_dim = env["action_dim"]  # 2
```

## Integration with Tracking System

The platform integrates with the existing tracking system (`experiments/tracking/`):

```python
from experiments.tracking.database import ExperimentDB
from experiments.tracking.logger import ExperimentLogger

db = ExperimentDB("tracking.db")

with ExperimentLogger(db, experiment_number=5, dataset="cartpole", ...) as logger:
    # Run experiment
    # Metrics are automatically logged
    pass
```

## Examples

### 1. Quick 10-minute test

```bash
python -m platform.runner \
  --dataset cartpole \
  --method SGD_test \
  --model reservoir \
  --optimizer sgd \
  --max-time 600 \
  --no-logger
```

### 2. Full training with CL features

```bash
python -m platform.runner \
  --dataset lunarlander \
  --method GA_trainable_CL \
  --model trainable \
  --optimizer ga \
  --use-cl-info \
  --max-time 36000 \
  --population-size 100
```

### 3. Reproduce experiment 4 setup

```bash
python -m platform.runner \
  --dataset cartpole \
  --method SGD_reservoir \
  --model reservoir \
  --optimizer sgd \
  --use-cl-info \
  --subject sub01 \
  --seed 42 \
  --hidden-size 50 \
  --max-time 36000
```

## Design Principles

1. **Shared capabilities**: Models work with both SGD and GA
2. **GA-exclusive features**: Dynamic networks, F1 fitness (future)
3. **Modular design**: Easy to add new models/optimizers
4. **Type safety**: Extensive jaxtyping annotations with beartype
5. **GPU efficiency**: Batched operations for GA populations

## Future Enhancements

- Mamba state space models
- GAIL (adversarial imitation learning)
- Transfer learning for GA
- F1 score fitness for GA
- Hyperparameter tuning utilities
