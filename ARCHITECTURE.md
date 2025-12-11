# Architecture Overview

**Last Updated:** December 11, 2024

This document provides a comprehensive overview of the system architecture for the human behaviour modeling research platform.

---

## Table of Contents

1. [Research Context & Goals](#research-context--goals)
2. [High-Level System Design](#high-level-system-design)
3. [Module-by-Module Breakdown](#module-by-module-breakdown)
4. [Key Design Principles](#key-design-principles)
5. [Technology Stack](#technology-stack)
6. [Supported Combinations](#supported-combinations)
7. [Data Flow](#data-flow)

---

## Research Context & Goals

### Problem Statement

This platform investigates **behavioral cloning**: learning policies that imitate human behavior from demonstration data. The central research question is:

> **How do gradient-based methods (SGD) compare to evolutionary methods (GA/ES/CMA-ES) for learning human behavioral policies?**

### Approach

- **Input:** Human gameplay trajectories from classic control environments
- **Task:** Supervised learning to predict actions from observations
- **Methods:** Gradient descent vs evolution-based optimization
- **Metrics:** Cross-entropy loss (primary), F1 score (secondary)

### Environments

- **CartPole-v1** (4D state, 2 actions)
- **MountainCar-v0** (2D state, 3 actions)
- **Acrobot-v1** (6D state, 3 actions)
- **LunarLander-v2** (8D state, 4 actions)

### Key Hypothesis

Evolutionary methods may be competitive with gradient-based methods when:
- Limited data is available
- Human behavior is non-stationary
- Network architectures are simple

---

## High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    EXPERIMENTATION LAYER                     │
│  experiments/cli/ (submit, monitor, query)                   │
│  experiments/tracking/ (SQLite database)                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      PLATFORM LAYER                          │
│                   (Unified Framework)                        │
│                                                               │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐ │
│  │    Models     │  │  Optimizers   │  │   Evaluation    │ │
│  │               │  │               │  │                 │ │
│  │  - Feedforward│  │  - SGD        │  │  - Metrics      │ │
│  │  - Recurrent  │  │  - GA         │  │  - Comparison   │ │
│  │  - Dynamic    │  │  - ES         │  │                 │ │
│  │               │  │  - CMA-ES     │  │                 │ │
│  └───────────────┘  └───────────────┘  └─────────────────┘ │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                 Data Loading & Preprocessing           │  │
│  │  - HuggingFace datasets  - Human behavioral data      │  │
│  │  - Continual learning features  - Episode batching    │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    Configuration                       │  │
│  │  - Environment configs  - Device management           │  │
│  │  - Paths & constants                                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  platform/runner.py (Main Entry Point)                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                             │
│  data/ (Human behavioral trajectories - JSON)               │
└─────────────────────────────────────────────────────────────┘
```

### Design Philosophy

1. **Modular:** Clear separation between models, optimizers, data, evaluation
2. **Reusable:** Same models work with multiple optimizers
3. **Extensible:** Easy to add new models, optimizers, or datasets
4. **Type-Safe:** Extensive runtime type checking with jaxtyping + beartype
5. **Experiment-First:** Built for running hundreds of experiments efficiently

---

## Module-by-Module Breakdown

### platform/models/

**Purpose:** Neural network architectures for behavioral cloning

#### feedforward.py
- **MLP:** Two-layer feedforward network [input, 50, output]
- Tanh activation
- No temporal memory
- ~150 parameters for CartPole

#### recurrent.py
- **RecurrentMLPReservoir:** Echo state network with frozen recurrent weights
  - Random recurrent matrix (not trained)
  - Only input/output weights trainable
  - ~452 parameters for CartPole

- **RecurrentMLPTrainable:** Fully trainable recurrent network
  - Rank-1 recurrent weights: W_rec = u ⊗ v^T
  - All weights learnable
  - ~552 parameters for CartPole

#### dynamic.py
- **DynamicNetPopulation:** Evolving network topology (GA-exclusive)
  - Starts with minimal connectivity
  - Adds/removes neurons and connections via mutation
  - Non-differentiable (cannot use with SGD)

**Key Interfaces:**
```python
# Forward pass (feedforward)
actions = model(observations)  # [BS, action_dim]

# Forward pass (recurrent)
actions, hidden = model(observations, hidden_state)  # [BS, action_dim], [BS, hidden_dim]

# Reset hidden state
hidden = model.initial_state(batch_size)  # [BS, hidden_dim]
```

---

### platform/optimizers/

**Purpose:** Training algorithms

#### base.py
Shared utilities:
- `create_episode_list()` - Convert flat data to episode structure
- `save_checkpoint()` - Save model state and training progress
- `load_checkpoint()` - Resume from checkpoint
- Logging helpers

#### sgd.py
**Gradient-based optimization:**
- Adam optimizer
- Episode-based DataLoader with padding
- Periodic evaluation (every ~60 seconds)
- Time-based stopping (e.g., 10 hours)
- Checkpoint/resume support

**Training Loop:**
```python
for epoch in range(max_epochs):
    for batch in dataloader:
        loss = criterion(model(obs), actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### genetic.py
**Genetic algorithm for recurrent networks:**
- `BatchedRecurrentPopulation` - GPU-parallel population evaluation
- Mutation-based evolution (Gaussian noise)
- Simple selection (top 50% survive)
- Adaptive sigma mutations

**Evolution Loop:**
```python
population = BatchedRecurrentPopulation(...)
for generation in range(max_generations):
    fitnesses = population.evaluate_batch(obs, actions)
    population.selection_and_mutation(fitnesses)
```

#### evolution.py
**Evolution strategies for all model types:**

- **Simple ES (Evolution Strategies):**
  - Natural gradient estimation
  - Works with feedforward and recurrent models

- **CMA-ES (Covariance Matrix Adaptation):**
  - Diagonal covariance for efficiency
  - Adaptive step size
  - Works with feedforward and recurrent models

**Evolution Loop:**
```python
population = ESPopulation(...)  # or CMAESPopulation
for generation in range(max_generations):
    fitnesses = population.evaluate_batch(obs, actions)
    population.update(fitnesses)
```

---

### platform/data/

**Purpose:** Data loading and preprocessing

#### loaders.py
- `load_cartpole_data()` - HuggingFace CartPole-v1 expert trajectories
- `load_lunarlander_data()` - HuggingFace LunarLander-v2 expert trajectories
- `load_human_data()` - Local JSON files with human gameplay
  - Per-session random split (train/test)
  - Session/run metadata extraction

#### preprocessing.py
**Continual Learning (CL) Features:**
- `compute_session_run_ids()` - Detect session boundaries from timestamps
- `normalize_session_run_features()` - Normalize session/run IDs to [-1, 1]

**Episode Batching:**
- `EpisodeDataset` - PyTorch Dataset that preserves episode boundaries
- `episode_collate_fn()` - Collate with padding for variable-length episodes

---

### platform/evaluation/

**Purpose:** Metrics and behavioral comparison

#### metrics.py
- `compute_cross_entropy()` - Cross-entropy loss (primary metric)
- `compute_macro_f1()` - Macro F1 score with action sampling

#### comparison.py
- `evaluate_progression_recurrent()` - Compare model vs human behavior in environment
- Computes:
  - Average return over episodes
  - Behavioral similarity metrics
- Requires gymnasium for environment loading

---

### platform/config.py

**Purpose:** Centralized configuration

**Key Components:**
- `DEVICE` - Global torch device (CPU/GPU)
- `set_device(gpu_index)` - Set device dynamically
- `ENV_CONFIGS` - Environment specifications:
  ```python
  {
      "cartpole": {
          "obs_dim": 4,
          "action_dim": 2,
          "env_name": "CartPole-v1"
      },
      ...
  }
  ```
- `ExperimentConfig` - Dataclass for experiment parameters
- Path constants (RESULTS_DIR, DATA_DIR, etc.)

---

### platform/runner.py

**Purpose:** Main CLI entry point

**Responsibilities:**
- Parse command-line arguments
- Load configuration
- Initialize model
- Select and run optimizer
- Log results to tracking database
- Save checkpoints

**Key Arguments:**
```bash
--dataset       # cartpole | lunarlander | HF:CartPole-v1
--model         # mlp | reservoir | trainable | dynamic
--optimizer     # sgd | ga | es | cmaes
--method        # Experiment method name
--subject       # sub01 | sub02 (for human data)
--use-cl-info   # Include continual learning features
--max-time      # Maximum training time (seconds)
--seed          # Random seed
--gpu           # GPU index (-1 for CPU)
```

**Auto-Dispatch:** Runner automatically selects appropriate implementation based on model type:
- Feedforward → Use `BatchedPopulation` for GA/ES/CMA-ES
- Recurrent → Use `BatchedRecurrentPopulation` for GA/ES/CMA-ES
- Dynamic → Use `DynamicNetPopulation` for GA only

---

### experiments/tracking/

**Purpose:** Experiment tracking and monitoring

#### database.py
SQLite schema:
- `experiment_runs` - Metadata, config, SLURM job info
- `run_metrics` - Time-series training metrics (logged every ~60s)
- `run_results` - Final aggregated results
- `run_errors` - Error logs with tracebacks
- `slurm_jobs` - Cluster job tracking

#### logger.py
- `ExperimentLogger` - Context manager for automatic logging
- Logs metrics, errors, and results to database
- Integrates with platform/runner.py

---

### experiments/cli/

**Purpose:** Command-line tools for experiment management

- **submit_jobs.py** - Submit experiments to SLURM cluster
- **monitor_jobs.py** - Real-time job status monitoring
- **query_results.py** - Query and analyze results from database

---

### common/

**Purpose:** Shared components used across experiments

#### dynamic_net/
- `Net` class with evolving topology
- `WelfordRunningStandardizer` for online normalization
- Used by `platform/models/dynamic.py`

---

## Key Design Principles

### 1. Shared Models

**Principle:** Models should work with both SGD and evolutionary methods

**Implementation:**
- MLP, RecurrentMLPReservoir, RecurrentMLPTrainable all have:
  - Standard PyTorch `forward()` method (for SGD)
  - Vectorizable parameters (for GA/ES)

**Exception:** DynamicNet is GA-exclusive (non-differentiable topology)

### 2. Type Safety

**Principle:** Catch tensor shape errors at runtime

**Implementation:**
```python
from jaxtyping import Float, Int
from beartype import beartype

@beartype
def forward(
    self,
    obs: Float[Tensor, "BS obs_dim"],
) -> Float[Tensor, "BS action_dim"]:
    ...
```

**Benefits:**
- Clear documentation of expected shapes
- Runtime validation prevents silent bugs
- Self-documenting code

### 3. GPU Efficiency

**Principle:** Evolutionary methods should scale to large populations

**Implementation:**
- Batched population evaluation: evaluate all individuals in parallel
- `BatchedRecurrentPopulation` evaluates 50-100 networks simultaneously
- Vectorized operations throughout

**Performance:**
- 50-agent population: ~10x faster than sequential
- Enables large-scale evolutionary experiments

### 4. Modular & Extensible

**Principle:** Easy to add new components without breaking existing code

**Implementation:**
- Factory pattern: `create_model(model_type, ...)`
- Consistent interfaces across models and optimizers
- Configuration-driven (ENV_CONFIGS)

**Adding a new model:**
1. Create class in `platform/models/`
2. Add to factory in `__init__.py`
3. Works with all optimizers automatically

### 5. Experiment-First Design

**Principle:** Built for running hundreds of experiments

**Implementation:**
- Database tracking instead of JSON files
- SLURM integration for cluster computing
- Time-based stopping (not epoch-based)
- Automatic checkpointing and resumption

---

## Technology Stack

### Core
- **PyTorch 2.0+** - Neural networks and training
- **NumPy** - Numerical operations
- **Python 3.9+** - Core language

### Type Safety
- **jaxtyping** - Tensor shape annotations
- **beartype** - Runtime type checking

### Data
- **HuggingFace datasets** - RL expert trajectories
- **gymnasium** - Environment simulation (optional)

### Experiment Management
- **SQLite** - Experiment tracking database
- **SLURM** - Cluster job scheduling
- **argparse** - CLI interface

### Utilities
- **einops** - Tensor operations
- **scikit-learn** - Metrics and utilities
- **matplotlib** - Visualization (optional)

---

## Supported Combinations

### Model × Optimizer Matrix

| Model Type | SGD | GA | ES | CMA-ES |
|------------|-----|----|----|--------|
| **MLP (feedforward)** | ✅ | ✅ | ✅ | ✅ |
| **RecurrentMLPReservoir** | ✅ | ✅ | ✅ | ✅ |
| **RecurrentMLPTrainable** | ✅ | ✅ | ✅ | ✅ |
| **DynamicNet** | ❌ | ✅ | ❌ | ❌ |

**Notes:**
- DynamicNet is GA-exclusive (non-differentiable topology changes)
- All other models work with all optimizers

### Dataset × Model Compatibility

All models work with all datasets (CartPole, MountainCar, Acrobot, LunarLander).

Environment configs automatically handle:
- State dimensionality (obs_dim)
- Action dimensionality (action_dim)
- Network input/output sizing

### Continual Learning Features

**Optional:** Add session/run IDs as features with `--use-cl-info`

**Compatible with:**
- ✅ Human behavioral data (has temporal structure)
- ❌ HuggingFace datasets (no session structure)

---

## Data Flow

### Training Pipeline

```
1. Data Loading
   └─> load_human_data() or load_cartpole_data()
       └─> observations, actions, metadata

2. Preprocessing
   └─> compute_session_run_ids() (if --use-cl-info)
   └─> normalize_session_run_features()
   └─> train/test split

3. Model Initialization
   └─> create_model(model_type, obs_dim, action_dim, hidden_size)

4. Optimization
   ├─> SGD: optimize_sgd()
   │   └─> EpisodeDataset → DataLoader → training loop
   │
   └─> GA/ES: optimize_ga() / optimize_es()
       └─> BatchedPopulation → evaluate_batch() → selection/mutation

5. Evaluation
   └─> compute_cross_entropy(model, test_obs, test_actions)
   └─> compute_macro_f1(model, test_obs, test_actions)

6. Logging & Checkpointing
   └─> ExperimentLogger.log_metrics()
   └─> save_checkpoint()
```

### Experiment Tracking Flow

```
1. Submit Job
   └─> experiments/cli/submit_jobs.py
       └─> Creates SLURM job script
       └─> Submits to cluster
       └─> Logs job info to database

2. Job Runs
   └─> platform/runner.py executes
       └─> Logs start time, config to database
       └─> Periodically logs metrics
       └─> Logs final results
       └─> Logs any errors

3. Monitor
   └─> experiments/cli/monitor_jobs.py
       └─> Queries database for job status
       └─> Displays real-time progress

4. Query Results
   └─> experiments/cli/query_results.py
       └─> Aggregate results across runs
       └─> Statistical comparisons
       └─> Export to CSV
```

---

## Summary

This architecture supports a comprehensive research platform for comparing gradient-based and evolutionary training methods on behavioral cloning tasks. Key strengths:

1. **Modular Design:** Easy to extend with new models or optimizers
2. **Type Safety:** Runtime validation prevents common bugs
3. **Scalability:** Database tracking + cluster integration
4. **GPU Efficiency:** Parallel evaluation for evolutionary methods
5. **Experiment-First:** Built for running many experiments efficiently

The platform consolidates 4 previous experiments into a unified, reusable framework that maintains backwards compatibility while providing a clean foundation for future research.

---

**Related Documentation:**
- [Usage Guide](USAGE_GUIDE.md) - How to run experiments
- [Navigation Guide](NAVIGATION.md) - Where to find specific code
- [Platform README](../platform/README.md) - Platform-specific details
