# Plan: Build Unified Experimentation Platform

## Goal
Refactor from discrete experiment folders (1, 2, 3, 4) to a unified, reusable experimentation platform.

## Current State
- Experiments 1-4 each have duplicated code (main.py, src/, configs)
- experiments/ has infrastructure: cli/, slurm/, tracking/
- common/ has some shared utilities (dynamic_net)

## Proposed Structure

```
platform/                    # NEW: Core experimentation code
├── __init__.py
├── models/                  # Network architectures
│   ├── __init__.py
│   ├── feedforward.py       # MLP variants
│   ├── recurrent.py         # RecurrentMLP, reservoir, trainable
│   ├── dynamic.py           # Dynamic networks (GA-exclusive)
│   └── mamba.py             # Mamba (future)
├── optimizers/              # Weight update mechanisms
│   ├── __init__.py
│   ├── sgd.py              # Backprop optimization
│   ├── genetic.py          # Mutation-based evolution
│   └── base.py             # Shared utilities
├── methods/                 # Training methods/objectives (extensible)
│   ├── __init__.py
│   ├── supervised.py        # Standard behavioral cloning
│   ├── gail.py             # GAIL adversarial imitation (future)
│   ├── imitation.py        # Generator/discriminator (future, GA)
│   ├── transfer.py         # Transfer learning (future, GA)
│   └── ...                 # Easy to add: IRL, AIRL, other RL methods
├── data/                    # Data loading & processing
│   ├── __init__.py
│   ├── loaders.py          # HuggingFace, local JSON loaders
│   └── preprocessing.py    # Normalization, CL features, splits
├── evaluation/              # Metrics & analysis
│   ├── __init__.py
│   ├── metrics.py          # Loss, F1, behavioral metrics
│   └── comparison.py       # Human-model comparison
├── config.py               # Configuration schemas
└── runner.py               # Main experiment runner

experiments/
├── cli/                    # Keep as-is
├── slurm/                  # Keep as-is
├── tracking/               # Keep as-is
├── configs/                # NEW: YAML experiment configs
└── archive/                # NEW: Move 1,2,3,4 here
    ├── 1_dl_vs_ga_scaling_dataset_size_flops/
    ├── 2_dl_vs_ga_es/
    ├── 3_cl_info_dl_vs_ga/
    └── 4_add_recurrence/
```

## Key Design Principles

**Capability Hierarchy**:
- **Shared**: Models/utilities that both SGD and GA can use
  - MLP (feedforward)
  - RecurrentMLP (reservoir, trainable)
  - Mamba (future)
  - Cross-entropy loss
  - Data loading, preprocessing, evaluation

- **Optimizers** (weight update mechanisms):
  - SGD: Backpropagation with gradient descent
  - GA: Mutation-based population evolution

- **Training Methods** (work with both optimizers):
  - Standard supervised: Direct behavioral cloning
  - GAIL (future): Adversarial imitation learning (generator vs discriminator)
    - Can use SGD or GA for weight updates
  - Imitation learning (future): Generator/discriminator co-evolution
    - Primarily for GA, inspired by old_imitate.py
  - Transfer learning (future): Env/memory/fitness transfer
    - GA-exclusive, inspired by old_gen_transfer.py

- **GA-exclusive features**:
  - Dynamic networks (graph-based, evolving topology)
  - F1 score fitness (not just cross-entropy)
  - Population-based features
  - Transfer learning across generations

**Design rule**: Anything built for SGD must also work with GA. GA can have exclusive features that SGD doesn't use.

**Current Scope**: Output classification only (no temporal prediction yet)

## Implementation Steps

### 1. Create Platform Structure
- Create `platform/` directory with subdirectories
- Add `__init__.py` files for proper Python package
- Design for extensibility (easy to add new optimizers/models)

### 2. Extract Models

**Shared models (SGD + GA):**
- `platform/models/feedforward.py`: MLP (from experiments 2/3)
  - Used by both SGD and GA
- `platform/models/recurrent.py`: RecurrentMLPReservoir, RecurrentMLPTrainable
  - Used by both SGD and GA
- `platform/models/mamba.py`: Mamba architecture (future)
  - Will be used by both SGD and GA

**GA-exclusive models:**
- `platform/models/dynamic.py`: DynamicNetPopulation wrapper
  - Graph-based networks with evolving topology
  - Only works with GA (no gradients for topology changes)
  - Extracted from experiment 4 + common/dynamic_net

**Source files:**
- `experiments/4_add_recurrence/src/models.py`: Recurrent models
- `experiments/2_dl_vs_ga_es/main.py`: Feedforward MLP
- `common/dynamic_net/`: Dynamic network implementation

### 3. Extract Optimizers

**Shared infrastructure:**
- Data loading and batching
- Model forward pass (inference)
- Evaluation and metrics
- Checkpointing and logging

**SGD optimizer** (`platform/optimizers/sgd.py`):
- `optimize_sgd(model, data, config, ...)`
- Works with: MLP, RecurrentMLP, Mamba (shared models only)
- Loss: Cross-entropy only
- Update mechanism: Backpropagation with gradient descent
- Standard supervised learning

**GA optimizer** (`platform/optimizers/genetic.py`):
- `optimize_ga(model, data, config, ...)`
- Works with: MLP, RecurrentMLP, Mamba (shared) + DynamicNet (GA-exclusive)
- Loss/Fitness: Cross-entropy OR F1 score
- Update mechanism: Mutation-based population evolution
- Features:
  - Population-based optimization
  - Adaptive mutation (sigma adaptation)
  - F1 score fitness (future)
  - Imitation learning (generator/discriminator, future)
  - Transfer learning (env/mem/fit transfer, future)

**Shared utilities** (`platform/optimizers/base.py`):
- Evaluation functions (behavioral metrics, comparison)
- Checkpoint management
- Logging helpers
- Episode batching for recurrent models

**Source files:**
- `experiments/4_add_recurrence/src/optim.py`: SGD and GA implementations
- `common/old_imitate.py`: Reference for imitation learning (future)
- `common/old_gen_transfer.py`: Reference for transfer learning (future)

### 4. Extract Data Loading
**From experiments 2, 3, 4:**
- Create `platform/data/loaders.py`:
  - `load_huggingface_data()`: CartPole-v1, LunarLander-v2 (exp 2)
  - `load_human_behavioral_data()`: Local JSON files (exp 3, 4)
  - Dataset registry for easy access

- Create `platform/data/preprocessing.py`:
  - Train/val/test splits
  - Session/run ID computation (CL features)
  - Episode dataset creation for recurrent models
  - Normalization utilities

### 5. Extract Evaluation
**From experiment 4:**
- `platform/evaluation/metrics.py`:
  - Cross-entropy loss computation
  - F1 score (macro)
  - Behavioral metrics (percentage difference from human)

- `platform/evaluation/comparison.py`:
  - Human vs model behavior comparison
  - Episode rollout evaluation

### 6. Configuration System
- `platform/config.py`: Consolidate all hyperparameters
  - Model configs (hidden_size, etc.)
  - Optimizer configs (lr, population_size, sigma, etc.)
  - Training configs (batch_size, max_time, etc.)
  - Data configs (split ratios, CL features, etc.)

### 7. Unified Runner
- `platform/runner.py`: Main experiment execution
  - Load config (from YAML or command-line)
  - Initialize dataset
  - Initialize model
  - Initialize optimizer
  - Run optimization with tracking/logging
  - Save results via tracking system

### 8. Archive Old Experiments
- Create `experiments/archive/`
- Move experiments 1, 2, 3, 4 there
- Keep for reference but no longer active development

### 9. Experiment Configs
- Create `experiments/configs/` for YAML configs
- Example configs for common experiment types:
  - `cartpole_sgd_reservoir.yaml`
  - `lunarlander_ga_trainable.yaml`
  - Can define sweeps here

### 10. Update CLI Tools
- Modify `experiments/cli/submit_jobs.py` to use platform runner
- Point to YAML configs instead of hardcoded sweeps
- Keep monitoring and query tools as-is

## Critical Files to Extract

**From experiment 4 (primary source):**
- `experiments/4_add_recurrence/src/models.py` (836 lines)
- `experiments/4_add_recurrence/src/optim.py` (836 lines)
- `experiments/4_add_recurrence/src/config.py`
- `experiments/4_add_recurrence/main.py` (logic → runner.py)

**From experiment 2:**
- Data loading functions for HuggingFace datasets
- Feedforward model variants

**From experiment 3:**
- CL feature extraction logic
- Human behavioral data loading

## Benefits
1. **Single source of truth**: No more duplicated code across experiments
2. **Iterative development**: Add features once, use everywhere
3. **Easy experimentation**: Just write YAML config, no code changes
4. **Maintainable**: Changes to platform propagate automatically
5. **Version control**: Platform evolves, configs are versioned
6. **Infrastructure intact**: Keep tracking, SLURM, CLI systems

## Future Enhancements (Not in Initial Platform)

**Reference implementations to draw from:**
- `common/old_imitate.py`: Adversarial imitation learning setup
  - Generator/discriminator co-evolution
  - Generator tries to fool discriminator (match target behavior)
  - Discriminator tries to distinguish generator from target
  - Used for GAIL and imitation-based GA

- `common/old_gen_transfer.py`: Transfer learning for GA
  - `env_transfer`: Transfer environment state across generations
  - `mem_transfer`: Transfer agent memory/internal state
  - `fit_transfer`: Accumulate fitness across generations (continual learning)
  - Allows building on previous generations

**When implementing these:**
1. Add `platform/methods/gail.py` for adversarial imitation (works with both SGD and GA)
2. Extend `platform/optimizers/genetic.py` with imitation and transfer variants
3. Add `platform/models/mamba.py` for state space models
4. Update config system to support transfer flags and GAIL parameters

## Next Steps After Platform Creation
1. Test platform with a simple experiment (CartPole + SGD)
2. Verify tracking integration works
3. Run sanity check experiments (exp 2 datasets with exp 4 methods)
4. Document platform API for future use
5. Iterate: Add Mamba, GAIL, transfer learning as needed
