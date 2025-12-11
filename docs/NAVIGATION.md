# Navigation Guide

**Last Updated:** December 11, 2024

Quick reference for finding files and understanding the codebase structure.

---

## Table of Contents

1. [Directory Tree](#directory-tree)
2. [Key Files Cheat Sheet](#key-files-cheat-sheet)
3. [Where to Find X](#where-to-find-x)
4. [Import Patterns](#import-patterns)
5. [File Complexity Guide](#file-complexity-guide)

---

## Directory Tree

```
human_behaviour_modeling_research/
│
├── platform/                      # CORE: Unified experimentation framework
│   ├── __init__.py               # Package initialization with beartype
│   ├── config.py                 # Configuration, ENV_CONFIGS, device management
│   ├── runner.py                 # Main CLI entry point (279 LOC)
│   │
│   ├── models/                   # Neural network architectures
│   │   ├── __init__.py          # Model factory: create_model()
│   │   ├── feedforward.py       # MLP (two-layer feedforward)
│   │   ├── recurrent.py         # RecurrentMLPReservoir, RecurrentMLPTrainable
│   │   └── dynamic.py           # DynamicNetPopulation (GA-exclusive)
│   │
│   ├── optimizers/              # Training algorithms
│   │   ├── __init__.py          # Optimizer exports
│   │   ├── base.py              # Shared utilities (checkpointing, episodes)
│   │   ├── sgd.py               # SGD with Adam optimizer
│   │   ├── genetic.py           # GA for recurrent models
│   │   └── evolution.py         # ES and CMA-ES for all models (~1200 LOC)
│   │
│   ├── data/                    # Data loading & preprocessing
│   │   ├── __init__.py          # Data module exports
│   │   ├── loaders.py           # HuggingFace + human data loaders
│   │   └── preprocessing.py     # CL features, EpisodeDataset, collate_fn
│   │
│   ├── evaluation/              # Metrics and behavioral comparison
│   │   ├── __init__.py          # Evaluation exports
│   │   ├── metrics.py           # Cross-entropy, F1 score
│   │   └── comparison.py        # evaluate_progression_recurrent()
│   │
│   └── README.md                # Platform-specific documentation
│
├── experiments/                 # Experiment infrastructure & tracking
│   ├── tracking/                # Database for experiment tracking
│   │   ├── database.py          # SQLite schema (ExperimentDB)
│   │   ├── logger.py            # ExperimentLogger (context manager)
│   │   └── tracking.db          # SQLite database file
│   │
│   ├── cli/                     # Command-line tools
│   │   ├── submit_jobs.py       # Submit experiments to SLURM
│   │   ├── monitor_jobs.py      # Monitor job status
│   │   └── query_results.py     # Query and analyze results
│   │
│   ├── slurm/                   # SLURM templates and configs
│   │   └── job_template.sh      # SLURM job script template
│   │
│   ├── configs/                 # YAML configs (future - not yet implemented)
│   │
│   └── [1-4]_*/                 # Archived experiments (pre-platform)
│       ├── 1_dl_vs_ga_scaling_dataset_size_flops/
│       ├── 2_dl_vs_ga_es/
│       ├── 3_cl_info_dl_vs_ga/
│       └── 4_add_recurrence/    # Most recent, reference implementation
│
├── data/                        # Human behavioral data (JSON)
│   ├── sub01_data_cartpole.json       (3.2 MB)
│   ├── sub01_data_mountaincar.json    (2.7 MB)
│   ├── sub01_data_acrobot.json        (7.4 MB)
│   ├── sub01_data_lunarlander.json    (5.5 MB)
│   ├── sub02_data_cartpole.json       (2.7 MB)
│   ├── sub02_data_mountaincar.json    (2.5 MB)
│   ├── sub02_data_acrobot.json        (5.7 MB)
│   └── sub02_data_lunarlander.json    (8.5 MB)
│
├── common/                      # Shared components
│   └── dynamic_net/             # Dynamic complexity networks
│       ├── __init__.py
│       ├── evolution.py         # Net class with evolving topology
│       └── computation.py       # WelfordRunningStandardizer
│
├── utils/                       # Utilities
│   └── type_utils.py            # Beartype validators
│
├── docs/                        # Documentation
│   ├── README.md                # Documentation index
│   ├── ARCHITECTURE.md          # System design
│   ├── USAGE_GUIDE.md           # How to run experiments
│   ├── NAVIGATION.md            # This file
│   └── STATUS.md                # Current implementation status
│
├── results/                     # Checkpoints and outputs (auto-generated)
│   └── *_checkpoint.pt          # Model checkpoints
│
├── README.md                    # Project entry point
├── CLAUDE.md                    # Instructions for Claude
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Code style config (black, isort)
│
└── .claude/                     # Claude Code internal (ignore)
    └── plans/
```

---

## Key Files Cheat Sheet

### Entry Points

| Purpose | File | LOC |
|---------|------|-----|
| Run experiments | `platform/runner.py` | 279 |
| Submit to cluster | `experiments/cli/submit_jobs.py` | ~200 |
| Monitor jobs | `experiments/cli/monitor_jobs.py` | ~150 |
| Query results | `experiments/cli/query_results.py` | ~300 |

### Core Platform

| Component | File | LOC | Description |
|-----------|------|-----|-------------|
| Config | `platform/config.py` | ~200 | ENV_CONFIGS, device management |
| Model factory | `platform/models/__init__.py` | ~50 | create_model() |
| MLP | `platform/models/feedforward.py` | ~50 | Two-layer feedforward |
| Recurrent | `platform/models/recurrent.py` | ~250 | Reservoir + trainable RNN |
| Dynamic nets | `platform/models/dynamic.py` | ~350 | Evolving topology (GA only) |
| SGD optimizer | `platform/optimizers/sgd.py` | ~300 | Adam with backprop |
| GA optimizer | `platform/optimizers/genetic.py` | ~450 | Genetic algorithm |
| ES/CMA-ES | `platform/optimizers/evolution.py` | ~1200 | Evolution strategies |
| Data loaders | `platform/data/loaders.py` | ~200 | HuggingFace + human data |
| Preprocessing | `platform/data/preprocessing.py` | ~200 | CL features, episodes |
| Metrics | `platform/evaluation/metrics.py` | ~100 | Cross-entropy, F1 |

### Experiment Tracking

| Component | File | LOC | Description |
|-----------|------|-----|-------------|
| Database | `experiments/tracking/database.py` | ~400 | SQLite schema |
| Logger | `experiments/tracking/logger.py` | ~200 | ExperimentLogger class |

### Reference Implementations

| Experiment | Directory | Description |
|------------|-----------|-------------|
| Experiment 4 | `experiments/4_add_recurrence/` | Most recent, has recurrent models |
| Experiment 3 | `experiments/3_cl_info_dl_vs_ga/` | Continual learning features |
| Experiment 2 | `experiments/2_dl_vs_ga_es/` | Evolution strategies |
| Experiment 1 | `experiments/1_dl_vs_ga_scaling_dataset_size_flops/` | Initial DL vs GA |

---

## Where to Find X

### "I want to..."

| Task | Go to |
|------|-------|
| **Run an experiment** | `platform/runner.py` |
| **Add a new model** | `platform/models/` → Create new file → Register in `__init__.py` |
| **Add a new optimizer** | `platform/optimizers/` → Create new file → Update `runner.py` |
| **Modify training loop** | `platform/optimizers/sgd.py` or `genetic.py` |
| **Change evaluation metrics** | `platform/evaluation/metrics.py` |
| **Add a new dataset** | `platform/data/loaders.py` → Add loader function |
| **Modify data preprocessing** | `platform/data/preprocessing.py` |
| **Change environment configs** | `platform/config.py` → Update `ENV_CONFIGS` |
| **Query experiment results** | `experiments/cli/query_results.py` |
| **Monitor running jobs** | `experiments/cli/monitor_jobs.py` |
| **Understand database schema** | `experiments/tracking/database.py` |
| **See reference implementation** | `experiments/4_add_recurrence/` |

### "Where is the code for..."

| Feature | File | Function/Class |
|---------|------|----------------|
| **MLP forward pass** | `platform/models/feedforward.py` | `MLP.forward()` |
| **Recurrent forward pass** | `platform/models/recurrent.py` | `RecurrentMLPReservoir.forward()` |
| **SGD training loop** | `platform/optimizers/sgd.py` | `optimize_sgd()` |
| **GA evolution loop** | `platform/optimizers/genetic.py` | `optimize_ga()` |
| **ES evolution loop** | `platform/optimizers/evolution.py` | `ESPopulation` class |
| **Population evaluation** | `platform/optimizers/genetic.py` | `BatchedRecurrentPopulation.evaluate_batch()` |
| **Cross-entropy loss** | `platform/evaluation/metrics.py` | `compute_cross_entropy()` |
| **F1 score** | `platform/evaluation/metrics.py` | `compute_macro_f1()` |
| **Load human data** | `platform/data/loaders.py` | `load_human_data()` |
| **Load HF data** | `platform/data/loaders.py` | `load_cartpole_data()` |
| **Session/run detection** | `platform/data/preprocessing.py` | `compute_session_run_ids()` |
| **Episode batching** | `platform/data/preprocessing.py` | `EpisodeDataset`, `episode_collate_fn()` |
| **Checkpoint saving** | `platform/optimizers/base.py` | `save_checkpoint()` |
| **Checkpoint loading** | `platform/optimizers/base.py` | `load_checkpoint()` |
| **Experiment logging** | `experiments/tracking/logger.py` | `ExperimentLogger` |
| **Database queries** | `experiments/tracking/database.py` | `ExperimentDB` methods |

---

## Import Patterns

### Platform Modules

```python
# Models
from platform.models import create_model
from platform.models.recurrent import RecurrentMLPReservoir, RecurrentMLPTrainable
from platform.models.feedforward import MLP
from platform.models.dynamic import DynamicNetPopulation

# Optimizers
from platform.optimizers.sgd import optimize_sgd
from platform.optimizers.genetic import optimize_ga, BatchedRecurrentPopulation
from platform.optimizers.evolution import ESPopulation, CMAESPopulation

# Data
from platform.data.loaders import load_human_data, load_cartpole_data
from platform.data.preprocessing import (
    compute_session_run_ids,
    normalize_session_run_features,
    EpisodeDataset,
    episode_collate_fn
)

# Evaluation
from platform.evaluation.metrics import compute_cross_entropy, compute_macro_f1
from platform.evaluation.comparison import evaluate_progression_recurrent

# Config
from platform.config import (
    DEVICE,
    set_device,
    ENV_CONFIGS,
    RESULTS_DIR,
    DATA_DIR,
    ExperimentConfig
)
```

### Tracking System

```python
# Database
from experiments.tracking.database import ExperimentDB

# Logger
from experiments.tracking.logger import ExperimentLogger

# Usage
db = ExperimentDB("experiments/tracking/tracking.db")
with ExperimentLogger(db, experiment_number=5, ...) as logger:
    logger.log_metrics({"loss": 0.5, "f1": 0.8})
```

### Common Components

```python
# Dynamic networks
from common.dynamic_net import Net
from common.dynamic_net.computation import WelfordRunningStandardizer
```

---

## File Complexity Guide

### Simple Entry Points (Good Starting Points)

| File | LOC | Complexity | Notes |
|------|-----|------------|-------|
| `platform/runner.py` | 279 | Medium | Main entry point, mostly argument parsing |
| `platform/config.py` | ~200 | Low | Just data structures and constants |
| `platform/models/feedforward.py` | ~50 | Low | Simple MLP, good reference |
| `platform/evaluation/metrics.py` | ~100 | Low | Just loss functions |

### Moderate Complexity

| File | LOC | Complexity | Notes |
|------|-----|------------|-------|
| `platform/models/recurrent.py` | ~250 | Medium | Two RNN variants, clear structure |
| `platform/data/loaders.py` | ~200 | Medium | Data loading, straightforward |
| `platform/data/preprocessing.py` | ~200 | Medium | CL features, episode handling |
| `platform/optimizers/sgd.py` | ~300 | Medium | Standard PyTorch training loop |
| `experiments/tracking/database.py` | ~400 | Medium | SQL schema, query methods |

### Complex (Deep Dive Required)

| File | LOC | Complexity | Notes |
|------|-----|------------|-------|
| `platform/optimizers/evolution.py` | ~1200 | High | ES and CMA-ES algorithms, batched ops |
| `platform/optimizers/genetic.py` | ~450 | High | GA with population management |
| `platform/models/dynamic.py` | ~350 | High | Evolving topology, mutation operators |
| `experiments/cli/query_results.py` | ~300 | Medium-High | Complex SQL queries, aggregations |

### Legacy (Archived, Reference Only)

| Directory | Status | Notes |
|-----------|--------|-------|
| `experiments/1_*` | Archived | Initial experiments |
| `experiments/2_*` | Archived | ES introduction |
| `experiments/3_*` | Archived | CL features added |
| `experiments/4_*` | Archived | Reference for platform validation |

---

## Quick Navigation Tips

### 1. Finding Implementations

**Pattern:** Most functionality follows this structure:
```
platform/{module}/{file}.py
```

**Example:**
- Want recurrent models? → `platform/models/recurrent.py`
- Want SGD optimizer? → `platform/optimizers/sgd.py`
- Want data loading? → `platform/data/loaders.py`

### 2. Understanding Data Flow

**Trace the path:**
1. `platform/runner.py` - Entry point
2. `platform/data/loaders.py` - Load data
3. `platform/models/__init__.py` - Create model
4. `platform/optimizers/{sgd|genetic|evolution}.py` - Train
5. `platform/evaluation/metrics.py` - Evaluate
6. `experiments/tracking/logger.py` - Log results

### 3. Finding Examples

**Best references:**
1. `platform/runner.py` - Shows how all pieces fit together
2. `experiments/4_add_recurrence/` - Full reference implementation
3. `platform/README.md` - Usage examples
4. `docs/USAGE_GUIDE.md` - Complete command examples

### 4. Debugging

**Check these in order:**
1. `platform/runner.py` - Argument parsing and initialization
2. `platform/config.py` - Configuration values
3. `platform/data/loaders.py` - Data loading
4. `platform/optimizers/*.py` - Training loops
5. `experiments/tracking/logger.py` - Logging issues

---

## Code Organization Patterns

### Naming Conventions

**Files:**
- Lowercase with underscores: `recurrent.py`, `data_loaders.py`
- Descriptive: File name matches main class/function

**Classes:**
- PascalCase: `RecurrentMLPReservoir`, `BatchedPopulation`
- Descriptive: Name indicates purpose

**Functions:**
- Lowercase with underscores: `load_human_data()`, `compute_cross_entropy()`
- Verb-noun pattern: `create_model()`, `evaluate_progression()`

### Module Structure

**Typical module layout:**
```python
# Imports (standard lib, third-party, local)
import torch
from jaxtyping import Float
from platform.config import DEVICE

# Constants
HIDDEN_SIZE = 50

# Classes
class RecurrentMLPReservoir(nn.Module):
    def __init__(self, ...):
        ...

    def forward(self, ...):
        ...

# Functions
def create_model(...):
    ...

# Module exports (in __init__.py)
__all__ = ["RecurrentMLPReservoir", "create_model"]
```

### Type Annotations

**Extensive use of jaxtyping:**
```python
from jaxtyping import Float, Int
from torch import Tensor

def forward(
    obs: Float[Tensor, "BS obs_dim"]
) -> Float[Tensor, "BS action_dim"]:
    ...
```

**Legend:**
- `BS` = Batch size
- `SL` = Sequence length
- `HD` = Hidden dimension
- `obs_dim` = Observation dimension
- `action_dim` = Action dimension

---

## Summary

**Quick Reference Card:**
```
Run experiment       → platform/runner.py
Add model            → platform/models/
Add optimizer        → platform/optimizers/
Load data            → platform/data/loaders.py
Modify metrics       → platform/evaluation/metrics.py
Query results        → experiments/cli/query_results.py
Reference code       → experiments/4_add_recurrence/
```

**Most Important Files (Start Here):**
1. `README.md` - Project overview
2. `docs/ARCHITECTURE.md` - System design
3. `docs/USAGE_GUIDE.md` - How to run things
4. `platform/runner.py` - Main entry point
5. `platform/config.py` - Configuration

---

**Related Documentation:**
- [Architecture Guide](ARCHITECTURE.md) - System design details
- [Usage Guide](USAGE_GUIDE.md) - How to run experiments
- [Status](STATUS.md) - What's implemented, what's next
