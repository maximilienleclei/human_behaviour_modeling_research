# Unified Experimentation Platform - Implementation Plan

**Created:** 2025-12-10  
**Status:** Draft - Ready for Implementation  
**Based on:** PLATFORM_PLAN.md and experiments 2, 3, 4

---

## Executive Summary

This plan details the step-by-step implementation of a unified experimentation platform that consolidates experiments 2, 3, and 4 into a reusable, maintainable architecture. The platform will separate concerns into models, optimizers, data handling, and evaluation while maintaining full compatibility with existing tracking infrastructure.

### Key Design Principles

1. **Incremental Implementation**: Build and test each module before moving to the next
2. **Backward Compatibility**: Keep experiment 4 functional during transition as validation reference
3. **Zero Infrastructure Changes**: experiments/tracking/, experiments/cli/, experiments/slurm/ remain untouched
4. **Testing at Each Step**: Validate each module against experiment 4's behavior before proceeding

---

## Phase 1: Foundation - Core Platform Structure

### Step 1.1: Create Directory Structure

**Action:** Create the platform/ directory hierarchy

```
platform/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── feedforward.py       # From exp 2, 3
│   ├── recurrent.py          # From exp 4
│   └── dynamic.py            # From exp 4 + common/dynamic_net
├── optimizers/
│   ├── __init__.py
│   ├── base.py              # Shared utilities
│   ├── sgd.py               # From exp 4 deeplearn_recurrent
│   └── genetic.py           # From exp 4 neuroevolve_*
├── data/
│   ├── __init__.py
│   ├── loaders.py           # From exp 2, 3, 4
│   └── preprocessing.py     # CL features, normalization
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py           # CE, F1, behavioral metrics
│   └── comparison.py        # Human-model comparison
├── config.py                # Consolidated configuration
└── runner.py                # Main execution engine
```

**Files to Create:**
- All `__init__.py` files with proper imports
- Placeholder files with docstrings

**Success Criteria:**
- `import platform.models` works
- `import platform.optimizers` works
- Directory structure matches PLATFORM_PLAN.md

---

## Phase 2: Models Module - Architecture Definitions

### Step 2.1: Extract Feedforward Models

**Source Files:**
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/3_cl_info_dl_vs_ga/src/models.py` (lines 12-36: MLP class)
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/2_dl_vs_ga_es/main.py` (MLP definition)

**Target:** `platform/models/feedforward.py`

**Content:**
```python
"""Feedforward neural network architectures.

Shared by both SGD and GA optimizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


class MLP(nn.Module):
    """Two-layer feedforward MLP with tanh activation.
    
    Architecture: [input_size, hidden_size, output_size]
    
    Used by:
    - SGD optimizer (backpropagation)
    - GA optimizer (mutation-based evolution)
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(
        self, 
        x: Float[Tensor, "BS input_size"]
    ) -> Float[Tensor, "BS output_size"]:
        """Forward pass returning logits."""
        h = torch.tanh(self.fc1(x))
        logits = self.fc2(h)
        return logits
    
    def get_probs(
        self, 
        x: Float[Tensor, "BS input_size"]
    ) -> Float[Tensor, "BS output_size"]:
        """Get probability distribution over actions."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs
```

**Testing:**
```python
# Quick validation test
model = MLP(4, 50, 2)
x = torch.randn(32, 4)
logits = model(x)
assert logits.shape == (32, 2)
probs = model.get_probs(x)
assert torch.allclose(probs.sum(dim=1), torch.ones(32))
```

**Success Criteria:**
- MLP class instantiates correctly
- Forward pass produces correct shapes
- Probability sums to 1

---

### Step 2.2: Extract Recurrent Models

**Source Files:**
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/4_add_recurrence/src/models.py` (lines 15-213)

**Target:** `platform/models/recurrent.py`

**Content:** Copy directly from experiment 4:
- `RecurrentMLPReservoir` (lines 15-111)
- `RecurrentMLPTrainable` (lines 113-213)

**Key Features to Preserve:**
- Frozen reservoir weights (W_hh buffer)
- Rank-1 factorization for trainable variant (u ⊗ v^T)
- `forward_step()` and `forward()` methods
- `get_probs()` for action sampling
- Hidden state management

**Testing:**
```python
# Test reservoir model
model = RecurrentMLPReservoir(4, 50, 2)
x_seq = torch.randn(8, 10, 4)  # batch=8, seq_len=10, input=4
logits, h_final = model(x_seq)
assert logits.shape == (8, 10, 2)
assert h_final.shape == (8, 50)

# Test trainable model
model2 = RecurrentMLPTrainable(4, 50, 2)
logits2, h_final2 = model2(x_seq)
assert logits2.shape == (8, 10, 2)
```

**Success Criteria:**
- Both recurrent models instantiate
- Sequence processing works correctly
- Hidden states propagate properly

---

### Step 2.3: Extract Dynamic Networks

**Source Files:**
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/4_add_recurrence/src/models.py` (lines 607-923: DynamicNetPopulation)
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/common/dynamic_net/evolution.py` (Net class)
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/common/dynamic_net/computation.py` (WelfordRunningStandardizer)

**Target:** `platform/models/dynamic.py`

**Strategy:** 
- Copy `DynamicNetPopulation` wrapper from exp 4
- Import from `common/dynamic_net` (keep that module as-is)
- This is GA-exclusive (no gradients through topology changes)

**Content:**
```python
"""Dynamic complexity networks for neuroevolution.

These networks have evolving topologies and are GA-exclusive.
Cannot be used with SGD due to non-differentiable topology changes.
"""

from common.dynamic_net.evolution import Net
from common.dynamic_net.computation import WelfordRunningStandardizer

# Copy DynamicNetPopulation class from exp 4
# (lines 607-923 of models.py)
```

**Testing:**
```python
# Test dynamic network population
pop = DynamicNetPopulation(
    input_size=4,
    output_size=2,
    pop_size=10,
    initial_mutations=5
)
obs = torch.randn(32, 4)
logits = pop.forward_batch(obs)
assert logits.shape == (10, 32, 2)
```

**Success Criteria:**
- DynamicNetPopulation instantiates
- Batched forward pass works
- Mutation and selection operations work

---

## Phase 3: Data Module - Loading and Preprocessing

### Step 3.1: Create Data Loaders

**Source Files:**
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/4_add_recurrence/src/data.py` (lines 124-359: load_human_data)
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/2_dl_vs_ga_es/main.py` (HuggingFace loaders)

**Target:** `platform/data/loaders.py`

**Content:**
```python
"""Data loading functions for various sources.

Supports:
- Human behavioral data (local JSON files)
- HuggingFace datasets (CartPole-v1, LunarLander-v2)
"""

def load_human_data(
    env_name: str,
    use_cl_info: bool,
    subject: str = "sub01",
    holdout_pct: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
    """Load human behavioral data from JSON files.
    
    Returns:
        (optim_obs, optim_act, test_obs, test_act, metadata)
    """
    # Copy from exp 4 data.py lines 124-359
    pass


def load_huggingface_data(
    dataset_name: str,
    dataset_size: int,
    train_split: float = 0.9,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load data from HuggingFace datasets hub.
    
    Supported: CartPole-v1, LunarLander-v2
    
    Returns:
        (train_obs, train_act, test_obs, test_act)
    """
    # Extract from exp 2 main.py
    pass
```

**Testing:**
```python
# Test human data loading
optim_obs, optim_act, test_obs, test_act, metadata = load_human_data(
    "cartpole", use_cl_info=True, subject="sub01"
)
assert optim_obs.shape[1] == 6  # 4 obs + 2 CL features
assert "optim_episode_boundaries" in metadata
```

---

### Step 3.2: Create Preprocessing Utilities

**Source Files:**
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/4_add_recurrence/src/data.py` (lines 16-122: session/run computation, normalization)
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/4_add_recurrence/src/data.py` (lines 362-429: EpisodeDataset, collate_fn)

**Target:** `platform/data/preprocessing.py`

**Content:**
```python
"""Data preprocessing utilities.

Functions:
- compute_session_run_ids(): Extract continual learning features from timestamps
- normalize_session_run_features(): Map session/run IDs to [-1, 1]
- EpisodeDataset: PyTorch Dataset for episodic data
- episode_collate_fn(): Collate with padding for variable-length episodes
"""

# Copy from exp 4 data.py:
# - compute_session_run_ids (lines 16-62)
# - normalize_session_run_features (lines 65-122)
# - EpisodeDataset class (lines 362-392)
# - episode_collate_fn (lines 394-429)
```

**Testing:**
```python
# Test CL feature extraction
timestamps = ["2024-01-01T10:00:00", "2024-01-01T10:15:00", "2024-01-01T11:00:00"]
session_ids, run_ids = compute_session_run_ids(timestamps)
assert len(session_ids) == 3
assert session_ids[0] == session_ids[1]  # Same session
assert session_ids[2] > session_ids[1]   # New session after 1 hour
```

---

## Phase 4: Evaluation Module - Metrics and Comparison

### Step 4.1: Create Metrics Functions

**Source Files:**
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/4_add_recurrence/src/metrics.py` (if exists) or inline from optim.py

**Target:** `platform/evaluation/metrics.py`

**Content:**
```python
"""Evaluation metrics for behavioral cloning.

Metrics:
- Cross-entropy loss (primary fitness for both SGD and GA)
- Macro F1 score (secondary metric)
- Behavioral similarity (percentage difference from human performance)
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from sklearn.metrics import f1_score


def compute_cross_entropy(
    logits: Float[Tensor, "N output_size"],
    targets: Int[Tensor, " N"],
) -> float:
    """Compute mean cross-entropy loss."""
    loss = F.cross_entropy(logits, targets)
    return loss.item()


def compute_macro_f1(
    predictions: list[int],
    targets: list[int],
    num_classes: int,
) -> float:
    """Compute macro-averaged F1 score."""
    return f1_score(
        targets,
        predictions,
        average="macro",
        labels=list(range(num_classes)),
        zero_division=0,
    )


def compute_behavioral_similarity(
    model_returns: list[float],
    human_returns: list[float],
) -> tuple[float, float]:
    """Compute percentage difference between model and human returns.
    
    Returns:
        (mean_pct_diff, std_pct_diff)
    """
    import numpy as np
    pct_diffs = [
        ((m - h) / abs(h)) * 100 if h != 0 else m * 100
        for m, h in zip(model_returns, human_returns)
    ]
    return float(np.mean(pct_diffs)), float(np.std(pct_diffs))
```

---

### Step 4.2: Create Comparison Utilities

**Source Files:**
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/4_add_recurrence/src/optim.py` (lines 27-122: evaluate_progression_recurrent)

**Target:** `platform/evaluation/comparison.py`

**Content:**
```python
"""Human-model behavior comparison utilities.

Functions for evaluating how closely models match human behavior
by rolling out episodes and comparing returns.
"""

import gymnasium as gym
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor


def evaluate_model_on_human_episodes(
    model,  # RecurrentMLPReservoir | RecurrentMLPTrainable | MLP
    episode_details: list[dict],
    env: gym.Env,
    max_steps: int = 1000,
    use_cl_features: bool = False,
    model_type: str = "recurrent",  # "recurrent" or "feedforward"
) -> tuple[float, float, list[float]]:
    """Evaluate model by replaying human episode seeds.
    
    Returns:
        (mean_pct_diff, std_pct_diff, model_returns)
    """
    # Copy from evaluate_progression_recurrent in optim.py
    # Adapt for both recurrent and feedforward models
    pass
```

---

## Phase 5: Optimizers Module - Training Algorithms

### Step 5.1: Create Base Utilities

**Target:** `platform/optimizers/base.py`

**Content:**
```python
"""Shared utilities for optimization algorithms.

Functions used by both SGD and GA optimizers:
- Checkpoint management
- Episode creation and batching
- Progress tracking
- Logging helpers
"""

from pathlib import Path
import torch
from torch import Tensor


def create_episode_list(
    observations: Tensor,
    actions: Tensor,
    episode_boundaries: list[tuple[int, int]],
) -> list[dict]:
    """Convert flat data into list of episode dicts.
    
    Used by both SGD (for DataLoader) and GA (for fitness eval).
    """
    episodes = []
    for start, length in episode_boundaries:
        episodes.append({
            "observations": observations[start:start + length],
            "actions": actions[start:start + length],
        })
    return episodes


def save_checkpoint(
    checkpoint_path: Path,
    epoch_or_gen: int,
    model_or_population,
    optimizer_state: dict | None,
    metrics: dict,
    elapsed_time: float,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "epoch_or_generation": epoch_or_gen,
        "model_or_population_state": (
            model_or_population.state_dict() 
            if hasattr(model_or_population, "state_dict")
            else model_or_population.get_state_dict()
        ),
        "optimizer_state": optimizer_state,
        "metrics": metrics,
        "elapsed_time": elapsed_time,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load training checkpoint."""
    return torch.load(checkpoint_path, weights_only=False)
```

---

### Step 5.2: Extract SGD Optimizer

**Source Files:**
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/4_add_recurrence/src/optim.py` (lines 150-453: deeplearn_recurrent)

**Target:** `platform/optimizers/sgd.py`

**Content:**
```python
"""SGD (backpropagation) optimizer for behavioral cloning.

Supports:
- Feedforward models (MLP)
- Recurrent models (RecurrentMLPReservoir, RecurrentMLPTrainable)

Training loop:
- Episode-based DataLoader with padding/masking
- Adam optimizer (or configurable)
- Time-based evaluation intervals
- Checkpoint and behavioral evaluation
- Integration with ExperimentLogger
"""

from pathlib import Path
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from platform.data.preprocessing import EpisodeDataset, episode_collate_fn
from platform.evaluation.metrics import compute_cross_entropy, compute_macro_f1
from platform.optimizers.base import create_episode_list, save_checkpoint


def optimize_sgd(
    model,  # MLP | RecurrentMLPReservoir | RecurrentMLPTrainable
    optim_obs: Tensor,
    optim_act: Tensor,
    test_obs: Tensor,
    test_act: Tensor,
    config,  # ExperimentConfig
    metadata: dict,
    checkpoint_path: Path,
    logger=None,  # ExperimentLogger
    device: torch.device = torch.device("cuda:0"),
) -> dict:
    """Run SGD optimization.
    
    Returns:
        Dictionary with training history
    """
    # Copy structure from deeplearn_recurrent (lines 150-453)
    # Generalize to work with both feedforward and recurrent models
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create episode dataset (for recurrent) or standard dataset (for feedforward)
    if "episode_boundaries" in metadata:
        # Recurrent: use episodes
        dataset = EpisodeDataset(
            optim_obs, optim_act, metadata["optim_episode_boundaries"]
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=episode_collate_fn
        )
    else:
        # Feedforward: use flat samples
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(optim_obs, optim_act)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training loop with time-based stopping
    # ... (full implementation copied from exp 4)
    
    pass
```

**Key Adaptations:**
- Support both episodic (recurrent) and flat (feedforward) data
- Unified checkpoint format
- Consistent logging interface
- Time-based evaluation intervals

---

### Step 5.3: Extract GA Optimizer

**Source Files:**
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/4_add_recurrence/src/optim.py` (lines 456-703: neuroevolve_recurrent)
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/4_add_recurrence/src/optim.py` (lines 706-836: neuroevolve_dynamic)
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/3_cl_info_dl_vs_ga/src/models.py` (lines 39-200: BatchedPopulation for feedforward)
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/4_add_recurrence/src/models.py` (lines 216-605: BatchedRecurrentPopulation)

**Target:** `platform/optimizers/genetic.py`

**Content:**
```python
"""Genetic Algorithm optimizer for behavioral cloning.

Supports:
- Feedforward models (BatchedPopulation)
- Recurrent models (BatchedRecurrentPopulation)
- Dynamic networks (DynamicNetPopulation)

Features:
- GPU-parallel batched evaluation
- Adaptive sigma mutation
- Simple GA selection (top 50% survive and duplicate)
- Cross-entropy fitness
- F1 score fitness (future)

GA-exclusive capabilities:
- Works with non-differentiable models (DynamicNet)
- Population-based optimization
- Evolutionary operators (mutation, selection)
"""

import random
import time
from pathlib import Path
import torch

from platform.optimizers.base import create_episode_list, save_checkpoint


class BatchedPopulation:
    """Batched feedforward population for efficient GPU-parallel evolution."""
    # Copy from exp 3 models.py lines 39-200
    pass


class BatchedRecurrentPopulation:
    """Batched recurrent population for efficient GPU-parallel evolution."""
    # Copy from exp 4 models.py lines 216-605
    pass


def optimize_ga(
    model_type: str,  # "feedforward", "recurrent_reservoir", "recurrent_trainable", "dynamic"
    input_size: int,
    output_size: int,
    hidden_size: int,
    optim_obs: Tensor,
    optim_act: Tensor,
    test_obs: Tensor,
    test_act: Tensor,
    config,  # ExperimentConfig
    metadata: dict,
    checkpoint_path: Path,
    logger=None,  # ExperimentLogger
    device: torch.device = torch.device("cuda:0"),
) -> dict:
    """Run genetic algorithm optimization.
    
    Returns:
        Dictionary with evolution history
    """
    # Initialize appropriate population type
    if model_type == "feedforward":
        population = BatchedPopulation(
            input_size, hidden_size, output_size,
            config.population_size,
            config.adaptive_sigma_init,
            config.adaptive_sigma_noise,
        )
    elif model_type in ["recurrent_reservoir", "recurrent_trainable"]:
        from platform.models.recurrent import BatchedRecurrentPopulation
        population = BatchedRecurrentPopulation(
            input_size, hidden_size, output_size,
            config.population_size,
            model_type.replace("recurrent_", ""),
            config.adaptive_sigma_init,
            config.adaptive_sigma_noise,
        )
    elif model_type == "dynamic":
        from platform.models.dynamic import DynamicNetPopulation
        population = DynamicNetPopulation(
            input_size, output_size, config.population_size
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Evolution loop
    # ... (combine structure from neuroevolve_recurrent and neuroevolve_dynamic)
    
    pass
```

**Key Consolidation:**
- Single `optimize_ga()` function handles all model types
- Batched population classes moved here (closer to where they're used)
- Consistent interface with `optimize_sgd()`

---

## Phase 6: Configuration System

### Step 6.1: Consolidate Configuration

**Source Files:**
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/4_add_recurrence/src/config.py`
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/2_dl_vs_ga_es/main.py` (ExperimentConfig)

**Target:** `platform/config.py`

**Content:**
```python
"""Unified configuration system for experimentation platform.

Defines:
- ExperimentConfig: Hyperparameters for training
- ENV_CONFIGS: Environment-specific parameters
- Device management
- Directory paths
"""

from dataclasses import dataclass
from pathlib import Path
import torch


# Device configuration
DEVICE = torch.device("cuda:0")  # Default, overridable


def set_device(gpu_index: int) -> None:
    """Set global DEVICE variable."""
    global DEVICE
    DEVICE = torch.device(f"cuda:{gpu_index}")


@dataclass
class ExperimentConfig:
    """Unified experiment configuration."""
    
    # Data parameters
    batch_size: int = 32
    holdout_pct: float = 0.1
    
    # Model parameters
    hidden_size: int = 50
    
    # Optimizer parameters - SGD
    learning_rate: float = 1e-3
    optimizer_type: str = "adam"  # "adam", "sgd", "adamw"
    
    # Optimizer parameters - GA
    population_size: int = 50
    adaptive_sigma_init: float = 1e-3
    adaptive_sigma_noise: float = 1e-2
    
    # Training parameters
    max_optim_time: int = 36000  # 10 hours in seconds
    loss_eval_interval_seconds: int = 60  # Evaluate every minute
    ckpt_and_behav_eval_interval_seconds: int = 300  # Checkpoint every 5 minutes
    
    # Evaluation parameters
    num_f1_samples: int = 10
    num_eval_episodes: int = 100
    
    # Reproducibility
    seed: int = 42


# Environment configurations
ENV_CONFIGS = {
    "cartpole": {
        "obs_dim": 4,
        "action_dim": 2,
        "gym_name": "CartPole-v1",
        "data_files": {
            "sub01": "sub01_data_cartpole.json",
            "sub02": "sub02_data_cartpole.json",
        },
    },
    "mountaincar": {
        "obs_dim": 2,
        "action_dim": 3,
        "gym_name": "MountainCar-v0",
        "data_files": {
            "sub01": "sub01_data_mountaincar.json",
            "sub02": "sub02_data_mountaincar.json",
        },
    },
    "acrobot": {
        "obs_dim": 6,
        "action_dim": 3,
        "gym_name": "Acrobot-v1",
        "data_files": {
            "sub01": "sub01_data_acrobot.json",
            "sub02": "sub02_data_acrobot.json",
        },
    },
    "lunarlander": {
        "obs_dim": 8,
        "action_dim": 4,
        "gym_name": "LunarLander-v2",
        "data_files": {
            "sub01": "sub01_data_lunarlander.json",
            "sub02": "sub02_data_lunarlander.json",
        },
    },
}


# Directory structure
PLATFORM_DIR = Path(__file__).parent
PROJECT_ROOT = PLATFORM_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
```

**Success Criteria:**
- All parameters from exp 2, 3, 4 are included
- No duplicate or conflicting configurations
- Clear documentation of what each parameter controls

---

## Phase 7: Runner - Main Execution Engine

### Step 7.1: Create Unified Runner

**Source Files:**
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/4_add_recurrence/main.py` (structure and flow)

**Target:** `platform/runner.py`

**Content:**
```python
"""Main experiment runner for the unified platform.

Orchestrates:
1. Configuration loading (from YAML or command-line)
2. Data loading
3. Model initialization
4. Optimizer selection
5. Training execution
6. Results logging

Usage:
    python -m platform.runner --config experiments/configs/cartpole_sgd_reservoir.yaml
    
Or programmatically:
    from platform.runner import run_experiment
    run_experiment(config_dict)
"""

import argparse
import sys
from pathlib import Path
import yaml

import torch

from platform.config import ExperimentConfig, ENV_CONFIGS, set_device, RESULTS_DIR
from platform.data.loaders import load_human_data, load_huggingface_data
from platform.models.feedforward import MLP
from platform.models.recurrent import RecurrentMLPReservoir, RecurrentMLPTrainable
from platform.models.dynamic import DynamicNetPopulation
from platform.optimizers.sgd import optimize_sgd
from platform.optimizers.genetic import optimize_ga

# Import tracking infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments" / "tracking"))
from database import ExperimentDB
from logger import ExperimentLogger


def run_experiment(
    dataset: str,
    method: str,
    model_type: str,
    optimizer_type: str,
    use_cl_info: bool = False,
    subject: str = "sub01",
    config: ExperimentConfig | None = None,
    gpu_id: int = 0,
) -> dict:
    """Run a single experiment.
    
    Args:
        dataset: Environment name (cartpole, lunarlander, etc.)
        method: Method name (for tracking)
        model_type: "feedforward", "recurrent_reservoir", "recurrent_trainable", "dynamic"
        optimizer_type: "sgd" or "ga"
        use_cl_info: Whether to include CL features
        subject: Subject identifier
        config: ExperimentConfig instance (or use default)
        gpu_id: GPU index
        
    Returns:
        Dictionary with training results
    """
    # 1. Setup
    if config is None:
        config = ExperimentConfig()
    
    set_device(gpu_id)
    torch.manual_seed(config.seed)
    
    env_config = ENV_CONFIGS[dataset]
    
    # 2. Load data
    print(f"Loading {dataset} data...")
    optim_obs, optim_act, test_obs, test_act, metadata = load_human_data(
        dataset, use_cl_info, subject, config.holdout_pct
    )
    
    input_size = optim_obs.shape[1]
    output_size = env_config["action_dim"]
    
    # 3. Initialize tracking
    db_path = Path(__file__).parent.parent / "experiments" / "tracking.db"
    db = ExperimentDB(db_path)
    
    logger = ExperimentLogger(
        db=db,
        experiment_number=5,  # Platform experiments start at 5
        dataset=dataset,
        method=method,
        subject=subject,
        use_cl_info=use_cl_info,
        seed=config.seed,
        config=config,
        gpu_id=gpu_id,
    )
    
    # 4. Run optimization
    checkpoint_path = RESULTS_DIR / f"{dataset}_{method}_{subject}_checkpoint.pt"
    
    with logger:
        if optimizer_type == "sgd":
            # Initialize model
            if model_type == "feedforward":
                model = MLP(input_size, config.hidden_size, output_size)
            elif model_type == "recurrent_reservoir":
                model = RecurrentMLPReservoir(input_size, config.hidden_size, output_size)
            elif model_type == "recurrent_trainable":
                model = RecurrentMLPTrainable(input_size, config.hidden_size, output_size)
            else:
                raise ValueError(f"Model type {model_type} not supported with SGD")
            
            model = model.to(torch.device(f"cuda:{gpu_id}"))
            
            results = optimize_sgd(
                model=model,
                optim_obs=optim_obs,
                optim_act=optim_act,
                test_obs=test_obs,
                test_act=test_act,
                config=config,
                metadata=metadata,
                checkpoint_path=checkpoint_path,
                logger=logger,
                device=torch.device(f"cuda:{gpu_id}"),
            )
            
        elif optimizer_type == "ga":
            results = optimize_ga(
                model_type=model_type,
                input_size=input_size,
                output_size=output_size,
                hidden_size=config.hidden_size,
                optim_obs=optim_obs,
                optim_act=optim_act,
                test_obs=test_obs,
                test_act=test_act,
                config=config,
                metadata=metadata,
                checkpoint_path=checkpoint_path,
                logger=logger,
                device=torch.device(f"cuda:{gpu_id}"),
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return results


def main():
    """Command-line interface for running experiments."""
    parser = argparse.ArgumentParser(description="Run unified platform experiment")
    
    # Either load from YAML config
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    
    # Or specify parameters directly
    parser.add_argument("--dataset", type=str, choices=list(ENV_CONFIGS.keys()))
    parser.add_argument("--method", type=str, help="Method name for tracking")
    parser.add_argument("--model", type=str, 
                       choices=["feedforward", "recurrent_reservoir", "recurrent_trainable", "dynamic"])
    parser.add_argument("--optimizer", type=str, choices=["sgd", "ga"])
    parser.add_argument("--use-cl-info", action="store_true")
    parser.add_argument("--subject", type=str, default="sub01")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.config:
        # Load from YAML
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
        
        run_experiment(**config_dict)
    else:
        # Use command-line arguments
        if not all([args.dataset, args.method, args.model, args.optimizer]):
            parser.error("Must provide either --config or all of: --dataset, --method, --model, --optimizer")
        
        config = ExperimentConfig(seed=args.seed)
        
        run_experiment(
            dataset=args.dataset,
            method=args.method,
            model_type=args.model,
            optimizer_type=args.optimizer,
            use_cl_info=args.use_cl_info,
            subject=args.subject,
            config=config,
            gpu_id=args.gpu,
        )


if __name__ == "__main__":
    main()
```

**Success Criteria:**
- Can run via `python -m platform.runner --config <yaml>`
- Can run via command-line arguments
- Integrates with ExperimentLogger
- Checkpointing works correctly

---

## Phase 8: YAML Configuration System

### Step 8.1: Create Example Configs

**Target:** `experiments/configs/`

**Create these example files:**

**`experiments/configs/cartpole_sgd_reservoir.yaml`:**
```yaml
dataset: cartpole
method: SGD_reservoir_CL
model_type: recurrent_reservoir
optimizer_type: sgd
use_cl_info: true
subject: sub01
gpu_id: 0

config:
  seed: 42
  batch_size: 32
  hidden_size: 50
  learning_rate: 0.001
  max_optim_time: 36000  # 10 hours
  loss_eval_interval_seconds: 60
  ckpt_and_behav_eval_interval_seconds: 300
```

**`experiments/configs/lunarlander_ga_trainable.yaml`:**
```yaml
dataset: lunarlander
method: GA_trainable_noCL
model_type: recurrent_trainable
optimizer_type: ga
use_cl_info: false
subject: sub01
gpu_id: 0

config:
  seed: 42
  population_size: 50
  adaptive_sigma_init: 0.001
  adaptive_sigma_noise: 0.01
  max_optim_time: 36000
```

**`experiments/configs/cartpole_ga_dynamic.yaml`:**
```yaml
dataset: cartpole
method: GA_dynamic_CL
model_type: dynamic
optimizer_type: ga
use_cl_info: true
subject: sub01
gpu_id: 0

config:
  seed: 42
  population_size: 50
  max_optim_time: 36000
```

---

## Phase 9: Testing and Validation

### Step 9.1: Unit Tests

**Target:** `platform/tests/`

Create test files for each module:

**`platform/tests/test_models.py`:**
```python
"""Unit tests for model architectures."""

import torch
from platform.models.feedforward import MLP
from platform.models.recurrent import RecurrentMLPReservoir, RecurrentMLPTrainable


def test_mlp():
    model = MLP(4, 50, 2)
    x = torch.randn(32, 4)
    logits = model(x)
    assert logits.shape == (32, 2)


def test_recurrent_reservoir():
    model = RecurrentMLPReservoir(4, 50, 2)
    x = torch.randn(8, 10, 4)
    logits, h = model(x)
    assert logits.shape == (8, 10, 2)
    assert h.shape == (8, 50)


def test_recurrent_trainable():
    model = RecurrentMLPTrainable(4, 50, 2)
    x = torch.randn(8, 10, 4)
    logits, h = model(x)
    assert logits.shape == (8, 10, 2)
    assert h.shape == (8, 50)
```

**Run tests:**
```bash
cd /scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research
python -m pytest platform/tests/
```

---

### Step 9.2: Integration Test - Reproduce Experiment 4

**Goal:** Verify platform produces same results as experiment 4

**Test:** `platform/tests/test_integration_exp4.py`

```python
"""Integration test: verify platform matches experiment 4 behavior."""

import torch
from platform.runner import run_experiment
from platform.config import ExperimentConfig


def test_sgd_reservoir_matches_exp4():
    """Run platform version and compare with exp4 checkpoint."""
    
    config = ExperimentConfig(
        seed=42,
        max_optim_time=60,  # Short test run
    )
    
    results = run_experiment(
        dataset="cartpole",
        method="test_SGD_reservoir",
        model_type="recurrent_reservoir",
        optimizer_type="sgd",
        use_cl_info=True,
        subject="sub01",
        config=config,
        gpu_id=0,
    )
    
    # Verify results structure
    assert "loss_history" in results
    assert "test_loss_history" in results
    assert len(results["loss_history"]) > 0
    
    print("✓ Platform SGD matches experiment 4 structure")


def test_ga_reservoir_matches_exp4():
    """Test GA optimizer matches experiment 4."""
    
    config = ExperimentConfig(
        seed=42,
        population_size=10,  # Small for fast test
        max_optim_time=60,
    )
    
    results = run_experiment(
        dataset="cartpole",
        method="test_GA_reservoir",
        model_type="recurrent_reservoir",
        optimizer_type="ga",
        use_cl_info=True,
        subject="sub01",
        config=config,
        gpu_id=0,
    )
    
    assert "fitness_history" in results
    assert len(results["fitness_history"]) > 0
    
    print("✓ Platform GA matches experiment 4 structure")
```

**Run integration tests:**
```bash
python -m pytest platform/tests/test_integration_exp4.py -v -s
```

---

## Phase 10: CLI Update and Archive

### Step 10.1: Update CLI Tools

**Files to modify:**
- `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments/cli/submit_jobs.py`

**Changes:**
```python
# Add option to use platform runner instead of individual experiment scripts

def submit_platform_job(
    dataset: str,
    method: str,
    model_type: str,
    optimizer_type: str,
    config_path: str | None = None,
    **kwargs
) -> int:
    """Submit job using platform runner.
    
    Returns:
        SLURM job ID
    """
    if config_path:
        cmd = f"python -m platform.runner --config {config_path}"
    else:
        cmd = f"python -m platform.runner --dataset {dataset} --method {method} --model {model_type} --optimizer {optimizer_type}"
        for key, value in kwargs.items():
            cmd += f" --{key.replace('_', '-')} {value}"
    
    # Submit via existing SLURM infrastructure
    job_id = submit_slurm_job(cmd, ...)
    return job_id
```

**Success Criteria:**
- Can submit jobs using `submit_jobs.py --use-platform ...`
- Jobs appear in database via ExperimentLogger
- `monitor_jobs.py` shows platform jobs
- `query_results.py` can query platform results

---

### Step 10.2: Archive Old Experiments

**Action:** Move experiments 1-4 to archive

```bash
cd /scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/experiments
mkdir -p archive

# Move old experiments
mv 1_dl_vs_ga_scaling_dataset_size_flops archive/
mv 2_dl_vs_ga_es archive/
mv 3_cl_info_dl_vs_ga archive/
mv 4_add_recurrence archive/

# Create README in archive
cat > archive/README.md << 'EOF'
# Archived Experiments

These experiments have been consolidated into the unified platform/ module.

## Experiments

1. **1_dl_vs_ga_scaling_dataset_size_flops**: Initial DL vs GA comparison
2. **2_dl_vs_ga_es**: HuggingFace dataset experiments
3. **3_cl_info_dl_vs_ga**: Continual learning features added
4. **4_add_recurrence**: Recurrent and dynamic networks added

## Migration

All functionality has been extracted into:
- `platform/models/`: Model architectures
- `platform/optimizers/`: Training algorithms
- `platform/data/`: Data loading and preprocessing
- `platform/evaluation/`: Metrics and comparison

## Usage

To reproduce these experiments, use the platform runner:
```bash
python -m platform.runner --config experiments/configs/<experiment>.yaml
```

See `experiments/configs/` for example configurations.
EOF
```

**Success Criteria:**
- All 4 experiments moved to archive/
- Archive has README explaining migration
- Git history preserved (use `git mv`)

---

## Phase 11: Documentation

### Step 11.1: Platform Documentation

**Target:** `platform/README.md`

```markdown
# Unified Experimentation Platform

A modular, reusable platform for behavioral cloning experiments comparing deep learning (SGD) and genetic algorithms (GA) across various model architectures.

## Architecture

```
platform/
├── models/           # Neural network architectures
│   ├── feedforward.py    # MLP (SGD + GA)
│   ├── recurrent.py      # Reservoir & trainable RNNs (SGD + GA)
│   └── dynamic.py        # Dynamic complexity networks (GA-only)
├── optimizers/       # Training algorithms
│   ├── sgd.py           # Backpropagation
│   ├── genetic.py       # Evolutionary algorithms
│   └── base.py          # Shared utilities
├── data/            # Data loading & preprocessing
│   ├── loaders.py       # HuggingFace + local JSON
│   └── preprocessing.py # CL features, episodes
├── evaluation/      # Metrics & comparison
│   ├── metrics.py       # CE, F1, behavioral similarity
│   └── comparison.py    # Human-model comparison
├── config.py        # Configuration system
└── runner.py        # Main execution engine
```

## Quick Start

### 1. Run from YAML config
```bash
python -m platform.runner --config experiments/configs/cartpole_sgd_reservoir.yaml
```

### 2. Run from command line
```bash
python -m platform.runner \
  --dataset cartpole \
  --method SGD_reservoir_CL \
  --model recurrent_reservoir \
  --optimizer sgd \
  --use-cl-info \
  --gpu 0
```

### 3. Run programmatically
```python
from platform.runner import run_experiment
from platform.config import ExperimentConfig

config = ExperimentConfig(
    seed=42,
    max_optim_time=3600,  # 1 hour
)

results = run_experiment(
    dataset="cartpole",
    method="my_experiment",
    model_type="recurrent_reservoir",
    optimizer_type="sgd",
    config=config,
)
```

## Supported Combinations

| Model Type | SGD | GA | Notes |
|-----------|-----|----|-------|
| feedforward | ✓ | ✓ | Standard MLP |
| recurrent_reservoir | ✓ | ✓ | Frozen recurrent weights |
| recurrent_trainable | ✓ | ✓ | Trainable recurrent (rank-1) |
| dynamic | ✗ | ✓ | GA-only (evolving topology) |

## Configuration

### YAML Format
```yaml
dataset: cartpole
method: SGD_reservoir
model_type: recurrent_reservoir
optimizer_type: sgd
use_cl_info: true
subject: sub01
gpu_id: 0

config:
  seed: 42
  batch_size: 32
  hidden_size: 50
  learning_rate: 0.001
  max_optim_time: 36000
  population_size: 50  # GA only
  adaptive_sigma_init: 0.001  # GA only
```

## Tracking and Logging

All experiments are logged to SQLite database (`experiments/tracking.db`):

```python
# Query results
from experiments.tracking.query import ExperimentQuery

query = ExperimentQuery("experiments/tracking.db")
runs = query.get_runs(dataset="cartpole", method="SGD_reservoir")
```

## Testing

Run unit tests:
```bash
python -m pytest platform/tests/
```

Run integration tests:
```bash
python -m pytest platform/tests/test_integration_exp4.py -v
```

## Migration from Old Experiments

Experiments 1-4 have been archived. To reproduce:

| Old Experiment | Platform Config |
|---------------|-----------------|
| Exp 2: DL vs GA (CartPole) | `configs/cartpole_sgd_feedforward.yaml` |
| Exp 3: CL features | Set `use_cl_info: true` |
| Exp 4: Recurrent | Use `recurrent_reservoir` or `recurrent_trainable` |
| Exp 4: Dynamic | Use `model_type: dynamic` with `optimizer: ga` |

## Design Principles

1. **Shared Models**: MLP and recurrent models work with both SGD and GA
2. **GA-Exclusive**: Dynamic networks only work with GA
3. **Optimizer Separation**: Weight updates (SGD vs GA) separated from model definitions
4. **Extensibility**: Easy to add new models, optimizers, or training methods
5. **Infrastructure Preservation**: tracking/, cli/, slurm/ remain unchanged
```

---

### Step 11.2: Example Sweep Configuration

**Target:** `experiments/configs/sweeps/cartpole_recurrent_sweep.yaml`

```yaml
# Sweep configuration: test all recurrent variants on CartPole

sweep_name: cartpole_recurrent_comparison
base_config:
  dataset: cartpole
  subject: sub01
  gpu_id: 0
  config:
    seed: 42
    max_optim_time: 36000
    hidden_size: 50

experiments:
  # SGD experiments
  - method: SGD_reservoir_CL
    model_type: recurrent_reservoir
    optimizer_type: sgd
    use_cl_info: true
  
  - method: SGD_reservoir_noCL
    model_type: recurrent_reservoir
    optimizer_type: sgd
    use_cl_info: false
  
  - method: SGD_trainable_CL
    model_type: recurrent_trainable
    optimizer_type: sgd
    use_cl_info: true
  
  - method: SGD_trainable_noCL
    model_type: recurrent_trainable
    optimizer_type: sgd
    use_cl_info: false
  
  # GA experiments
  - method: GA_reservoir_CL
    model_type: recurrent_reservoir
    optimizer_type: ga
    use_cl_info: true
    config:
      population_size: 50
  
  - method: GA_reservoir_noCL
    model_type: recurrent_reservoir
    optimizer_type: ga
    use_cl_info: false
    config:
      population_size: 50
  
  - method: GA_trainable_CL
    model_type: recurrent_trainable
    optimizer_type: ga
    use_cl_info: true
    config:
      population_size: 50
  
  - method: GA_trainable_noCL
    model_type: recurrent_trainable
    optimizer_type: ga
    use_cl_info: false
    config:
      population_size: 50
  
  - method: GA_dynamic_CL
    model_type: dynamic
    optimizer_type: ga
    use_cl_info: true
    config:
      population_size: 50
```

**Sweep Runner:** `experiments/cli/run_sweep.py`

```python
"""Run a sweep of experiments from YAML configuration."""

import argparse
import yaml
from pathlib import Path
import copy

from experiments.cli.submit_jobs import submit_platform_job


def run_sweep(sweep_config_path: Path) -> list[int]:
    """Submit all experiments in a sweep.
    
    Returns:
        List of SLURM job IDs
    """
    with open(sweep_config_path) as f:
        sweep = yaml.safe_load(f)
    
    base_config = sweep.get("base_config", {})
    job_ids = []
    
    for exp in sweep["experiments"]:
        # Merge base config with experiment-specific config
        config = copy.deepcopy(base_config)
        config.update(exp)
        
        # Submit job
        job_id = submit_platform_job(**config)
        job_ids.append(job_id)
        print(f"Submitted {exp['method']}: job {job_id}")
    
    return job_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_config", type=Path, help="Path to sweep YAML")
    args = parser.parse_args()
    
    job_ids = run_sweep(args.sweep_config)
    print(f"\nSubmitted {len(job_ids)} jobs")
```

---

## Phase 12: Final Validation and Cleanup

### Step 12.1: End-to-End Test

**Test all combinations:**

```bash
# Test 1: SGD + Recurrent Reservoir
python -m platform.runner \
  --dataset cartpole \
  --method test_sgd_reservoir \
  --model recurrent_reservoir \
  --optimizer sgd \
  --use-cl-info \
  --gpu 0

# Test 2: GA + Recurrent Trainable
python -m platform.runner \
  --dataset cartpole \
  --method test_ga_trainable \
  --model recurrent_trainable \
  --optimizer ga \
  --gpu 0

# Test 3: GA + Dynamic
python -m platform.runner \
  --dataset cartpole \
  --method test_ga_dynamic \
  --model dynamic \
  --optimizer ga \
  --use-cl-info \
  --gpu 0

# Test 4: Feedforward MLP + SGD
python -m platform.runner \
  --dataset lunarlander \
  --method test_sgd_feedforward \
  --model feedforward \
  --optimizer sgd \
  --gpu 0
```

**Success Criteria:**
- All 4 tests complete without errors
- Results appear in tracking.db
- Checkpoints are created and resumable
- Behavioral evaluation works (if implemented)

---

### Step 12.2: Code Quality Checks

**Run formatters and linters:**
```bash
# Format code
black platform/

# Type checking
mypy platform/ --ignore-missing-imports

# Linting
ruff check platform/
```

**Fix any issues identified**

---

### Step 12.3: Git Commit Strategy

**Commit the platform in logical chunks:**

```bash
# Commit 1: Directory structure
git add platform/__init__.py platform/*/_.py
git commit -m "platform: create directory structure and module stubs"

# Commit 2: Models
git add platform/models/
git commit -m "platform: extract model architectures from experiments 2-4"

# Commit 3: Data
git add platform/data/
git commit -m "platform: consolidate data loading and preprocessing"

# Commit 4: Evaluation
git add platform/evaluation/
git commit -m "platform: extract evaluation metrics and comparison utilities"

# Commit 5: Optimizers
git add platform/optimizers/
git commit -m "platform: implement SGD and GA optimizers"

# Commit 6: Config and Runner
git add platform/config.py platform/runner.py
git commit -m "platform: add configuration system and main runner"

# Commit 7: Example configs
git add experiments/configs/
git commit -m "experiments: add example YAML configurations"

# Commit 8: Tests
git add platform/tests/
git commit -m "platform: add unit and integration tests"

# Commit 9: Documentation
git add platform/README.md PLATFORM_IMPLEMENTATION_PLAN.md
git commit -m "docs: add platform documentation and implementation plan"

# Commit 10: CLI updates
git add experiments/cli/
git commit -m "cli: update tools to support platform runner"

# Commit 11: Archive
git mv experiments/1_dl_vs_ga_scaling_dataset_size_flops experiments/archive/
git mv experiments/2_dl_vs_ga_es experiments/archive/
git mv experiments/3_cl_info_dl_vs_ga experiments/archive/
git mv experiments/4_add_recurrence experiments/archive/
git add experiments/archive/README.md
git commit -m "experiments: archive experiments 1-4 after platform migration"
```

---

## Critical Implementation Notes

### 1. Import Paths

**Critical:** All imports must use absolute paths from project root:

```python
# CORRECT
from platform.models.feedforward import MLP
from platform.optimizers.sgd import optimize_sgd

# WRONG
from models.feedforward import MLP  # Will fail
from ..optimizers.sgd import optimize_sgd  # Will fail
```

**Ensure PYTHONPATH:**
```bash
export PYTHONPATH=/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research:$PYTHONPATH
```

---

### 2. Device Management

**Critical:** All tensors must respect DEVICE configuration:

```python
# In each function that creates tensors
from platform.config import DEVICE

def some_function(...):
    tensor = torch.zeros(10, 20, device=DEVICE)  # Use DEVICE
    model = model.to(DEVICE)  # Move model to DEVICE
```

---

### 3. Checkpoint Compatibility

**Critical:** Maintain checkpoint format compatibility:

```python
# Save checkpoint with version info
checkpoint = {
    "version": "platform_v1",
    "model_type": model_type,
    "epoch_or_generation": epoch,
    "model_state": ...,
    "optimizer_state": ...,
    "metrics": ...,
}
```

This allows:
1. Resuming experiments seamlessly
2. Comparing with experiment 4 checkpoints
3. Future migrations

---

### 4. ExperimentLogger Integration

**Critical:** All optimizer functions must accept and use logger:

```python
def optimize_sgd(..., logger=None):
    if logger is not None:
        logger.log_progress(
            epoch=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            f1_score=f1,
        )
```

This ensures:
- Database tracking works
- CLI monitoring tools function
- Results are queryable

---

### 5. Episode vs Step Handling

**Critical:** Distinguish between episodic (recurrent) and step (feedforward) data:

```python
# Check metadata for episode boundaries
if "episode_boundaries" in metadata:
    # Recurrent: use EpisodeDataset
    dataset = EpisodeDataset(obs, act, metadata["episode_boundaries"])
    dataloader = DataLoader(dataset, collate_fn=episode_collate_fn, ...)
else:
    # Feedforward: use TensorDataset
    dataset = TensorDataset(obs, act)
    dataloader = DataLoader(dataset, ...)
```

---

## Testing Strategy

### Validation Hierarchy

1. **Unit Tests** (fastest, most focused)
   - Model forward passes produce correct shapes
   - Optimizers initialize correctly
   - Data loaders return expected formats

2. **Integration Tests** (medium speed)
   - Full training loops run without errors
   - Checkpointing and resumption work
   - Logging integrates with database

3. **Comparison Tests** (slowest, most comprehensive)
   - Platform results match experiment 4 results
   - Same random seed produces same outputs
   - Final metrics are within tolerance

### Test Execution Order

```bash
# 1. Fast unit tests (< 1 minute)
python -m pytest platform/tests/test_models.py -v

# 2. Integration tests (5-10 minutes)
python -m pytest platform/tests/test_integration_exp4.py -v

# 3. Full comparison (1-2 hours)
# Run platform and exp4 side-by-side, compare checkpoints
```

---

## Potential Pitfalls and Solutions

### Pitfall 1: Circular Imports

**Problem:** Models import from optimizers, optimizers import from models

**Solution:** Use dependency injection and import at function level:

```python
# In optimizers/genetic.py
def optimize_ga(model_type: str, ...):
    if model_type == "dynamic":
        from platform.models.dynamic import DynamicNetPopulation
        population = DynamicNetPopulation(...)
```

---

### Pitfall 2: GPU Memory Issues

**Problem:** Batched GA populations use lots of memory

**Solution:** 
- Monitor GPU memory usage
- Reduce population size or batch size if needed
- Add memory clearing between generations:

```python
# After selection in GA
torch.cuda.empty_cache()
```

---

### Pitfall 3: Behavioral Evaluation Slows Training

**Problem:** evaluate_progression_recurrent is expensive

**Solution:**
- Make it truly optional (controlled by flag)
- Run less frequently (every 5-10 minutes)
- Use smaller episode sample for quick validation

```python
if track_progression and (elapsed - last_eval) > eval_interval:
    # Run behavioral evaluation
    pass
```

---

### Pitfall 4: Configuration Conflicts

**Problem:** YAML config and command-line args both provided

**Solution:** Clear precedence rules:

```python
# In runner.py
if args.config:
    # YAML takes full precedence
    config_dict = load_yaml(args.config)
    # Override ONLY explicitly provided command-line args
    if args.gpu is not None:
        config_dict["gpu_id"] = args.gpu
```

---

## Sequencing and Dependencies

### Critical Path

The following must be completed in order:

1. **Phase 1** (Foundation) → enables imports
2. **Phase 2** (Models) → required by optimizers
3. **Phase 3** (Data) → required by optimizers
4. **Phase 4** (Evaluation) → required by optimizers
5. **Phase 5** (Optimizers) → required by runner
6. **Phase 6** (Config) → required by runner
7. **Phase 7** (Runner) → ties everything together

Phases 8-12 can be done concurrently after Phase 7.

### Parallelizable Work

After Phase 7 is complete:
- Documentation (Phase 11) can be written
- Tests (Phase 9) can be developed
- CLI updates (Phase 10) can be made
- Archive preparation (Phase 10.2) can proceed

---

## Success Metrics

### Phase Completion Criteria

Each phase is considered complete when:

1. All files are created
2. Unit tests pass
3. Integration with previous phases works
4. Documentation is written

### Overall Project Success

The platform is ready when:

1. All 12 phases complete
2. Integration tests pass (platform matches exp 4)
3. End-to-end tests pass (all model/optimizer combos work)
4. CLI tools function correctly
5. Documentation is comprehensive
6. Experiments 1-4 are archived
7. Git history is clean and logical

---

## Timeline Estimate

**Assuming one person working focused:**

- Phase 1 (Foundation): 1 hour
- Phase 2 (Models): 3-4 hours (careful extraction)
- Phase 3 (Data): 2-3 hours
- Phase 4 (Evaluation): 2 hours
- Phase 5 (Optimizers): 6-8 hours (most complex)
- Phase 6 (Config): 1 hour
- Phase 7 (Runner): 3-4 hours
- Phase 8 (YAML): 1 hour
- Phase 9 (Testing): 4-6 hours
- Phase 10 (CLI/Archive): 2-3 hours
- Phase 11 (Docs): 2-3 hours
- Phase 12 (Validation): 2-4 hours

**Total: 30-40 hours (approximately 1 week of focused work)**

---

## Next Steps After Platform Completion

Once the platform is stable:

1. **Experiment 5:** Run comprehensive comparison across all datasets
2. **Add Mamba:** Implement state space models (`platform/models/mamba.py`)
3. **Add GAIL:** Adversarial imitation learning (`platform/methods/gail.py`)
4. **Add Transfer:** GA transfer learning (`platform/optimizers/genetic.py` extensions)
5. **Optimize Performance:** Profile and optimize hotspots
6. **Extend Datasets:** Add more environments
7. **Hyperparameter Tuning:** Systematic sweep over hyperparameters

---

## Critical Files for Implementation

### Highest Priority (Core Functionality)

1. `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/platform/models/recurrent.py`
   - **Why:** Contains RecurrentMLPReservoir and RecurrentMLPTrainable (primary models for current experiments)
   - **Source:** Extract from experiments/4_add_recurrence/src/models.py lines 15-213
   - **Testing:** Must verify hidden state propagation and parameter counts

2. `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/platform/optimizers/sgd.py`
   - **Why:** SGD optimizer is critical path for validation against exp 4
   - **Source:** Extract from experiments/4_add_recurrence/src/optim.py lines 150-453
   - **Testing:** Must match exp 4 training curves

3. `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/platform/optimizers/genetic.py`
   - **Why:** GA optimizer with batched populations (most complex component)
   - **Source:** Combine neuroevolve_recurrent (lines 456-703) and neuroevolve_dynamic (lines 706-836)
   - **Testing:** Must verify GPU-parallel evaluation works correctly

4. `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/platform/data/loaders.py`
   - **Why:** Data loading must preserve episode boundaries and CL features
   - **Source:** Extract from experiments/4_add_recurrence/src/data.py lines 124-359
   - **Testing:** Must verify optim/test split matches exp 4

5. `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/platform/runner.py`
   - **Why:** Main entry point that orchestrates everything
   - **Source:** Structure from experiments/4_add_recurrence/main.py
   - **Testing:** End-to-end execution must work

---

## Final Checklist Before Archiving Experiments

- [ ] All unit tests pass
- [ ] Integration tests confirm platform matches exp 4
- [ ] CLI tools work with platform runner
- [ ] Documentation is complete and accurate
- [ ] Example configs run successfully
- [ ] Sweep configuration tested
- [ ] Git commits are logical and well-documented
- [ ] Experiments 1-4 archived with README
- [ ] Team reviewed and approved migration
- [ ] Backup of experiments directory made

---

## Appendix: File Mapping Reference

### From Experiment 4 to Platform

| Exp 4 File | Platform Destination | Lines | Notes |
|-----------|---------------------|-------|-------|
| `src/models.py` | `models/recurrent.py` | 15-213 | RecurrentMLP classes |
| `src/models.py` | `models/dynamic.py` | 607-923 | DynamicNetPopulation |
| `src/models.py` | `optimizers/genetic.py` | 216-605 | BatchedRecurrentPopulation |
| `src/optim.py` | `optimizers/sgd.py` | 150-453 | deeplearn_recurrent → optimize_sgd |
| `src/optim.py` | `optimizers/genetic.py` | 456-703 | neuroevolve_recurrent → optimize_ga |
| `src/optim.py` | `optimizers/genetic.py` | 706-836 | neuroevolve_dynamic → optimize_ga |
| `src/optim.py` | `optimizers/base.py` | 124-148 | create_episode_list |
| `src/optim.py` | `evaluation/comparison.py` | 27-122 | evaluate_progression_recurrent |
| `src/data.py` | `data/loaders.py` | 124-359 | load_human_data |
| `src/data.py` | `data/preprocessing.py` | 16-122 | CL features computation |
| `src/data.py` | `data/preprocessing.py` | 362-429 | EpisodeDataset |
| `src/config.py` | `config.py` | All | Merge configurations |
| `main.py` | `runner.py` | Structure | Adapt main flow |

### From Experiment 3 to Platform

| Exp 3 File | Platform Destination | Lines | Notes |
|-----------|---------------------|-------|-------|
| `src/models.py` | `models/feedforward.py` | 12-36 | MLP class |
| `src/models.py` | `optimizers/genetic.py` | 39-200 | BatchedPopulation (feedforward) |

### From Experiment 2 to Platform

| Exp 2 File | Platform Destination | Lines | Notes |
|-----------|---------------------|-------|-------|
| `main.py` | `data/loaders.py` | HF loaders | load_huggingface_data |

---

**End of Implementation Plan**

This plan should be saved to `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/PLATFORM_IMPLEMENTATION_PLAN.md` for reference during implementation.
