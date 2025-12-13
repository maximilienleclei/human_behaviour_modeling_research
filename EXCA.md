# exca Integration Guide

**A comprehensive guide to using Meta's exca library for experiment orchestration in human behavior modeling research**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Integration Architecture](#2-integration-architecture)
3. [Configuration Management](#3-configuration-management)
4. [Concrete Integration Examples](#4-concrete-integration-examples)
5. [Slurm Integration Guide](#5-slurm-integration-guide)
6. [Migration Guide](#6-migration-guide)
7. [Best Practices and Patterns](#7-best-practices-and-patterns)

---

## 1. Overview

### 1.1 What is exca?

**exca** (Execute and Cache) is a Python library developed by Meta FAIR (Facebook AI Research) that provides seamless execution and caching capabilities for Python functions. It's designed to simplify ML pipeline workflows by eliminating boilerplate code for:

- **Job submission** to compute clusters (Slurm via submitit)
- **Result caching** to disk and/or RAM
- **Configuration management** with hierarchical configs and automatic UID generation

The library is built on top of `pydantic` for type-safe configuration and `submitit` for cluster execution.

**Source**: https://github.com/facebookresearch/exca

**Version**: 0.5.7+

### 1.2 The Problem exca Solves

In ML research, running a simple experiment often requires cumbersome overhead:

```python
# WITHOUT exca - Manual boilerplate
import pickle
from pathlib import Path
import submitit

def train_model(config):
    # 1. Check if result already exists
    cache_file = Path(f"results/{config_hash}.pkl")
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # 2. Submit to cluster
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(timeout_min=720, gpus_per_node=1)
    job = executor.submit(actual_training_function, config)
    result = job.result()

    # 3. Cache result
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

    return result
```

**WITH exca - Clean and automatic:**

```python
import exca
import pydantic

class TrainingTask(pydantic.BaseModel):
    learning_rate: float = 0.001
    infra: exca.TaskInfra = exca.TaskInfra()

    @infra.apply
    def train(self) -> dict:
        return actual_training_function(self.learning_rate)

# Usage
task = TrainingTask(learning_rate=0.01, infra={"folder": "results", "cluster": "auto"})
result = task.train()  # Automatically handles caching, cluster submission, UID generation
```

### 1.3 Why Use exca for This Project?

This human behavior modeling research repository benefits from exca in several ways:

1. **Long-running experiments**: Neuroevolution (GA/ES/CMA-ES) can run for 10+ hours. Slurm integration enables easy cluster execution.

2. **Expensive computations**: Training neural networks to imitate human gameplay is computationally intensive. Caching prevents recomputation.

3. **Hyperparameter sweeps**: Comparing different model architectures, population sizes, and learning rates requires running many experiments. MapInfra enables parallel execution.

4. **Reproducibility**: UID-based organization ensures every experiment configuration has a unique identifier. No more "results_final_v3_FIXED.pt".

5. **Data preprocessing**: Loading and preprocessing human gameplay data is slow. Caching it once speeds up all subsequent experiments.

### 1.4 Installation

exca is already in `requirements.txt`:

```bash
pip install exca>=0.5.7
```

**Dependencies** (automatically installed):
- `pydantic>=2.5.0` - Type-safe configuration
- `submitit>=1.5.1` - Slurm cluster integration
- `numpy>=1.19` - Array operations
- `pyyaml>=6.0` - YAML serialization
- `orjson` - Fast JSON for cache metadata
- `cloudpickle` - Robust pickling

### 1.5 Core Concepts

#### TaskInfra - Single Task Execution

For methods that take **no parameters** (except `self`). The entire task is defined by the configuration.

```python
class MyTask(pydantic.BaseModel):
    param: int = 12
    infra: exca.TaskInfra = exca.TaskInfra()

    @infra.apply
    def process(self) -> float:
        return self.param * np.random.rand()
```

**Key features**:
- Modes: `"cached"`, `"retry"`, `"force"`, `"read-only"`
- `job()` method for submission
- `status()` for checking progress
- `job_array()` context for batch submission

#### MapInfra - Batch Processing

For methods that process **sequences/iterables** with per-item caching.

```python
class DataProcessor(pydantic.BaseModel):
    infra: exca.MapInfra = exca.MapInfra()

    @infra.apply(item_uid=str)
    def process_items(self, items: list[str]) -> Iterator[dict]:
        for item in items:
            yield process_single_item(item)
```

**Key features**:
- Per-item caching with customizable `item_uid`
- Parallel execution: `max_jobs`, `min_samples_per_job`
- Multiple backends: `None`, `"auto"`, `"slurm"`, `"local"`, `"threadpool"`, `"processpool"`

#### ConfDict - Hierarchical Configuration

Dictionary that splits keys on `"."` for nested configs:

```python
ConfDict({"training.optim.lr": 0.01})
# Becomes: {"training": {"optim": {"lr": 0.01}}}
```

**Key features**:
- `to_yaml()` / `from_yaml()` for serialization
- `to_uid()` for unique identifier generation
- `from_model()` to convert pydantic models

#### CacheDict - Dual-Layer Caching

Disk + RAM caching with JSONL metadata:

```python
cache = CacheDict(folder="cache", keep_in_ram=True)
cache["key1"] = np.array([1, 2, 3])  # Stored to disk + RAM
value = cache["key1"]  # Loads from RAM (fast!)
```

**Supported data types**:
- `NumpyArray` / `NumpyMemmapArray` - numpy arrays
- `TorchTensor` - PyTorch tensors
- `PandasDataFrame` / `ParquetPandasDataFrame` - pandas DataFrames
- `MemmapArrayFile` - Multiple arrays in single memmap (efficient for many small arrays)
- `Pickle` - Fallback for any Python object

#### UID System

Every configuration gets a **unique identifier** generated from its parameters:

```python
config = {"dataset": "cartpole", "method": "ga_feedforward", "seed": 42}
uid = ConfDict(config).to_uid()
# Result: "dataset=cartpole,method=ga_feedforward,seed=42-a3f8b2c1"
#         └─────────────── human-readable ────────────┘ └── hash ──┘
```

This ensures:
- **No collisions**: Different configs get different UIDs
- **Reproducibility**: Same config always generates same UID
- **Organization**: Results organized by UID in cache folders

### 1.6 Comparison with Other Tools

| Feature | exca | lru_cache | hydra | submitit |
|---------|------|-----------|-------|----------|
| RAM caching | ✅ | ✅ | ❌ | ❌ |
| Disk caching | ✅ | ❌ | ❌ | ❌ |
| Hierarchical config | ✅ | ❌ | ✅ | ❌ |
| Slurm execution | ✅ | ❌ | ❌ | ✅ |
| Pure Python (no CLI) | ✅ | ✅ | ❌ | ✅ |
| UID generation | ✅ | ❌ | ❌ | ❌ |
| pydantic integration | ✅ | ❌ | ❌ | ❌ |

**Key advantages**:
- **Unified interface**: Config, execution, and caching in one place
- **Type safety**: Built on pydantic for validation
- **Transparent**: No magic, just decorators on pydantic models

---

## 2. Integration Architecture

### 2.1 Current Architecture

Before exca integration:

```
config/experiments.py (dataclass configs)
    ↓
ne/eval/supervised.py → train_supervised(population, data, optimizer="ga")
    ↓
ne/optim/ga.py → optimize_ga() → fitness evaluations → parameter updates
    ↓
results/checkpoints/  (manual checkpoint management)
```

**Limitations**:
- No automatic caching (manual checkpoint files)
- No UID-based organization (naming like "exp1_seed42.pt")
- No cluster execution infrastructure
- Config management via dataclasses (no validation)

### 2.2 New Architecture with exca

After integration:

```
experiments/configs.py (pydantic configs with infra)
    ↓
experiments/tasks/neuroevolution.py → GAExperiment.run()
    ↓
    ├─→ [exca checks cache by UID]
    ├─→ [exca submits to Slurm if cluster="auto"]
    ├─→ experiments/tasks/data.py → load_preprocessed() [CACHED]
    ├─→ ne/eval/supervised.py → train_supervised() [UNCHANGED]
    └─→ [exca caches result by UID]
    ↓
results/exca_cache/cartpole/ga_feedforward/
    └─ dataset=cartpole,method=ga_feedforward,seed=42-a3f8b2c1/
        ├─ config.yaml          (full config)
        ├─ uid.yaml            (UID config)
        ├─ result.pkl          (cached result)
        ├─ submitit/           (Slurm job logs - symlink)
        └─ code/              (code snapshot - symlink)
```

**Improvements**:
- ✅ Automatic caching with UID-based organization
- ✅ Slurm cluster execution
- ✅ Hierarchical config with validation
- ✅ Traditional checkpointing still works (for resuming mid-training)

### 2.3 File Organization

New `experiments/` directory structure:

```
experiments/
├── __init__.md              # Directory overview
├── __init__.py             # Exports
├── configs.py              # Pydantic experiment configs
│
├── tasks/                  # Task wrappers for training
│   ├── __init__.md
│   ├── __init__.py
│   ├── neuroevolution.py   # GA/ES/CMA-ES tasks
│   ├── deep_learning.py    # SGD tasks
│   └── data.py            # Data preprocessing with caching
│
├── sweeps/                 # Hyperparameter optimization
│   ├── __init__.md
│   ├── __init__.py
│   └── hyperparameters.py  # MapInfra for param sweeps
│
├── utils/                  # Helper utilities
│   ├── __init__.md
│   ├── __init__.py
│   ├── paths.py           # Cache folder management
│   └── batch_submit.py    # Job array utilities
│
└── scripts/                # Example usage scripts
    ├── run_ga_cartpole.py
    ├── run_sgd_cartpole.py
    ├── sweep_hidden_size.py
    └── compare_methods.py
```

### 2.4 Integration with Existing Code

**Key principle**: The `experiments/` directory is a **thin wrapper layer**.

**What stays the same** (NO modifications):
- `config/device.py` - Device management
- `config/paths.py` - Path definitions
- `config/state.py` - State persistence
- `data/` - All data loading code
- `dl/` - All deep learning code (models, optimizers)
- `ne/` - All neuroevolution code (networks, evaluation, optimizers)
- `metrics/` - All evaluation metrics

**What's new** (orchestration layer):
- `experiments/configs.py` - Pydantic versions of configs (extends `config/experiments.py`)
- `experiments/tasks/` - exca wrappers around existing training functions
- `experiments/sweeps/` - Hyperparameter sweep infrastructure
- `experiments/utils/` - Helper utilities

**Example**: Wrapping `train_supervised()`

```python
# OLD: Direct call (still works!)
from ne.eval.supervised import train_supervised

results = train_supervised(
    population=population,
    train_data=(train_obs, train_act),
    test_data=(test_obs, test_act),
    optimizer="ga",
    max_time=36000,
)

# NEW: With exca wrapper (adds caching + Slurm)
from experiments.tasks.neuroevolution import GAExperiment

exp = GAExperiment(dataset="cartpole", method="ga_feedforward", seed=42)
results = exp.run()  # Calls train_supervised() internally
```

---

## 3. Configuration Management

### 3.1 Converting from Dataclass to Pydantic

#### Current (config/experiments.py):

```python
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    hidden_size: int = 50

@dataclass
class ExperimentConfig:
    experiment_number: int
    dataset: str
    method: str
    model: ModelConfig = field(default_factory=ModelConfig)
```

#### New (experiments/configs.py):

```python
import pydantic
from typing import ClassVar
import exca

class ModelConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    hidden_size: int = 50

class ExperimentConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    # Experiment metadata
    experiment_number: int
    dataset: str  # cartpole, mountaincar, acrobot, lunarlander
    method: str   # ga_feedforward, es_recurrent, sgd_reservoir, etc.
    subject: str = "sub01"
    seed: int = 42

    # Sub-configs
    model: ModelConfig = pydantic.Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = pydantic.Field(default_factory=OptimizerConfig)
    training: TrainingConfig = pydantic.Field(default_factory=TrainingConfig)
    data: DataConfig = pydantic.Field(default_factory=DataConfig)

    # exca infrastructure
    infra: exca.TaskInfra = exca.TaskInfra(
        version="1",
        folder=None,  # Set at runtime
        cluster="auto",  # Auto-detect Slurm
        timeout_min=720,  # 12 hours
        slurm_partition="normal",
        cpus_per_task=4,
        gpus_per_node=1,
        mem_gb=16,
    )

    # Exclude from UID (doesn't affect computation)
    _exclude_from_cls_uid: ClassVar[tuple[str, ...]] = ("experiment_number",)
```

**Benefits of pydantic**:
- ✅ **Validation**: Catches type errors before execution
- ✅ **Defaults**: Clear default values
- ✅ **Serialization**: Easy JSON/YAML export
- ✅ **IDE support**: Better autocomplete and type checking

### 3.2 UID Management

#### Class UID vs Cache UID

exca has two types of UIDs:

1. **Class UID**: Identifies the experiment configuration
   - Controlled by `_exclude_from_cls_uid` class variable
   - Determines the cache folder name

2. **Cache UID**: Identifies what gets cached
   - Controlled by `exclude_from_cache_uid` in `@infra.apply()`
   - Can exclude parameters that don't affect the final result

**Example**:

```python
class GAExperiment(ExperimentConfig):
    # Class-level exclusions (from folder UID)
    _exclude_from_cls_uid: ClassVar[tuple[str, ...]] = (
        "experiment_number",  # Just for labeling, doesn't affect computation
        "infra",             # Infrastructure params auto-excluded by exca
    )

    # Method-level exclusions (from cache UID)
    @infra.apply(exclude_from_cache_uid=("training.eval_interval",))
    def run(self) -> dict:
        # eval_interval affects logging frequency but not final result
        return train_supervised(...)
```

**Result**:
- Experiments with different `experiment_number` share the same cache folder
- Experiments with different `eval_interval` share the same cached result
- Experiments with different `seed` get different cached results

#### UID Structure

```
dataset=cartpole,method=ga_feedforward,seed=42-a3f8b2c1
└────────────── readable params ──────────────┘ └─ hash ─┘
```

- **Readable part**: Key parameters in sorted order
- **Hash**: MD5 hash (8 chars) to avoid collisions

#### Controlling What's in the UID

```python
# Example: These two configs should share the same cache
config1 = GAExperiment(experiment_number=1, seed=42, dataset="cartpole")
config2 = GAExperiment(experiment_number=2, seed=42, dataset="cartpole")

# Because experiment_number is excluded from UID:
assert config1.infra.uid() == config2.infra.uid()
```

### 3.3 Version Control for Cache Invalidation

When computation logic changes, increment `version` to invalidate old cache:

```python
class GAExperiment(ExperimentConfig):
    infra: exca.TaskInfra = exca.TaskInfra(
        version="2",  # Changed from "1" due to bugfix in fitness evaluation
    )
```

**When to increment**:
- 🔴 **YES**: Changed the algorithm, fixed a bug, updated dependencies
- 🟢 **NO**: Added logging, changed checkpoint frequency, renamed variables

**Effect**: New version creates new UID, so old cached results are ignored.

### 3.4 YAML Config Loading

```python
# experiments/configs.py
import yaml
from pathlib import Path

def load_experiment_config(yaml_path: Path) -> GAExperiment:
    """Load experiment config from YAML file."""
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)
    return GAExperiment(**config_dict)
```

**Example YAML** (`experiments/configs/cartpole_ga.yaml`):

```yaml
dataset: cartpole
method: ga_feedforward
subject: sub01
seed: 42

model:
  hidden_size: 64

optimizer:
  population_size: 100
  adaptive_sigma_init: 0.01

training:
  max_optim_time: 36000  # 10 hours

infra:
  folder: /scratch/results/exca_cache
  cluster: auto
  slurm_partition: gpu
  gpus_per_node: 1
  timeout_min: 720
```

**Usage**:

```python
config = load_experiment_config("experiments/configs/cartpole_ga.yaml")
results = config.run()
```

### 3.5 Example Configs for Each Experiment Type

#### GA Experiment:

```python
GAExperiment(
    experiment_number=1,
    dataset="cartpole",
    method="ga_feedforward",
    subject="sub01",
    seed=42,
    model={"hidden_size": 64},
    optimizer={
        "population_size": 100,
        "adaptive_sigma_init": 0.01,
        "adaptive_sigma_noise": 0.001,
    },
    training={
        "max_optim_time": 36000,
        "loss_eval_interval_seconds": 60,
    },
    data={
        "use_cl_info": False,
        "holdout_pct": 0.1,
    },
    infra={
        "folder": "/scratch/results/exca_cache",
        "cluster": "auto",
    },
)
```

#### SGD Experiment:

```python
SGDExperiment(
    experiment_number=1,
    dataset="lunarlander",
    method="sgd_reservoir",
    seed=42,
    model={"hidden_size": 128},
    optimizer={"learning_rate": 0.001},
    training={
        "max_optim_time": 36000,
        "batch_size": 32,
    },
    infra={
        "folder": "/scratch/results/exca_cache",
        "cluster": "auto",
        "gpus_per_node": 1,
    },
)
```

---

## 4. Concrete Integration Examples

### 4.1 Example 1: GA Training with TaskInfra

**File**: `experiments/tasks/neuroevolution.py`

Full implementation of wrapping GA training:

```python
"""Neuroevolution experiment tasks wrapped with exca."""

from pathlib import Path
from typing import ClassVar
import pydantic
import exca
import torch

from config.device import DEVICE
from config.paths import DATA_DIR, RESULTS_DIR
from data.simexp_control_tasks.loaders import load_human_data
from ne.net.feedforward import BatchedFeedforward
from ne.pop.population import Population
from ne.eval.supervised import train_supervised
from experiments.configs import ExperimentConfig
from experiments.tasks.data import DataPreprocessingTask


class GAExperiment(ExperimentConfig):
    """Genetic Algorithm experiment with exca caching."""

    infra: exca.TaskInfra = exca.TaskInfra(
        version="1",
        folder=None,  # Set in __init__
        cluster="auto",  # Auto-detect Slurm
        timeout_min=720,  # 12 hours
        slurm_partition="normal",
        cpus_per_task=4,
        gpus_per_node=1,
        mem_gb=16,
    )

    # Exclude non-deterministic params from UID
    _exclude_from_cls_uid: ClassVar[tuple[str, ...]] = (
        "experiment_number",  # Just for labeling
    )

    def __init__(self, **data):
        super().__init__(**data)
        # Set cache folder based on dataset/method
        if self.infra.folder is None:
            cache_dir = RESULTS_DIR / "exca_cache" / self.dataset / self.method
            self.infra = self.infra.model_copy(update={"folder": cache_dir})

    @infra.apply(exclude_from_cache_uid=("training.eval_interval",))
    def run(self) -> dict:
        """Run complete GA experiment.

        Returns:
            dict with:
                - fitness_history: list[float]
                - test_loss_history: list[float]
                - final_generation: int
                - checkpoint_path: str
        """
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)

        # Load data (uses DataPreprocessingTask for caching)
        data_task = DataPreprocessingTask(
            dataset=self.dataset,
            subject=self.subject,
            use_cl_info=self.data.use_cl_info,
            holdout_pct=self.data.holdout_pct,
        )
        data = data_task.load_preprocessed()

        # Create network population
        population = self._create_population(
            input_size=data["train_obs"].shape[1],
            output_size=data["metadata"]["num_actions"],
        )

        # Setup checkpoint path (traditional checkpointing in addition to exca cache)
        checkpoint_path = self._get_checkpoint_path()

        # Run training using existing function (UNCHANGED)
        results = train_supervised(
            population=population,
            train_data=(data["train_obs"], data["train_act"]),
            test_data=(data["test_obs"], data["test_act"]),
            optimizer="ga",
            max_time=self.training.max_optim_time,
            eval_interval=self.training.loss_eval_interval_seconds,
            checkpoint_path=checkpoint_path,
            logger=None,
        )

        # Add checkpoint path to results
        results["checkpoint_path"] = str(checkpoint_path)
        return results

    def _create_population(self, input_size: int, output_size: int) -> Population:
        """Create batched feedforward population."""
        from ne.net.feedforward import BatchedFeedforward
        from ne.pop.population import Population

        nets = BatchedFeedforward(
            num_nets=self.optimizer.population_size,
            input_size=input_size,
            hidden_size=self.model.hidden_size,
            output_size=output_size,
            adaptive_sigma_init=self.optimizer.adaptive_sigma_init,
            adaptive_sigma_noise=self.optimizer.adaptive_sigma_noise,
            device=DEVICE,
        )
        return Population(nets)

    def _get_checkpoint_path(self) -> Path:
        """Get checkpoint path (separate from exca cache for resuming)."""
        ckpt_dir = RESULTS_DIR / "checkpoints" / self.dataset / self.method
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir / f"exp{self.experiment_number}_seed{self.seed}.pt"


class ESExperiment(GAExperiment):
    """Evolution Strategies experiment."""

    @infra.apply(exclude_from_cache_uid=("training.eval_interval",))
    def run(self) -> dict:
        # Same as GA but with optimizer="es"
        # [Implementation similar to GAExperiment.run() but calls train_supervised with optimizer="es"]
        pass


class CMAESExperiment(GAExperiment):
    """CMA-ES experiment."""

    @infra.apply(exclude_from_cache_uid=("training.eval_interval",))
    def run(self) -> dict:
        # Same as GA but with optimizer="cmaes"
        # [Implementation similar to GAExperiment.run() but calls train_supervised with optimizer="cmaes"]
        pass
```

**Usage**:

```python
# Create experiment
exp = GAExperiment(
    experiment_number=1,
    dataset="cartpole",
    method="ga_feedforward",
    seed=42,
    model={"hidden_size": 64},
    optimizer={"population_size": 100},
)

# First call: Runs on Slurm (if available), caches result
results = exp.run()
# Output: Submitting to cluster, running for 10 hours...

# Second call: Loads from cache instantly!
results = exp.run()
# Output: Loading from cache... Done in 0.1s!

# Access results
print(f"Best fitness: {min(results['fitness_history'])}")
print(f"Final test loss: {results['test_loss_history'][-1]}")
print(f"Checkpoint: {results['checkpoint_path']}")
```

### 4.2 Example 2: Hyperparameter Sweep with MapInfra

**File**: `experiments/sweeps/hyperparameters.py`

```python
"""Hyperparameter sweeps using MapInfra."""

from typing import Iterator, ClassVar
import pydantic
import exca
from pathlib import Path

from experiments.tasks.neuroevolution import GAExperiment


class HyperparameterSweep(pydantic.BaseModel):
    """Run multiple experiments with different hyperparameters in parallel."""

    base_config: GAExperiment
    param_name: str  # e.g., "model.hidden_size"
    param_values: list[int | float | str]

    infra: exca.MapInfra = exca.MapInfra(
        version="1",
        folder=None,  # Set in __init__
        cluster="auto",
        cpus_per_task=4,
        gpus_per_node=1,
        max_jobs=128,  # Run up to 128 jobs in parallel
        min_samples_per_job=1,  # Each job processes 1 config
        max_num_timeout=3,  # Retry failed jobs up to 3 times
    )

    _exclude_from_cls_uid: ClassVar[tuple[str, ...]] = ("infra",)

    def __init__(self, **data):
        super().__init__(**data)
        if self.infra.folder is None:
            cache_dir = Path("/scratch/results/exca_cache/sweeps")
            self.infra = self.infra.model_copy(update={"folder": cache_dir})

    @infra.apply(item_uid=str)
    def run_sweep(self, param_values: list) -> Iterator[dict]:
        """Run experiment for each parameter value.

        Args:
            param_values: List of parameter values to try

        Yields:
            Results dict for each parameter value
        """
        for value in param_values:
            # Create modified config
            config_dict = self.base_config.model_dump()

            # Set parameter value (supports nested params like "model.hidden_size")
            self._set_nested_param(config_dict, self.param_name, value)

            # Create and run experiment
            exp = type(self.base_config)(**config_dict)
            result = exp.run()  # Uses exca caching!

            # Add metadata
            result["param_name"] = self.param_name
            result["param_value"] = value

            yield result

    def _set_nested_param(self, config: dict, param_path: str, value):
        """Set nested parameter like 'model.hidden_size'."""
        parts = param_path.split(".")
        current = config
        for part in parts[:-1]:
            current = current[part]
        current[parts[-1]] = value
```

**Usage**:

```python
# Base experiment config
base = GAExperiment(
    experiment_number=1,
    dataset="cartpole",
    method="ga_feedforward",
    seed=42,
)

# Sweep hidden size
sweep = HyperparameterSweep(
    base_config=base,
    param_name="model.hidden_size",
    param_values=[32, 64, 128, 256],
)

# Run sweep (submits 4 jobs to Slurm in parallel)
results = list(sweep.run_sweep(sweep.param_values))

# Analyze results
for r in results:
    print(f"hidden_size={r['param_value']}: "
          f"best_fitness={min(r['fitness_history']):.4f}")
```

**Output**:
```
Submitted 4 jobs for sweeps into 4 jobs on cluster 'slurm' (eg: 123456)
Waiting for processing 4 samples for sweeps
Finished processing 4 samples for sweeps
hidden_size=32: best_fitness=0.1234
hidden_size=64: best_fitness=0.0987
hidden_size=128: best_fitness=0.0876
hidden_size=256: best_fitness=0.0923
```

### 4.3 Example 3: Data Preprocessing Caching

**File**: `experiments/tasks/data.py`

```python
"""Data preprocessing tasks with caching."""

from typing import ClassVar
import pydantic
import exca
from pathlib import Path

from config.paths import DATA_DIR
from data.simexp_control_tasks.loaders import load_human_data


class DataPreprocessingTask(pydantic.BaseModel):
    """Preprocess and cache human behavioral data."""

    dataset: str  # cartpole, mountaincar, acrobot, lunarlander
    subject: str = "sub01"
    use_cl_info: bool = False
    holdout_pct: float = 0.1

    infra: exca.TaskInfra = exca.TaskInfra(
        version="1",
        folder=None,  # Set in __init__
        cluster=None,  # Run locally (preprocessing is fast)
        keep_in_ram=True,  # Cache in RAM after first load
    )

    _exclude_from_cls_uid: ClassVar[tuple[str, ...]] = ("infra",)

    def __init__(self, **data):
        super().__init__(**data)
        if self.infra.folder is None:
            cache_dir = DATA_DIR / "preprocessed_cache"
            self.infra = self.infra.model_copy(update={"folder": cache_dir})

    @infra.apply()
    def load_preprocessed(self) -> dict:
        """Load and preprocess data with caching.

        Returns:
            dict with train_obs, train_act, test_obs, test_act, metadata
        """
        print(f"Loading {self.subject}'s {self.dataset} data...")

        train_obs, train_act, test_obs, test_act, metadata = load_human_data(
            env_name=self.dataset,
            use_cl_info=self.use_cl_info,
            subject=self.subject,
            holdout_pct=self.holdout_pct,
        )

        # Add input size to metadata for convenience
        metadata["input_size"] = train_obs.shape[1]
        metadata["num_actions"] = len(set(train_act.tolist()))

        return {
            "train_obs": train_obs,
            "train_act": train_act,
            "test_obs": test_obs,
            "test_act": test_act,
            "metadata": metadata,
        }
```

**Usage**:

```python
# First call: Loads from JSON, preprocesses, caches to disk
data_task = DataPreprocessingTask(
    dataset="cartpole",
    subject="sub01",
    use_cl_info=True,
)
data = data_task.load_preprocessed()
# Output: Loading sub01's cartpole data...
#         Loaded 150 episodes
#         ... (processing output)

# Second call: Loads from disk cache (fast!)
data = data_task.load_preprocessed()
# Output: [loads instantly from cache]

# Third call: Loads from RAM cache (even faster!)
data = data_task.load_preprocessed()
# Output: [instant - already in RAM]

# Access data
print(f"Train size: {len(data['train_obs'])}")
print(f"Test size: {len(data['test_obs'])}")
print(f"Input dim: {data['metadata']['input_size']}")
```

### 4.4 Example 4: SGD Training with exca

**File**: `experiments/tasks/deep_learning.py`

```python
"""Deep learning (SGD) experiment tasks."""

from typing import ClassVar
import pydantic
import exca
import torch

from config.device import DEVICE
from config.paths import RESULTS_DIR
from dl.models.recurrent import RecurrentMLPReservoir
from dl.optim.sgd import optimize_sgd
from experiments.configs import ExperimentConfig
from experiments.tasks.data import DataPreprocessingTask


class SGDExperiment(ExperimentConfig):
    """SGD training with recurrent networks."""

    infra: exca.TaskInfra = exca.TaskInfra(
        version="1",
        folder=None,
        cluster="auto",
        timeout_min=720,
        slurm_partition="gpu",
        cpus_per_task=8,
        gpus_per_node=1,
        mem_gb=32,
    )

    _exclude_from_cls_uid: ClassVar[tuple[str, ...]] = ("experiment_number",)

    def __init__(self, **data):
        super().__init__(**data)
        if self.infra.folder is None:
            cache_dir = RESULTS_DIR / "exca_cache" / self.dataset / self.method
            self.infra = self.infra.model_copy(update={"folder": cache_dir})

    @infra.apply(exclude_from_cache_uid=("training.eval_interval",))
    def run(self) -> dict:
        """Run SGD training.

        Returns:
            dict with loss_history, test_loss_history, checkpoint_path
        """
        torch.manual_seed(self.seed)

        # Load preprocessed data (uses caching!)
        data_task = DataPreprocessingTask(
            dataset=self.dataset,
            subject=self.subject,
            use_cl_info=self.data.use_cl_info,
            holdout_pct=self.data.holdout_pct,
        )
        data = data_task.load_preprocessed()

        # Create model
        model = self._create_model(
            input_size=data["metadata"]["input_size"],
            output_size=data["metadata"]["num_actions"],
        )

        # Setup checkpoint
        checkpoint_path = self._get_checkpoint_path()

        # Run SGD optimization (existing function)
        loss_history, test_loss_history = optimize_sgd(
            model=model,
            optim_obs=data["train_obs"],
            optim_act=data["train_act"],
            test_obs=data["test_obs"],
            test_act=data["test_act"],
            output_size=data["metadata"]["num_actions"],
            metadata=data["metadata"],
            checkpoint_path=checkpoint_path,
            max_optim_time=self.training.max_optim_time,
            batch_size=self.training.batch_size,
            learning_rate=self.optimizer.learning_rate,
            loss_eval_interval_seconds=self.training.loss_eval_interval_seconds,
            logger=None,
        )

        return {
            "loss_history": loss_history,
            "test_loss_history": test_loss_history,
            "checkpoint_path": str(checkpoint_path),
        }

    def _create_model(self, input_size: int, output_size: int):
        """Create recurrent model based on method."""
        if "reservoir" in self.method:
            from dl.models.recurrent import RecurrentMLPReservoir
            return RecurrentMLPReservoir(
                input_size=input_size,
                hidden_size=self.model.hidden_size,
                output_size=output_size,
            )
        elif "trainable" in self.method:
            from dl.models.recurrent import RecurrentMLPTrainable
            return RecurrentMLPTrainable(
                input_size=input_size,
                hidden_size=self.model.hidden_size,
                output_size=output_size,
            )
        else:
            from dl.models.feedforward import FeedforwardMLP
            return FeedforwardMLP(
                input_size=input_size,
                hidden_size=self.model.hidden_size,
                output_size=output_size,
            )

    def _get_checkpoint_path(self):
        """Get checkpoint path."""
        ckpt_dir = RESULTS_DIR / "checkpoints" / self.dataset / self.method
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir / f"exp{self.experiment_number}_seed{self.seed}.pt"
```

### 4.5 Example 5: Multiple Seeds on Slurm

**File**: `experiments/utils/batch_submit.py`

```python
"""Utilities for submitting batches of experiments."""

from typing import Iterator
import exca
from experiments.tasks.neuroevolution import GAExperiment


def run_multiple_seeds(
    base_config: GAExperiment,
    seeds: list[int],
) -> list[dict]:
    """Run same experiment with multiple seeds on Slurm.

    Uses exca's job_array context for efficient submission.

    Args:
        base_config: Base experiment configuration
        seeds: List of random seeds to try

    Returns:
        List of results dicts, one per seed
    """
    results = []

    # Create experiment configs with different seeds
    experiments = []
    for seed in seeds:
        config_dict = base_config.model_dump()
        config_dict["seed"] = seed
        exp = type(base_config)(**config_dict)
        experiments.append(exp)

    # Submit all jobs as a job array
    with base_config.infra.job_array() as array:
        for exp in experiments:
            array.append(exp)

    # Wait for all jobs to complete and collect results
    for exp in experiments:
        exp.infra.job().wait()
        result = exp.run()  # Load from cache
        results.append(result)

    return results
```

**Usage**:

```python
# Base config
base = GAExperiment(
    experiment_number=1,
    dataset="cartpole",
    method="ga_feedforward",
    model={"hidden_size": 64},
)

# Run with 10 different seeds
results = run_multiple_seeds(base, seeds=list(range(10)))

# Aggregate results
best_fitnesses = [min(r["fitness_history"]) for r in results]
print(f"Mean best fitness: {np.mean(best_fitnesses):.4f} ± {np.std(best_fitnesses):.4f}")
```

---

## 5. Slurm Integration Guide

### 5.1 Slurm Configuration

exca auto-detects Slurm via submitit. Configure cluster parameters in the `infra` field:

```python
infra: exca.TaskInfra = exca.TaskInfra(
    # Cluster selection
    cluster="auto",  # Auto-detect Slurm, fallback to local
    # Options: None (local), "auto", "slurm", "local", "debug", "threadpool", "processpool"

    # Resource requests
    cpus_per_task=4,       # Number of CPUs per job
    gpus_per_node=1,       # Number of GPUs per job
    mem_gb=16,            # Memory in GB

    # Slurm-specific
    slurm_partition="normal",  # Partition name (e.g., "gpu", "long", "debug")
    slurm_constraint="",       # Node constraints (e.g., "volta" for specific GPU)
    slurm_account="",         # Account name (if required by cluster)
    slurm_qos="",             # Quality of service

    # Timeouts
    timeout_min=720,          # Max job duration (12 hours)

    # Error handling
    max_num_timeout=3,        # Retry failed jobs 3 times
)
```

### 5.2 Monitoring Jobs

#### Check Job Status

```python
exp = GAExperiment(...)

# Get job object
job = exp.infra.job()

# Check status
status = exp.infra.status()
# Returns: "not submitted", "running", "completed", or "failed"

print(f"Status: {status}")

if status == "running":
    # Get job ID
    print(f"Job ID: {job.job_id}")

    # View logs
    print("=== STDOUT ===")
    print(job.stdout())

    print("=== STDERR ===")
    print(job.stderr())

    # Cancel job if needed
    # job.cancel()

elif status == "completed":
    # Load result
    result = exp.run()
    print(f"Result: {result}")

elif status == "failed":
    # Check error
    print(f"Job failed: {job.exception()}")
```

#### Wait for Completion

```python
# Submit job
exp = GAExperiment(...)
job = exp.infra.job()

# Wait for completion (blocking)
job.wait()

# Load result
result = exp.run()
```

### 5.3 Cache and Job Folders

exca organizes files automatically:

```
results/exca_cache/cartpole/ga_feedforward/
└─ dataset=cartpole,method=ga_feedforward,seed=42-a3f8b2c1/
    ├─ config.yaml          # Full configuration
    ├─ uid.yaml            # Minimal config (for UID generation)
    ├─ result.pkl          # Cached result
    ├─ job.pkl             # Pickled job object
    ├─ submitit/           # Slurm job files (symlink)
    │   ├─ 123456_0_log.out      # stdout
    │   ├─ 123456_0_log.err      # stderr
    │   └─ 123456_0_submitted.pkl
    └─ code/              # Working directory (symlink)
```

**Key files**:
- `config.yaml` - Full experiment configuration
- `uid.yaml` - Minimal config used for UID (excludes defaults and excluded fields)
- `result.pkl` - Cached result (loaded by `exp.run()`)
- `submitit/` - Slurm logs and metadata
- `code/` - Code snapshot (if `workdir` is configured)

### 5.4 Best Practices for Long-Running Jobs

#### 1. Use Appropriate Timeouts

Evolution can run for 10+ hours:

```python
infra={"timeout_min": 720}  # 12 hours
```

If timeout is too short, jobs will be killed before completion.

#### 2. Enable Retries

Handle transient cluster issues:

```python
infra={"max_num_timeout": 3}  # Retry failed jobs up to 3 times
```

#### 3. Use Checkpointing + exca Cache

Combine traditional checkpointing (for resuming) with exca caching (for final results):

```python
@infra.apply()
def run(self) -> dict:
    # Traditional checkpoint for resuming mid-training
    checkpoint_path = self._get_checkpoint_path()

    results = train_supervised(
        ...,
        checkpoint_path=checkpoint_path,  # Can resume from here
    )

    # exca caches final results
    return results
```

**Benefits**:
- **Traditional checkpoint**: Resume interrupted training
- **exca cache**: Skip completed experiments entirely

#### 4. Monitor with `squeue`

```bash
# View your jobs
squeue -u $USER

# View specific job
squeue -j 123456

# Cancel job
scancel 123456
```

Or use exca's helper:

```python
from exca.helpers import find_slurm_job

job = find_slurm_job(job_id="123456", folder="/scratch/results/exca_cache")
print(job.stdout())
```

#### 5. Check Cache Before Resubmitting

Avoid duplicate submissions:

```python
exp = GAExperiment(...)

# Check if already completed
if exp.infra.status() == "completed":
    print("Already completed, loading from cache...")
    result = exp.run()
else:
    print("Submitting to cluster...")
    result = exp.run()
```

#### 6. Use Job Arrays for Sweeps

More efficient than individual jobs:

```python
# Bad: Submit 100 individual jobs
for seed in range(100):
    exp = GAExperiment(seed=seed)
    exp.run()  # 100 separate jobs

# Good: Use job array
with base_exp.infra.job_array() as array:
    for seed in range(100):
        exp = GAExperiment(seed=seed)
        array.append(exp)
# Single job array with 100 tasks
```

### 5.5 Troubleshooting

**Job fails immediately**:
- Check `job.stderr()` for errors
- Verify resource requests (memory, GPUs)
- Check partition availability

**Job hangs**:
- Check if waiting for I/O
- Verify GPU is actually used (`nvidia-smi`)
- Check logs for deadlocks

**Cache not working**:
- Verify `folder` parameter is set
- Check UID generation (different configs should have different UIDs)
- Look for file permission issues

---

## 6. Migration Guide

### 6.1 Backward Compatibility

exca integration is **fully backward compatible**. Old code continues to work:

```python
# OLD WAY (still works!)
from ne.eval.supervised import train_supervised
from ne.net.feedforward import BatchedFeedforward
from ne.pop.population import Population

# Manual setup
nets = BatchedFeedforward(...)
population = Population(nets)

# Direct call
results = train_supervised(
    population=population,
    train_data=train_data,
    test_data=test_data,
    optimizer="ga",
)
```

```python
# NEW WAY (with exca benefits)
from experiments.tasks.neuroevolution import GAExperiment

exp = GAExperiment(dataset="cartpole", method="ga_feedforward")
results = exp.run()  # Calls train_supervised() internally
```

**Both work!** Use the new way for new experiments, keep old scripts as-is.

### 6.2 Migration Path

**Phase 1: Add Infrastructure** (Week 1)
- Install exca
- Create `experiments/` directory structure
- Add pydantic configs
- **Keep old code unchanged**

**Phase 2: Wrap Existing Experiments** (Week 2)
- Create TaskInfra wrappers for GA, ES, CMA-ES, SGD
- Test on small experiments
- Compare results with old method (should be identical)

**Phase 3: Adopt for New Experiments** (Ongoing)
- Use exca for all new experiments
- Gradually migrate old experiment scripts when revisiting them
- Build up library of configs in `experiments/configs/`

### 6.3 When to Use exca vs Direct Calls

**Use exca for**:
- ✅ Production experiments (need caching, reproducibility)
- ✅ Slurm cluster execution
- ✅ Hyperparameter sweeps
- ✅ Long-running experiments (hours to days)
- ✅ Experiments you might want to re-run with same config

**Use direct calls for**:
- ✅ Quick debugging
- ✅ Interactive development in notebooks
- ✅ One-off tests
- ✅ Prototyping new algorithms

**Example workflow**:
```python
# Phase 1: Prototype with direct calls
def test_new_fitness_function():
    population = create_test_population()
    results = train_supervised(population, data, optimizer="ga", max_time=60)
    assert results["fitness_history"][-1] < 0.1

# Phase 2: Once working, wrap with exca
class NewFitnessExperiment(GAExperiment):
    @infra.apply()
    def run(self) -> dict:
        # Production version with caching
        ...
```

---

## 7. Best Practices and Patterns

### 7.1 UID Management

#### DO: Exclude Non-Deterministic Parameters

```python
# Good
_exclude_from_cls_uid: ClassVar[tuple[str, ...]] = (
    "experiment_number",  # Just for labeling
    "result_dir",        # Doesn't affect computation
)
```

```python
# Bad - Don't exclude parameters that affect results!
_exclude_from_cls_uid: ClassVar[tuple[str, ...]] = (
    "seed",              # ❌ WRONG! Seed affects results
    "learning_rate",     # ❌ WRONG! LR affects results
)
```

#### DO: Use exclude_from_cache_uid for Logging Parameters

```python
# Good - eval_interval affects logging but not final result
@infra.apply(exclude_from_cache_uid=("training.eval_interval",))
def run(self) -> dict:
    return train_supervised(..., eval_interval=self.training.eval_interval)
```

```python
# Bad - max_time affects when training stops!
@infra.apply(exclude_from_cache_uid=("training.max_time",))  # ❌ WRONG
def run(self) -> dict:
    return train_supervised(..., max_time=self.training.max_time)
```

### 7.2 Versioning

Increment `version` when **computation changes**:

```python
# Version 1: Original implementation
infra: exca.TaskInfra = exca.TaskInfra(version="1")

# Later: Fixed bug in fitness calculation
infra: exca.TaskInfra = exca.TaskInfra(version="2")  # New cache!
```

**When to increment**:
- 🔴 Changed algorithm
- 🔴 Fixed bug that affects results
- 🔴 Updated dependency versions
- 🔴 Changed data preprocessing
- 🟢 Added logging
- 🟢 Renamed variables
- 🟢 Refactored code (no logic change)

### 7.3 Folder Organization

Organize cache folders hierarchically:

```python
# Good
cache_dir = RESULTS_DIR / "exca_cache" / self.dataset / self.method

# Result:
# results/exca_cache/
#   ├─ cartpole/
#   │   ├─ ga_feedforward/
#   │   ├─ es_recurrent/
#   │   └─ sgd_reservoir/
#   ├─ lunarlander/
#   │   ├─ ga_feedforward/
#   │   └─ ...
```

### 7.4 Combining exca Cache with Traditional Checkpoints

**Pattern**: Use both for different purposes

```python
@infra.apply()
def run(self) -> dict:
    # 1. Traditional checkpoint (for resuming interrupted training)
    checkpoint_path = RESULTS_DIR / "checkpoints" / f"exp{self.id}.pt"

    # 2. Run training (can resume from checkpoint)
    results = train_supervised(
        ...,
        checkpoint_path=checkpoint_path,
    )

    # 3. exca caches FINAL results (skip completed experiments entirely)
    return results
```

**Benefits**:
- **Traditional checkpoint**: Resume if job times out mid-training
- **exca cache**: Skip if already completed successfully

### 7.5 Data Preprocessing Pattern

**Always cache expensive preprocessing**:

```python
# Bad: Reload every time
class Experiment(pydantic.BaseModel):
    @infra.apply()
    def run(self) -> dict:
        data = load_human_data(...)  # Slow! Reloads from JSON every time
        results = train(data)
        return results

# Good: Cache preprocessing
class Experiment(pydantic.BaseModel):
    @infra.apply()
    def run(self) -> dict:
        data_task = DataPreprocessingTask(...)
        data = data_task.load_preprocessed()  # Cached! Fast after first load
        results = train(data)
        return results
```

### 7.6 Error Handling

Let exca handle retries:

```python
# Good: Let exca retry
infra: exca.TaskInfra = exca.TaskInfra(
    max_num_timeout=3,  # Retry 3 times
)

@infra.apply()
def run(self) -> dict:
    # Don't wrap in try/except
    # Let errors propagate to exca
    return train_supervised(...)
```

```python
# Bad: Swallow errors
@infra.apply()
def run(self) -> dict:
    try:
        return train_supervised(...)
    except Exception:
        return {"error": True}  # ❌ exca will cache this error!
```

### 7.7 Memory Management

For large results, avoid keeping everything in RAM:

```python
# Bad: Keeps all arrays in RAM
infra: exca.TaskInfra = exca.TaskInfra(keep_in_ram=True)  # ❌ For large data

# Good: Use RAM cache only for small, frequently accessed data
class DataPreprocessingTask(pydantic.BaseModel):
    infra: exca.TaskInfra = exca.TaskInfra(
        keep_in_ram=True,  # ✅ Data is small and reused often
    )
```

### 7.8 Testing Configs Before Long Runs

Test with short timeout first:

```python
# Development: Short timeout for testing
exp = GAExperiment(
    ...,
    training={"max_optim_time": 60},  # 1 minute
    infra={"cluster": None},          # Local
)
exp.run()  # Quick test

# Production: Long timeout
exp = GAExperiment(
    ...,
    training={"max_optim_time": 36000},  # 10 hours
    infra={"cluster": "auto"},           # Slurm
)
exp.run()  # Real run
```

---

## Conclusion

This guide covered the complete integration of exca into the human behavior modeling research codebase. Key takeaways:

1. **exca provides**: Automatic caching, Slurm execution, hierarchical configs
2. **Integration is thin**: Existing training code (ne/, dl/, data/) unchanged
3. **experiments/ directory**: New orchestration layer with pydantic configs
4. **Backward compatible**: Old scripts still work
5. **Best practices**: Proper UID management, versioning, error handling

**Next steps**:
1. Review examples in `experiments/scripts/`
2. Try running a simple experiment
3. Create your own configs in `experiments/configs/`
4. Build up a library of reusable experiment templates

**Resources**:
- exca GitHub: https://github.com/facebookresearch/exca
- exca docs: https://facebookresearch.github.io/exca/
- This repo: `experiments/` directory for examples

Happy experimenting! 🚀
