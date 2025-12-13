# exca Integration Plan

**Status**: In Progress
**Goal**: Extract exca patterns, document comprehensively, create experiment orchestration layer
**Date Started**: 2025-12-12
**Last Updated**: 2025-12-12

---

## Progress Tracker

### ✅ Completed Tasks

- [x] **Task 1**: Create EXCA.md Documentation (~2000 lines) - **DONE**
  - Created comprehensive 7-section guide
  - Includes all examples, best practices, and integration patterns
  - File: `EXCA.md`

### 📋 TODO Tasks

- [ ] **Task 2**: Create experiments/ Directory Structure
  - [ ] 2.1: Core Files
    - [ ] `experiments/__init__.md`
    - [ ] `experiments/__init__.py`
    - [ ] `experiments/configs.py`
  - [ ] 2.2: Task Wrappers
    - [ ] `experiments/tasks/__init__.md`
    - [ ] `experiments/tasks/__init__.py`
    - [ ] `experiments/tasks/neuroevolution.py` (GA/ES/CMA-ES)
    - [ ] `experiments/tasks/deep_learning.py` (SGD)
    - [ ] `experiments/tasks/data.py` (Data preprocessing)
  - [ ] 2.3: Hyperparameter Sweeps
    - [ ] `experiments/sweeps/__init__.md`
    - [ ] `experiments/sweeps/__init__.py`
    - [ ] `experiments/sweeps/hyperparameters.py`
  - [ ] 2.4: Utilities
    - [ ] `experiments/utils/__init__.md`
    - [ ] `experiments/utils/__init__.py`
    - [ ] `experiments/utils/paths.py`
    - [ ] `experiments/utils/batch_submit.py`
  - [ ] 2.5: Example Scripts
    - [ ] `experiments/scripts/run_ga_cartpole.py`
    - [ ] `experiments/scripts/run_sgd_cartpole.py`
    - [ ] `experiments/scripts/sweep_hidden_size.py`
    - [ ] `experiments/scripts/compare_methods.py`

- [ ] **Task 3**: Update Supporting Files
  - [ ] Update `.gitignore` (add exca cache directories)
  - [ ] Verify `requirements.txt` (exca>=0.5.7)

- [ ] **Task 4**: Delete exca/ Directory
  - [ ] Remove cloned exca directory after all documentation complete

### 📊 Summary

- **Total Tasks**: 4 major tasks, 22 files to create
- **Completed**: 1/4 major tasks (25%)
- **Files Created**: 1/22 files (EXCA.md)
- **Remaining**: 21 files across experiments/ directory + supporting files

---

## Overview

Integrate Meta's `exca` library into the human behavior modeling research repository to provide:
- Automatic result caching with UID-based organization
- Slurm cluster execution for long-running experiments
- Hierarchical configuration management via pydantic
- Professional experiment orchestration layer

The user cloned exca/ into the repo for reference. We will:
1. Create comprehensive EXCA.md documentation with all integration patterns
2. Create new experiments/ orchestration layer
3. Delete the exca/ directory (install via pip instead)

---

## Implementation Tasks

### Task 1: Create EXCA.md Documentation (~2000 lines)

**File**: `EXCA.md`

Create comprehensive documentation with 7 sections:

#### 1. Overview (~200 lines)
- What is exca? (Execute and Cache library from Meta FAIR)
- Why use it for this project?
- Installation: `pip install exca`
- Core concepts: TaskInfra, MapInfra, ConfDict, CacheDict, UID system
- Comparison with other tools (lru_cache, hydra, submitit)

#### 2. Integration Architecture (~300 lines)
- Current architecture diagram
- New architecture with exca
- File organization for experiments/ directory
- Integration with existing code (no modifications to dl/, ne/, data/)

#### 3. Configuration Management (~400 lines)
- Converting dataclasses to pydantic BaseModel
- UID management (_exclude_from_cls_uid, exclude_from_cache_uid)
- Version control for cache invalidation
- YAML config loading
- Example configs for each experiment type

#### 4. Concrete Integration Examples (~800 lines)

**Example 1: GA training with TaskInfra**
```python
class GAExperiment(ExperimentConfig):
    infra: exca.TaskInfra = exca.TaskInfra(...)

    @infra.apply(exclude_from_cache_uid=("training.eval_interval",))
    def run(self) -> dict:
        # Load data (cached)
        data_task = DataPreprocessingTask(...)
        data = data_task.load_preprocessed()

        # Create population
        population = self._create_population(...)

        # Call existing function
        results = train_supervised(population, train_data, test_data, ...)

        return results
```

**Example 2: Hyperparameter sweep with MapInfra**
```python
class HyperparameterSweep(pydantic.BaseModel):
    infra: exca.MapInfra = exca.MapInfra(...)

    @infra.apply(item_uid=str)
    def run_sweep(self, param_values: list) -> Iterator[dict]:
        for value in param_values:
            exp = GAExperiment(...)
            yield exp.run()
```

**Example 3: Data preprocessing caching**
```python
class DataPreprocessingTask(pydantic.BaseModel):
    infra: exca.TaskInfra = exca.TaskInfra(keep_in_ram=True)

    @infra.apply()
    def load_preprocessed(self) -> dict:
        return load_human_data(...)  # Cached!
```

**Example 4: SGD training**
**Example 5: Multiple seeds on Slurm**

#### 5. Slurm Integration Guide (~300 lines)
- Slurm configuration (partition, CPUs, GPUs, memory, timeout)
- Job monitoring (status, logs, job IDs)
- Cache and job folder organization
- Best practices for long-running jobs

#### 6. Migration Guide (~200 lines)
- Backward compatibility (old code still works)
- Migration path: Phase 1 (add infra) → Phase 2 (wrap) → Phase 3 (adopt)
- When to use exca vs direct calls

#### 7. Best Practices (~200 lines)
- UID management dos and don'ts
- Versioning for cache invalidation
- Folder organization strategies
- Combining exca cache with traditional checkpoints
- Data preprocessing patterns

---

### Task 2: Create experiments/ Directory Structure

```
experiments/
├── __init__.md              # Overview (~100 lines)
├── __init__.py             # Exports (~20 lines)
├── configs.py              # Pydantic configs (~200 lines)
├── tasks/
│   ├── __init__.md        # (~80 lines)
│   ├── __init__.py        # (~15 lines)
│   ├── neuroevolution.py  # GA/ES/CMA-ES wrappers (~300 lines)
│   ├── deep_learning.py   # SGD wrapper (~200 lines)
│   ├── data.py           # Data preprocessing cache (~100 lines)
│   └── evaluation.py     # Behavioral comparison (~150 lines)
├── sweeps/
│   ├── __init__.md       # (~60 lines)
│   ├── __init__.py       # (~10 lines)
│   └── hyperparameters.py # MapInfra sweeps (~150 lines)
├── utils/
│   ├── __init__.md       # (~40 lines)
│   ├── __init__.py       # (~10 lines)
│   ├── paths.py         # Cache folder management (~50 lines)
│   └── batch_submit.py  # Job array helpers (~80 lines)
└── scripts/
    ├── run_ga_cartpole.py      # Example (~50 lines)
    ├── run_sgd_cartpole.py     # Example (~50 lines)
    ├── sweep_hidden_size.py    # Example sweep (~80 lines)
    └── compare_methods.py      # Example comparison (~100 lines)
```

#### Key Files

**experiments/configs.py**
```python
class ExperimentConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    # Metadata
    experiment_number: int
    dataset: str
    method: str
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
        folder=None,
        cluster="auto",
        timeout_min=720,
        gpus_per_node=1,
    )

    _exclude_from_cls_uid: ClassVar[tuple[str, ...]] = ("experiment_number",)
```

**experiments/tasks/neuroevolution.py**
- GAExperiment - wraps `train_supervised(optimizer="ga")`
- ESExperiment - wraps `train_supervised(optimizer="es")`
- CMAESExperiment - wraps `train_supervised(optimizer="cmaes")`

**experiments/tasks/deep_learning.py**
- SGDExperiment - wraps `optimize_sgd()`

**experiments/tasks/data.py**
- DataPreprocessingTask - wraps `load_human_data()` with caching

**experiments/sweeps/hyperparameters.py**
- HyperparameterSweep - MapInfra for parallel sweeps

---

### Task 3: Update Supporting Files

**.gitignore**
```
# exca cache directories
results/exca_cache/
data/preprocessed_cache/

# exca temporary files
.exca_tmp/
```

**requirements.txt**
```
exca>=0.5.7
pydantic>=2.0.0
```

---

### Task 4: Delete exca/ Directory

After EXCA.md is complete:
```bash
rm -rf exca/
```

Rationale: exca should be installed via pip as external dependency.

---

## Critical Files Reference

### Existing Files (NO modifications):
1. `config/experiments.py` - Dataclass configs to convert to pydantic
2. `ne/eval/supervised.py` - train_supervised() to wrap
3. `dl/optim/sgd.py` - optimize_sgd() to wrap
4. `data/simexp_control_tasks/loaders.py` - load_human_data() to cache
5. `ne/net/feedforward.py` - BatchedFeedforward for population creation
6. `config/paths.py` - Path constants to use

### Files to Create (20 files):
1. `EXCA.md` - Comprehensive documentation (~2000 lines)
2-17. `experiments/` directory files (see Task 2 structure)
18-19. `.gitignore`, `requirements.txt` updates
20. N/A (deletion task)

---

## Key Design Principles

1. **Thin wrapper layer**: experiments/ only wraps existing code, doesn't duplicate logic
2. **Backward compatible**: Old code (direct calls to train_supervised, optimize_sgd) still works
3. **Clean separation**:
   - config/ - Shared config utilities (device, paths, state)
   - data/ - Data loading (unchanged)
   - dl/ - Deep learning training (unchanged)
   - ne/ - Neuroevolution training (unchanged)
   - experiments/ - NEW orchestration layer with exca
4. **Progressive adoption**: Can use exca for new experiments while keeping old scripts
5. **Documentation first**: EXCA.md provides complete guide for all integration patterns

---

## Implementation Order

1. **EXCA.md** (largest file, ~2000 lines) - Create comprehensive documentation first
2. **experiments/configs.py** - Foundation for all tasks
3. **experiments/tasks/data.py** - Data caching (used by other tasks)
4. **experiments/tasks/neuroevolution.py** - GA/ES/CMA-ES wrappers
5. **experiments/tasks/deep_learning.py** - SGD wrapper
6. **experiments/sweeps/hyperparameters.py** - Hyperparameter infrastructure
7. **experiments/utils/** - Helper utilities
8. **experiments/scripts/** - Example scripts
9. **__init__.md and __init__.py files** - Documentation and exports for each directory
10. **.gitignore** - Add cache directories
11. **Delete exca/** - Remove cloned directory

---

## Integration Flow Examples

### Current Flow (ne/eval/supervised.py):
```
user script
  → create BatchedFeedforward
  → create Population
  → call train_supervised(population, data, optimizer="ga")
    → create fitness evaluators
    → call optimize_ga()
      → optimize() base loop
```

### With exca:
```
user script
  → create GAExperiment config
  → call exp.run()  [decorated with @infra.apply]
    → [exca checks cache]
    → [exca submits to slurm if cluster="auto"]
    → create BatchedFeedforward
    → create Population
    → call train_supervised() [UNCHANGED]
    → [exca caches result]
```

---

## Example Usage

### Simple GA Experiment:
```python
from experiments.tasks.neuroevolution import GAExperiment

exp = GAExperiment(
    experiment_number=1,
    dataset="cartpole",
    method="ga_feedforward",
    seed=42,
    model={"hidden_size": 64},
    optimizer={"population_size": 100},
    training={"max_optim_time": 36000},
)

# First call: runs on Slurm if available, caches result
results = exp.run()

# Second call: loads from cache instantly!
results = exp.run()

print(f"Best fitness: {min(results['fitness_history'])}")
```

### Hyperparameter Sweep:
```python
from experiments.sweeps.hyperparameters import HyperparameterSweep

sweep = HyperparameterSweep(
    base_config=exp,
    param_name="model.hidden_size",
    param_values=[32, 64, 128, 256],
)

# Runs 4 experiments in parallel on Slurm
results = list(sweep.run_sweep(sweep.param_values))
```

---

## Expected Outcome

After implementation:
- ✅ EXCA.md provides complete integration guide
- ✅ experiments/ directory with professional orchestration layer
- ✅ Can run experiments with automatic caching: `GAExperiment(...).run()`
- ✅ Can submit to Slurm automatically: `infra={"cluster": "auto"}`
- ✅ Can sweep hyperparameters in parallel: `HyperparameterSweep(...).run_sweep(...)`
- ✅ Data preprocessing cached for fast reloading
- ✅ All existing code still works unchanged
- ✅ exca/ directory removed, installed via pip
- ✅ Clean, well-documented integration ready for research use

---

## Notes

- All .py files > 30 lines need corresponding .md files (per CLAUDE.md)
- Each directory needs __init__.md with overview
- Must use exact paths (watch for "maximilienleclei" typo)
- Preserve all existing checkpointing logic (exca adds layer on top)
- Use `config.DEVICE` consistently (imported from config/device.py)

---

## Quick Reference: exca Core Concepts

### TaskInfra
- For single tasks (methods with no params except self)
- Modes: "cached", "retry", "force", "read-only"
- `@infra.apply` decorator
- `job()` method for submission
- `status()` for checking ("not submitted", "running", "completed", "failed")
- `job_array()` context for batch submission

### MapInfra
- For batch processing (methods that take iterables)
- Per-item caching with `item_uid` function
- `max_jobs`, `min_samples_per_job` for parallelization
- Multiple backends: None, "auto", "slurm", "local", "threadpool", "processpool"

### ConfDict
- Hierarchical dict with "." key splitting
- `to_yaml()` / `from_yaml()` for serialization
- `to_uid()` for unique identifier generation
- `from_model()` to convert pydantic models

### CacheDict
- Dual-layer caching (disk + RAM)
- JSONL metadata files for key tracking
- Multiple cache types: NumpyArray, NumpyMemmapArray, TorchTensor, PandasDataFrame, etc.
- Thread-safe concurrent writes

### Helper Functions
- `to_config()` / `to_config_model()` - Convert functions to pydantic configs
- `with_infra()` - Decorator to add infra to functions
- `find_slurm_job()` - Retrieve job by ID from cache
- `DiscriminatedModel` - Type-preserving pydantic serialization
