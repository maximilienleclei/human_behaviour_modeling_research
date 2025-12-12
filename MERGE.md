# Repository Merge Analysis and Plan

**Date**: 2025-12-11
**Purpose**: Merge `ai_repo/` and `claude_repo/` into unified behavior cloning research platform

---

# Part 1: Understanding ai_repo/

**Location**: `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/ai_repo`

**Status**: ARCHIVED (as noted in README)

## Overview

Production-quality ML research framework designed to reduce boilerplate and streamline ML workflows. Supports two optimization paradigms:
- **Deep Learning (DL)**: PyTorch Lightning-based
- **Neuroevolution (NE)**: MPI-parallelized genetic algorithms

**Scale**: ~9,856 lines of Python across 112 files

## Directory Structure

```
ai_repo/
├── pyproject.toml              # Production config (204 lines, comprehensive)
├── LICENSE
├── README.md
├── renovate.json               # Automated dependency updates
│
├── docker/                     # Container support
│   ├── cpu/                    # CPU Docker configs
│   ├── cuda/                   # GPU Docker configs
│   └── rocm/                   # AMD GPU configs
│
├── docs/                       # Sphinx documentation
│
├── common/                     # Reusable ML engines (core library)
│   ├── config.py              # Base Hydra configs
│   ├── runner.py              # BaseTaskRunner (Hydra-Zen integration)
│   ├── store.py               # Storage utilities
│   │
│   ├── optim/                 # Optimization engines
│   │   ├── dl/               # Deep Learning engine
│   │   │   ├── train.py      # Lightning training loop
│   │   │   ├── datamodule/   # Lightning DataModule bases
│   │   │   ├── litmodule/    # Lightning Module bases
│   │   │   └── utils/        # Lightning utilities
│   │   │
│   │   └── ne/               # Neuroevolution engine (MPI-based)
│   │       ├── fit.py        # Main evolution loop (305 LOC)
│   │       ├── agent/        # Agent base classes
│   │       ├── space/        # Environment wrappers
│   │       ├── net/          # Neural network implementations
│   │       └── utils/        # NE utilities (MPI, evaluation, logging)
│   │
│   ├── infer/                # Inference engines
│   │   └── lightning/        # Lightning checkpoint inference
│   │
│   └── utils/                # Shared utilities
│       ├── beartype.py       # Dynamic type checking (119 LOC, well-documented)
│       ├── hydra_zen.py      # Hydra configuration helpers
│       ├── wandb.py          # W&B logging utilities
│       ├── torch.py          # PyTorch utilities (RunningStandardization)
│       ├── mpi4py.py         # MPI communication helpers
│       ├── misc.py           # Miscellaneous utilities
│       └── runner.py         # Task execution utilities
│
└── projects/                 # Application projects
    ├── classify_mnist/       # MNIST classification benchmark
    │   ├── train.py
    │   ├── infer.py
    │   ├── datamodule.py
    │   ├── litmodule.py
    │   └── task/             # Hydra task configs
    │
    ├── ne_control_score/     # Neuroevolution for RL control
    │   ├── __main__.py
    │   ├── agent.py
    │   ├── space.py
    │   └── task/
    │
    └── haptic_pred/          # Haptic track generation (most complex)
        ├── litmodule/
        ├── datamodule/
        ├── task/
        └── [extensive project structure]
```

## Key Technologies

### Configuration & Execution
- **hydra-core** (1.3.2): Configuration management
- **hydra-zen** (0.15.0): Structured configs
- **submitit** (forked): Local & SLURM job launching with MPI
- **hydra-submitit-launcher**: SLURM integration

### Deep Learning Stack
- **torch** (2.6.0): PyTorch
- **lightning** (2.5.2): Training framework
- **transformers** (4.53.3): HuggingFace models
- **diffusers** (0.34.0): Diffusion models
- **timm** (1.0.19): Image models
- **x-transformers** (2.4.14): Transformer utilities
- **mambapy** (forked): Mamba state-space models

### Neuroevolution Stack
- **mpi4py** (4.1.0): Inter-process communication
- **torchrl** (0.9.2): RL/IL tasks
- **gymnasium[mujoco]** (1.2.0): RL environments

### Type Safety & Quality
- **beartype** (0.21.0): Dynamic type checking
- **mypy** (forked): Static type checking
- **jaxtyping** (0.3.2): Tensor annotations
- **ruff** (0.12.5): Fast linting
- **black** (25.1.0): Formatting
- **pre-commit**: Git hooks

### Logging & Monitoring
- **wandb** (0.21.0): Experiment tracking

## Architecture Patterns

### 1. Service-Engine-Project-Task Hierarchy
- **Services**: Execution types (optim, infer, serve)
- **Engines**: Specific implementations (dl, ne)
- **Projects**: Domain-specific code
- **Tasks**: YAML configs for specific runs

### 2. Abstract Base Classes
- `BaseTaskRunner` → `DeepLearningTaskRunner` / `NeuroevolutionTaskRunner`
- `BaseLitModule` → `BaseClassificationLitModule` → Project-specific modules
- `BaseAgent` → `GymAgent`
- `BaseSpace` → `GymReinforcementSpace`

### 3. Configuration as Code
- Hydra-Zen for structured configs
- YAML overrides for task variations
- Config interpolation (e.g., `${config.env_transfer}`)

### 4. Parallel Execution
- MPI for neuroevolution (scatter/gather patterns)
- Multi-GPU support for deep learning
- SLURM integration for cluster computing

## CI/CD Infrastructure

- **GitHub Actions**: format-lint, on-push, on-pr workflows
- **Pre-commit hooks**: black, ruff, doc8, yamllint
- **Renovate**: Automated dependency updates
- **Docker/Apptainer**: Container-first development
- **Comprehensive linting**: Black (line-length 79), Ruff with extensive rules, strict mypy

## Key Strengths

✅ **Production-ready infrastructure**: CI/CD, Docker, comprehensive tooling
✅ **Flexible configuration**: Hydra + Hydra-Zen for complex configs
✅ **Type safety**: Dynamic (beartype) + static (mypy) checking
✅ **Scalability**: MPI for distributed computing, SLURM integration
✅ **Best practices**: Linting, formatting, pre-commit hooks

## Key Weaknesses for Our Use Case

❌ **MPI-based neuroevolution**: Complex, CPU-focused, not GPU-batched
❌ **Heavy dependencies**: Many libraries not needed for behavior cloning
❌ **Over-engineered for simple experiments**: Service-Engine-Project hierarchy is overkill

---

# Part 2: Understanding claude_repo/

**Location**: `/scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research/claude_repo`

**Status**: Active research codebase (7/12 phases complete, fully functional)

## Overview

Research platform for investigating behavioral cloning methods comparing gradient-based (SGD) vs evolutionary (GA/ES/CMA-ES) optimization. Central question: **How do these methods differ in learning human behavioral policies?**

**Scale**: 62 Python files, ~4,000+ LOC of platform code, 56 Markdown documentation files

## Directory Structure

```
claude_repo/
├── pyproject.toml              # Minimal (6 lines, black/isort only)
├── requirement.txt             # 9 core dependencies
├── LICENSE
├── README.md
├── PLATFORM_IMPLEMENTATION_PLAN.md
├── CLAUDE.md
│
├── platform/                   # Core research platform (HEART OF CODEBASE)
│   ├── config.py              # Environment configs, device management
│   ├── runner.py              # Main entry point (argparse CLI, 279 LOC)
│   │
│   ├── models/                # Neural network architectures
│   │   ├── feedforward.py    # Simple 2-layer MLP (~150 params)
│   │   ├── recurrent.py      # Reservoir & trainable RNNs (~450-550 params)
│   │   └── dynamic.py        # Evolving topology networks (experimental)
│   │
│   ├── optimizers/            # Training algorithms (~1,600 LOC total)
│   │   ├── base.py           # Shared utilities (checkpointing, episode processing)
│   │   ├── sgd.py            # Adam optimizer, episode-based batching
│   │   ├── genetic.py        # GA for recurrent networks (batched GPU evaluation)
│   │   └── evolution.py      # Simple GA, ES, CMA-ES for feedforward (~1,200 LOC)
│   │
│   ├── data/                  # Data loading & preprocessing
│   │   ├── loaders.py        # HuggingFace + local human data support
│   │   └── preprocessing.py  # Continual learning features, batching
│   │
│   └── evaluation/            # Metrics & analysis
│       ├── metrics.py        # Cross-entropy loss, macro F1 score
│       └── comparison.py     # Behavioral comparison in environments
│
├── data/                      # Human behavioral data (38+ MB)
│   ├── collect.py            # Interactive GUI for data collection
│   ├── data_analysis.py      # Session statistics
│   ├── replay.py             # Deterministic episode replay
│   └── [8 JSON files]        # sub01/sub02 × 4 environments
│       ├── sub01_data_cartpole_run1.json
│       ├── sub01_data_mountaincar_run1.json
│       ├── sub01_data_acrobot_run1.json
│       ├── sub01_data_lunarlander_run1.json
│       └── [4 more for sub02]
│
├── common/                    # Shared utilities
│   └── dynamic_net/          # Dynamic topology networks (UNIQUE RESEARCH)
│       ├── computation.py    # Batched forward passes (~400 LOC)
│       └── evolution.py      # Network topology mutations (~600 LOC)
│
├── experiments/               # Experiment management
│   ├── 1_dl_vs_ga_scaling_dataset_size_flops/
│   ├── 2_dl_vs_ga_es/        # Has results
│   ├── 3_cl_info_dl_vs_ga/   # Has extensive results (20+ files)
│   ├── 4_add_recurrence/     # Reference implementation
│   ├── 5_adversarial_imitation/  # Planned GAIL
│   │
│   ├── tracking/             # Custom SQLite experiment tracking
│   │   ├── database.py       # SQLite schema (16K LOC)
│   │   ├── logger.py         # ExperimentLogger context manager (4.6K LOC)
│   │   ├── query.py          # Result querying (10K LOC)
│   │   └── slurm_manager.py  # SLURM job management (11K LOC)
│   │
│   ├── cli/                  # Command-line tools
│   │   ├── submit_jobs.py    # Submit to SLURM
│   │   ├── monitor_jobs.py   # Job monitoring
│   │   └── query_results.py  # Result analysis
│   │
│   └── slurm/templates/      # SLURM job templates
│
├── utils/                     # Supporting utilities
│   └── beartype.py           # Custom type validators (58 LOC, basic)
│
└── docs/                     # Comprehensive documentation
    ├── README.md             # Documentation index
    ├── ARCHITECTURE.md       # System design
    ├── USAGE_GUIDE.md        # How to run experiments
    ├── NAVIGATION.md         # Code navigation
    └── STATUS.md             # Implementation status
```

## Key Technologies

### Core Dependencies
- **numpy**: Numerical operations
- **torch**: PyTorch (assumed installed, not in requirements.txt)
- **scikit-learn**: Metrics
- **matplotlib**: Visualization
- **datasets**: HuggingFace datasets
- **einops**: Tensor operations
- **jaxtyping**: Tensor shape annotations
- **beartype**: Runtime type checking
- **ordered_set**: Utilities
- **gymnasium[classic-control,box2d]**: RL environments

### Configuration
- **argparse**: Command-line interface
- Simple Python config dictionaries in `config.py`

## Key Innovations

### 1. GPU-Batched Population Evaluation ⭐

**Most important contribution**: All networks in a population evaluated in parallel on GPU, achieving ~10x speedup.

```python
# Instead of:
for agent in population:
    fitness = evaluate(agent)  # Sequential, slow

# Do:
batch_logits = population.forward_all(observations)  # Parallel on GPU
fitnesses = compute_all_losses(batch_logits, actions)  # Vectorized
```

Located in:
- `platform/optimizers/evolution.py`: `BatchedPopulation` class
- `platform/optimizers/genetic.py`: Recurrent batched evaluation

### 2. Dynamic Topology Networks ⭐

**Unique research**: Networks that grow/prune nodes and connections during evolution.

Located in:
- `common/dynamic_net/computation.py`: Forward pass through graph
- `common/dynamic_net/evolution.py`: Mutation operations

### 3. Human Behavioral Data ⭐

**Irreplaceable**: 38+ MB of actual human gameplay data from 2 subjects across 4 environments.

Format: JSON files with episodes containing:
- observations, actions, rewards, timestamps, seeds
- Ready for behavioral cloning

### 4. Continual Learning Features

Session/run IDs as additional inputs to model non-stationary behavior (humans learning over time).

Located in:
- `platform/data/preprocessing.py`: `compute_session_id()`, `compute_run_id()`

## Architecture Patterns

### 1. Factory Pattern
```python
model = create_model(model_type, input_size, hidden_size, output_size)
```

### 2. Strategy Pattern
Different optimizers share common interfaces but implement different strategies.

### 3. Context Manager
```python
with ExperimentLogger(...) as logger:
    # Automatic logging and error handling
```

### 4. Type-Safe Tensors
```python
from jaxtyping import Float, Int
def forward(obs: Float[Tensor, "BS obs_dim"]) -> Float[Tensor, "BS action_dim"]:
```

### 5. Configuration-Driven
```python
ENV_CONFIGS = {
    "cartpole": {"obs_dim": 4, "action_dim": 2, "env_name": "CartPole-v1"},
    ...
}
```

## Experiment Results

**Experiment 3** (continual learning) has **20+ result files** showing:
- SGD vs GA comparisons
- With/without continual learning ablations
- Multiple environments and subjects
- Training curves, checkpoints, metadata

**Validates**: The platform works end-to-end and produces research results.

## Key Strengths

✅ **GPU-batched neuroevolution**: Unique, fast, innovative
✅ **Simple, focused**: Easy to understand and modify
✅ **Complete**: All pieces work together
✅ **Well-documented**: Every file has .md documentation
✅ **Type-safe**: jaxtyping + beartype for runtime validation
✅ **Real data**: 38MB human behavioral data collected
✅ **Proven**: Experiment 3 shows it works

## Key Weaknesses

❌ **Argparse CLI**: Less flexible than Hydra for sweeps
❌ **Custom tracking**: Reinventing wheel vs using W&B
❌ **Custom SLURM**: Hydra submitit is more mature
❌ **Minimal dependencies**: Missing production tooling (linting, CI/CD, Docker)

---

# Part 3: Comparison & Merge Strategy

## What to Keep from Each Repo

### From ai_repo ✅
- Production infrastructure: Docker, CI/CD, pre-commit hooks
- Hydra + Hydra-Zen configuration system
- PyTorch Lightning framework for SGD
- Enhanced utilities (hydra_zen, torch_utils, wandb_utils)
- W&B integration
- SLURM integration via Hydra submitit
- Comprehensive pyproject.toml
- Type checking and linting setup

### From claude_repo ✅
- **GPU-batched neuroevolution** (CRITICAL - key innovation)
- All models (feedforward, recurrent, dynamic topology)
- **Human behavioral data** (38MB, irreplaceable)
- Data loading and preprocessing
- Evaluation metrics
- **Dynamic topology networks** (unique research)
- Experiment results (archive)

### To Discard ❌

From ai_repo:
- MPI-based neuroevolution (replaced by claude_repo's GPU version)
- Inference engine (not needed)
- All projects (classify_mnist, haptic_pred, ne_control_score)
- Sphinx docs (use Markdown)

From claude_repo:
- Argparse CLI (migrate to Hydra)
- Custom SQLite tracking (use W&B)
- Custom SLURM scripts (use Hydra submitit)
- CLI tools (replaced by Hydra)

## Key Conflicts & Resolutions

### 1. Directory Name: `common/`
**Conflict**: Both repos have `common/` with different purposes
- claude_repo: Dynamic topology networks
- ai_repo: Infrastructure library

**Resolution**: Rename claude_repo's `common/` → `networks/dynamic_topology/`

### 2. File: `utils/beartype.py`
**Conflict**: Nearly identical validators
- ai_repo: 119 LOC with docstrings
- claude_repo: 58 LOC, checks both strings and lists

**Resolution**: Use ai_repo version (better docs), add list support to `not_empty()`

### 3. Neuroevolution Implementations
**Conflict**: Two completely different approaches
- claude_repo: GPU-batched, simple, fast
- ai_repo: MPI-distributed, complex, CPU-focused

**Resolution**: Keep ONLY claude_repo's GPU-batched version (user preference)

### 4. Configuration Systems
**Conflict**: argparse vs Hydra
- claude_repo: Simple argparse CLI
- ai_repo: Hydra + Hydra-Zen

**Resolution**: Migrate to Hydra (user preference), provide example command mappings

### 5. Experiment Tracking
**Conflict**: SQLite vs W&B
- claude_repo: Custom SQLite database
- ai_repo: W&B integration

**Resolution**: Use W&B only (user preference: no optional features)

## User Preferences (from questions)

1. ✅ **Incremental implementation**: Complete each phase, validate, then proceed
2. ✅ **Keep original repos**: claude_repo/ and ai_repo/ remain as reference (.gitignored)
3. ✅ **Core features only**: No optional features (SQLite, argparse wrapper, extended tests)

---

# Part 4: Detailed Merge Plan

## Proposed Directory Structure

```
behavior_cloning_research/              # Merged repo root
├── pyproject.toml                      # From ai_repo (enhanced)
├── README.md                           # New unified README
├── .gitignore                          # Merged, ignore claude_repo/ and ai_repo/
├── .pre-commit-config.yaml             # From ai_repo
├── .devcontainer.json                  # From ai_repo
│
├── docker/                             # From ai_repo
│   ├── cpu/
│   ├── cuda/
│   └── rocm/
│
├── .github/workflows/                  # From ai_repo (updated)
│   ├── format-lint.yaml
│   ├── on-push.yaml
│   └── on-pr.yaml
│
├── docs/                               # Merged documentation
│   ├── README.md
│   ├── ARCHITECTURE.md                 # Updated from claude_repo
│   ├── USAGE_GUIDE.md                  # Rewritten for Hydra
│   └── MIGRATION.md                    # New: command mappings
│
├── src/behavior_cloning/               # Main package
│   ├── __init__.py
│   ├── config.py                       # Base Hydra configs
│   ├── runner.py                       # Base task runner
│   │
│   ├── models/                         # Neural architectures
│   │   ├── __init__.py
│   │   ├── feedforward.py              # From claude_repo
│   │   ├── recurrent.py                # From claude_repo
│   │   └── dynamic.py                  # From claude_repo
│   │
│   ├── data/                           # Data handling
│   │   ├── __init__.py
│   │   ├── datamodules.py              # NEW: Lightning DataModules
│   │   ├── loaders.py                  # From claude_repo
│   │   ├── preprocessing.py            # From claude_repo
│   │   └── human_behavior/             # Human data (38MB)
│   │       ├── sub01_data_cartpole_run1.json
│   │       ├── sub01_data_mountaincar_run1.json
│   │       ├── sub01_data_acrobot_run1.json
│   │       ├── sub01_data_lunarlander_run1.json
│   │       ├── sub02_data_cartpole_run1.json
│   │       ├── sub02_data_mountaincar_run1.json
│   │       ├── sub02_data_acrobot_run1.json
│   │       └── sub02_data_lunarlander_run1.json
│   │
│   ├── optimizers/                     # Training algorithms
│   │   ├── __init__.py
│   │   │
│   │   ├── dl/                         # Deep Learning (SGD)
│   │   │   ├── __init__.py
│   │   │   ├── litmodule.py            # NEW: Lightning wrapper (~200 LOC)
│   │   │   └── trainer.py              # NEW: Lightning Trainer setup (~100 LOC)
│   │   │
│   │   └── neuroevolution/             # Evolutionary methods (GPU-batched)
│   │       ├── __init__.py
│   │       ├── base.py                 # From claude_repo
│   │       ├── batched_population.py   # ⭐ KEY: GPU-parallel evaluation
│   │       ├── feedforward_ga.py       # Refactored from evolution.py
│   │       ├── feedforward_es.py       # Refactored from evolution.py
│   │       ├── feedforward_cmaes.py    # Refactored from evolution.py
│   │       ├── recurrent_ga.py         # Refactored from genetic.py
│   │       ├── recurrent_es.py         # Refactored from genetic.py
│   │       └── recurrent_cmaes.py      # Refactored from genetic.py
│   │
│   ├── networks/                       # Special research contributions
│   │   ├── __init__.py
│   │   ├── dynamic_topology/           # From claude_repo/common/dynamic_net
│   │   │   ├── __init__.py
│   │   │   ├── computation.py          # ~400 LOC
│   │   │   └── evolution.py            # ~600 LOC
│   │   └── README.md
│   │
│   ├── evaluation/                     # Metrics and comparison
│   │   ├── __init__.py
│   │   ├── metrics.py                  # From claude_repo
│   │   └── comparison.py               # From claude_repo
│   │
│   └── utils/                          # Shared utilities
│       ├── __init__.py
│       ├── beartype.py                 # From ai_repo (better docs)
│       ├── hydra_zen.py                # From ai_repo
│       ├── torch_utils.py              # From ai_repo (torch.py renamed)
│       ├── wandb_utils.py              # From ai_repo (wandb.py renamed)
│       ├── misc.py                     # From ai_repo
│       └── runner.py                   # From ai_repo
│
├── experiments/                        # Experiment definitions
│   ├── behavior_cloning/               # Main BC research
│   │   ├── __init__.py
│   │   ├── __main__.py                 # Entry point
│   │   ├── runner.py                   # ⭐ Hydra task runner (~200 LOC)
│   │   │
│   │   ├── configs/                    # ⭐ Hydra configuration
│   │   │   ├── config.yaml             # Base config (~100 lines)
│   │   │   │
│   │   │   ├── model/                  # Model configs
│   │   │   │   ├── mlp.yaml
│   │   │   │   ├── reservoir.yaml
│   │   │   │   ├── trainable.yaml
│   │   │   │   └── dynamic.yaml
│   │   │   │
│   │   │   ├── optimizer/              # Optimizer configs
│   │   │   │   ├── sgd.yaml
│   │   │   │   ├── ga.yaml
│   │   │   │   ├── es.yaml
│   │   │   │   └── cmaes.yaml
│   │   │   │
│   │   │   ├── dataset/                # Dataset configs
│   │   │   │   ├── cartpole.yaml
│   │   │   │   ├── mountaincar.yaml
│   │   │   │   ├── acrobot.yaml
│   │   │   │   └── lunarlander.yaml
│   │   │   │
│   │   │   ├── experiment/             # Pre-composed configs
│   │   │   │   ├── sgd_cartpole_test.yaml      (10 min test)
│   │   │   │   ├── sgd_cartpole_full.yaml      (1 hour)
│   │   │   │   ├── ga_lunarlander_full.yaml    (1 hour)
│   │   │   │   └── [more presets...]
│   │   │   │
│   │   │   └── hydra/                  # Hydra runtime configs
│   │   │       ├── local.yaml
│   │   │       └── slurm.yaml
│   │   │
│   │   ├── datamodules.py              # BC-specific DataModules
│   │   ├── litmodules.py               # BC-specific LitModules
│   │   └── models.py                   # BC-specific model instantiation
│   │
│   └── archive/                        # Archived experiments (read-only)
│       ├── experiment_1/               # From claude_repo
│       ├── experiment_2/
│       ├── experiment_3/
│       └── experiment_4/
│
├── scripts/                            # Utility scripts
│   ├── collect_data.py                 # From claude_repo
│   ├── analyze_data.py                 # From claude_repo
│   └── replay_behavior.py              # From claude_repo
│
├── results/                            # Experiment outputs (.gitignored)
│   ├── checkpoints/
│   ├── logs/
│   └── plots/
│
├── tests/                              # Test suite
│   ├── test_models.py
│   ├── test_data.py
│   └── test_optimizers.py
│
├── claude_repo/                        # Original repo (reference, .gitignored)
└── ai_repo/                            # Original repo (reference, .gitignored)
```

---

## Implementation Phases

### Phase 0: Preparation (1-2 hours)

**Goal**: Set up fresh repo structure without breaking existing repos

**Tasks**:
1. Create backup of both repos:
   ```bash
   tar -czf claude_repo_backup_$(date +%Y%m%d).tar.gz claude_repo/
   tar -czf ai_repo_backup_$(date +%Y%m%d).tar.gz ai_repo/
   ```

2. Create new branch:
   ```bash
   git checkout -b repo-merge
   ```

3. Create directory structure:
   ```bash
   mkdir -p src/behavior_cloning/{models,data,optimizers/{dl,neuroevolution},networks/dynamic_topology,evaluation,utils}
   mkdir -p experiments/behavior_cloning/{configs/{model,optimizer,dataset,experiment,hydra},archive}
   mkdir -p scripts tests docs docker/.github/workflows results/{checkpoints,logs,plots}
   ```

4. Copy infrastructure files:
   ```bash
   # From ai_repo
   cp ai_repo/.pre-commit-config.yaml .
   cp -r ai_repo/.github/ .
   cp -r ai_repo/docker/ .
   cp ai_repo/.devcontainer.json .
   ```

5. Create base `pyproject.toml`:
   - Use ai_repo's as base
   - Add claude_repo dependencies: `gymnasium[classic-control,box2d]`, `ordered_set`
   - Update package name to `behavior_cloning`
   - Remove unnecessary dependencies: `mpi4py`, deep learning libraries not needed

6. Create `.gitignore`:
   ```
   # Ignore original repos
   claude_repo/
   ai_repo/

   # Python
   __pycache__/
   *.py[cod]
   *$py.class
   *.so
   .Python
   build/
   develop-eggs/
   dist/
   downloads/
   eggs/
   .eggs/
   lib/
   lib64/
   parts/
   sdist/
   var/
   wheels/
   *.egg-info/
   .installed.cfg
   *.egg

   # Virtual environments
   venv/
   ENV/
   env/

   # IDEs
   .vscode/
   .idea/
   *.swp
   *.swo
   *~

   # Results
   results/
   checkpoints/
   logs/
   *.db

   # Hydra
   outputs/
   multirun/
   .hydra/

   # OS
   .DS_Store
   Thumbs.db
   ```

7. Initialize package:
   ```python
   # src/behavior_cloning/__init__.py
   """Behavior Cloning Research Platform

   A unified platform for comparing gradient-based and evolutionary
   optimization methods for learning human behavioral policies.
   """

   __version__ = "1.0.0-unified"
   ```

**Validation**:
- [ ] Directory structure created
- [ ] Can install package: `pip install -e .`
- [ ] Pre-commit hooks work: `pre-commit install && pre-commit run --all-files`

**Rollback**: `git branch -D repo-merge`

---

### Phase 1: Core Models & Data (4-6 hours)

**Goal**: Migrate models and data loading (foundation)

#### Step 1.1: Copy Models

```bash
# Copy model files
cp claude_repo/platform/models/feedforward.py src/behavior_cloning/models/
cp claude_repo/platform/models/recurrent.py src/behavior_cloning/models/
cp claude_repo/platform/models/dynamic.py src/behavior_cloning/models/
```

Create `src/behavior_cloning/models/__init__.py`:
```python
"""Neural network architectures for behavior cloning."""

from .feedforward import MLP
from .recurrent import RecurrentMLPReservoir, RecurrentMLPTrainable
from .dynamic import DynamicNetwork

__all__ = [
    "MLP",
    "RecurrentMLPReservoir",
    "RecurrentMLPTrainable",
    "DynamicNetwork",
]
```

**Minor edits needed**:
- Update import paths (if any relative imports)
- Ensure all models are pure `nn.Module` (no Lightning dependencies yet)

#### Step 1.2: Migrate Human Behavioral Data

```bash
# Copy data files (38MB)
cp claude_repo/data/sub*.json src/behavior_cloning/data/human_behavior/

# Verify integrity
cd claude_repo/data && md5sum sub*.json > /tmp/original_checksums.txt
cd ../../src/behavior_cloning/data/human_behavior && md5sum sub*.json > /tmp/new_checksums.txt
diff /tmp/original_checksums.txt /tmp/new_checksums.txt  # Should be empty!
```

#### Step 1.3: Copy Data Loading

```bash
cp claude_repo/platform/data/loaders.py src/behavior_cloning/data/
cp claude_repo/platform/data/preprocessing.py src/behavior_cloning/data/
```

**Edits needed in `loaders.py`**:
- Update path to human behavior data:
  ```python
  # OLD
  DATA_DIR = Path(__file__).parent.parent.parent / "data"

  # NEW
  DATA_DIR = Path(__file__).parent / "human_behavior"
  ```

#### Step 1.4: Copy Utilities from ai_repo

```bash
# Copy all utilities
cp ai_repo/common/utils/beartype.py src/behavior_cloning/utils/
cp ai_repo/common/utils/hydra_zen.py src/behavior_cloning/utils/
cp ai_repo/common/utils/torch.py src/behavior_cloning/utils/torch_utils.py
cp ai_repo/common/utils/wandb.py src/behavior_cloning/utils/wandb_utils.py
cp ai_repo/common/utils/misc.py src/behavior_cloning/utils/
cp ai_repo/common/utils/runner.py src/behavior_cloning/utils/

# Copy base runner and config
cp ai_repo/common/runner.py src/behavior_cloning/
cp ai_repo/common/config.py src/behavior_cloning/
```

Create `src/behavior_cloning/utils/__init__.py`:
```python
"""Shared utilities."""

from .beartype import *
from .hydra_zen import *
from .torch_utils import *
from .wandb_utils import *
from .misc import *
```

#### Step 1.5: Create Lightning DataModule

Create `src/behavior_cloning/data/datamodules.py`:
```python
"""Lightning DataModules for behavior cloning."""

from pathlib import Path
from typing import Optional, Literal
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from lightning.pytorch import LightningDataModule

from .loaders import load_human_data
from .preprocessing import preprocess_episodes, add_cl_info


class BehaviorCloningDataModule(LightningDataModule):
    """DataModule for human behavioral cloning.

    Wraps claude_repo's data loading functions with Lightning interface.
    """

    def __init__(
        self,
        dataset_name: Literal["cartpole", "mountaincar", "acrobot", "lunarlander"],
        subject: Literal["sub01", "sub02"],
        use_cl_info: bool = False,
        holdout_pct: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.subject = subject
        self.use_cl_info = use_cl_info
        self.holdout_pct = holdout_pct
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Load and split data."""
        # Load data using claude_repo's loader
        episodes = load_human_data(self.dataset_name, self.subject)

        # Preprocess
        obs, actions = preprocess_episodes(episodes)

        # Add continual learning info if requested
        if self.use_cl_info:
            obs = add_cl_info(obs, episodes)

        # Convert to tensors
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)

        # Create dataset
        full_dataset = TensorDataset(obs_tensor, actions_tensor)

        # Split train/val/test
        n_total = len(full_dataset)
        n_test = int(n_total * self.holdout_pct)
        n_val = int((n_total - n_test) * 0.1)  # 10% of training for validation
        n_train = n_total - n_test - n_val

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
```

Create `src/behavior_cloning/data/__init__.py`:
```python
"""Data loading and preprocessing."""

from .loaders import load_human_data
from .preprocessing import preprocess_episodes, add_cl_info
from .datamodules import BehaviorCloningDataModule

__all__ = [
    "load_human_data",
    "preprocess_episodes",
    "add_cl_info",
    "BehaviorCloningDataModule",
]
```

#### Step 1.6: Write Basic Tests

Create `tests/test_models.py`:
```python
"""Tests for neural network models."""

import torch
import pytest
from behavior_cloning.models import MLP, RecurrentMLPReservoir, RecurrentMLPTrainable


def test_mlp_forward():
    """Test MLP forward pass."""
    model = MLP(input_size=4, hidden_size=50, output_size=2)
    x = torch.randn(32, 4)
    out = model(x)
    assert out.shape == (32, 2)


def test_reservoir_forward():
    """Test reservoir RNN forward pass."""
    model = RecurrentMLPReservoir(input_size=4, hidden_size=50, output_size=2)
    x = torch.randn(32, 4)
    out = model(x)
    assert out.shape == (32, 2)


def test_trainable_recurrent_forward():
    """Test trainable RNN forward pass."""
    model = RecurrentMLPTrainable(input_size=4, hidden_size=50, output_size=2, rank=10)
    x = torch.randn(32, 4)
    out = model(x)
    assert out.shape == (32, 2)
```

Create `tests/test_data.py`:
```python
"""Tests for data loading."""

import pytest
from behavior_cloning.data import BehaviorCloningDataModule


def test_load_cartpole_data():
    """Test loading CartPole data."""
    dm = BehaviorCloningDataModule(
        dataset_name="cartpole",
        subject="sub01",
        batch_size=32,
    )
    dm.setup("fit")

    # Check dataloaders exist
    assert dm.train_dataloader() is not None
    assert dm.val_dataloader() is not None
    assert dm.test_dataloader() is not None

    # Check batch shapes
    batch = next(iter(dm.train_dataloader()))
    obs, actions = batch
    assert obs.shape[0] == 32  # batch size
    assert obs.shape[1] == 4   # CartPole observation dim
    assert actions.shape[0] == 32


def test_continual_learning_info():
    """Test loading data with CL info."""
    dm = BehaviorCloningDataModule(
        dataset_name="cartpole",
        subject="sub01",
        use_cl_info=True,
        batch_size=32,
    )
    dm.setup("fit")

    batch = next(iter(dm.train_dataloader()))
    obs, actions = batch
    # With CL info, observation should have 2 extra dimensions
    assert obs.shape[1] == 4 + 2  # obs_dim + session_id + run_id
```

#### Step 1.7: Commit

```bash
git add src/behavior_cloning/models/ src/behavior_cloning/data/ src/behavior_cloning/utils/ tests/
git commit -m "Phase 1: Add models, data loading, and basic tests

- Copy models (MLP, RecurrentMLP, DynamicNetwork) from claude_repo
- Migrate human behavioral data (38MB, 8 JSON files)
- Copy data loaders and preprocessing from claude_repo
- Copy utilities from ai_repo
- Create Lightning DataModule wrapper
- Add basic tests for models and data loading
"
```

**Validation**:
- [ ] Models import: `python -c "from behavior_cloning.models import MLP; print(MLP)"`
- [ ] Data loads: `python -c "from behavior_cloning.data import BehaviorCloningDataModule; dm = BehaviorCloningDataModule('cartpole', 'sub01'); dm.setup('fit'); print(len(dm.train_dataloader()))"`
- [ ] Tests pass: `pytest tests/test_models.py tests/test_data.py -v`
- [ ] Data integrity: Verify MD5 checksums match

**Rollback**: `git reset --hard HEAD~1`

---

### Phase 2: SGD Optimizer (Lightning) (6-8 hours)

**Goal**: Get SGD training working with PyTorch Lightning

#### Step 2.1: Create Lightning LitModule

Create `src/behavior_cloning/optimizers/dl/litmodule.py`:
```python
"""Lightning module for behavior cloning."""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics import Accuracy


class BehaviorCloningLitModule(LightningModule):
    """Lightning module for behavior cloning with any model.

    This wraps any PyTorch nn.Module (MLP, RecurrentMLP, DynamicNetwork)
    and provides the training/validation/test loop logic.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=model.output_size)
        self.val_acc = Accuracy(task="multiclass", num_classes=model.output_size)
        self.test_acc = Accuracy(task="multiclass", num_classes=model.output_size)

        # Save hyperparameters (for checkpointing)
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        """Forward pass through model."""
        return self.model(x)

    def step(self, batch, stage: str):
        """Shared step for train/val/test.

        Args:
            batch: (observations, actions)
            stage: "train", "val", or "test"
        """
        obs, actions = batch

        # Forward pass
        logits = self(obs)

        # Compute loss
        loss = F.cross_entropy(logits, actions)

        # Compute accuracy
        preds = logits.argmax(dim=-1)
        if stage == "train":
            acc = self.train_acc(preds, actions)
        elif stage == "val":
            acc = self.val_acc(preds, actions)
        else:  # test
            acc = self.test_acc(preds, actions)

        # Log metrics
        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/accuracy", acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        """Configure Adam optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
```

Create `src/behavior_cloning/optimizers/dl/__init__.py`:
```python
"""Deep learning optimizers (SGD via Lightning)."""

from .litmodule import BehaviorCloningLitModule

__all__ = ["BehaviorCloningLitModule"]
```

#### Step 2.2: Create Trainer Wrapper

Create `src/behavior_cloning/optimizers/dl/trainer.py`:
```python
"""Lightning Trainer setup for behavior cloning."""

from typing import Optional
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


def create_trainer(
    max_epochs: int,
    output_dir: Path,
    wandb_project: str = "behavior-cloning-research",
    wandb_name: Optional[str] = None,
    enable_wandb: bool = True,
    early_stopping_patience: int = 50,
    device: str = "cuda:0",
):
    """Create Lightning Trainer with logging and callbacks.

    Args:
        max_epochs: Maximum number of training epochs
        output_dir: Directory for checkpoints and logs
        wandb_project: W&B project name
        wandb_name: W&B run name
        enable_wandb: Whether to enable W&B logging
        early_stopping_patience: Patience for early stopping
        device: Device to train on ("cuda:0", "cpu", etc.)

    Returns:
        Configured Lightning Trainer
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = None
    if enable_wandb:
        logger = WandbLogger(
            project=wandb_project,
            name=wandb_name,
            save_dir=output_dir,
        )

    # Callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
        # Save last model
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="last",
            monitor=None,
        ),
        # Early stopping
        EarlyStopping(
            monitor="val/loss",
            patience=early_stopping_patience,
            mode="min",
        ),
    ]

    # Create trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu" if "cuda" in device else "cpu",
        devices=1,
        deterministic=True,
    )

    return trainer
```

#### Step 2.3: Create Hydra Configs

Create `experiments/behavior_cloning/configs/config.yaml`:
```yaml
defaults:
  - model: mlp
  - optimizer: sgd
  - dataset: cartpole
  - hydra: local
  - _self_

# Global settings
seed: 42
device: "cuda:0"
max_optim_time: 36000  # 10 hours
output_dir: "./results/${now:%Y-%m-%d}/${now:%H-%M-%S}"

# W&B logging
wandb:
  project: "behavior-cloning-research"
  entity: null  # Set via env var or override
  enabled: true
```

Create `experiments/behavior_cloning/configs/model/mlp.yaml`:
```yaml
# @package _global_
model:
  type: "mlp"
  hidden_size: 50
  activation: "tanh"
```

Create `experiments/behavior_cloning/configs/model/reservoir.yaml`:
```yaml
# @package _global_
model:
  type: "reservoir"
  hidden_size: 50
  activation: "tanh"
```

Create `experiments/behavior_cloning/configs/model/trainable.yaml`:
```yaml
# @package _global_
model:
  type: "trainable"
  hidden_size: 50
  rank: 10
  activation: "tanh"
```

Create `experiments/behavior_cloning/configs/optimizer/sgd.yaml`:
```yaml
# @package _global_
optimizer:
  type: "sgd"
  learning_rate: 1e-3
  weight_decay: 0.0
  batch_size: 32
  max_epochs: 1000
  early_stopping_patience: 50
```

Create `experiments/behavior_cloning/configs/dataset/cartpole.yaml`:
```yaml
# @package _global_
dataset:
  name: "cartpole"
  subject: "sub01"
  use_cl_info: false
  holdout_pct: 0.1
  input_size: 4
  output_size: 2
```

Create `experiments/behavior_cloning/configs/dataset/lunarlander.yaml`:
```yaml
# @package _global_
dataset:
  name: "lunarlander"
  subject: "sub01"
  use_cl_info: false
  holdout_pct: 0.1
  input_size: 8
  output_size: 4
```

Create `experiments/behavior_cloning/configs/hydra/local.yaml`:
```yaml
# @package hydra
run:
  dir: ${output_dir}

job:
  chdir: true
```

Create `experiments/behavior_cloning/configs/experiment/sgd_cartpole_test.yaml`:
```yaml
# Quick 10-minute test run
defaults:
  - /model: mlp
  - /optimizer: sgd
  - /dataset: cartpole
  - /hydra: local
  - _self_

max_optim_time: 600  # 10 minutes
seed: 42

optimizer:
  batch_size: 32
  learning_rate: 1e-3
  max_epochs: 100

dataset:
  subject: "sub01"
  use_cl_info: false
```

#### Step 2.4: Create Runner

Create `experiments/behavior_cloning/runner.py`:
```python
"""Main runner for behavior cloning experiments."""

from pathlib import Path
import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import store, zen

from behavior_cloning.runner import BaseTaskRunner
from behavior_cloning.models import MLP, RecurrentMLPReservoir, RecurrentMLPTrainable
from behavior_cloning.data import BehaviorCloningDataModule
from behavior_cloning.optimizers.dl import BehaviorCloningLitModule
from behavior_cloning.optimizers.dl.trainer import create_trainer


class BehaviorCloningRunner(BaseTaskRunner):
    """Runner for behavior cloning experiments."""

    @classmethod
    def store_configs(cls, store: ConfigStore):
        """Register Hydra configs (called automatically)."""
        # Configs are in YAML files, so nothing to register programmatically
        pass

    @staticmethod
    def run_subtask(cfg):
        """Run a single experiment based on Hydra config."""
        # Set seed
        torch.manual_seed(cfg.seed)

        # Create DataModule
        datamodule = BehaviorCloningDataModule(
            dataset_name=cfg.dataset.name,
            subject=cfg.dataset.subject,
            use_cl_info=cfg.dataset.use_cl_info,
            holdout_pct=cfg.dataset.holdout_pct,
            batch_size=cfg.optimizer.batch_size,
        )

        # Create model
        if cfg.model.type == "mlp":
            model = MLP(
                input_size=cfg.dataset.input_size + (2 if cfg.dataset.use_cl_info else 0),
                hidden_size=cfg.model.hidden_size,
                output_size=cfg.dataset.output_size,
            )
        elif cfg.model.type == "reservoir":
            model = RecurrentMLPReservoir(
                input_size=cfg.dataset.input_size + (2 if cfg.dataset.use_cl_info else 0),
                hidden_size=cfg.model.hidden_size,
                output_size=cfg.dataset.output_size,
            )
        elif cfg.model.type == "trainable":
            model = RecurrentMLPTrainable(
                input_size=cfg.dataset.input_size + (2 if cfg.dataset.use_cl_info else 0),
                hidden_size=cfg.model.hidden_size,
                output_size=cfg.dataset.output_size,
                rank=cfg.model.rank,
            )
        else:
            raise ValueError(f"Unknown model type: {cfg.model.type}")

        # Check if using SGD or neuroevolution
        if cfg.optimizer.type == "sgd":
            # SGD path: Use Lightning
            litmodule = BehaviorCloningLitModule(
                model=model,
                learning_rate=cfg.optimizer.learning_rate,
                weight_decay=cfg.optimizer.weight_decay,
            )

            trainer = create_trainer(
                max_epochs=cfg.optimizer.max_epochs,
                output_dir=Path(cfg.output_dir),
                wandb_project=cfg.wandb.project,
                wandb_name=f"{cfg.dataset.name}_{cfg.optimizer.type}_{cfg.model.type}",
                enable_wandb=cfg.wandb.enabled,
                early_stopping_patience=cfg.optimizer.early_stopping_patience,
                device=cfg.device,
            )

            # Train
            trainer.fit(litmodule, datamodule)

            # Test
            results = trainer.test(litmodule, datamodule)
            print(f"Test results: {results}")

        elif cfg.optimizer.type in ["ga", "es", "cmaes"]:
            # Neuroevolution path (to be implemented in Phase 3)
            raise NotImplementedError("Neuroevolution not yet implemented")

        else:
            raise ValueError(f"Unknown optimizer type: {cfg.optimizer.type}")


if __name__ == "__main__":
    BehaviorCloningRunner.run_task()
```

Create `experiments/behavior_cloning/__main__.py`:
```python
"""Entry point for behavior cloning experiments."""

from .runner import BehaviorCloningRunner

if __name__ == "__main__":
    BehaviorCloningRunner.run_task()
```

Create `experiments/behavior_cloning/__init__.py`:
```python
"""Behavior cloning experiments."""
```

#### Step 2.5: Test SGD Training

```bash
# Quick 10-minute test
python -m experiments.behavior_cloning \
  --config-name=experiment/sgd_cartpole_test

# Or with CLI overrides
python -m experiments.behavior_cloning \
  model=mlp \
  optimizer=sgd \
  dataset=cartpole \
  max_optim_time=600
```

#### Step 2.6: Commit

```bash
git add src/behavior_cloning/optimizers/dl/ experiments/behavior_cloning/
git commit -m "Phase 2: Add SGD optimizer with Lightning

- Create BehaviorCloningLitModule (wraps any model)
- Create Lightning Trainer setup with W&B logging
- Create Hydra configs (model, optimizer, dataset, experiment)
- Create BehaviorCloningRunner (Hydra-based entry point)
- Test SGD training on CartPole
"
```

**Validation**:
- [ ] Can run SGD training: `python -m experiments.behavior_cloning --config-name=experiment/sgd_cartpole_test`
- [ ] W&B logging works (check dashboard at wandb.ai)
- [ ] Checkpoints saved: `ls results/*/checkpoints/`
- [ ] Test accuracy >80% on CartPole (comparable to claude_repo baseline)
- [ ] Can resume from checkpoint

**Rollback**: `git reset --hard HEAD~1`

---

### Phase 3: Neuroevolution Optimizers (8-12 hours)

**Goal**: Migrate GPU-batched GA, ES, CMA-ES from claude_repo

#### Step 3.1: Copy Base Files

```bash
cp claude_repo/platform/optimizers/base.py src/behavior_cloning/optimizers/neuroevolution/
```

#### Step 3.2: Extract BatchedPopulation Class

The `BatchedPopulation` class is the key innovation - it enables GPU-parallel evaluation of entire populations.

From `claude_repo/platform/optimizers/evolution.py`, extract the core batching logic into:

`src/behavior_cloning/optimizers/neuroevolution/batched_population.py`:
```python
"""GPU-batched population evaluation for neuroevolution.

This is the KEY INNOVATION from claude_repo: instead of evaluating each
network in a population sequentially, we stack all parameters and do a
single batched forward pass on the GPU.

Performance: ~10x speedup over sequential evaluation.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from jaxtyping import Float


class BatchedPopulation:
    """Population with GPU-batched forward pass.

    All networks in the population share the same architecture but have
    different parameters. We stack all parameters into batched tensors
    and do a single forward pass for the entire population.
    """

    def __init__(
        self,
        population_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: str = "cuda:0",
    ):
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device

        # Initialize population parameters
        # Shape: (population_size, num_params)
        self.params = self._initialize_params()

    def _initialize_params(self) -> torch.Tensor:
        """Initialize population with random parameters."""
        # Calculate total number of parameters
        num_params = (
            self.input_size * self.hidden_size + self.hidden_size +  # Layer 1
            self.hidden_size * self.output_size + self.output_size   # Layer 2
        )

        # Xavier initialization for each network
        params = []
        for _ in range(self.population_size):
            p = self._xavier_init(num_params)
            params.append(p)

        return torch.stack(params).to(self.device)

    def _xavier_init(self, num_params: int) -> torch.Tensor:
        """Xavier initialization for one network."""
        # Simplified - actual implementation should respect layer boundaries
        scale = (2.0 / (self.input_size + self.output_size)) ** 0.5
        return torch.randn(num_params) * scale

    def forward_all(
        self,
        obs: Float[torch.Tensor, "batch obs_dim"]
    ) -> Float[torch.Tensor, "pop_size batch action_dim"]:
        """Forward pass for ALL networks in population (batched on GPU).

        Args:
            obs: Observations, shape (batch_size, input_size)

        Returns:
            Logits for all networks, shape (population_size, batch_size, output_size)
        """
        batch_size = obs.shape[0]

        # Expand observations for all networks: (pop_size, batch_size, input_size)
        obs_expanded = obs.unsqueeze(0).expand(self.population_size, -1, -1)

        # Unpack parameters for each layer
        w1, b1, w2, b2 = self._unpack_params()

        # Layer 1: (pop_size, batch_size, hidden_size)
        h = torch.tanh(torch.bmm(obs_expanded, w1) + b1.unsqueeze(1))

        # Layer 2: (pop_size, batch_size, output_size)
        logits = torch.bmm(h, w2) + b2.unsqueeze(1)

        return logits

    def _unpack_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack flat parameters into weight matrices and biases."""
        # Calculate indices
        w1_size = self.input_size * self.hidden_size
        b1_size = self.hidden_size
        w2_size = self.hidden_size * self.output_size
        b2_size = self.output_size

        idx = 0

        # Layer 1 weights: (pop_size, input_size, hidden_size)
        w1 = self.params[:, idx:idx+w1_size].view(
            self.population_size, self.input_size, self.hidden_size
        )
        idx += w1_size

        # Layer 1 biases: (pop_size, hidden_size)
        b1 = self.params[:, idx:idx+b1_size]
        idx += b1_size

        # Layer 2 weights: (pop_size, hidden_size, output_size)
        w2 = self.params[:, idx:idx+w2_size].view(
            self.population_size, self.hidden_size, self.output_size
        )
        idx += w2_size

        # Layer 2 biases: (pop_size, output_size)
        b2 = self.params[:, idx:idx+b2_size]

        return w1, b1, w2, b2

    def evaluate_population(
        self,
        obs: Float[torch.Tensor, "n_episodes episode_len obs_dim"],
        actions: Float[torch.Tensor, "n_episodes episode_len"],
    ) -> Float[torch.Tensor, "pop_size"]:
        """Evaluate fitness for all networks (batched).

        Args:
            obs: Observations from episodes
            actions: True actions from episodes

        Returns:
            Fitness (negative loss) for each network, shape (population_size,)
        """
        # Flatten episodes: (n_episodes * episode_len, obs_dim)
        obs_flat = obs.view(-1, obs.shape[-1])
        actions_flat = actions.view(-1)

        # Forward pass for all networks: (pop_size, n_steps, action_dim)
        logits = self.forward_all(obs_flat)

        # Compute cross-entropy loss for each network
        losses = []
        for i in range(self.population_size):
            loss = nn.functional.cross_entropy(logits[i], actions_flat)
            losses.append(loss)

        losses = torch.stack(losses)

        # Fitness is negative loss (higher is better)
        fitnesses = -losses

        return fitnesses

    def get_params(self, idx: int) -> torch.Tensor:
        """Get parameters for network at index."""
        return self.params[idx]

    def set_params(self, idx: int, params: torch.Tensor):
        """Set parameters for network at index."""
        self.params[idx] = params
```

This is a simplified version - the actual implementation in claude_repo is more sophisticated, but this captures the key idea.

#### Step 3.3: Implement Feedforward GA/ES/CMA-ES

Refactor `claude_repo/platform/optimizers/evolution.py` into 3 files:

`src/behavior_cloning/optimizers/neuroevolution/feedforward_ga.py`:
```python
"""Simple Genetic Algorithm for feedforward networks."""

import torch
from pathlib import Path
from typing import Tuple, List
import wandb

from .batched_population import BatchedPopulation


def optimize_ga(
    input_size: int,
    hidden_size: int,
    output_size: int,
    population_size: int,
    observations: torch.Tensor,
    actions: torch.Tensor,
    max_generations: int,
    mutation_sigma: float = 0.1,
    elitism_pct: float = 0.1,
    device: str = "cuda:0",
    log_to_wandb: bool = True,
) -> Tuple[torch.Tensor, List[float]]:
    """Train with Simple GA using GPU-batched evaluation.

    Args:
        input_size: Input dimension
        hidden_size: Hidden layer size
        output_size: Output dimension
        population_size: Number of individuals
        observations: Training data observations
        actions: Training data actions
        max_generations: Number of generations
        mutation_sigma: Mutation standard deviation
        elitism_pct: Percentage of elite individuals to keep
        device: Device to train on
        log_to_wandb: Whether to log to W&B

    Returns:
        best_params: Parameters of best individual
        fitness_history: Fitness curve over generations
    """
    # Create population
    population = BatchedPopulation(
        population_size=population_size,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        device=device,
    )

    fitness_history = []
    n_elite = int(population_size * elitism_pct)

    for generation in range(max_generations):
        # Evaluate population (GPU-batched!)
        fitnesses = population.evaluate_population(observations, actions)

        # Track best
        best_fitness = fitnesses.max().item()
        fitness_history.append(best_fitness)

        if log_to_wandb:
            wandb.log({
                "generation": generation,
                "fitness/best": best_fitness,
                "fitness/mean": fitnesses.mean().item(),
                "fitness/std": fitnesses.std().item(),
            })

        print(f"Generation {generation}: best fitness = {best_fitness:.4f}")

        # Selection and mutation
        # Sort by fitness
        sorted_indices = torch.argsort(fitnesses, descending=True)

        # Keep elite
        elite_params = population.params[sorted_indices[:n_elite]].clone()

        # Generate offspring by mutating elite
        offspring = []
        for i in range(population_size - n_elite):
            # Select random elite parent
            parent_idx = torch.randint(0, n_elite, (1,)).item()
            parent = elite_params[parent_idx]

            # Mutate
            noise = torch.randn_like(parent) * mutation_sigma
            child = parent + noise
            offspring.append(child)

        # Replace non-elite with offspring
        population.params[sorted_indices[n_elite:]] = torch.stack(offspring)

    # Return best individual
    final_fitnesses = population.evaluate_population(observations, actions)
    best_idx = torch.argmax(final_fitnesses)
    best_params = population.get_params(best_idx)

    return best_params, fitness_history
```

Similarly, create:
- `feedforward_es.py` (Natural Evolution Strategies)
- `feedforward_cmaes.py` (CMA-ES)

And for recurrent networks (refactored from `genetic.py`):
- `recurrent_ga.py`
- `recurrent_es.py`
- `recurrent_cmaes.py`

(These follow the same pattern but handle recurrent network evaluation)

#### Step 3.4: Integrate with Runner

Update `experiments/behavior_cloning/runner.py` to handle neuroevolution:

```python
# In run_subtask method, add neuroevolution branch:

elif cfg.optimizer.type in ["ga", "es", "cmaes"]:
    # Neuroevolution path
    from behavior_cloning.optimizers.neuroevolution import train_neuroevolution

    # Prepare data (no DataLoader, just tensors)
    from behavior_cloning.data import load_human_data, preprocess_episodes
    episodes = load_human_data(cfg.dataset.name, cfg.dataset.subject)
    obs, actions = preprocess_episodes(episodes)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(cfg.device)
    actions_tensor = torch.tensor(actions, dtype=torch.long).to(cfg.device)

    # Run neuroevolution
    best_params, fitness_history = train_neuroevolution(
        algorithm=cfg.optimizer.type,
        model_type=cfg.model.type,
        input_size=cfg.dataset.input_size,
        hidden_size=cfg.model.hidden_size,
        output_size=cfg.dataset.output_size,
        observations=obs_tensor,
        actions=actions_tensor,
        population_size=cfg.optimizer.population_size,
        max_generations=cfg.optimizer.max_generations,
        device=cfg.device,
        log_to_wandb=cfg.wandb.enabled,
    )

    # Save best individual
    torch.save(best_params, Path(cfg.output_dir) / "best_params.pt")
    print(f"Final best fitness: {fitness_history[-1]:.4f}")
```

#### Step 3.5: Create NE Hydra Configs

Create `experiments/behavior_cloning/configs/optimizer/ga.yaml`:
```yaml
# @package _global_
optimizer:
  type: "ga"
  population_size: 50
  max_generations: 1000
  mutation_sigma: 0.1
  elitism_pct: 0.1
```

Create `experiments/behavior_cloning/configs/optimizer/es.yaml`:
```yaml
# @package _global_
optimizer:
  type: "es"
  population_size: 50
  max_generations: 1000
  sigma: 0.1
  learning_rate: 0.01
```

Create `experiments/behavior_cloning/configs/optimizer/cmaes.yaml`:
```yaml
# @package _global_
optimizer:
  type: "cmaes"
  population_size: 50
  max_generations: 1000
  sigma_init: 0.3
```

#### Step 3.6: Test Neuroevolution

```bash
# Test GA
python -m experiments.behavior_cloning \
  model=mlp \
  optimizer=ga \
  dataset=cartpole \
  max_optim_time=600

# Test ES
python -m experiments.behavior_cloning \
  model=reservoir \
  optimizer=es \
  dataset=lunarlander \
  max_optim_time=600

# Test CMA-ES
python -m experiments.behavior_cloning \
  model=trainable \
  optimizer=cmaes \
  dataset=acrobot \
  max_optim_time=600
```

#### Step 3.7: Commit

```bash
git add src/behavior_cloning/optimizers/neuroevolution/ experiments/behavior_cloning/configs/optimizer/
git commit -m "Phase 3: Add GPU-batched neuroevolution (GA, ES, CMA-ES)

- Extract BatchedPopulation class (key innovation: GPU-parallel evaluation)
- Implement Simple GA for feedforward and recurrent networks
- Implement Simple ES for feedforward and recurrent networks
- Implement CMA-ES for feedforward and recurrent networks
- Integrate with Hydra runner
- Add W&B logging for fitness curves
- Test all algorithm × model combinations
"
```

**Validation**:
- [ ] GA works for MLP: `python -m experiments.behavior_cloning model=mlp optimizer=ga dataset=cartpole max_optim_time=600`
- [ ] ES works for reservoir: `python -m experiments.behavior_cloning model=reservoir optimizer=es dataset=lunarlander max_optim_time=600`
- [ ] CMA-ES works: Test on any model/dataset combination
- [ ] GPU utilization >80% during evolution (check `nvidia-smi`)
- [ ] Fitness improves over generations (check W&B dashboard)
- [ ] Performance matches claude_repo baseline (±10%)

**Rollback**: `git reset --hard HEAD~1`

---

### Phase 4: Hydra Features (SLURM, Sweeps) (4-6 hours)

**Goal**: Enable SLURM cluster execution and hyperparameter sweeps

#### Step 4.1: Create SLURM Config

Create `experiments/behavior_cloning/configs/hydra/slurm.yaml`:
```yaml
# @package hydra
defaults:
  - override /hydra/launcher: submitit_slurm

run:
  dir: ${output_dir}

job:
  chdir: true

launcher:
  timeout_min: 600  # 10 hours
  cpus_per_task: 2
  gpus_per_node: 1
  mem_gb: 15
  partition: "gpu"
  account: "rrg-pbellec"  # Update with your account
  setup:
    - "module load python/3.12"
    - "source $HOME/venv/bin/activate"
```

#### Step 4.2: Test SLURM Submission

```bash
# Submit single job
python -m experiments.behavior_cloning \
  --config-name=experiment/sgd_cartpole_test \
  hydra=slurm

# Check job status
squeue -u $USER
```

#### Step 4.3: Create Sweep Config

Create `experiments/behavior_cloning/configs/sweep/learning_rate.yaml`:
```yaml
# Hyperparameter sweep for learning rate
defaults:
  - /experiment: sgd_cartpole_test
  - override /hydra/sweeper: optuna

hydra:
  sweeper:
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
    direction: minimize
    study_name: "lr_sweep"
    storage: null
    n_trials: 20
    n_jobs: 1

    params:
      optimizer.learning_rate: tag(log, interval(0.0001, 0.01))
```

#### Step 4.4: Test Sweep

```bash
# Local sweep (sequential)
python -m experiments.behavior_cloning \
  --config-name=sweep/learning_rate \
  --multirun

# SLURM sweep (parallel)
python -m experiments.behavior_cloning \
  --config-name=sweep/learning_rate \
  hydra=slurm \
  --multirun
```

#### Step 4.5: Create Multi-Run Examples

Create `experiments/behavior_cloning/configs/experiment/sgd_all_datasets.yaml`:
```yaml
# Run SGD on all datasets
defaults:
  - /model: mlp
  - /optimizer: sgd
  - /dataset: cartpole
  - /hydra: slurm
  - _self_

# Will be overridden in multirun
seed: 42
max_optim_time: 3600  # 1 hour each
```

```bash
# Submit 4 jobs (one per dataset) to SLURM
python -m experiments.behavior_cloning \
  --config-name=experiment/sgd_all_datasets \
  --multirun \
  dataset=cartpole,mountaincar,acrobot,lunarlander \
  seed=1,2,3  # 3 seeds per dataset = 12 jobs total
```

#### Step 4.6: Commit

```bash
git add experiments/behavior_cloning/configs/hydra/slurm.yaml experiments/behavior_cloning/configs/sweep/
git commit -m "Phase 4: Add SLURM and hyperparameter sweep support

- Create SLURM launcher config
- Test job submission
- Add Optuna sweep configs
- Create multi-run examples
"
```

**Validation**:
- [ ] Can submit to SLURM: `python -m experiments.behavior_cloning hydra=slurm ...`
- [ ] Jobs appear in queue: `squeue -u $USER`
- [ ] Jobs complete successfully
- [ ] Checkpoints saved to correct location
- [ ] W&B logs from SLURM jobs
- [ ] Hyperparameter sweeps work

**Rollback**: `git reset --hard HEAD~1`

---

### Phase 5: Advanced Features (6-8 hours)

**Goal**: Migrate dynamic topology networks, evaluation tools, archive experiments

#### Step 5.1: Migrate Dynamic Topology Networks

```bash
# Copy dynamic topology code
cp -r claude_repo/common/dynamic_net/ src/behavior_cloning/networks/dynamic_topology/
```

Create `src/behavior_cloning/networks/dynamic_topology/README.md`:
```markdown
# Dynamic Topology Networks

This module implements neural networks with evolving topology - the network structure
(number of neurons, connections) changes during evolution.

## Key Features

- **Graph-based representation**: Networks are directed graphs with nodes (neurons) and edges (connections)
- **Topology mutations**: Add/remove neurons, add/remove connections
- **Parameter mutations**: Mutate connection weights
- **Welford running standardization**: Online observation normalization

## Usage

See `computation.py` for forward pass through dynamic graph.
See `evolution.py` for mutation operations.

## Status

Experimental - implemented but not extensively tested.
```

Create `experiments/behavior_cloning/configs/model/dynamic.yaml`:
```yaml
# @package _global_
model:
  type: "dynamic"
  hidden_size: 50  # Initial hidden size
  max_neurons: 100  # Maximum neurons
  activation: "tanh"
```

Update runner to handle dynamic models (may require additional work depending on how different they are).

#### Step 5.2: Migrate Evaluation Tools

```bash
cp claude_repo/platform/evaluation/metrics.py src/behavior_cloning/evaluation/
cp claude_repo/platform/evaluation/comparison.py src/behavior_cloning/evaluation/
```

Create `src/behavior_cloning/evaluation/__init__.py`:
```python
"""Evaluation metrics and comparison tools."""

from .metrics import compute_cross_entropy, compute_f1_score
from .comparison import compare_to_human

__all__ = [
    "compute_cross_entropy",
    "compute_f1_score",
    "compare_to_human",
]
```

Create evaluation script `scripts/evaluate_checkpoint.py`:
```python
"""Evaluate a trained checkpoint."""

import argparse
from pathlib import Path
import torch
from behavior_cloning.models import MLP, RecurrentMLPReservoir
from behavior_cloning.data import load_human_data, preprocess_episodes
from behavior_cloning.evaluation import compute_cross_entropy, compute_f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subject", type=str, required=True)
    args = parser.parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    # ... load model and evaluate

    print(f"Cross-entropy: {ce:.4f}")
    print(f"F1 score: {f1:.4f}")


if __name__ == "__main__":
    main()
```

#### Step 5.3: Archive Old Experiments

```bash
# Move experiment directories
mv claude_repo/experiments/1_dl_vs_ga_scaling_dataset_size_flops experiments/archive/experiment_1
mv claude_repo/experiments/2_dl_vs_ga_es experiments/archive/experiment_2
mv claude_repo/experiments/3_cl_info_dl_vs_ga experiments/archive/experiment_3
mv claude_repo/experiments/4_add_recurrence experiments/archive/experiment_4
```

Create `experiments/archive/README.md`:
```markdown
# Archived Experiments

This directory contains experiments from the original claude_repo before the merge.

These are preserved for reference but are not actively maintained. To reproduce
these experiments with the new unified codebase, see the experiment configs in
`experiments/behavior_cloning/configs/experiment/`.

## Experiment Descriptions

- **experiment_1**: Scaling analysis (dataset size vs FLOPs)
- **experiment_2**: DL vs GA vs ES comparison
- **experiment_3**: Continual learning ablation (20+ result files)
- **experiment_4**: Recurrent network reference implementation

## Results

Experiment 3 has extensive results showing:
- SGD vs GA comparisons
- With/without continual learning features
- Multiple environments (CartPole, LunarLander, etc.)
- Training curves and checkpoints
```

#### Step 5.4: Create Full Experiment Configs

Create configs that reproduce key experiments from claude_repo:

`experiments/behavior_cloning/configs/experiment/dl_vs_ga_cartpole.yaml`:
```yaml
# Reproduce experiment comparing DL and GA on CartPole
defaults:
  - /model: mlp
  - /optimizer: sgd  # Will override for GA run
  - /dataset: cartpole
  - /hydra: slurm
  - _self_

max_optim_time: 3600  # 1 hour
seed: 42

# Run with: python -m experiments.behavior_cloning \
#   --config-name=experiment/dl_vs_ga_cartpole \
#   --multirun \
#   optimizer=sgd,ga
```

#### Step 5.5: Commit

```bash
git add src/behavior_cloning/networks/ src/behavior_cloning/evaluation/ experiments/archive/ scripts/evaluate_checkpoint.py
git commit -m "Phase 5: Add dynamic networks, evaluation tools, archive experiments

- Copy dynamic topology networks from claude_repo
- Add evaluation metrics and comparison tools
- Create checkpoint evaluation script
- Archive old experiments (experiment_1 through experiment_4)
- Create experiment configs reproducing key comparisons
"
```

**Validation**:
- [ ] Dynamic topology code is present
- [ ] Evaluation scripts work
- [ ] Archived experiments are accessible
- [ ] Can view old results

**Rollback**: `git reset --hard HEAD~1`

---

### Phase 6: Documentation & Cleanup (4-6 hours)

**Goal**: Complete documentation, clean up code

#### Step 6.1: Write Unified README

Create `README.md`:
```markdown
# Behavior Cloning Research Platform

A unified platform for comparing gradient-based (SGD) and evolutionary (GA/ES/CMA-ES)
optimization methods for learning human behavioral policies.

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd behavior_cloning_research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Run Your First Experiment

```bash
# Local 10-minute test (SGD on CartPole)
python -m experiments.behavior_cloning \
  --config-name=experiment/sgd_cartpole_test

# SLURM cluster (1-hour GA on LunarLander)
python -m experiments.behavior_cloning \
  --config-name=experiment/ga_lunarlander_full \
  hydra=slurm
```

## Features

- ✅ **GPU-batched neuroevolution**: 10x faster than sequential evaluation
- ✅ **PyTorch Lightning**: Production-quality deep learning
- ✅ **Hydra configuration**: Flexible experiment management
- ✅ **W&B logging**: Automatic experiment tracking
- ✅ **SLURM integration**: Easy cluster computing
- ✅ **Human behavioral data**: 38MB of real gameplay data
- ✅ **Multiple optimizers**: SGD, GA, ES, CMA-ES
- ✅ **Multiple architectures**: Feedforward, recurrent, dynamic topology

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Usage Guide](docs/USAGE_GUIDE.md)
- [Migration from Old Code](docs/MIGRATION.md)

## Citation

If you use this code in your research, please cite:

```
[Citation to be added]
```

## License

[License information]
```

#### Step 6.2: Update ARCHITECTURE.md

Update `docs/ARCHITECTURE.md` with new structure, explain key design decisions.

#### Step 6.3: Rewrite USAGE_GUIDE.md

Create comprehensive guide with:
- Installation instructions
- Configuration system explanation
- Example experiments
- SLURM usage
- Hyperparameter sweeps
- Checkpoint evaluation
- Troubleshooting

#### Step 6.4: Create MIGRATION.md

Create `docs/MIGRATION.md`:
```markdown
# Migration Guide: Old Code → New Unified Platform

This guide helps you migrate from the old argparse-based CLI to the new Hydra-based system.

## Command Mappings

### SGD Training

**Old**:
```bash
python -m platform.runner \
  --dataset cartpole \
  --method SGD_test \
  --model reservoir \
  --optimizer sgd \
  --subject sub01 \
  --max-time 600
```

**New**:
```bash
python -m experiments.behavior_cloning \
  model=reservoir \
  optimizer=sgd \
  dataset=cartpole \
  dataset.subject=sub01 \
  max_optim_time=600
```

[More examples...]

## Key Differences

1. **Configuration**: YAML files instead of argparse
2. **Logging**: W&B instead of SQLite
3. **SLURM**: Hydra submitit instead of custom scripts
4. **Models**: Wrapped in Lightning for SGD, unchanged for NE

[More details...]
```

#### Step 6.5: Move Data Collection Scripts

```bash
cp claude_repo/data/collect.py scripts/collect_data.py
cp claude_repo/data/data_analysis.py scripts/analyze_data.py
cp claude_repo/data/replay.py scripts/replay_behavior.py
```

Add docstrings and usage instructions to each script.

#### Step 6.6: Run Linters and Formatters

```bash
# Format code
black src/ experiments/ scripts/ tests/

# Lint
ruff check src/ experiments/ scripts/ tests/ --fix

# Type check
mypy src/

# Check pre-commit
pre-commit run --all-files
```

Fix any issues found.

#### Step 6.7: Ensure All Tests Pass

```bash
pytest tests/ -v --cov=src/behavior_cloning
```

#### Step 6.8: Commit

```bash
git add README.md docs/ scripts/ src/ experiments/ tests/
git commit -m "Phase 6: Documentation and cleanup

- Write unified README with quick start
- Update ARCHITECTURE.md for new structure
- Rewrite USAGE_GUIDE.md for Hydra
- Create MIGRATION.md (command mappings)
- Move data collection scripts to scripts/
- Run linters (black, ruff, mypy)
- Ensure all tests pass
"
```

**Validation**:
- [ ] README is clear and comprehensive
- [ ] Can follow quick start from scratch
- [ ] All documentation is up-to-date
- [ ] All linters pass: `ruff check .`, `black --check .`, `mypy src/`
- [ ] All tests pass: `pytest tests/`

**Rollback**: `git reset --hard HEAD~1`

---

### Phase 7: Final Validation & Merge (2-4 hours)

**Goal**: Final testing, create PR, merge to main

#### Step 7.1: Run Full Test Suite

```bash
# Unit tests
pytest tests/ -v --cov=src/behavior_cloning

# Linters
ruff check .
black --check .
mypy src/

# Pre-commit
pre-commit run --all-files
```

Fix any issues.

#### Step 7.2: Run Integration Tests

```bash
# Quick 10-min runs on all datasets to verify everything works
for dataset in cartpole mountaincar acrobot lunarlander; do
  echo "Testing SGD on $dataset..."
  python -m experiments.behavior_cloning \
    model=mlp \
    optimizer=sgd \
    dataset=$dataset \
    max_optim_time=600 \
    wandb.enabled=false
done

# Test GA
python -m experiments.behavior_cloning \
  model=mlp \
  optimizer=ga \
  dataset=cartpole \
  max_optim_time=600 \
  wandb.enabled=false
```

Verify:
- All runs complete successfully
- Results are reasonable (loss decreases, accuracy >70%)
- No errors or warnings

#### Step 7.3: Test on Cluster

```bash
# Submit test job to SLURM
python -m experiments.behavior_cloning \
  --config-name=experiment/sgd_cartpole_test \
  hydra=slurm

# Wait for completion
squeue -u $USER

# Check logs and results
cat results/<timestamp>/slurm_*.out
```

Verify:
- Job completed successfully
- Logs look correct
- Checkpoints were saved

#### Step 7.4: Code Review

Self-review all changes:
- [ ] No hardcoded paths
- [ ] All configs are correct
- [ ] Documentation is accurate
- [ ] No TODOs or FIXMEs left
- [ ] All imports are correct
- [ ] No dead code

#### Step 7.5: Update .gitignore

Ensure `.gitignore` includes:
```
# Original repos (kept as reference)
claude_repo/
ai_repo/

# Results
results/
outputs/
multirun/
.hydra/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# IDEs
.vscode/
.idea/

# Checkpoints
*.ckpt
*.pt

# Logs
*.log
*.db
```

#### Step 7.6: Create PR

```bash
# Push branch
git push origin repo-merge

# Create PR on GitHub with description:
```

**PR Title**: Merge claude_repo and ai_repo into unified behavior cloning platform

**PR Description**:
```markdown
## Summary

This PR merges `claude_repo` and `ai_repo` into a unified behavior cloning research platform.

## Key Changes

### Architecture
- Fresh directory structure with `src/behavior_cloning/` as main package
- Experiments in `experiments/behavior_cloning/`
- Original repos preserved in `claude_repo/` and `ai_repo/` (gitignored)

### From claude_repo (✅ Kept)
- GPU-batched neuroevolution (key innovation)
- All models (feedforward, recurrent, dynamic topology)
- Human behavioral data (38MB, 8 JSON files)
- Data loading and preprocessing
- Evaluation metrics

### From ai_repo (✅ Kept)
- Production infrastructure (Docker, CI/CD, pre-commit)
- Hydra + Hydra-Zen configuration
- PyTorch Lightning for SGD
- Enhanced utilities
- W&B integration
- SLURM integration

### Migration
- Argparse CLI → Hydra configuration
- Custom SQLite tracking → W&B
- Custom SLURM scripts → Hydra submitit
- All models wrapped in Lightning for SGD

### Testing
- ✅ All unit tests pass
- ✅ Integration tests pass (10-min runs on all datasets)
- ✅ SLURM submission works
- ✅ All linters pass (ruff, black, mypy)

### Documentation
- ✅ Unified README with quick start
- ✅ Updated ARCHITECTURE.md
- ✅ Rewritten USAGE_GUIDE.md for Hydra
- ✅ Migration guide (argparse → Hydra)

## Migration Impact

### Breaking Changes
- Old argparse commands no longer work (see MIGRATION.md)
- Results stored in new location
- Configuration system completely different

### Preserved Functionality
- All models work identically
- All optimizers work identically
- Human data intact (verified with checksums)
- Old experiment results archived

## Validation

Tested on:
- [ ] Local CPU
- [ ] Local GPU
- [ ] SLURM cluster

Performance:
- [ ] SGD matches claude_repo baseline (±5%)
- [ ] GA matches claude_repo baseline (±10%)
- [ ] GPU utilization >80% during NE

## Next Steps

After merge:
1. Run full experiment comparing DL vs GA on all datasets
2. Validate results against archived Experiment 3
3. Update any external documentation
4. Announce migration to collaborators
```

#### Step 7.7: Merge to Main

After PR approval:

```bash
git checkout main
git merge repo-merge
git push origin main
```

#### Step 7.8: Tag Release

```bash
git tag -a v1.0.0-unified -m "Unified behavior cloning platform (claude_repo + ai_repo merge)"
git push origin v1.0.0-unified
```

#### Step 7.9: Final Cleanup

```bash
# Delete merge branch
git branch -D repo-merge
git push origin --delete repo-merge

# Verify main branch works
pip install -e .
python -m experiments.behavior_cloning --config-name=experiment/sgd_cartpole_test
```

**Validation**:
- [ ] Main branch works
- [ ] Can clone fresh and run experiments
- [ ] Documentation is accessible
- [ ] Release is tagged

---

## Summary

### Total Effort: 35-52 hours

- Phase 0: 1-2 hours (preparation)
- Phase 1: 4-6 hours (models & data)
- Phase 2: 6-8 hours (SGD with Lightning)
- Phase 3: 8-12 hours (neuroevolution)
- Phase 4: 4-6 hours (SLURM & sweeps)
- Phase 5: 6-8 hours (advanced features)
- Phase 6: 4-6 hours (documentation)
- Phase 7: 2-4 hours (final validation)

### Key Success Criteria

1. ✅ GPU-batched neuroevolution preserved (key innovation)
2. ✅ Human behavioral data safe (38MB, checksums verified)
3. ✅ Performance matches baselines (SGD ±5%, NE ±10%)
4. ✅ Hydra configuration works
5. ✅ SLURM submission works
6. ✅ Documentation complete
7. ✅ All tests pass

### Most Critical Files

1. `src/behavior_cloning/optimizers/neuroevolution/batched_population.py` (~400 LOC)
2. `experiments/behavior_cloning/runner.py` (~200 LOC)
3. `src/behavior_cloning/data/datamodules.py` (~300 LOC)
4. `src/behavior_cloning/optimizers/dl/litmodule.py` (~200 LOC)
5. `experiments/behavior_cloning/configs/config.yaml` (~100 lines YAML)

### Rollback Strategy

Each phase:
- Committed separately with descriptive message
- Has validation criteria
- Can be reverted with `git reset --hard HEAD~1`

Overall:
- Work on separate branch (`repo-merge`)
- Original repos backed up as tarballs
- Original repos preserved in-place (.gitignored)
- Can always return to commit before merge

---

## Post-Merge

After successful merge:

1. Run full experiment:
   ```bash
   python -m experiments.behavior_cloning \
     --config-name=experiment/dl_vs_ga_cartpole \
     --multirun \
     hydra=slurm \
     optimizer=sgd,ga \
     seed=1,2,3,4,5
   ```

2. Compare results to archived Experiment 3

3. Update external documentation

4. Announce migration to collaborators

5. Consider follow-up improvements:
   - Add more comprehensive tests
   - Add behavioral evaluation integration
   - Optimize performance
   - Add more documentation

---

**END OF MERGE PLAN**
