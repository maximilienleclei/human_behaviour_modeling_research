# Human Behaviour Modeling Research

A research platform for comparing gradient-based and evolutionary training methods for behavioral cloning from human demonstrations in classic control environments.

## Overview

This codebase implements a comprehensive experimentation framework to investigate how different neural network training approaches (SGD, genetic algorithms, evolution strategies) perform when learning to imitate human behavior in reinforcement learning environments (CartPole, MountainCar, Acrobot, LunarLander).

**Key Research Question:** How do gradient-based methods compare to evolutionary methods for learning human behavioral policies?

## Quick Start

```bash
# Quick 10-minute test on human CartPole data
python -m platform.runner \
  --dataset cartpole \
  --method SGD_test \
  --model reservoir \
  --optimizer sgd \
  --subject sub01 \
  --max-time 600

# Full training with continual learning features
python -m platform.runner \
  --dataset lunarlander \
  --method GA_trainable \
  --model trainable \
  --optimizer ga \
  --use-cl-info \
  --subject sub01 \
  --max-time 36000
```

## Repository Structure

```
human_behaviour_modeling_research/
├── platform/              # Unified experimentation framework (CORE)
│   ├── models/           # Neural network architectures
│   ├── optimizers/       # Training algorithms (SGD, GA, ES, CMA-ES)
│   ├── data/             # Data loading and preprocessing
│   ├── evaluation/       # Metrics and comparison
│   ├── config.py         # Configuration management
│   └── runner.py         # Main CLI entry point
│
├── experiments/          # Experiment infrastructure
│   ├── tracking/         # SQLite database for experiment tracking
│   ├── cli/              # Command-line tools (submit, monitor, query)
│   ├── slurm/            # SLURM cluster job management
│   └── [1-4]_*/          # Archived experiments (consolidated into platform/)
│
├── data/                 # Human behavioral data (JSON files)
├── common/               # Shared components (dynamic networks)
├── docs/                 # Documentation (you are here!)
└── utils/                # Type checking utilities
```

## Key Features

- **Multiple Training Methods:** SGD (backpropagation), GA (genetic algorithm), ES (evolution strategies), CMA-ES
- **Multiple Architectures:** Feedforward MLP, recurrent networks (reservoir + trainable), dynamic topology
- **Type-Safe:** Extensive jaxtyping + beartype validation
- **GPU-Efficient:** Batched population operations for evolutionary methods
- **Experiment Tracking:** SQLite database for scalable experiment management
- **Cluster Integration:** SLURM job submission and monitoring

## Documentation

📖 **[Complete Documentation Index](docs/README.md)** - Start here for a guided tour of all documentation

**Quick Links:**
- New to the project? → [Architecture Guide](docs/ARCHITECTURE.md)
- Running experiments? → [Usage Guide](docs/USAGE_GUIDE.md)
- Need to find code? → [Navigation Guide](docs/NAVIGATION.md)
- Current status? → [Status & Next Steps](docs/STATUS.md)

## Prerequisites

### Required Dependencies
```
torch>=2.0
numpy
scikit-learn
datasets (HuggingFace)
einops
jaxtyping
beartype
ordered_set
gymnasium[classic-control,box2d]
matplotlib
```

### Optional Dependencies
```
matplotlib (for plotting)
seaborn (for visualization)
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd human_behaviour_modeling_research

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (if not already installed)
# Visit https://pytorch.org for platform-specific instructions
```

## Supported Configurations

| Model Type | SGD | GA | ES | CMA-ES |
|------------|-----|----|----|--------|
| Feedforward MLP | ✅ | ✅ | ✅ | ✅ |
| Recurrent (Reservoir) | ✅ | ✅ | ✅ | ✅ |
| Recurrent (Trainable) | ✅ | ✅ | ✅ | ✅ |
| Dynamic Topology | ❌ | ✅ | ❌ | ❌ |

## Datasets

**Human Behavioral Data:**
- CartPole-v1 (2 subjects)
- MountainCar-v0 (2 subjects)
- Acrobot-v1 (2 subjects)
- LunarLander-v2 (2 subjects)

**HuggingFace Datasets:**
- Expert trajectories from trained RL agents
- Used for scaling experiments

## Common Commands

### Run Local Experiment
```bash
python -m platform.runner \
  --dataset cartpole \
  --method SGD_reservoir \
  --model reservoir \
  --optimizer sgd \
  --use-cl-info \
  --subject sub01
```

### Submit to Cluster
```bash
python experiments/cli/submit_jobs.py \
  --dataset cartpole \
  --method GA_trainable \
  --model trainable \
  --optimizer ga
```

### Monitor Jobs
```bash
python experiments/cli/monitor_jobs.py --watch
```

### Query Results
```bash
python experiments/cli/query_results.py --experiment 5 --summary
```

## Project Status

**Platform Status:** Core implementation complete (7/12 phases)
- ✅ Models, data loading, evaluation, optimizers (SGD, GA, ES, CMA-ES)
- ✅ CLI runner with full argument parsing
- ⬜ YAML configuration system (optional)
- ⬜ Comprehensive testing suite

See [docs/STATUS.md](docs/STATUS.md) for detailed status and next steps.
