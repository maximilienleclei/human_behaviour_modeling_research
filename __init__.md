# Human Behaviour Modeling Research

Research codebase for modeling human behavior in reinforcement learning tasks using both gradient-based and evolutionary optimization methods.

## Overview

This repository contains a complete ecosystem for training neural networks to predict human actions in RL environments (CartPole, MountainCar, Acrobot, LunarLander). The codebase supports multiple model architectures, optimization algorithms, and experimental paradigms with integrated experiment tracking and cluster computing support.

## Repository Structure

### platform/
Unified platform for behavior modeling experiments:
- **models/** - Neural architectures (feedforward, recurrent, dynamic)
- **optimizers/** - Training algorithms (SGD, GA, ES, CMA-ES)
- **data/** - Data loading and preprocessing
- **evaluation/** - Metrics and behavioral comparison
- **runner.py** - Main experiment orchestrator
- **config.py** - Configuration schemas

### experiments/
Experiment management and individual research studies:
- **tracking/** - SQLite database logging with SLURM integration
- **cli/** - Command-line tools for job management and result querying
- **[numbered experiments]** - Individual research studies (e.g., 2_dl_vs_ga_es/)

### data/
Human behavioral data collection and analysis:
- **collect.py** - Interactive data collection tool
- **data_analysis.py** - Session statistics and visualization
- **replay.py** - Deterministic episode replay
- **[JSON files]** - Human gameplay data

### common/
Shared utilities and specialized implementations:
- **dynamic_net/** - Graph-based evolving neural networks

### utils/
Supporting utilities:
- **beartype.py** - Custom type validators

### docs/
Documentation and project planning

## Key Features

**Multiple Architectures**: Feedforward MLPs, recurrent networks (reservoir and trainable), dynamic networks with evolving topology.

**Flexible Optimization**: Gradient-based (SGD with backprop) and gradient-free (Simple GA, Simple ES, diagonal CMA-ES).

**GPU Efficiency**: Batched population evaluation for evolutionary methods, dramatically faster than sequential computation.

**Experiment Tracking**: SQLite database with SLURM integration for large-scale cluster experiments.

**Continual Learning**: Optional session/run features to model temporal adaptation.

**Human Data**: Tools for collecting, replaying, and analyzing human gameplay.

## Quick Start

**Train a model:**
```bash
python platform/runner.py --dataset cartpole --method SGD_reservoir --seed 42
```

**Submit SLURM sweep:**
```bash
python experiments/cli/submit_jobs.py --exp 4 --sweep-all
```

**Monitor jobs:**
```bash
python experiments/cli/monitor_jobs.py --exp 4 --watch
```

**Query results:**
```bash
python experiments/cli/query_results.py --exp 4 --best --metric test_loss
```

## Research Workflow

1. **Data Collection**: Use data/collect.py to gather human gameplay
2. **Data Analysis**: Visualize sessions with data/data_analysis.py
3. **Model Training**: Run experiments via platform/runner.py or submit to cluster
4. **Experiment Tracking**: Automatic logging to database via experiments/tracking/
5. **Result Analysis**: Query and compare methods via experiments/cli/

## Method Naming Convention

Methods follow pattern: `[ALGORITHM]_[MODEL_TYPE]`

Examples:
- `SGD_reservoir` - SGD with recurrent reservoir network
- `adaptive_ga_trainable` - Adaptive GA with trainable recurrent network
- `adaptive_ga_dynamic` - Adaptive GA with dynamic evolving network

## Environment Support

- **CartPole-v1**: Balance pole with left/right actions
- **MountainCar-v0**: Build momentum to reach goal
- **Acrobot-v1**: Swing acrobot to target height
- **LunarLander-v3**: Land spacecraft safely

## Technical Details

**Type Safety**: Extensive use of type hints with beartype enforcement and jaxtyping for tensors.

**Reproducibility**: Seed control across all random sources (Python, NumPy, PyTorch).

**Modularity**: Clean separation between models, optimizers, data, and evaluation enables easy extension.

**Documentation**: Every Python file over 30 lines has accompanying .md file explaining purpose and contents.

## Dependencies

- PyTorch (neural networks, GPU computation)
- Gymnasium (RL environments)
- HuggingFace Datasets (benchmark datasets)
- beartype + jaxtyping (type safety)
- matplotlib (visualization)
- sklearn (metrics)
- pygame (data collection GUI)

See requirements file for complete list.

## Project Status

Active research project with ongoing experiments exploring:
- Comparison of gradient vs gradient-free methods
- Continual learning features for temporal adaptation
- Dynamic network topology evolution
- Recurrent architectures for sequential decision-making

## References

For detailed information on specific components, see the corresponding __init__.md files in each directory.
