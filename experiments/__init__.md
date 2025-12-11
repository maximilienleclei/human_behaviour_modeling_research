# Experiments Directory

Experiment management infrastructure and individual experiment implementations.

## Overview

This directory contains both the tracking infrastructure for managing experiments at scale and individual experiment scripts for specific research questions.

## Directory Structure

### tracking/
Database-backed experiment tracking with SLURM integration:
- **database.py** - SQLite interface for runs, metrics, results, errors
- **logger.py** - Context manager for automatic experiment logging
- **query.py** - High-level result querying and analysis
- **slurm_manager.py** - SLURM job submission and monitoring

### cli/
Command-line tools for experiment management:
- **submit_jobs.py** - Submit single jobs or parameter sweeps to SLURM
- **monitor_jobs.py** - Real-time job status monitoring
- **query_results.py** - Query and analyze experiment results

### Individual Experiments

**2_dl_vs_ga_es/**
Systematic comparison of deep learning (SGD) vs genetic algorithms (GA) vs evolution strategies (ES) on CartPole and LunarLander. Self-contained experiment with its own implementation (predates unified platform).

**Other experiments (1, 3, 4, 5+)**
Various experiments exploring different aspects of behavior modeling. Each experiment directory contains:
- Experiment-specific code
- Configuration files
- Analysis scripts
- Results and plots

## Tracking System Workflow

1. **Setup**: Initialize experiment configuration
2. **Submit**: Use CLI to submit jobs to SLURM cluster
3. **Execute**: Jobs run with automatic database logging via ExperimentLogger
4. **Monitor**: Track progress via monitoring CLI
5. **Analyze**: Query results and generate comparisons

## Naming Convention

Experiments are numbered sequentially (1, 2, 3, ...) with descriptive names:
- Experiment 2: DL vs GA/ES comparison
- Experiment 4: Recurrent models with continual learning
- Etc.

## Usage

**Typical experiment workflow:**
```bash
# Submit sweep
python experiments/cli/submit_jobs.py --exp 4 --sweep-all

# Monitor progress
python experiments/cli/monitor_jobs.py --exp 4 --watch

# Query results
python experiments/cli/query_results.py --exp 4 --best --metric test_loss
```

## Integration

Experiments use platform/ for model/optimizer implementations and data/ for human behavioral data. The tracking system is independent but can wrap platform/runner.py for unified experiments.
