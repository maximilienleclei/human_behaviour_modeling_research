# SLURM Job Submission CLI

Command-line tool for submitting single experiment jobs or full parameter sweeps to SLURM.

## What it's for

Provides convenient interface for launching experiments on SLURM clusters. Handles both single job submission and full parameter sweep arrays with configurable resource allocation.

## What it contains

### CLI Interface
- `main()` - Entry point with argument parsing and submission logic

### Submission Modes
- Single job: Specify dataset, method, subject, seed individually
- Sweep mode (`--sweep-all`): Submit full parameter grid as SLURM array job

### Configuration
- Experiment parameters (dataset, method, subject, CL features, seed)
- SLURM resources (time limit, GPU type, memory, CPUs, account)
- Array job control (max concurrent jobs)

## Key Details

Uses experiments/tracking/slurm_manager.py to generate SLURM scripts and submit jobs. Sweep mode automatically generates all combinations of predefined parameter grids (experiment-specific configurations hardcoded in script, e.g., exp 4 sweeps over 4 datasets × 5 methods × 2 CL settings). SLURM array jobs use `--max-concurrent` to limit parallel execution. Default resources match CLAUDE.md specifications (30min, h100_1g.10gb:1, 15G memory, 2 CPUs). Both submission modes return SLURM job IDs for monitoring. Jobs are linked to database runs for tracking via experiments/tracking/logger.py.
