# SLURM Job Monitoring CLI

Command-line tool for monitoring SLURM jobs and experiment run status with optional continuous refresh.

## What it's for

Provides real-time visibility into experiment execution status across SLURM jobs. Shows summary statistics (pending/running/completed/failed/timeout counts), currently running jobs, and recent failures.

## What it contains

### CLI Interface
- `main()` - Entry point with argument parsing and monitoring loop

### Features
- Summary statistics by experiment number or all experiments
- List of currently running jobs with SLURM job IDs
- Recent failures (up to 5) with run details
- Watch mode with configurable refresh interval
- Screen clearing for continuous monitoring

## Key Details

Uses experiments/tracking/database.py to query run status and experiments/tracking/slurm_manager.py to aggregate job statistics. In watch mode (`--watch`), continuously refreshes display at specified interval (default 30s) until interrupted. Can filter by experiment number via `--exp` flag. Displays formatted tables showing run IDs, datasets, methods, and SLURM job identifiers (including array task IDs). Handles graceful shutdown on Ctrl+C.
