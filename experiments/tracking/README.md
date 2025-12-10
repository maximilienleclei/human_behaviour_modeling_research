# Experiment Tracking

SQLite database that stores experiment runs, metrics, and errors efficiently.

## What it does

- Replaces 180MB JSON files with lightweight database (<10KB queries)
- Tracks run metadata: dataset, method, parameters, SLURM job info
- Logs time-series metrics during training (every ~60s)
- Stores final results and aggregated statistics
- Records errors with full tracebacks for debugging

## Database tables

- `experiment_runs`: Metadata for each run (status, config, SLURM info)
- `run_metrics`: Training metrics over time (loss, F1, etc.)
- `run_results`: Final aggregated results (best loss, convergence, etc.)
- `run_errors`: Error logs with tracebacks
- `slurm_jobs`: SLURM job tracking

## Files

- `database.py`: Database interface
- `logger.py`: Automatic tracking during experiments
- `query.py`: Query interface for analysis
- `slurm_manager.py`: SLURM job management
