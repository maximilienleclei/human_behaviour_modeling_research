# Experiment Tracking Database

SQLite database interface for persisting experiment runs, metrics, results, errors, and SLURM job information.

## What it's for

Provides centralized storage for all experiment tracking data. Enables querying historical results, monitoring active jobs, debugging failures, and analyzing performance across methods and datasets.

## What it contains

### Main Class
- `ExperimentDB` - SQLite interface with CRUD operations for all tables

### Database Schema (5 tables)
- `experiment_runs` - Core table with run metadata (experiment number, dataset, method, subject, seed, SLURM job info, status, timestamps)
- `run_metrics` - Time-series metrics during training (elapsed time, epoch, loss, F1, fitness, behavioral comparison, GPU memory)
- `run_results` - Final aggregated results per run (best metrics, final values)
- `run_errors` - Error logs with exception type, message, and traceback
- `slurm_jobs` - SLURM job metadata and resource allocation

### Key Methods
- `create_run()` - Creates new run entry, returns run_id
- `update_run_status()` - Updates run status (pending/running/completed/failed/timeout) and exit code
- `log_metrics()` - Records training metrics at specific elapsed time
- `log_error()` - Stores error information with traceback
- `get_all_runs()` - Retrieves all runs with optional experiment number filter
- `get_run_metrics()` - Returns metric time series for specific run

## Key Details

Uses SQLite with row_factory=sqlite3.Row for dict-style access. The UNIQUE constraint on experiment_runs prevents duplicate runs for same (experiment, dataset, method, subject, cl_info, seed) combination. SLURM integration stores job IDs and array task IDs to link database runs with cluster jobs. Config and results are stored as JSON blobs for flexibility. Status lifecycle: pending (created) → running (started) → completed/failed/timeout (finished). Used by experiments/tracking/logger.py for automatic logging, experiments/tracking/query.py for analysis, and experiments/tracking/slurm_manager.py for job monitoring.
