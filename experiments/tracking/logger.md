# Experiment Logger

Context manager for logging experiment runs to the tracking database with automatic status updates and error handling.

## What it's for

Provides a clean interface for wrapping experiment execution with database logging. Automatically creates run entries, updates status, tracks SLURM job information, and logs metrics/errors during training.

## What it contains

### Main Class
- `ExperimentLogger` - Context manager that handles experiment lifecycle logging

### Methods
- `__enter__()` - Creates database run entry, extracts SLURM environment variables, sets status to "running", starts timer
- `__exit__()` - Updates final status ("completed" or "failed"), logs exceptions and tracebacks if errors occur
- `log_progress()` - Records training metrics (loss, F1, fitness, behavioral comparison, GPU memory) with elapsed time

## Key Details

Designed to be used with Python's `with` statement for automatic cleanup. On entry, it reads SLURM_JOB_ID and SLURM_ARRAY_TASK_ID environment variables to link database runs with SLURM jobs. If an exception occurs during execution, it catches the error, logs it to the database with full traceback, and marks the run as "failed". The logger is passed to optimizer functions (platform/optimizers/) to enable periodic metric logging via `log_progress()`. Integrates with experiments/tracking/database.py for all database operations.
