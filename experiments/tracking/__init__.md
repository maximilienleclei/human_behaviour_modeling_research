# Experiments Tracking Directory

Database-backed experiment tracking system with SLURM integration for cluster computing.

## Overview

This directory provides infrastructure for logging experiments to SQLite database, managing SLURM job submission/monitoring, and querying results for analysis.

## Files

### Database Interface
- **database.py** - SQLite database for experiment tracking
  - `ExperimentDB` - Core database interface
  - Schema: runs, metrics, results, errors, SLURM jobs
  - CRUD operations, status updates, metric logging

### Logging Interface
- **logger.py** - Context manager for automatic logging
  - `ExperimentLogger` - Wraps experiments with database logging
  - Automatic status tracking (pending → running → completed/failed)
  - Error handling and traceback logging
  - SLURM environment variable extraction

### Query Interface
- **query.py** - High-level result querying
  - `ExperimentQuery` - Analysis-friendly query methods
  - Summary statistics, best runs, method comparisons
  - Failed run analysis, CSV export

### SLURM Manager
- **slurm_manager.py** - SLURM job submission and monitoring
  - `SlurmManager` - Job lifecycle management
  - Template-based script generation, array job support
  - Status monitoring and database synchronization

## Workflow

1. **Submission**: CLI tool (experiments/cli/submit_jobs.py) → SlurmManager → sbatch
2. **Execution**: Experiment script → ExperimentLogger context manager → Database
3. **Monitoring**: CLI tool (experiments/cli/monitor_jobs.py) → SlurmManager → Database queries
4. **Analysis**: CLI tool (experiments/cli/query_results.py) → ExperimentQuery → Results

## Usage

The tracking system integrates with platform/runner.py via ExperimentLogger. SLURM tools are used from CLI for cluster workflows.
