# Experiment Results Query Interface

High-level interface for querying and analyzing experiment results from the tracking database.

## What it's for

Provides convenient Python API for accessing experiment data without writing SQL. Used by CLI tools (experiments/cli/query_results.py) and analysis notebooks. Optimized for minimal context usage with concise summaries.

## What it contains

### Main Class
- `ExperimentQuery` - Query interface with methods for different analysis patterns

### Query Methods
- `summarize_all_experiments()` - Overview of all experiments with status counts
- `summarize_experiment()` - Detailed summary of single experiment (datasets, methods, runs)
- `get_best_runs()` - Returns top N runs ranked by specified metric
- `compare_methods()` - Aggregates statistics across methods on specific dataset
- `get_failed_runs()` - Returns failed/timeout runs with error details
- `export_results_csv()` - Exports full results to CSV file

## Key Details

All queries operate on experiments/tracking/database.py's ExperimentDB interface. The `compare_methods()` function groups runs by method and computes mean/std across seeds for specified metric (test_loss, f1_score, mean_pct_diff, etc.). Best runs query fetches final metric values from run_results table. Summary functions count runs by status (pending, running, completed, failed, timeout) and group by experiment number, dataset, or method as needed. CSV export includes all run metadata (config, timestamps, SLURM job info) plus final results for downstream analysis in R/Python/Excel.
