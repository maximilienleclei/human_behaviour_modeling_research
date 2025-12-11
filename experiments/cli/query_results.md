# Experiment Results Query CLI

Command-line tool for querying and analyzing experiment results from the tracking database.

## What it's for

Provides convenient interface for analyzing completed experiments without writing SQL queries. Supports summary statistics, best run rankings, method comparisons, failure analysis, and CSV export.

## What it contains

### CLI Interface
- `main()` - Entry point with argument parsing and query dispatch

### Query Modes
- `--summary` - Overall experiment statistics or specific experiment details
- `--best` - Top N runs ranked by specified metric (default: test_loss)
- `--compare` - Compare methods on specific dataset with aggregated statistics
- `--failed` - List failed runs with error messages and tracebacks
- `--export` - Export full results to CSV file

## Key Details

Uses experiments/tracking/query.py as the backend query interface. Best runs can be ranked by any logged metric (test_loss, f1_score, mean_pct_diff, etc.) with configurable top N (default 5). Method comparison aggregates results across seeds/runs to show mean and std for each method. Failed run analysis includes truncated error messages (200 chars) and SLURM job identifiers for debugging. CSV export includes all run metadata and final metrics for post-processing analysis.
