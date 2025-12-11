# Experiments CLI Directory

Command-line tools for managing SLURM jobs and querying experiment results.

## Overview

This directory provides CLI scripts for interacting with the experiment tracking system: submitting jobs to SLURM, monitoring job status, and analyzing results.

## Files

### Job Submission
- **submit_jobs.py** - Submit experiments to SLURM
  - Single job submission with custom parameters
  - Full parameter sweep submission (array jobs)
  - Resource configuration (time, GPU, memory, CPUs)
  - Experiment-specific sweep definitions

### Job Monitoring
- **monitor_jobs.py** - Monitor SLURM job status
  - Real-time status summary (pending/running/completed/failed/timeout)
  - Currently running jobs list
  - Recent failures display
  - Watch mode with auto-refresh

### Result Querying
- **query_results.py** - Query and analyze results
  - Experiment summaries and statistics
  - Best run rankings by metric
  - Method comparison on datasets
  - Failed run analysis with error details
  - CSV export for external analysis

## Typical Usage Patterns

**Submit sweep:**
```bash
python experiments/cli/submit_jobs.py --exp 4 --sweep-all
```

**Monitor progress:**
```bash
python experiments/cli/monitor_jobs.py --exp 4 --watch
```

**Query best results:**
```bash
python experiments/cli/query_results.py --exp 4 --best --metric test_loss --top 10
```

**Compare methods:**
```bash
python experiments/cli/query_results.py --exp 4 --compare --dataset cartpole
```

## Integration

These tools build on experiments/tracking/ modules (database.py, slurm_manager.py, query.py) to provide user-friendly interfaces for common operations.
