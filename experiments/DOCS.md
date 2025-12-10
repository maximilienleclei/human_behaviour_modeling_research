# Experiment Infrastructure

System for running and tracking large-scale experiments on the HPC cluster.

## Purpose

Replaces manual experiment management and large JSON files with:
- Automated SLURM job submission (single jobs or parameter sweeps)
- Efficient database tracking (queries are <10KB vs 180MB files)
- Monitoring and analysis tools

## Components

**`tracking/`** - Database for experiment tracking
- Stores run metadata, metrics, results, and errors
- 5 tables: runs, metrics, results, errors, slurm_jobs
- Enables context-efficient queries for analysis

**`cli/`** - Command-line tools
- `submit_jobs.py`: Submit experiments to cluster
- `monitor_jobs.py`: Monitor job status
- `query_results.py`: Analyze results and debug failures

**`slurm/`** - Cluster job management
- Templates for single jobs and job arrays
- Automated resource allocation
- Structured logging

## Benefits

- Scale to 1000+ parallel experiments
- LLM-friendly: queries fit in context window
- Complete error tracking for debugging
- Reproducible: all configs stored
