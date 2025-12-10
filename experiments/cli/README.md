# Command-Line Interface

Three tools for managing experiments on the cluster.

## Tools

**`submit_jobs.py`** - Submit experiments to SLURM
- Can submit single jobs or full parameter sweeps
- Handles job arrays for parallel execution
- Configurable resources (time, memory, GPU)

**`monitor_jobs.py`** - Monitor job status
- Shows pending/running/completed/failed counts
- Lists currently running jobs
- Displays recent failures
- Can run continuously with auto-refresh

**`query_results.py`** - Analyze results
- Summarize experiment statistics
- Find best-performing models
- Compare methods on datasets
- Debug failed runs with error details
- Export results to CSV

All tools interact with the tracking database to read/write experiment data.
