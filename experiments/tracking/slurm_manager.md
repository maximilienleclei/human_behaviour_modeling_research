# SLURM Job Manager

Manager for submitting SLURM jobs and monitoring their status with database integration.

## What it's for

Handles SLURM job submission (single jobs and array sweeps), script generation from templates, and job status monitoring. Links SLURM jobs to database runs for tracking and maintains job array configurations.

## What it contains

### Main Class
- `SlurmManager` - Manager with methods for job submission and monitoring

### Submission Methods
- `submit_single_job()` - Submits single SLURM job with specified parameters
- `submit_sweep()` - Generates job array for full parameter sweep (all combinations)
- `_generate_job_script()` - Creates SLURM script from template with variable substitution
- `_save_array_config()` - Saves job array parameter mapping to JSON file

### Monitoring Methods
- `monitor_jobs()` - Queries database and returns status summary (pending/running/completed/failed/timeout counts)
- `_update_job_statuses()` - Updates database run statuses based on SLURM job states (via `scontrol`)

## Key Details

Uses Jinja2-style templates from template_dir to generate SLURM batch scripts with resource specifications (time, GPU, memory, CPUs). Job array sweeps create all parameter combinations (dataset × method × subject × CL × seed) and save mapping to config_dir as JSON for array task indexing. Logs are written to log_dir with naming pattern: {experiment}_{job_id}_{array_task_id}.out. The manager calls `sbatch` via subprocess to submit jobs and captures SLURM job IDs. Links jobs to database via experiments/tracking/database.py by storing SLURM job ID and array task ID in run entries. Monitoring updates run statuses by comparing database state with SLURM state (squeue/scontrol). Used by experiments/cli/submit_jobs.py and experiments/cli/monitor_jobs.py.
