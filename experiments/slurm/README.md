# SLURM Job Management

Templates and configurations for running experiments on the HPC cluster.

## Structure

- `templates/`: Bash script templates for SLURM jobs
  - `job_single.sh`: Single experiment submission
  - `job_array.sh`: Parameter sweeps (runs many configs in parallel)
- `configs/`: Generated JSON configs for job arrays (auto-created)
- `logs/`: SLURM output and error logs (auto-created)

## What templates do

1. Load Python environment and activate venv
2. Set up SLURM resources (GPU, memory, time limit)
3. Run experiment with specified parameters
4. Log job metadata (ID, node, timestamps, exit codes)

## Default resources (Compute Canada)

- GPU: `h100_1g.10gb:1` (10GB GPU memory)
- Memory: `15G` RAM
- CPUs: `2` cores
- Time: `00:30:00` (30 minutes)
- Account: `rrg-pbellec`

Resources are configurable per job based on experiment needs.
