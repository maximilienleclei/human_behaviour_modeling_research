#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/slurm-%A_%a.out
#SBATCH --error={log_dir}/slurm-%A_%a.err
#SBATCH --array=0-{max_task_id}%{max_concurrent}
#SBATCH --time={time_limit}
#SBATCH --gpus={gpu_type}
#SBATCH --account={account}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}

# Print job info
echo "===================================="
echo "SLURM Job Array Information"
echo "===================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "===================================="

# Environment setup
module load python/3.11

# Activate virtual environment
if [ -d "/scratch/mleclei/venv" ]; then
    source /scratch/mleclei/venv/bin/activate
else
    echo "Warning: Virtual environment not found at /scratch/mleclei/venv, using system Python"
fi

# Change to project root
cd {project_root}

# Print Python info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Read configuration for this array task
CONFIG_FILE="{config_file}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Reading configuration from: $CONFIG_FILE"

# Parse JSON config and build command
python -c "
import json
import sys

with open('$CONFIG_FILE') as f:
    configs = json.load(f)

task_id = int('$SLURM_ARRAY_TASK_ID')

if task_id >= len(configs):
    print(f'Error: Task ID {task_id} out of range (0-{len(configs)-1})', file=sys.stderr)
    sys.exit(1)

config = configs[task_id]

# Build command
cmd_parts = [
    'python', '-u', '{experiment_main}',
    '--dataset', config['dataset'],
    '--method', config['method'],
    '--subject', config['subject'],
    '--seed', str(config['seed']),
    '--gpu', '0',
]

if config.get('use_cl_info', False):
    cmd_parts.append('--use-cl-info')

print(' '.join(cmd_parts))
" > /tmp/cmd_$SLURM_ARRAY_JOB_ID\_$SLURM_ARRAY_TASK_ID.sh

# Execute the command
echo "===================================="
echo "Running experiment..."
echo "Command: $(cat /tmp/cmd_$SLURM_ARRAY_JOB_ID\_$SLURM_ARRAY_TASK_ID.sh)"
echo "===================================="

bash /tmp/cmd_$SLURM_ARRAY_JOB_ID\_$SLURM_ARRAY_TASK_ID.sh

# Capture exit code
EXIT_CODE=$?

# Cleanup
rm -f /tmp/cmd_$SLURM_ARRAY_JOB_ID\_$SLURM_ARRAY_TASK_ID.sh

echo ""
echo "===================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "===================================="

exit $EXIT_CODE
