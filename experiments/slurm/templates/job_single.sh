#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/slurm-%j.out
#SBATCH --error={log_dir}/slurm-%j.err
#SBATCH --time={time_limit}
#SBATCH --gpus={gpu_type}
#SBATCH --account={account}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}

# Print job info
echo "===================================="
echo "SLURM Job Information"
echo "===================================="
echo "Job ID: $SLURM_JOB_ID"
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

# Change to experiment directory
cd {experiment_dir}

# Print Python info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Run experiment
echo "===================================="
echo "Running experiment..."
echo "===================================="
python -u main.py \
    --dataset {dataset} \
    --method {method} \
    --subject {subject} \
    {use_cl_flag} \
    --seed {seed} \
    --gpu 0

# Capture exit code
EXIT_CODE=$?

echo ""
echo "===================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "===================================="

exit $EXIT_CODE
