# Usage Guide

**Last Updated:** December 11, 2024

Complete reference for running experiments, monitoring jobs, and querying results.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Platform Runner](#platform-runner)
3. [Common Workflows](#common-workflows)
4. [Cluster Usage (SLURM)](#cluster-usage-slurm)
5. [Monitoring & Results](#monitoring--results)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Working Example

```bash
# 30-second smoke test
python -m platform.runner \
  --dataset cartpole \
  --method test \
  --model reservoir \
  --optimizer sgd \
  --max-time 30 \
  --no-logger
```

### Quick Test (10 minutes)

```bash
# SGD on human CartPole data
python -m platform.runner \
  --dataset cartpole \
  --method SGD_test \
  --model reservoir \
  --optimizer sgd \
  --subject sub01 \
  --max-time 600
```

### Full Training Run

```bash
# Full 10-hour training with continual learning features
python -m platform.runner \
  --dataset lunarlander \
  --method GA_trainable_CL \
  --model trainable \
  --optimizer ga \
  --use-cl-info \
  --subject sub01 \
  --max-time 36000
```

---

## Platform Runner

The main entry point is `platform/runner.py`. Run experiments via:

```bash
python -m platform.runner [OPTIONS]
```

### Required Arguments

#### --dataset
Specify which dataset to use.

**Human behavioral data:**
- `cartpole` - CartPole-v1 human gameplay
- `mountaincar` - MountainCar-v0 human gameplay
- `acrobot` - Acrobot-v1 human gameplay
- `lunarlander` - LunarLander-v2 human gameplay

**HuggingFace datasets:**
- `HF:CartPole-v1` - Expert RL trajectories
- `HF:LunarLander-v2` - Expert RL trajectories

**Example:**
```bash
--dataset cartpole                # Human data
--dataset HF:CartPole-v1          # HuggingFace data
```

#### --method
Experiment method name (for tracking and checkpoint naming).

**Format:** `{OPTIMIZER}_{MODEL}[_CL]`

**Examples:**
```bash
--method SGD_reservoir            # SGD with reservoir model
--method GA_trainable_CL          # GA with trainable + CL features
--method ES_mlp                   # ES with feedforward MLP
```

**Note:** This is just a string identifier, not parsed by the system.

#### --model
Neural network architecture.

**Options:**
- `mlp` - Two-layer feedforward network (~150 params)
- `reservoir` - Recurrent with frozen reservoir (~452 params)
- `trainable` - Recurrent with trainable weights (~552 params)
- `dynamic` - Evolving topology (GA only, experimental)

**Examples:**
```bash
--model reservoir                 # Recurrent reservoir
--model mlp                       # Simple feedforward
```

#### --optimizer
Training algorithm.

**Options:**
- `sgd` - Stochastic gradient descent (Adam optimizer)
- `ga` - Genetic algorithm (mutation-based evolution)
- `es` - Evolution strategies
- `cmaes` - Covariance Matrix Adaptation ES

**Compatibility:**
- All models work with `sgd`, `ga`, `es`, `cmaes`
- Exception: `dynamic` only works with `ga`

**Examples:**
```bash
--optimizer sgd                   # Gradient descent
--optimizer ga                    # Genetic algorithm
--optimizer cmaes                 # CMA-ES
```

---

### Optional Arguments

#### Data & Features

**--subject**
Subject identifier for human behavioral data (required for human data).

```bash
--subject sub01                   # Subject 1
--subject sub02                   # Subject 2
```

**--use-cl-info**
Include continual learning features (session ID, run ID).

Only works with human data (has temporal structure).

```bash
--use-cl-info                     # Add CL features
```

#### Model Configuration

**--hidden-size**
Hidden layer dimension (default: 50).

```bash
--hidden-size 50                  # Default
--hidden-size 100                 # Larger hidden layer
```

#### Training Configuration

**--max-time**
Maximum training time in seconds (default: 36000 = 10 hours).

```bash
--max-time 600                    # 10 minutes
--max-time 3600                   # 1 hour
--max-time 36000                  # 10 hours (default)
```

**--seed**
Random seed for reproducibility.

```bash
--seed 42                         # Reproducible results
```

**--batch-size** (SGD only)
Batch size for gradient descent (default: 32).

```bash
--batch-size 32                   # Default
--batch-size 64                   # Larger batches
```

**--population-size** (GA/ES/CMA-ES only)
Population size for evolutionary methods (default: 50).

```bash
--population-size 50              # Default
--population-size 100             # Larger population
```

#### System Configuration

**--gpu**
GPU index to use (-1 for CPU, default: 0).

```bash
--gpu 0                           # Use GPU 0
--gpu 1                           # Use GPU 1
--gpu -1                          # Use CPU
```

**--no-logger**
Disable experiment tracking database (useful for quick tests).

```bash
--no-logger                       # No database logging
```

---

### Complete Example

```bash
python -m platform.runner \
  --dataset lunarlander \
  --method GA_trainable_CL_full \
  --model trainable \
  --optimizer ga \
  --subject sub01 \
  --use-cl-info \
  --hidden-size 50 \
  --max-time 36000 \
  --seed 42 \
  --population-size 100 \
  --gpu 0
```

---

## Common Workflows

### 1. Quick Smoke Test (30 seconds)

**Purpose:** Verify imports and basic functionality.

```bash
cd /scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research

python -m platform.runner \
  --dataset cartpole \
  --method smoke_test \
  --model reservoir \
  --optimizer sgd \
  --max-time 30 \
  --no-logger
```

**Expected:** Training runs without errors, creates checkpoint in `results/`.

---

### 2. Development Test (10 minutes)

**Purpose:** Test changes before long runs.

```bash
python -m platform.runner \
  --dataset cartpole \
  --method dev_test_$(date +%s) \
  --model reservoir \
  --optimizer sgd \
  --subject sub01 \
  --max-time 600 \
  --seed 42
```

**Note:** Using timestamp in method name prevents checkpoint conflicts.

---

### 3. Full Training Run (10 hours)

**Purpose:** Production experiment.

```bash
python -m platform.runner \
  --dataset lunarlander \
  --method GA_trainable_CL \
  --model trainable \
  --optimizer ga \
  --use-cl-info \
  --subject sub01 \
  --max-time 36000 \
  --seed 42
```

**Checkpointing:** Saves every ~300 seconds to `results/`.

**Resume:** If interrupted, rerun same command to resume from checkpoint.

---

### 4. Reproduce Experiment 4 Setup

**Purpose:** Validate platform matches old experiments.

```bash
python -m platform.runner \
  --dataset cartpole \
  --method SGD_reservoir_exp4_validation \
  --model reservoir \
  --optimizer sgd \
  --use-cl-info \
  --subject sub01 \
  --seed 42 \
  --hidden-size 50 \
  --max-time 36000
```

---

### 5. Parameter Sweep (Manual)

**Purpose:** Test multiple configurations.

```bash
# Loop over seeds
for seed in 42 43 44 45 46; do
  python -m platform.runner \
    --dataset cartpole \
    --method SGD_reservoir_seed${seed} \
    --model reservoir \
    --optimizer sgd \
    --subject sub01 \
    --seed $seed \
    --max-time 3600
done
```

**Note:** For large sweeps, use SLURM job arrays (see below).

---

### 6. Compare Optimizers

**Purpose:** Run same config with different optimizers.

```bash
# SGD
python -m platform.runner \
  --dataset cartpole --method SGD_comparison \
  --model reservoir --optimizer sgd \
  --subject sub01 --max-time 3600 --seed 42

# GA
python -m platform.runner \
  --dataset cartpole --method GA_comparison \
  --model reservoir --optimizer ga \
  --subject sub01 --max-time 3600 --seed 42

# ES
python -m platform.runner \
  --dataset cartpole --method ES_comparison \
  --model reservoir --optimizer es \
  --subject sub01 --max-time 3600 --seed 42

# CMA-ES
python -m platform.runner \
  --dataset cartpole --method CMAES_comparison \
  --model reservoir --optimizer cmaes \
  --subject sub01 --max-time 3600 --seed 42
```

---

## Cluster Usage (SLURM)

For running large-scale experiments on a compute cluster.

### Submit Single Job

```bash
python experiments/cli/submit_jobs.py \
  --dataset cartpole \
  --method SGD_reservoir \
  --model reservoir \
  --optimizer sgd \
  --subject sub01 \
  --use-cl-info \
  --seed 42
```

**SLURM Configuration (from CLAUDE.md):**
```bash
--time=00:30:00                   # 30 minutes
--gpus=h100_1g.10gb:1            # 1 GPU
--account=rrg-pbellec            # Account name
--mem=15G                        # 15GB memory
--cpus-per-task=2                # 2 CPUs
```

### Submit Job Array (Parameter Sweep)

**Example:** Run 5 seeds in parallel

```bash
# Create sweep config (manual for now)
for seed in 42 43 44 45 46; do
  python experiments/cli/submit_jobs.py \
    --dataset cartpole \
    --method SGD_reservoir_seed${seed} \
    --model reservoir \
    --optimizer sgd \
    --subject sub01 \
    --seed $seed
done
```

**Future:** YAML config support for easier sweeps.

### Check Job Status

```bash
# SLURM native
squeue -u $USER

# Or use tracking system
python experiments/cli/monitor_jobs.py
```

### Cancel Jobs

```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel jobs matching name
scancel -n <job_name>
```

---

## Monitoring & Results

### Real-Time Monitoring

```bash
# Watch mode (auto-refresh every 10s)
python experiments/cli/monitor_jobs.py --watch

# Single check
python experiments/cli/monitor_jobs.py
```

**Output:**
- Job status (running, completed, failed)
- Current metrics (loss, F1 score)
- Time elapsed
- Error messages (if failed)

---

### Query Results

#### Summary Statistics

```bash
python experiments/cli/query_results.py \
  --experiment 5 \
  --summary
```

**Output:**
- Mean/std of final metrics across runs
- Best/worst runs
- Completion rate

#### Filter by Method

```bash
python experiments/cli/query_results.py \
  --experiment 5 \
  --method SGD_reservoir
```

#### Export to CSV

```bash
python experiments/cli/query_results.py \
  --experiment 5 \
  --output results.csv
```

#### Compare Methods

```bash
python experiments/cli/query_results.py \
  --experiment 5 \
  --compare-methods SGD_reservoir GA_trainable
```

---

### Manual Checkpoint Inspection

Checkpoints are saved to `results/` directory:

```bash
# List checkpoints
ls -lh results/

# Load checkpoint in Python
import torch
checkpoint = torch.load("results/cartpole_SGD_reservoir_sub01_checkpoint.pt")
print(checkpoint.keys())  # ['model_state', 'optimizer_state', 'epoch', 'elapsed_time', ...]
```

---

## Troubleshooting

### Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'platform'
```

**Solution:**
```bash
# Make sure you're in project root
cd /scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research

# Verify imports work
python -c "from platform.models import RecurrentMLPReservoir; print('OK')"
```

---

### GPU Errors

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size (SGD):**
   ```bash
   --batch-size 16                 # Instead of 32
   ```

2. **Reduce population size (GA/ES):**
   ```bash
   --population-size 25            # Instead of 50
   ```

3. **Use CPU:**
   ```bash
   --gpu -1
   ```

4. **Check GPU availability:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   nvidia-smi
   ```

---

### Data Not Found

**Problem:**
```
FileNotFoundError: data/sub01_data_cartpole.json not found
```

**Solution:**

1. **Check data directory exists:**
   ```bash
   ls -la data/
   ```

2. **Verify file names match:**
   ```
   data/
   ├── sub01_data_cartpole.json
   ├── sub01_data_mountaincar.json
   ├── sub01_data_acrobot.json
   ├── sub01_data_lunarlander.json
   ├── sub02_data_cartpole.json
   └── ...
   ```

3. **Use HuggingFace data instead:**
   ```bash
   --dataset HF:CartPole-v1        # No local files needed
   ```

---

### Checkpoint Conflicts

**Problem:**
```
Resuming from checkpoint... (but you want to start fresh)
```

**Solutions:**

1. **Delete checkpoint:**
   ```bash
   rm results/cartpole_SGD_reservoir_sub01_checkpoint.pt
   ```

2. **Use different method name:**
   ```bash
   --method SGD_reservoir_v2       # Creates new checkpoint
   ```

3. **Checkpoint naming:**
   ```
   results/{dataset}_{method}_{subject}_checkpoint.pt
   ```

---

### Type Errors (jaxtyping/beartype)

**Problem:**
```
beartype.roar.BeartypeCallHintParamViolation: ... expected Float[Tensor, "BS 4"] but got Float[Tensor, "1 4"]
```

**Explanation:**
- Type checking caught a tensor shape mismatch
- Expected batch dimension (BS) but got singleton dimension (1)

**Solution:**
- This usually indicates a bug in the code
- Check that data loading preserves batch dimensions
- Verify model input/output shapes match expected

---

### Training Not Converging

**Problem:**
- Loss not decreasing
- F1 score stays at 0

**Debugging Steps:**

1. **Check data loaded correctly:**
   ```bash
   python -c "
   from platform.data.loaders import load_human_data
   obs, act, _, _, _ = load_human_data('cartpole', False, 'sub01')
   print(f'Train: {obs.shape}, {act.shape}')
   print(f'Obs range: [{obs.min():.2f}, {obs.max():.2f}]')
   print(f'Actions: {set(act.tolist())}')
   "
   ```

2. **Reduce problem size:**
   ```bash
   --dataset cartpole              # Simplest environment
   --model mlp                     # Simplest model
   --optimizer sgd                 # Most reliable
   ```

3. **Check for NaNs:**
   - Look for `loss: nan` in output
   - May indicate learning rate too high or gradient explosion

4. **Increase training time:**
   ```bash
   --max-time 7200                 # 2 hours instead of 1
   ```

---

### SLURM Job Fails Immediately

**Problem:**
- Job submitted but fails within seconds

**Debugging Steps:**

1. **Check SLURM output:**
   ```bash
   cat slurm-<job_id>.out          # Standard output
   cat slurm-<job_id>.err          # Error output
   ```

2. **Check tracking database:**
   ```bash
   python experiments/cli/query_results.py --errors
   ```

3. **Test locally first:**
   ```bash
   # Run same command without SLURM
   python -m platform.runner ... --max-time 60 --no-logger
   ```

4. **Common issues:**
   - Wrong account name
   - Insufficient memory
   - GPU not available
   - Python environment not activated

---

## Advanced Usage

### Custom Data Files

To add your own human behavioral data:

1. **Format:** JSON file with structure:
   ```json
   {
     "observations": [[...], [...], ...],
     "actions": [0, 1, 0, ...],
     "rewards": [1.0, 1.0, 0.0, ...],
     "timestamps": [0.0, 0.1, 0.2, ...]
   }
   ```

2. **Naming:** `{subject}_data_{env}.json`

3. **Location:** `data/` directory

4. **Usage:**
   ```bash
   --dataset {env} --subject {subject}
   ```

---

### Custom Models

To add a new model architecture:

1. **Create:** `platform/models/my_model.py`
   ```python
   class MyModel(nn.Module):
       def forward(self, obs):
           return actions
   ```

2. **Register:** Add to `platform/models/__init__.py`
   ```python
   from .my_model import MyModel

   def create_model(model_type, ...):
       if model_type == "mymodel":
           return MyModel(...)
   ```

3. **Use:**
   ```bash
   --model mymodel
   ```

---

### Custom Optimizers

To add a new optimization method:

1. **Create:** `platform/optimizers/my_optimizer.py`
   ```python
   def optimize_my_method(model, train_obs, train_actions, ...):
       # Training loop
       return final_model
   ```

2. **Register:** Add to `platform/runner.py`
   ```python
   if args.optimizer == "myoptimizer":
       model = optimize_my_method(...)
   ```

3. **Use:**
   ```bash
   --optimizer myoptimizer
   ```

---

## Environment Variables

**PYTHONPATH:**
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/human_behaviour_modeling_research"
```

**CUDA_VISIBLE_DEVICES:**
```bash
export CUDA_VISIBLE_DEVICES=0     # Use only GPU 0
```

---

## Summary

**Quick Reference:**
```bash
# Test
python -m platform.runner --dataset cartpole --method test \
  --model reservoir --optimizer sgd --max-time 60 --no-logger

# Development
python -m platform.runner --dataset cartpole --method dev \
  --model reservoir --optimizer sgd --subject sub01 --max-time 600

# Production
python -m platform.runner --dataset lunarlander --method production \
  --model trainable --optimizer ga --use-cl-info --subject sub01 --max-time 36000

# Cluster
python experiments/cli/submit_jobs.py --dataset cartpole --method cluster_test \
  --model reservoir --optimizer sgd --subject sub01
```

---

**Related Documentation:**
- [Architecture Guide](ARCHITECTURE.md) - System design
- [Navigation Guide](NAVIGATION.md) - Where to find code
- [Status](STATUS.md) - Current implementation status
