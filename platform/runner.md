# Platform Experiment Runner

Main entry point for running experiments with automatic data loading, model creation, and optimizer dispatch.

## What it's for

Provides unified CLI for running all experiment configurations. Handles the complete experiment pipeline: seeds, data loading, model instantiation, optimizer selection, checkpoint management, and optional database logging.

## What it contains

### Core Functions
- `run_experiment()` - Main orchestration function that coordinates data, model, and optimizer
- `load_data()` - Loads data from HuggingFace or local JSON files with optional CL features
- `create_model()` - Factory function for instantiating model by type (reservoir, trainable, mlp)
- `set_random_seeds()` - Sets reproducible random seeds across all libraries
- `main()` - CLI entry point with argument parsing

### Method Dispatch Logic
Parses method names to extract model_type and optimizer_type:
- SGD methods: "SGD_reservoir", "SGD_trainable" → model + SGD optimizer
- GA methods: "adaptive_ga_reservoir", "adaptive_ga_trainable", "adaptive_ga_dynamic" → model + evolutionary optimizer
- ES methods: Similar patterns for evolution strategies

## Key Details

The runner automatically determines which optimizer to use based on method name patterns (SGD vs GA vs ES) and dispatches to appropriate optimizer function from platform/optimizers/. Supports both HuggingFace datasets (prefix "HF:") and local human data (cartpole, mountaincar, acrobot, lunarlander). CL features (session/run IDs) are only added for human data. Creates checkpoint paths using RESULTS_DIR/exp{N}/{dataset}/{method}/seed{S}. Integrates with experiments/tracking/logger.py when database logging is enabled. Handles both recurrent models (with episode boundaries in metadata) and feedforward models. Referenced by experiments/cli/submit_jobs.py which wraps it in SLURM scripts.
