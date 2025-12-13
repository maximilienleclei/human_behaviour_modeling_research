# Environment-Based Evaluation Orchestration

High-level training interface for TorchRL environment tasks using neuroevolution.

Contains create_env_fitness_evaluator() creating fitness closure for environment rollouts (metrics: "return" for RL, "cross_entropy" for imitation), and train_environment() orchestrating episode rollouts and optimization. Accepts state_config for continual learning modes (env_transfer, mem_transfer, fit_transfer). Same architecture as supervised learning but uses environment rollouts instead of static datasets. Optimizer never sees environment or episode data, only fitness values. Returns training results after optimization loop.
