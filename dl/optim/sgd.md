# SGD Optimizer

Gradient-based optimization using backpropagation with cross-entropy loss between model predictions and human actions.

Contains optimize_sgd() handling feedforward and recurrent models. Uses batched gradient descent with configurable batch_size and learning_rate. Runs for time budget (default 10 hours) with periodic evaluations: loss/F1 every 60s, checkpoints/behavioral comparison every 300s. For recurrent models, creates episode batches via EpisodeDataset (data/simexp_control_tasks/preprocessing.py) preserving temporal structure and hidden state resets. Supports optional database logging. Checkpoints include model state, optimizer state, and iteration count for resumption.
