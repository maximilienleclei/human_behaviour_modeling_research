# Evaluation Metrics

Metric computation functions for quantifying how well models match human behavior.

## What it's for

Provides standardized metrics for evaluating model performance during training and testing. Used by all optimizers to compute training loss and test performance.

## What it contains

### Metrics
- `compute_cross_entropy()` - Computes cross-entropy loss between model predictions and human actions
- `compute_macro_f1()` - Computes macro-averaged F1 score with multiple sampling trials

## Key Details

Cross-entropy is the primary training objective, measuring how well the model's probability distribution matches human action choices. Macro F1 score provides a balanced measure of classification performance across all action classes, averaged over multiple sampling trials to account for stochasticity in action selection. Both metrics operate on model outputs and require the model to have `forward()` (for cross-entropy) and `get_probs()` (for F1) methods.
