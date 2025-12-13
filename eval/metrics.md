# Evaluation Metrics

Metric computation functions for model performance evaluation.

## Purpose
Provides standardized metrics (cross-entropy, F1 score) used across both dl/ and ne/ modules for evaluating model predictions.

## Contents
- `compute_cross_entropy()` - Computes cross-entropy loss given model, observations, and target actions
- `compute_macro_f1()` - Computes macro F1 score with multiple sampling trials for stochastic policies
