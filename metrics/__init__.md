# Metrics Module

Shared evaluation metrics and behavioral comparison functions used across both deep learning (dl/) and neuroevolution (ne/) modules.

## Overview

This directory provides evaluation functions for quantifying how well models match human behavior. Includes standard ML metrics (loss, F1) and behavior-specific metrics (return comparison).

## Files

### Standard Metrics
- **metrics.py** - Loss and classification metrics
  - `compute_cross_entropy()` - Cross-entropy loss (primary training objective)
  - `compute_macro_f1()` - Macro-averaged F1 score with sampling trials
  - Used during training to track test performance

### Behavioral Comparison
- **comparison.py** - Compare model vs human behavior in environment
  - `evaluate_progression_recurrent()` - Run model on matched episodes, compare returns
  - Uses episode seeds from human data for matched comparison

## Evaluation Philosophy

Models are evaluated on two dimensions:
1. **Predictive accuracy**: How well do model action probabilities match human choices? (CE loss, F1 score)
2. **Behavioral alignment**: How similar are model returns to human returns on same episodes? (percentage difference)

The first measures action-level imitation, the second measures outcome-level similarity.

## Usage

Metrics are called periodically during optimization to log training progress. Behavioral comparison runs less frequently due to computational cost of environment rollouts.
