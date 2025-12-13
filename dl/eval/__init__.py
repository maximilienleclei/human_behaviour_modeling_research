"""Evaluation metrics and behavioral comparison functions.

This module provides functions for computing training/test metrics and comparing
model behavior to human behavior in environment rollouts.

Functions:
    compute_cross_entropy: Compute cross-entropy loss
    compute_macro_f1: Compute macro F1 score
    evaluate_progression_recurrent: Evaluate recurrent model on environment episodes
"""

__all__ = [
    "compute_cross_entropy",
    "compute_macro_f1",
    "evaluate_progression_recurrent",
]
