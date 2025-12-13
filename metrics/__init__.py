"""Shared evaluation metrics and behavioral comparison module.

Provides metric computation functions and behavioral evaluation used by both
dl/ and ne/ modules.

Functions:
    compute_cross_entropy: Compute cross-entropy loss
    compute_macro_f1: Compute macro F1 score
    evaluate_progression_recurrent: Evaluate recurrent model on environment episodes
"""

from metrics.metrics import compute_cross_entropy, compute_macro_f1
from metrics.comparison import evaluate_progression_recurrent

__all__ = [
    "compute_cross_entropy",
    "compute_macro_f1",
    "evaluate_progression_recurrent",
]
