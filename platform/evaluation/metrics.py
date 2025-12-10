"""Metric computation functions for evaluating model performance.

This module provides functions for computing training/test metrics including
cross-entropy loss and F1 scores.
"""

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from sklearn.metrics import f1_score
from torch import Tensor, nn


def compute_cross_entropy(
    model: nn.Module,
    observations: Float[Tensor, "N input_size"],
    actions: Int[Tensor, " N"],
) -> Float[Tensor, ""]:
    """Compute cross-entropy loss.
    
    Args:
        model: Neural network model with forward() method
        observations: Input observations
        actions: Target actions
        
    Returns:
        Cross-entropy loss scalar
    """
    logits: Float[Tensor, "N output_size"] = model(observations)
    loss: Float[Tensor, ""] = F.cross_entropy(logits, actions)
    return loss


def compute_macro_f1(
    model: nn.Module,
    observations: Float[Tensor, "N input_size"],
    actions: Int[Tensor, " N"],
    num_samples: int = 10,
    num_classes: int = 2,
) -> float:
    """Compute macro F1 score with multiple sampling trials.
    
    Args:
        model: Neural network model with get_probs() method
        observations: Input observations
        actions: Target actions
        num_samples: Number of sampling trials for averaging
        num_classes: Number of action classes
        
    Returns:
        Mean F1 score across sampling trials
    """
    probs: Float[Tensor, "N output_size"] = model.get_probs(observations)

    f1_scores: list[float] = []
    for _ in range(num_samples):
        sampled_actions: Int[Tensor, " N"] = torch.multinomial(
            probs, num_samples=1
        ).squeeze(-1)
        f1: float = f1_score(
            actions.cpu().numpy(),
            sampled_actions.cpu().numpy(),
            average="macro",
            labels=list(range(num_classes)),
            zero_division=0.0,
        )
        f1_scores.append(f1)

    return float(np.mean(f1_scores))
