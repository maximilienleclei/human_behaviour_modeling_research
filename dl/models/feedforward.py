"""Feedforward neural network models.

This module contains feedforward MLP architectures that work with both SGD and GA optimizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class MLP(nn.Module):
    """Two-layer MLP with tanh activations.
    
    Architecture: [input_size, hidden_size, output_size]
    Activation: tanh
    
    Works with both SGD (backprop) and GA (evolutionary) optimizers.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int
    ) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_size, hidden_size)
        self.fc2: nn.Linear = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: Float[Tensor, "BS input_size"]
    ) -> Float[Tensor, "BS output_size"]:
        """Forward pass returning logits."""
        h: Float[Tensor, "BS hidden_size"] = torch.tanh(self.fc1(x))
        logits: Float[Tensor, "BS output_size"] = self.fc2(h)
        return logits

    def get_probs(
        self, x: Float[Tensor, "BS input_size"]
    ) -> Float[Tensor, "BS output_size"]:
        """Get probability distribution over actions."""
        logits: Float[Tensor, "BS output_size"] = self.forward(x)
        probs: Float[Tensor, "BS output_size"] = F.softmax(logits, dim=-1)
        return probs
