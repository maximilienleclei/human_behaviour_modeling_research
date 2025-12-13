"""Recurrent neural network models for behavior modeling.

This module contains recurrent MLP architectures with both frozen reservoir
and trainable recurrent connections.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class RecurrentMLPReservoir(nn.Module):
    """Recurrent MLP with frozen reservoir (echo state network style).

    Architecture: [input_size, 50 recurrent (frozen), output_size]

    Trainable Parameters (e.g., CartPole+CL: input=6, output=2):
    - W_ih: 6 × 50 = 300  (input-to-hidden weights)
    - b_h: 50             (hidden biases)
    - W_ho: 50 × 2 = 100  (hidden-to-output weights)
    - b_o: 2              (output biases)
    Total: 452 params (matches exp 3 baseline)

    Frozen Parameters:
    - W_hh: 50 × 50 = 2500 (recurrent weights, randomly initialized once)
    
    Works with both SGD and GA optimizers.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int
    ) -> None:
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size

        # Trainable parameters
        self.W_ih: nn.Linear = nn.Linear(input_size, hidden_size)
        self.W_ho: nn.Linear = nn.Linear(hidden_size, output_size)

        # Frozen recurrent weights (reservoir)
        W_hh: Float[Tensor, "hidden_size hidden_size"] = (
            torch.randn(hidden_size, hidden_size) / math.sqrt(hidden_size)
        )
        self.register_buffer("W_hh", W_hh)

    def forward_step(
        self,
        x: Float[Tensor, "BS input_size"],
        h: Float[Tensor, "BS hidden_size"],
    ) -> tuple[Float[Tensor, "BS output_size"], Float[Tensor, "BS hidden_size"]]:
        """Single timestep forward pass.

        Args:
            x: Input observations [batch_size, input_size]
            h: Previous hidden state [batch_size, hidden_size]

        Returns:
            logits: Output logits [batch_size, output_size]
            h_new: New hidden state [batch_size, hidden_size]
        """
        h_new: Float[Tensor, "BS hidden_size"] = torch.tanh(
            self.W_ih(x) + h @ self.W_hh
        )
        logits: Float[Tensor, "BS output_size"] = self.W_ho(h_new)
        return logits, h_new

    def forward(
        self,
        x: Float[Tensor, "BS seq_len input_size"],
        h_0: Float[Tensor, "BS hidden_size"] | None = None,
    ) -> tuple[Float[Tensor, "BS seq_len output_size"], Float[Tensor, "BS hidden_size"]]:
        """Sequence forward pass.

        Args:
            x: Input sequence [batch_size, seq_len, input_size]
            h_0: Initial hidden state [batch_size, hidden_size] or None

        Returns:
            logits: Output logits [batch_size, seq_len, output_size]
            h_final: Final hidden state [batch_size, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        if h_0 is None:
            h: Float[Tensor, "BS hidden_size"] = torch.zeros(
                batch_size, self.hidden_size, device=x.device
            )
        else:
            h = h_0

        logits_seq: list[Float[Tensor, "BS output_size"]] = []
        for t in range(seq_len):
            logits_t, h = self.forward_step(x[:, t], h)
            logits_seq.append(logits_t)

        logits: Float[Tensor, "BS seq_len output_size"] = torch.stack(
            logits_seq, dim=1
        )
        return logits, h

    def get_probs(
        self, x: Float[Tensor, "BS input_size"], h: Float[Tensor, "BS hidden_size"]
    ) -> tuple[Float[Tensor, "BS output_size"], Float[Tensor, "BS hidden_size"]]:
        """Get probability distribution over actions for a single step."""
        logits, h_new = self.forward_step(x, h)
        probs: Float[Tensor, "BS output_size"] = F.softmax(logits, dim=-1)
        return probs, h_new


class RecurrentMLPTrainable(nn.Module):
    """Recurrent MLP with trainable recurrent weights (rank-1 factorization).

    Architecture: [input_size, 50 recurrent (trainable), output_size]

    Trainable Parameters (e.g., CartPole+CL: input=6, output=2):
    - W_ih: 6 × 50 = 300        (input-to-hidden weights)
    - b_h: 50                   (hidden biases)
    - W_hh = u ⊗ v^T: 50+50=100 (rank-1 recurrent matrix)
    - W_ho: 50 × 2 = 100        (hidden-to-output weights)
    - b_o: 2                    (output biases)
    Total: 552 params (~22% more than baseline)
    
    Works with both SGD and GA optimizers.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int
    ) -> None:
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size

        # Trainable parameters
        self.W_ih: nn.Linear = nn.Linear(input_size, hidden_size)

        # Rank-1 factorization of recurrent matrix: W_hh = u ⊗ v^T
        self.u: nn.Parameter = nn.Parameter(
            torch.randn(hidden_size) / math.sqrt(hidden_size)
        )
        self.v: nn.Parameter = nn.Parameter(
            torch.randn(hidden_size) / math.sqrt(hidden_size)
        )

        self.W_ho: nn.Linear = nn.Linear(hidden_size, output_size)

    def forward_step(
        self,
        x: Float[Tensor, "BS input_size"],
        h: Float[Tensor, "BS hidden_size"],
    ) -> tuple[Float[Tensor, "BS output_size"], Float[Tensor, "BS hidden_size"]]:
        """Single timestep forward pass.

        Args:
            x: Input observations [batch_size, input_size]
            h: Previous hidden state [batch_size, hidden_size]

        Returns:
            logits: Output logits [batch_size, output_size]
            h_new: New hidden state [batch_size, hidden_size]
        """
        # Efficient rank-1 computation: (u ⊗ v^T) @ h = u * (v^T @ h)
        v_dot_h: Float[Tensor, " BS"] = (h * self.v.unsqueeze(0)).sum(dim=1)
        W_hh_h: Float[Tensor, "BS hidden_size"] = self.u.unsqueeze(0) * v_dot_h.unsqueeze(1)

        h_new: Float[Tensor, "BS hidden_size"] = torch.tanh(
            self.W_ih(x) + W_hh_h
        )
        logits: Float[Tensor, "BS output_size"] = self.W_ho(h_new)
        return logits, h_new

    def forward(
        self,
        x: Float[Tensor, "BS seq_len input_size"],
        h_0: Float[Tensor, "BS hidden_size"] | None = None,
    ) -> tuple[Float[Tensor, "BS seq_len output_size"], Float[Tensor, "BS hidden_size"]]:
        """Sequence forward pass.

        Args:
            x: Input sequence [batch_size, seq_len, input_size]
            h_0: Initial hidden state [batch_size, hidden_size] or None

        Returns:
            logits: Output logits [batch_size, seq_len, output_size]
            h_final: Final hidden state [batch_size, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        if h_0 is None:
            h: Float[Tensor, "BS hidden_size"] = torch.zeros(
                batch_size, self.hidden_size, device=x.device
            )
        else:
            h = h_0

        logits_seq: list[Float[Tensor, "BS output_size"]] = []
        for t in range(seq_len):
            logits_t, h = self.forward_step(x[:, t], h)
            logits_seq.append(logits_t)

        logits: Float[Tensor, "BS seq_len output_size"] = torch.stack(
            logits_seq, dim=1
        )
        return logits, h

    def get_probs(
        self, x: Float[Tensor, "BS input_size"], h: Float[Tensor, "BS hidden_size"]
    ) -> tuple[Float[Tensor, "BS output_size"], Float[Tensor, "BS hidden_size"]]:
        """Get probability distribution over actions for a single step."""
        logits, h_new = self.forward_step(x, h)
        probs: Float[Tensor, "BS output_size"] = F.softmax(logits, dim=-1)
        return probs, h_new
