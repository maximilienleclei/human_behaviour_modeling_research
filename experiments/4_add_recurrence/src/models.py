"""Model definitions for Experiment 4 - Recurrent and Dynamic Networks."""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from . import config


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


class BatchedRecurrentPopulation:
    """Batched population of recurrent networks for efficient GPU-parallel neuroevolution."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        pop_size: int,
        model_type: str = "reservoir",
        sigma_init: float = 1e-3,
        sigma_noise: float = 1e-2,
    ) -> None:
        """Initialize batched recurrent population.

        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
            output_size: Output dimension
            pop_size: Population size
            model_type: 'reservoir' (frozen W_hh) or 'trainable' (trainable W_hh)
            sigma_init: Initial mutation sigma
            sigma_noise: Noise for adaptive sigma
        """
        self.pop_size: int = pop_size
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size
        self.model_type: str = model_type
        self.sigma_init: float = sigma_init
        self.sigma_noise: float = sigma_noise

        # Initialize batched parameters [pop_size, ...]
        # Using Xavier initialization like nn.Linear default
        ih_std: float = (1.0 / input_size) ** 0.5
        ho_std: float = (1.0 / hidden_size) ** 0.5

        # Input-to-hidden weights and biases (trainable)
        self.W_ih_weight: Float[Tensor, "pop_size hidden_size input_size"] = (
            torch.randn(pop_size, hidden_size, input_size, device=config.DEVICE) * ih_std
        )
        self.W_ih_bias: Float[Tensor, "pop_size hidden_size"] = (
            torch.randn(pop_size, hidden_size, device=config.DEVICE) * ih_std
        )

        # Hidden-to-output weights and biases (trainable)
        self.W_ho_weight: Float[Tensor, "pop_size output_size hidden_size"] = (
            torch.randn(pop_size, output_size, hidden_size, device=config.DEVICE) * ho_std
        )
        self.W_ho_bias: Float[Tensor, "pop_size output_size"] = (
            torch.randn(pop_size, output_size, device=config.DEVICE) * ho_std
        )

        # Recurrent weights (frozen or trainable based on model_type)
        if model_type == "reservoir":
            # Frozen reservoir weights (not mutated)
            hh_std: float = 1.0 / math.sqrt(hidden_size)
            self.W_hh: Float[Tensor, "pop_size hidden_size hidden_size"] = (
                torch.randn(pop_size, hidden_size, hidden_size, device=config.DEVICE) * hh_std
            )
        elif model_type == "trainable":
            # Rank-1 factorization: W_hh = u ⊗ v^T (trainable)
            hh_std = 1.0 / math.sqrt(hidden_size)
            self.u: Float[Tensor, "pop_size hidden_size"] = (
                torch.randn(pop_size, hidden_size, device=config.DEVICE) * hh_std
            )
            self.v: Float[Tensor, "pop_size hidden_size"] = (
                torch.randn(pop_size, hidden_size, device=config.DEVICE) * hh_std
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Initialize adaptive sigmas for trainable parameters
        self.W_ih_weight_sigma: Float[Tensor, "pop_size hidden_size input_size"] = (
            torch.full_like(self.W_ih_weight, sigma_init)
        )
        self.W_ih_bias_sigma: Float[Tensor, "pop_size hidden_size"] = (
            torch.full_like(self.W_ih_bias, sigma_init)
        )
        self.W_ho_weight_sigma: Float[Tensor, "pop_size output_size hidden_size"] = (
            torch.full_like(self.W_ho_weight, sigma_init)
        )
        self.W_ho_bias_sigma: Float[Tensor, "pop_size output_size"] = (
            torch.full_like(self.W_ho_bias, sigma_init)
        )

        if model_type == "trainable":
            self.u_sigma: Float[Tensor, "pop_size hidden_size"] = (
                torch.full_like(self.u, sigma_init)
            )
            self.v_sigma: Float[Tensor, "pop_size hidden_size"] = (
                torch.full_like(self.v, sigma_init)
            )

    def forward_batch_step(
        self,
        x: Float[Tensor, "pop_size N input_size"],
        h: Float[Tensor, "pop_size N hidden_size"],
    ) -> tuple[Float[Tensor, "pop_size N output_size"], Float[Tensor, "pop_size N hidden_size"]]:
        """Single timestep forward pass for all networks in parallel.

        Args:
            x: Input for all networks [pop_size, N, input_size]
            h: Previous hidden states [pop_size, N, hidden_size]

        Returns:
            logits: Output logits [pop_size, N, output_size]
            h_new: New hidden states [pop_size, N, hidden_size]
        """
        # Input-to-hidden: h_new = tanh(W_ih @ x + W_hh @ h + b_h)
        ih_out: Float[Tensor, "pop_size N hidden_size"] = torch.bmm(
            x, self.W_ih_weight.transpose(-1, -2)
        )
        ih_out = ih_out + self.W_ih_bias.unsqueeze(1)

        # Recurrent connection
        if self.model_type == "reservoir":
            # Matrix multiplication with frozen W_hh
            hh_out: Float[Tensor, "pop_size N hidden_size"] = torch.bmm(
                h, self.W_hh.transpose(-1, -2)
            )
        elif self.model_type == "trainable":
            # Efficient rank-1: (u ⊗ v^T) @ h = u * (v^T @ h)
            # h: [pop_size, N, hidden_size], v: [pop_size, hidden_size]
            v_expanded: Float[Tensor, "pop_size N hidden_size"] = self.v.unsqueeze(1).expand(
                -1, x.shape[1], -1
            )
            v_dot_h: Float[Tensor, "pop_size N"] = (h * v_expanded).sum(dim=2)
            u_expanded: Float[Tensor, "pop_size N hidden_size"] = self.u.unsqueeze(1).expand(
                -1, x.shape[1], -1
            )
            hh_out = u_expanded * v_dot_h.unsqueeze(2)

        h_new: Float[Tensor, "pop_size N hidden_size"] = torch.tanh(ih_out + hh_out)

        # Hidden-to-output
        logits: Float[Tensor, "pop_size N output_size"] = torch.bmm(
            h_new, self.W_ho_weight.transpose(-1, -2)
        )
        logits = logits + self.W_ho_bias.unsqueeze(1)

        return logits, h_new

    def forward_batch_sequence(
        self,
        x: Float[Tensor, "seq_len input_size"],
        h_0: Float[Tensor, "pop_size hidden_size"] | None = None,
    ) -> tuple[Float[Tensor, "pop_size seq_len output_size"], Float[Tensor, "pop_size hidden_size"]]:
        """Batched forward pass for sequence across all networks.

        Args:
            x: Input sequence [seq_len, input_size]
            h_0: Initial hidden states [pop_size, hidden_size] or None

        Returns:
            logits: Output logits [pop_size, seq_len, output_size]
            h_final: Final hidden states [pop_size, hidden_size]
        """
        seq_len = x.shape[0]

        if h_0 is None:
            h: Float[Tensor, "pop_size 1 hidden_size"] = torch.zeros(
                self.pop_size, 1, self.hidden_size, device=x.device
            )
        else:
            h = h_0.unsqueeze(1)  # [pop_size, 1, hidden_size]

        all_logits: list[Float[Tensor, "pop_size 1 output_size"]] = []
        for t in range(seq_len):
            x_t: Float[Tensor, "pop_size 1 input_size"] = (
                x[t].unsqueeze(0).unsqueeze(0).expand(self.pop_size, 1, -1)
            )
            logits_t, h = self.forward_batch_step(x_t, h)
            all_logits.append(logits_t)

        logits: Float[Tensor, "pop_size seq_len output_size"] = torch.cat(
            all_logits, dim=1
        )
        h_final: Float[Tensor, "pop_size hidden_size"] = h.squeeze(1)

        return logits, h_final

    def evaluate_episodes(
        self,
        episodes: list[dict],
    ) -> Float[Tensor, " pop_size"]:
        """Evaluate fitness on complete episodes.

        Args:
            episodes: List of dicts with 'observations' and 'actions' tensors

        Returns:
            fitness: Mean cross-entropy across all episodes [pop_size]
        """
        episode_losses: list[Float[Tensor, " pop_size"]] = []

        with torch.no_grad():
            for episode in episodes:
                obs: Float[Tensor, "seq_len input_size"] = episode["observations"].to(
                    config.DEVICE
                )
                act: Int[Tensor, " seq_len"] = episode["actions"].to(config.DEVICE)

                # Reset hidden state for each episode
                h_0: Float[Tensor, "pop_size hidden_size"] = torch.zeros(
                    self.pop_size, self.hidden_size, device=config.DEVICE
                )

                # Forward pass
                logits, _ = self.forward_batch_sequence(obs, h_0)  # [pop_size, seq_len, output_size]

                # Compute cross-entropy for each network
                act_expanded: Int[Tensor, "pop_size seq_len"] = act.unsqueeze(0).expand(
                    self.pop_size, -1
                )

                # Reshape for cross_entropy
                flat_logits: Float[Tensor, "pop_sizexseq_len output_size"] = logits.reshape(
                    -1, self.output_size
                )
                flat_actions: Int[Tensor, " pop_sizexseq_len"] = act_expanded.reshape(-1)

                # Compute per-sample CE then reshape and mean per network
                per_sample_ce: Float[Tensor, " pop_sizexseq_len"] = F.cross_entropy(
                    flat_logits, flat_actions, reduction="none"
                )
                per_network_ce: Float[Tensor, "pop_size seq_len"] = per_sample_ce.view(
                    self.pop_size, -1
                )
                episode_loss: Float[Tensor, " pop_size"] = per_network_ce.mean(dim=1)

                episode_losses.append(episode_loss)

        # Mean across episodes
        fitness: Float[Tensor, " pop_size"] = torch.stack(episode_losses).mean(dim=0)
        return fitness

    def mutate(self) -> None:
        """Apply adaptive sigma mutations to all networks in parallel."""
        # Update W_ih_weight sigma and parameter
        xi: Float[Tensor, "pop_size hidden_size input_size"] = (
            torch.randn_like(self.W_ih_weight_sigma) * self.sigma_noise
        )
        self.W_ih_weight_sigma = self.W_ih_weight_sigma * (1 + xi)
        eps: Float[Tensor, "pop_size hidden_size input_size"] = (
            torch.randn_like(self.W_ih_weight) * self.W_ih_weight_sigma
        )
        self.W_ih_weight = self.W_ih_weight + eps

        # Update W_ih_bias sigma and parameter
        xi = torch.randn_like(self.W_ih_bias_sigma) * self.sigma_noise
        self.W_ih_bias_sigma = self.W_ih_bias_sigma * (1 + xi)
        eps = torch.randn_like(self.W_ih_bias) * self.W_ih_bias_sigma
        self.W_ih_bias = self.W_ih_bias + eps

        # Update W_ho_weight sigma and parameter
        xi = torch.randn_like(self.W_ho_weight_sigma) * self.sigma_noise
        self.W_ho_weight_sigma = self.W_ho_weight_sigma * (1 + xi)
        eps = torch.randn_like(self.W_ho_weight) * self.W_ho_weight_sigma
        self.W_ho_weight = self.W_ho_weight + eps

        # Update W_ho_bias sigma and parameter
        xi = torch.randn_like(self.W_ho_bias_sigma) * self.sigma_noise
        self.W_ho_bias_sigma = self.W_ho_bias_sigma * (1 + xi)
        eps = torch.randn_like(self.W_ho_bias) * self.W_ho_bias_sigma
        self.W_ho_bias = self.W_ho_bias + eps

        # If trainable recurrent weights, mutate them too
        if self.model_type == "trainable":
            # Update u sigma and parameter
            xi = torch.randn_like(self.u_sigma) * self.sigma_noise
            self.u_sigma = self.u_sigma * (1 + xi)
            eps = torch.randn_like(self.u) * self.u_sigma
            self.u = self.u + eps

            # Update v sigma and parameter
            xi = torch.randn_like(self.v_sigma) * self.sigma_noise
            self.v_sigma = self.v_sigma * (1 + xi)
            eps = torch.randn_like(self.v) * self.v_sigma
            self.v = self.v + eps

    def select_simple_ga(self, fitness: Float[Tensor, " pop_size"]) -> None:
        """Simple GA selection: top 50% survive and duplicate (vectorized)."""
        # Sort by fitness (minimize CE)
        sorted_indices: Int[Tensor, " pop_size"] = torch.argsort(fitness)

        # Top 50% survive
        num_survivors: int = self.pop_size // 2
        survivor_indices: Int[Tensor, " num_survivors"] = sorted_indices[:num_survivors]

        # Create replacement mapping
        num_losers: int = self.pop_size - num_survivors
        replacement_indices: Int[Tensor, " num_losers"] = survivor_indices[
            torch.arange(num_losers, device=config.DEVICE) % num_survivors
        ]

        # Full new indices
        new_indices: Int[Tensor, " pop_size"] = torch.cat(
            [survivor_indices, replacement_indices]
        )

        # Reorder parameters
        self.W_ih_weight = self.W_ih_weight[new_indices].clone()
        self.W_ih_bias = self.W_ih_bias[new_indices].clone()
        self.W_ho_weight = self.W_ho_weight[new_indices].clone()
        self.W_ho_bias = self.W_ho_bias[new_indices].clone()

        self.W_ih_weight_sigma = self.W_ih_weight_sigma[new_indices].clone()
        self.W_ih_bias_sigma = self.W_ih_bias_sigma[new_indices].clone()
        self.W_ho_weight_sigma = self.W_ho_weight_sigma[new_indices].clone()
        self.W_ho_bias_sigma = self.W_ho_bias_sigma[new_indices].clone()

        if self.model_type == "reservoir":
            self.W_hh = self.W_hh[new_indices].clone()
        elif self.model_type == "trainable":
            self.u = self.u[new_indices].clone()
            self.v = self.v[new_indices].clone()
            self.u_sigma = self.u_sigma[new_indices].clone()
            self.v_sigma = self.v_sigma[new_indices].clone()

    def create_best_model(
        self, fitness: Float[Tensor, " pop_size"]
    ) -> RecurrentMLPReservoir | RecurrentMLPTrainable:
        """Create a model from the best network's parameters."""
        best_idx: int = torch.argmin(fitness).item()

        if self.model_type == "reservoir":
            model: RecurrentMLPReservoir = RecurrentMLPReservoir(
                self.input_size, self.hidden_size, self.output_size
            ).to(config.DEVICE)
            model.W_ih.weight.data = self.W_ih_weight[best_idx]
            model.W_ih.bias.data = self.W_ih_bias[best_idx]
            model.W_ho.weight.data = self.W_ho_weight[best_idx]
            model.W_ho.bias.data = self.W_ho_bias[best_idx]
            model.W_hh.data = self.W_hh[best_idx]
        else:  # trainable
            model = RecurrentMLPTrainable(
                self.input_size, self.hidden_size, self.output_size
            ).to(config.DEVICE)
            model.W_ih.weight.data = self.W_ih_weight[best_idx]
            model.W_ih.bias.data = self.W_ih_bias[best_idx]
            model.W_ho.weight.data = self.W_ho_weight[best_idx]
            model.W_ho.bias.data = self.W_ho_bias[best_idx]
            model.u.data = self.u[best_idx]
            model.v.data = self.v[best_idx]

        return model

    def get_state_dict(self) -> dict[str, Tensor]:
        """Get state dict for checkpointing."""
        state: dict[str, Tensor] = {
            "W_ih_weight": self.W_ih_weight,
            "W_ih_bias": self.W_ih_bias,
            "W_ho_weight": self.W_ho_weight,
            "W_ho_bias": self.W_ho_bias,
            "W_ih_weight_sigma": self.W_ih_weight_sigma,
            "W_ih_bias_sigma": self.W_ih_bias_sigma,
            "W_ho_weight_sigma": self.W_ho_weight_sigma,
            "W_ho_bias_sigma": self.W_ho_bias_sigma,
        }

        if self.model_type == "reservoir":
            state["W_hh"] = self.W_hh
        elif self.model_type == "trainable":
            state["u"] = self.u
            state["v"] = self.v
            state["u_sigma"] = self.u_sigma
            state["v_sigma"] = self.v_sigma

        return state

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Load state dict from checkpoint."""
        self.W_ih_weight = state["W_ih_weight"]
        self.W_ih_bias = state["W_ih_bias"]
        self.W_ho_weight = state["W_ho_weight"]
        self.W_ho_bias = state["W_ho_bias"]
        self.W_ih_weight_sigma = state["W_ih_weight_sigma"]
        self.W_ih_bias_sigma = state["W_ih_bias_sigma"]
        self.W_ho_weight_sigma = state["W_ho_weight_sigma"]
        self.W_ho_bias_sigma = state["W_ho_bias_sigma"]

        if self.model_type == "reservoir":
            self.W_hh = state["W_hh"]
        elif self.model_type == "trainable":
            self.u = state["u"]
            self.v = state["v"]
            self.u_sigma = state["u_sigma"]
            self.v_sigma = state["v_sigma"]


class DynamicNetPopulation:
    """Wrapper for common/dynamic_net population with BatchedPopulation interface.

    Uses dynamic networks with graph-based recurrence (via cycles and multiple passes).
    Properly implements batched computation following common/dynamic_net/computation.py pattern.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        pop_size: int,
        initial_mutations: int = 5,
    ) -> None:
        """Initialize dynamic network population.

        Args:
            input_size: Input dimension
            output_size: Output dimension
            pop_size: Population size
            initial_mutations: Number of mutations to apply to initialize networks
        """
        from common.dynamic_net.evolution import Net
        from common.dynamic_net.computation import WelfordRunningStandardizer

        self.input_size: int = input_size
        self.output_size: int = output_size
        self.pop_size: int = pop_size

        # Create networks
        self.nets: list[Net] = [
            Net(input_size, output_size) for _ in range(pop_size)
        ]

        # Initialize networks with some mutations to get non-trivial architectures
        for net in self.nets:
            for _ in range(initial_mutations):
                net.mutate()

        print(
            f"  Initialized {pop_size} dynamic networks with {initial_mutations} mutations each"
        )

        # Build computation infrastructure
        self._build_computation_infrastructure()

    def _build_computation_infrastructure(self) -> None:
        """Build batched computation tensors following computation.py pattern."""
        from common.dynamic_net.computation import WelfordRunningStandardizer

        # Get number of nodes for each network
        nets_num_nodes: Int[Tensor, "pop_size"] = torch.tensor(
            [len(net.nodes.all) for net in self.nets], device=config.DEVICE
        )

        # Concatenate n_mean_m2_x_z with zero entry at front
        self.n_mean_m2_x_z: Float[Tensor, "TNNplus1 5"] = torch.cat(
            [torch.zeros(1, 5, device=config.DEVICE)]
            + [net.n_mean_m2_x_z.to(config.DEVICE) for net in self.nets]
        )

        # Create WelfordRunningStandardizer
        self.wrs: WelfordRunningStandardizer = WelfordRunningStandardizer(
            self.n_mean_m2_x_z, verbose=False
        )

        # Build index mappings
        self.input_nodes_start_indices: Int[Tensor, "pop_size"] = (
            torch.cat(
                (
                    torch.tensor([0], device=config.DEVICE),
                    torch.cumsum(nets_num_nodes[:-1], dim=0),
                )
            )
            + 1
        )

        self.input_nodes_indices: Int[Tensor, "input_sizexpop_size"] = (
            self.input_nodes_start_indices.unsqueeze(1)
            + torch.arange(self.input_size, device=config.DEVICE)
        ).flatten()

        output_nodes_start_indices: Int[Tensor, "pop_size"] = (
            self.input_nodes_start_indices + self.input_size
        )

        self.output_nodes_indices: Int[Tensor, "output_sizexpop_size"] = (
            output_nodes_start_indices.unsqueeze(1)
            + torch.arange(self.output_size, device=config.DEVICE)
        ).flatten()

        nodes_indices: Int[Tensor, "TNN"] = torch.arange(
            1, len(self.n_mean_m2_x_z), device=config.DEVICE
        )
        self.mutable_nodes_indices: Int[Tensor, "TNMN"] = nodes_indices[
            ~torch.isin(nodes_indices, self.input_nodes_indices)
        ]

        # Build in_nodes_indices and weights
        nets_num_mutable_nodes: Int[Tensor, "pop_size"] = (
            nets_num_nodes - self.input_size
        )
        nets_cum_num_mutable_nodes: Int[Tensor, "pop_size"] = torch.cumsum(
            nets_num_mutable_nodes, 0
        )

        in_nodes_indices: Int[Tensor, "TNMN 3"] = torch.empty(
            (nets_num_mutable_nodes.sum(), 3), dtype=torch.int32, device=config.DEVICE
        )

        for i in range(self.pop_size):
            start: int = 0 if i == 0 else nets_cum_num_mutable_nodes[i - 1].item()
            end: int = nets_cum_num_mutable_nodes[i].item()
            net_in_nodes_indices: Int[Tensor, "NET_NMN 3"] = self.nets[
                i
            ].in_nodes_indices
            in_nodes_indices[start:end] = (
                net_in_nodes_indices + (net_in_nodes_indices >= 0) * self.input_nodes_start_indices[i]
            ).to(config.DEVICE)

        in_nodes_indices = torch.relu(in_nodes_indices)  # Map -1s to 0s
        self.flat_in_nodes_indices: Int[Tensor, "TNMNx3"] = in_nodes_indices.flatten()

        self.weights: Float[Tensor, "TNMN 3"] = torch.cat(
            [torch.tensor(net.weights, device=config.DEVICE) for net in self.nets]
        )

        # Build mask for variable num_network_passes_per_input
        num_network_passes_per_input: Int[Tensor, "pop_size"] = torch.tensor(
            [net.num_network_passes_per_input for net in self.nets],
            device=config.DEVICE,
        )
        self.max_num_network_passes_per_input: int = max(
            num_network_passes_per_input
        ).item()

        self.num_network_passes_per_input_mask: Float[Tensor, "max_passes TNMN"] = (
            torch.zeros(
                (self.max_num_network_passes_per_input, nets_num_mutable_nodes.sum()),
                device=config.DEVICE,
            )
        )

        for i in range(self.max_num_network_passes_per_input):
            for j in range(self.pop_size):
                if self.nets[j].num_network_passes_per_input > i:
                    start: int = (
                        0 if j == 0 else nets_cum_num_mutable_nodes[j - 1].item()
                    )
                    end: int = nets_cum_num_mutable_nodes[j].item()
                    self.num_network_passes_per_input_mask[i][start:end] = 1

        self.num_network_passes_per_input_mask = (
            self.num_network_passes_per_input_mask.bool()
        )

    def forward_batch(
        self, observations: Float[Tensor, "N input_size"]
    ) -> Float[Tensor, "pop_size N output_size"]:
        """Batched forward pass for all networks.

        Args:
            observations: Batch of observations [N, input_size]

        Returns:
            logits: Output logits for all networks [pop_size, N, output_size]
        """
        batch_size: int = len(observations)
        all_outputs: list[Float[Tensor, "pop_size output_size"]] = []

        with torch.no_grad():
            for i in range(batch_size):
                obs: Float[Tensor, " input_size"] = observations[i]

                # Flatten observation for all networks
                flat_obs: Float[Tensor, "input_sizexpop_size"] = obs.repeat(
                    self.pop_size
                )

                # Initialize output with previous z-scores
                out: Float[Tensor, "TNNplus1"] = self.n_mean_m2_x_z[:, 4].clone()

                # Set input nodes
                out[self.input_nodes_indices] = flat_obs

                # Apply Welford standardization
                out = self.wrs(out)

                # Multiple passes through the graph
                for pass_idx in range(self.max_num_network_passes_per_input):
                    # Map inputs for each mutable node
                    mapped_out: Float[Tensor, "TNMN 3"] = torch.gather(
                        out, 0, self.flat_in_nodes_indices
                    ).reshape(-1, 3)

                    # Compute weighted sum
                    matmuld_mapped_out: Float[Tensor, "TNMN"] = (
                        mapped_out * self.weights
                    ).sum(dim=1)

                    # Update only nodes that need this pass
                    out[self.mutable_nodes_indices] = torch.where(
                        self.num_network_passes_per_input_mask[pass_idx],
                        matmuld_mapped_out,
                        out[self.mutable_nodes_indices],
                    )

                    # Apply Welford standardization
                    out = self.wrs(out)

                # Extract output nodes
                outputs: Float[Tensor, "pop_size output_size"] = out[
                    self.output_nodes_indices
                ].reshape(self.pop_size, self.output_size)

                all_outputs.append(outputs)

        # Stack to get [pop_size, N, output_size]
        logits: Float[Tensor, "pop_size N output_size"] = torch.stack(
            all_outputs, dim=1
        )

        return logits

    def evaluate(
        self,
        observations: Float[Tensor, "N input_size"],
        actions: Int[Tensor, " N"],
    ) -> Float[Tensor, " pop_size"]:
        """Evaluate fitness (cross-entropy) of all networks.

        Args:
            observations: Batch of observations
            actions: Batch of actions

        Returns:
            fitness: Cross-entropy for each network [pop_size]
        """
        with torch.no_grad():
            # Get logits for all networks
            logits: Float[Tensor, "pop_size N output_size"] = self.forward_batch(
                observations
            )

            # Expand actions for all networks
            actions_expanded: Int[Tensor, "pop_size N"] = actions.unsqueeze(0).expand(
                self.pop_size, -1
            )

            # Reshape for cross_entropy
            flat_logits: Float[Tensor, "pop_sizexN output_size"] = logits.reshape(
                -1, self.output_size
            )
            flat_actions: Int[Tensor, " pop_sizexN"] = actions_expanded.reshape(-1)

            # Compute per-sample CE then reshape and mean per network
            per_sample_ce: Float[Tensor, " pop_sizexN"] = F.cross_entropy(
                flat_logits, flat_actions, reduction="none"
            )
            per_network_ce: Float[Tensor, "pop_size N"] = per_sample_ce.view(
                self.pop_size, -1
            )
            fitness: Float[Tensor, " pop_size"] = per_network_ce.mean(dim=1)

        return fitness

    def mutate(self) -> None:
        """Apply mutations to all networks in population."""
        for net in self.nets:
            net.mutate()
        # Rebuild computation infrastructure after topology changes
        self._build_computation_infrastructure()

    def select_simple_ga(self, fitness: Float[Tensor, " pop_size"]) -> None:
        """Simple GA selection: top 50% survive and duplicate."""
        # Sort by fitness (minimize CE)
        sorted_indices: list[int] = torch.argsort(fitness).cpu().tolist()

        # Top 50% survive
        num_survivors: int = self.pop_size // 2
        survivor_indices: list[int] = sorted_indices[:num_survivors]

        # Create new population
        new_nets: list = []
        for idx in survivor_indices:
            new_nets.append(self.nets[idx])

        # Duplicate survivors to fill population
        num_losers: int = self.pop_size - num_survivors
        for i in range(num_losers):
            survivor_idx: int = survivor_indices[i % num_survivors]
            # Deep copy the network
            import copy

            new_nets.append(copy.deepcopy(self.nets[survivor_idx]))

        self.nets = new_nets
        # Rebuild after selection
        self._build_computation_infrastructure()

    def get_state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        # For dynamic nets, we need to save the network structures
        # This is complex due to variable architectures
        # Simplified version for now
        return {
            "num_nets": len(self.nets),
            "input_size": self.input_size,
            "output_size": self.output_size,
            # Would need to serialize each net's structure
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state dict from checkpoint."""
        # Would need to deserialize network structures
        pass
