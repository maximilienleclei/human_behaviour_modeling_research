"""Genetic algorithm optimizer for neural networks.

This module provides evolutionary optimization using mutation-based population evolution.
Includes GPU-parallelized population classes for efficient neuroevolution.
"""

import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from platform import config
from platform.models.recurrent import RecurrentMLPReservoir, RecurrentMLPTrainable
from platform.optimizers.base import create_episode_list


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
    ):
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
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model_type = model_type
        self.sigma_init = sigma_init
        self.sigma_noise = sigma_noise

        # Initialize batched parameters using Xavier initialization
        ih_std = (1.0 / input_size) ** 0.5
        ho_std = (1.0 / hidden_size) ** 0.5

        # Input-to-hidden weights and biases
        self.W_ih_weight = torch.randn(pop_size, hidden_size, input_size, device=config.DEVICE) * ih_std
        self.W_ih_bias = torch.randn(pop_size, hidden_size, device=config.DEVICE) * ih_std

        # Hidden-to-output weights and biases
        self.W_ho_weight = torch.randn(pop_size, output_size, hidden_size, device=config.DEVICE) * ho_std
        self.W_ho_bias = torch.randn(pop_size, output_size, device=config.DEVICE) * ho_std

        # Recurrent weights
        if model_type == "reservoir":
            hh_std = 1.0 / math.sqrt(hidden_size)
            self.W_hh = torch.randn(pop_size, hidden_size, hidden_size, device=config.DEVICE) * hh_std
        elif model_type == "trainable":
            hh_std = 1.0 / math.sqrt(hidden_size)
            self.u = torch.randn(pop_size, hidden_size, device=config.DEVICE) * hh_std
            self.v = torch.randn(pop_size, hidden_size, device=config.DEVICE) * hh_std
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Initialize adaptive sigmas
        self.W_ih_weight_sigma = torch.full_like(self.W_ih_weight, sigma_init)
        self.W_ih_bias_sigma = torch.full_like(self.W_ih_bias, sigma_init)
        self.W_ho_weight_sigma = torch.full_like(self.W_ho_weight, sigma_init)
        self.W_ho_bias_sigma = torch.full_like(self.W_ho_bias, sigma_init)

        if model_type == "trainable":
            self.u_sigma = torch.full_like(self.u, sigma_init)
            self.v_sigma = torch.full_like(self.v, sigma_init)

    def forward_batch_step(self, x, h):
        """Single timestep forward pass for all networks in parallel."""
        # Input-to-hidden
        ih_out = torch.bmm(x, self.W_ih_weight.transpose(-1, -2))
        ih_out = ih_out + self.W_ih_bias.unsqueeze(1)

        # Recurrent connection
        if self.model_type == "reservoir":
            hh_out = torch.bmm(h, self.W_hh.transpose(-1, -2))
        elif self.model_type == "trainable":
            v_expanded = self.v.unsqueeze(1).expand(-1, x.shape[1], -1)
            v_dot_h = (h * v_expanded).sum(dim=2)
            u_expanded = self.u.unsqueeze(1).expand(-1, x.shape[1], -1)
            hh_out = u_expanded * v_dot_h.unsqueeze(2)

        h_new = torch.tanh(ih_out + hh_out)

        # Hidden-to-output
        logits = torch.bmm(h_new, self.W_ho_weight.transpose(-1, -2))
        logits = logits + self.W_ho_bias.unsqueeze(1)

        return logits, h_new

    def forward_batch_sequence(self, x, h_0=None):
        """Batched forward pass for sequence across all networks."""
        seq_len = x.shape[0]

        if h_0 is None:
            h = torch.zeros(self.pop_size, 1, self.hidden_size, device=x.device)
        else:
            h = h_0.unsqueeze(1)

        all_logits = []
        for t in range(seq_len):
            x_t = x[t].unsqueeze(0).unsqueeze(0).expand(self.pop_size, 1, -1)
            logits_t, h = self.forward_batch_step(x_t, h)
            all_logits.append(logits_t)

        logits = torch.cat(all_logits, dim=1)
        h_final = h.squeeze(1)

        return logits, h_final

    def evaluate_episodes(self, episodes):
        """Evaluate fitness on complete episodes."""
        episode_losses = []

        with torch.no_grad():
            for episode in episodes:
                obs = episode["observations"].to(config.DEVICE)
                act = episode["actions"].to(config.DEVICE)

                h_0 = torch.zeros(self.pop_size, self.hidden_size, device=config.DEVICE)
                logits, _ = self.forward_batch_sequence(obs, h_0)

                act_expanded = act.unsqueeze(0).expand(self.pop_size, -1)
                flat_logits = logits.reshape(-1, self.output_size)
                flat_actions = act_expanded.reshape(-1)

                per_sample_ce = F.cross_entropy(flat_logits, flat_actions, reduction="none")
                per_network_ce = per_sample_ce.view(self.pop_size, -1)
                episode_loss = per_network_ce.mean(dim=1)

                episode_losses.append(episode_loss)

        fitness = torch.stack(episode_losses).mean(dim=0)
        return fitness

    def mutate(self):
        """Apply adaptive sigma mutations to all networks."""
        # W_ih_weight
        xi = torch.randn_like(self.W_ih_weight_sigma) * self.sigma_noise
        self.W_ih_weight_sigma = self.W_ih_weight_sigma * (1 + xi)
        eps = torch.randn_like(self.W_ih_weight) * self.W_ih_weight_sigma
        self.W_ih_weight = self.W_ih_weight + eps

        # W_ih_bias
        xi = torch.randn_like(self.W_ih_bias_sigma) * self.sigma_noise
        self.W_ih_bias_sigma = self.W_ih_bias_sigma * (1 + xi)
        eps = torch.randn_like(self.W_ih_bias) * self.W_ih_bias_sigma
        self.W_ih_bias = self.W_ih_bias + eps

        # W_ho_weight
        xi = torch.randn_like(self.W_ho_weight_sigma) * self.sigma_noise
        self.W_ho_weight_sigma = self.W_ho_weight_sigma * (1 + xi)
        eps = torch.randn_like(self.W_ho_weight) * self.W_ho_weight_sigma
        self.W_ho_weight = self.W_ho_weight + eps

        # W_ho_bias
        xi = torch.randn_like(self.W_ho_bias_sigma) * self.sigma_noise
        self.W_ho_bias_sigma = self.W_ho_bias_sigma * (1 + xi)
        eps = torch.randn_like(self.W_ho_bias) * self.W_ho_bias_sigma
        self.W_ho_bias = self.W_ho_bias + eps

        # Trainable recurrent weights
        if self.model_type == "trainable":
            xi = torch.randn_like(self.u_sigma) * self.sigma_noise
            self.u_sigma = self.u_sigma * (1 + xi)
            eps = torch.randn_like(self.u) * self.u_sigma
            self.u = self.u + eps

            xi = torch.randn_like(self.v_sigma) * self.sigma_noise
            self.v_sigma = self.v_sigma * (1 + xi)
            eps = torch.randn_like(self.v) * self.v_sigma
            self.v = self.v + eps

    def select_simple_ga(self, fitness):
        """Simple GA selection: top 50% survive and duplicate."""
        sorted_indices = torch.argsort(fitness)
        num_survivors = self.pop_size // 2
        survivor_indices = sorted_indices[:num_survivors]

        num_losers = self.pop_size - num_survivors
        replacement_indices = survivor_indices[
            torch.arange(num_losers, device=config.DEVICE) % num_survivors
        ]

        new_indices = torch.cat([survivor_indices, replacement_indices])

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

    def create_best_model(self, fitness):
        """Create a model from the best network's parameters."""
        best_idx = torch.argmin(fitness).item()

        if self.model_type == "reservoir":
            model = RecurrentMLPReservoir(
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

    def get_state_dict(self):
        """Get state dict for checkpointing."""
        state = {
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

    def load_state_dict(self, state):
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


def optimize_ga(
    input_size: int,
    hidden_size: int,
    output_size: int,
    optim_obs: Float[Tensor, "optim_size input_size"],
    optim_act: Int[Tensor, " optim_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    metadata: dict,
    checkpoint_path: Path,
    model_type: str = "reservoir",
    max_optim_time: int = 36000,
    population_size: int = 50,
    sigma_init: float = 1e-3,
    sigma_noise: float = 1e-2,
    loss_eval_interval_seconds: int = 60,
    logger=None,
) -> tuple[list[float], list[float]]:
    """Optimize using genetic algorithm with mutation-based evolution.

    Args:
        input_size: Input dimension
        hidden_size: Hidden dimension
        output_size: Output dimension
        optim_obs: Optimization observations
        optim_act: Optimization actions
        test_obs: Test observations
        test_act: Test actions
        metadata: Metadata dict with episode_boundaries
        checkpoint_path: Path to save checkpoints
        model_type: 'reservoir' or 'trainable'
        max_optim_time: Maximum optimization time in seconds
        population_size: Population size for GA
        sigma_init: Initial mutation sigma
        sigma_noise: Noise for adaptive sigma
        loss_eval_interval_seconds: Evaluation interval in seconds
        logger: Optional ExperimentLogger for database logging

    Returns:
        Tuple of (fitness_history, test_loss_history)
    """
    if "optim_episode_boundaries" not in metadata:
        raise ValueError("metadata with episode boundaries required")

    # Create episode lists
    optim_episodes = create_episode_list(
        optim_obs, optim_act, metadata["optim_episode_boundaries"]
    )
    test_episodes = create_episode_list(
        test_obs, test_act, metadata["test_episode_boundaries"]
    )

    # Initialize population
    population = BatchedRecurrentPopulation(
        input_size,
        hidden_size,
        output_size,
        population_size,
        model_type=model_type,
        sigma_init=sigma_init,
        sigma_noise=sigma_noise,
    )

    fitness_history = []
    test_loss_history = []

    # Try to resume from checkpoint
    start_gen = 0
    if checkpoint_path.exists():
        print(f"  Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        fitness_history = checkpoint["fitness_history"]
        test_loss_history = checkpoint.get("test_loss_history", [])
        start_gen = checkpoint["generation"] + 1
        population.load_state_dict(checkpoint["population_state"])
        print(f"  Resumed at generation {start_gen}")

    generation = start_gen
    start_time = time.time()
    last_eval_time = -loss_eval_interval_seconds

    print(f"  Optimizing for {max_optim_time}s ({max_optim_time/60:.1f} min)...")

    # Determine number of episodes for fitness evaluation
    num_eval_episodes = min(32, len(optim_episodes))

    while True:
        # Check time limit
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_optim_time:
            print(f"  Time limit reached ({elapsed_time:.1f}s)")
            break

        # Sample episodes for fitness evaluation
        sampled_episodes = random.sample(optim_episodes, k=num_eval_episodes)

        # Evaluate fitness
        fitness = population.evaluate_episodes(sampled_episodes)
        best_fitness = fitness.min().item()
        fitness_history.append(best_fitness)

        # Selection
        population.select_simple_ga(fitness)

        # Mutation
        population.mutate()

        # Periodic evaluation
        elapsed = time.time() - start_time
        if elapsed - last_eval_time >= loss_eval_interval_seconds:
            # Evaluate on test episodes
            test_sampled = random.sample(
                test_episodes, k=min(num_eval_episodes, len(test_episodes))
            )
            test_fitness = population.evaluate_episodes(test_sampled)
            test_loss = test_fitness.min().item()

            test_loss_history.append(test_loss)
            last_eval_time = elapsed
            remaining = max_optim_time - elapsed
            print(
                f"  Gen {generation} [{elapsed:.0f}s/{max_optim_time}s, {remaining:.0f}s left]: "
                f"Best={best_fitness:.4f}, Test={test_loss:.4f}"
            )

            # Log to database
            if logger is not None:
                logger.log_progress(
                    epoch=generation,
                    best_fitness=best_fitness,
                    test_loss=test_loss,
                )

        # Save checkpoint periodically (every 300s)
        if elapsed % 300 < (time.time() - start_time) % 300:
            checkpoint_data = {
                "generation": generation,
                "fitness_history": fitness_history,
                "test_loss_history": test_loss_history,
                "population_state": population.get_state_dict(),
                "optim_time": elapsed,
            }
            torch.save(checkpoint_data, checkpoint_path)

        generation += 1

    # Final checkpoint
    total_time = time.time() - start_time
    print(f"  Complete: {generation} generations in {total_time:.1f}s ({total_time/60:.1f} min)")
    
    checkpoint_data = {
        "generation": generation,
        "fitness_history": fitness_history,
        "test_loss_history": test_loss_history,
        "population_state": population.get_state_dict(),
        "optim_time": total_time,
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"  Final checkpoint saved to {checkpoint_path}")

    return fitness_history, test_loss_history
