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

from config.device import DEVICE
from dl.models.recurrent import RecurrentMLPReservoir, RecurrentMLPTrainable
from dl.optim.base import create_episode_list
from eval.metrics import compute_cross_entropy


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
        self.W_ih_weight = torch.randn(pop_size, hidden_size, input_size, device=DEVICE) * ih_std
        self.W_ih_bias = torch.randn(pop_size, hidden_size, device=DEVICE) * ih_std

        # Hidden-to-output weights and biases
        self.W_ho_weight = torch.randn(pop_size, output_size, hidden_size, device=DEVICE) * ho_std
        self.W_ho_bias = torch.randn(pop_size, output_size, device=DEVICE) * ho_std

        # Recurrent weights
        if model_type == "reservoir":
            hh_std = 1.0 / math.sqrt(hidden_size)
            self.W_hh = torch.randn(pop_size, hidden_size, hidden_size, device=DEVICE) * hh_std
        elif model_type == "trainable":
            hh_std = 1.0 / math.sqrt(hidden_size)
            self.u = torch.randn(pop_size, hidden_size, device=DEVICE) * hh_std
            self.v = torch.randn(pop_size, hidden_size, device=DEVICE) * hh_std
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
                obs = episode["observations"].to(DEVICE)
                act = episode["actions"].to(DEVICE)

                h_0 = torch.zeros(self.pop_size, self.hidden_size, device=DEVICE)
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
            torch.arange(num_losers, device=DEVICE) % num_survivors
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

    def select_simple_es(self, fitness, minimize=True):
        """Simple ES selection: weighted combination of all networks (soft selection)."""
        # Standardize fitness
        if minimize:
            fitness_std = (-fitness - (-fitness).mean()) / (fitness.std() + 1e-8)
        else:
            fitness_std = (fitness - fitness.mean()) / (fitness.std() + 1e-8)
        weights = F.softmax(fitness_std, dim=0)

        # Compute weighted average for each parameter tensor
        # W_ih weights: [pop_size, hidden_size, input_size]
        w_ih = weights.view(-1, 1, 1)
        avg_W_ih_weight = (w_ih * self.W_ih_weight).sum(dim=0)
        self.W_ih_weight = avg_W_ih_weight.unsqueeze(0).expand(self.pop_size, -1, -1).clone()

        w_ih_bias = weights.view(-1, 1)
        avg_W_ih_bias = (w_ih_bias * self.W_ih_bias).sum(dim=0)
        self.W_ih_bias = avg_W_ih_bias.unsqueeze(0).expand(self.pop_size, -1).clone()

        # W_ho weights: [pop_size, output_size, hidden_size]
        w_ho = weights.view(-1, 1, 1)
        avg_W_ho_weight = (w_ho * self.W_ho_weight).sum(dim=0)
        self.W_ho_weight = avg_W_ho_weight.unsqueeze(0).expand(self.pop_size, -1, -1).clone()

        w_ho_bias = weights.view(-1, 1)
        avg_W_ho_bias = (w_ho_bias * self.W_ho_bias).sum(dim=0)
        self.W_ho_bias = avg_W_ho_bias.unsqueeze(0).expand(self.pop_size, -1).clone()

        # Sigmas
        avg_W_ih_weight_sigma = (w_ih * self.W_ih_weight_sigma).sum(dim=0)
        self.W_ih_weight_sigma = avg_W_ih_weight_sigma.unsqueeze(0).expand(self.pop_size, -1, -1).clone()

        avg_W_ih_bias_sigma = (w_ih_bias * self.W_ih_bias_sigma).sum(dim=0)
        self.W_ih_bias_sigma = avg_W_ih_bias_sigma.unsqueeze(0).expand(self.pop_size, -1).clone()

        avg_W_ho_weight_sigma = (w_ho * self.W_ho_weight_sigma).sum(dim=0)
        self.W_ho_weight_sigma = avg_W_ho_weight_sigma.unsqueeze(0).expand(self.pop_size, -1, -1).clone()

        avg_W_ho_bias_sigma = (w_ho_bias * self.W_ho_bias_sigma).sum(dim=0)
        self.W_ho_bias_sigma = avg_W_ho_bias_sigma.unsqueeze(0).expand(self.pop_size, -1).clone()

        # Recurrent weights
        if self.model_type == "reservoir":
            # W_hh: [pop_size, hidden_size, hidden_size]
            w_hh = weights.view(-1, 1, 1)
            avg_W_hh = (w_hh * self.W_hh).sum(dim=0)
            self.W_hh = avg_W_hh.unsqueeze(0).expand(self.pop_size, -1, -1).clone()
        elif self.model_type == "trainable":
            # u and v: [pop_size, hidden_size]
            w_uv = weights.view(-1, 1)
            avg_u = (w_uv * self.u).sum(dim=0)
            self.u = avg_u.unsqueeze(0).expand(self.pop_size, -1).clone()

            avg_v = (w_uv * self.v).sum(dim=0)
            self.v = avg_v.unsqueeze(0).expand(self.pop_size, -1).clone()

            # u and v sigmas
            avg_u_sigma = (w_uv * self.u_sigma).sum(dim=0)
            self.u_sigma = avg_u_sigma.unsqueeze(0).expand(self.pop_size, -1).clone()

            avg_v_sigma = (w_uv * self.v_sigma).sum(dim=0)
            self.v_sigma = avg_v_sigma.unsqueeze(0).expand(self.pop_size, -1).clone()

    def create_best_model(self, fitness):
        """Create a model from the best network's parameters."""
        best_idx = torch.argmin(fitness).item()

        if self.model_type == "reservoir":
            model = RecurrentMLPReservoir(
                self.input_size, self.hidden_size, self.output_size
            ).to(DEVICE)
            model.W_ih.weight.data = self.W_ih_weight[best_idx]
            model.W_ih.bias.data = self.W_ih_bias[best_idx]
            model.W_ho.weight.data = self.W_ho_weight[best_idx]
            model.W_ho.bias.data = self.W_ho_bias[best_idx]
            model.W_hh.data = self.W_hh[best_idx]
        else:  # trainable
            model = RecurrentMLPTrainable(
                self.input_size, self.hidden_size, self.output_size
            ).to(DEVICE)
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


def optimize_es_recurrent(
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
    """Optimize recurrent models using Simple Evolution Strategies (soft selection).

    Uses fitness-weighted combination of all individuals where better networks
    contribute more to the next generation parameters.

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
        population_size: Population size for ES
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

    print(f"  Optimizing with ES for {max_optim_time}s ({max_optim_time/60:.1f} min)...")

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

        # Selection (ES - soft selection)
        population.select_simple_es(fitness, minimize=True)

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
                "algorithm": "es",
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
        "algorithm": "es",
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"  Final checkpoint saved to {checkpoint_path}")

    return fitness_history, test_loss_history


class CMAESRecurrentPopulation:
    """Diagonal CMA-ES population for recurrent MLPs.

    Implements separable/diagonal CMA-ES for recurrent networks (reservoir or trainable).
    Similar to CMAESPopulation but handles recurrent parameters.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        pop_size: int,
        model_type: str = "reservoir",
        sigma_init: float = 1e-1,
    ) -> None:
        """Initialize diagonal CMA-ES population for recurrent models.

        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
            output_size: Output dimension
            pop_size: Population size (lambda)
            model_type: 'reservoir' or 'trainable'
            sigma_init: Initial global step size
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pop_size = pop_size
        self.mu = pop_size // 2
        self.model_type = model_type

        # Calculate parameter dimensions
        W_ih_weight_size = hidden_size * input_size
        W_ih_bias_size = hidden_size
        W_ho_weight_size = output_size * hidden_size
        W_ho_bias_size = output_size

        if model_type == "reservoir":
            W_hh_size = hidden_size * hidden_size
            self.n_params = W_ih_weight_size + W_ih_bias_size + W_ho_weight_size + W_ho_bias_size + W_hh_size
        elif model_type == "trainable":
            u_size = hidden_size
            v_size = hidden_size
            self.n_params = W_ih_weight_size + W_ih_bias_size + W_ho_weight_size + W_ho_bias_size + u_size + v_size
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Store sizes for unflattening
        self.W_ih_weight_size = W_ih_weight_size
        self.W_ih_bias_size = W_ih_bias_size
        self.W_ho_weight_size = W_ho_weight_size
        self.W_ho_bias_size = W_ho_bias_size
        if model_type == "reservoir":
            self.W_hh_size = W_hh_size
        else:
            self.u_size = u_size
            self.v_size = v_size

        # Initialize mean
        self.mean = torch.randn(self.n_params, device=DEVICE) * 0.01

        # Global step size
        self.sigma = sigma_init

        # Diagonal covariance
        self.C_diag = torch.ones(self.n_params, device=DEVICE)

        # Evolution paths
        self.p_sigma = torch.zeros(self.n_params, device=DEVICE)
        self.p_c = torch.zeros(self.n_params, device=DEVICE)

        # Strategy parameters
        self.mu_eff = self._compute_mu_eff()
        self.c_sigma = (self.mu_eff + 2.0) / (self.n_params + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, ((self.mu_eff - 1.0) / (self.n_params + 1.0)) ** 0.5 - 1.0) + self.c_sigma
        self.c_c = (4.0 + self.mu_eff / self.n_params) / (self.n_params + 4.0 + 2.0 * self.mu_eff / self.n_params)
        self.c_1 = 2.0 / ((self.n_params + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1.0 - self.c_1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((self.n_params + 2.0) ** 2 + self.mu_eff))

        # Recombination weights
        self.weights = self._compute_weights()

        # Expectation of ||N(0,I)||
        self.chi_n = self.n_params ** 0.5 * (1.0 - 1.0 / (4.0 * self.n_params) + 1.0 / (21.0 * self.n_params ** 2))

        # Storage for offspring
        self.offspring = torch.zeros(self.pop_size, self.n_params, device=DEVICE)

    def _compute_mu_eff(self) -> float:
        """Compute variance effectiveness of weighted recombination."""
        weights_prime = torch.log(torch.tensor(self.mu + 0.5, device=DEVICE)) - torch.log(
            torch.arange(1, self.mu + 1, dtype=torch.float32, device=DEVICE)
        )
        weights_prime = weights_prime / weights_prime.sum()
        mu_eff = 1.0 / (weights_prime ** 2).sum()
        return mu_eff.item()

    def _compute_weights(self):
        """Compute recombination weights for top mu individuals."""
        weights = torch.log(torch.tensor(self.mu + 0.5, device=DEVICE)) - torch.log(
            torch.arange(1, self.mu + 1, dtype=torch.float32, device=DEVICE)
        )
        weights = weights / weights.sum()
        return weights

    def _unflatten_params(self, flat):
        """Unflatten parameter vector to recurrent network parameters."""
        idx = 0
        W_ih_weight = flat[idx : idx + self.W_ih_weight_size].reshape(self.hidden_size, self.input_size)
        idx += self.W_ih_weight_size

        W_ih_bias = flat[idx : idx + self.W_ih_bias_size]
        idx += self.W_ih_bias_size

        W_ho_weight = flat[idx : idx + self.W_ho_weight_size].reshape(self.output_size, self.hidden_size)
        idx += self.W_ho_weight_size

        W_ho_bias = flat[idx : idx + self.W_ho_bias_size]
        idx += self.W_ho_bias_size

        if self.model_type == "reservoir":
            W_hh = flat[idx : idx + self.W_hh_size].reshape(self.hidden_size, self.hidden_size)
            return W_ih_weight, W_ih_bias, W_ho_weight, W_ho_bias, W_hh, None, None
        else:  # trainable
            u = flat[idx : idx + self.u_size]
            idx += self.u_size
            v = flat[idx : idx + self.v_size]
            return W_ih_weight, W_ih_bias, W_ho_weight, W_ho_bias, None, u, v

    def sample_offspring(self) -> None:
        """Sample lambda offspring from current distribution."""
        for i in range(self.pop_size):
            z = torch.randn(self.n_params, device=DEVICE)
            self.offspring[i] = self.mean + self.sigma * torch.sqrt(self.C_diag) * z

    def evaluate_episodes(self, episodes):
        """Evaluate fitness of all offspring on episodes (cross-entropy loss)."""
        with torch.no_grad():
            fitness = torch.zeros(self.pop_size, device=DEVICE)

            for i in range(self.pop_size):
                # Unflatten parameters
                W_ih_weight, W_ih_bias, W_ho_weight, W_ho_bias, W_hh, u, v = self._unflatten_params(self.offspring[i])

                total_ce = 0.0
                total_samples = 0

                for episode in episodes:
                    obs_seq = episode["observations"]
                    act_seq = episode["actions"]
                    T = obs_seq.shape[0]

                    # Recurrent forward pass
                    h = torch.zeros(self.hidden_size, device=DEVICE)
                    logits_list = []

                    for t in range(T):
                        # Input-to-hidden
                        h_new = obs_seq[t] @ W_ih_weight.T + W_ih_bias

                        # Recurrent connection
                        if self.model_type == "reservoir":
                            h_new = h_new + h @ W_hh.T
                        else:  # trainable
                            h_new = h_new + u * (v @ h)

                        # Activation
                        h = torch.tanh(h_new)

                        # Hidden-to-output
                        logits = h @ W_ho_weight.T + W_ho_bias
                        logits_list.append(logits)

                    # Stack logits and compute loss
                    all_logits = torch.stack(logits_list)
                    ce = F.cross_entropy(all_logits, act_seq, reduction="sum")
                    total_ce += ce.item()
                    total_samples += T

                fitness[i] = total_ce / total_samples

        return fitness

    def update(self, fitness):
        """Update distribution parameters using CMA-ES update rules."""
        # Select top mu offspring
        sorted_indices = torch.argsort(fitness)
        elite_indices = sorted_indices[: self.mu]

        # Weighted recombination
        old_mean = self.mean.clone()
        self.mean = torch.zeros_like(self.mean)
        for i, idx in enumerate(elite_indices):
            self.mean += self.weights[i] * self.offspring[idx]

        mean_shift = self.mean - old_mean

        # Update evolution path for sigma
        c_sigma_factor = (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) ** 0.5
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + c_sigma_factor * mean_shift / (
            self.sigma * torch.sqrt(self.C_diag)
        )

        # Update global step size
        p_sigma_norm = torch.norm(self.p_sigma).item()
        self.sigma = self.sigma * math.exp(
            (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)
        )

        # Update evolution path for C
        h_sig = 1.0 if p_sigma_norm / ((1.0 - (1.0 - self.c_sigma) ** (2.0 * self.mu_eff)) ** 0.5) < 1.4 + 2.0 / (self.n_params + 1.0) else 0.0
        c_c_factor = (self.c_c * (2.0 - self.c_c) * self.mu_eff) ** 0.5
        self.p_c = (1.0 - self.c_c) * self.p_c + h_sig * c_c_factor * mean_shift / self.sigma

        # Update diagonal covariance
        self.C_diag = (
            (1.0 - self.c_1 - self.c_mu) * self.C_diag
            + self.c_1 * self.p_c ** 2
        )

        # Rank-mu update
        for i, idx in enumerate(elite_indices):
            y_i = (self.offspring[idx] - old_mean) / self.sigma
            self.C_diag += self.c_mu * self.weights[i] * y_i ** 2

        # Ensure C_diag stays positive
        self.C_diag = torch.clamp(self.C_diag, min=1e-10)

    def get_state_dict(self):
        """Get state dict for checkpointing."""
        return {
            "mean": self.mean,
            "sigma": torch.tensor(self.sigma, device=DEVICE),
            "C_diag": self.C_diag,
            "p_sigma": self.p_sigma,
            "p_c": self.p_c,
        }

    def load_state_dict(self, state):
        """Load state dict from checkpoint."""
        self.mean = state["mean"]
        self.sigma = state["sigma"].item()
        self.C_diag = state["C_diag"]
        self.p_sigma = state["p_sigma"]
        self.p_c = state["p_c"]


def optimize_cmaes_recurrent(
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
    sigma_init: float = 1e-1,
    loss_eval_interval_seconds: int = 60,
    logger=None,
) -> tuple[list[float], list[float]]:
    """Optimize recurrent models using Diagonal CMA-ES.

    Uses Covariance Matrix Adaptation Evolution Strategy with diagonal approximation
    for adaptive step size and coordinate-wise variance adaptation.

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
        population_size: Population size (lambda)
        sigma_init: Initial global step size
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
    population = CMAESRecurrentPopulation(
        input_size,
        hidden_size,
        output_size,
        population_size,
        model_type=model_type,
        sigma_init=sigma_init,
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

    print(f"  Optimizing with CMA-ES for {max_optim_time}s ({max_optim_time/60:.1f} min)...")

    # Determine number of episodes for fitness evaluation
    num_eval_episodes = min(32, len(optim_episodes))

    while True:
        # Check time limit
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_optim_time:
            print(f"  Time limit reached ({elapsed_time:.1f}s)")
            break

        # Sample offspring from current distribution
        population.sample_offspring()

        # Sample episodes for fitness evaluation
        sampled_episodes = random.sample(optim_episodes, k=num_eval_episodes)

        # Evaluate fitness
        fitness = population.evaluate_episodes(sampled_episodes)
        best_fitness = fitness.min().item()
        fitness_history.append(best_fitness)

        # Update distribution (CMA-ES update)
        population.update(fitness)

        # Periodic evaluation
        elapsed = time.time() - start_time
        if elapsed - last_eval_time >= loss_eval_interval_seconds:
            # Evaluate on test episodes
            population.sample_offspring()
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
                f"Best={best_fitness:.4f}, Test={test_loss:.4f}, Sigma={population.sigma:.4e}"
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
                "algorithm": "cmaes",
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
        "algorithm": "cmaes",
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"  Final checkpoint saved to {checkpoint_path}")

    return fitness_history, test_loss_history
