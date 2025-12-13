"""Evolution-based optimizers for feedforward neural networks.

This module provides evolutionary optimization using Simple GA, Simple ES, and CMA-ES
for feedforward MLP architectures with GPU-parallel population evolution.
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
from dl.models.feedforward import MLP
from dl.optim.base import create_episode_list
from eval.metrics import compute_cross_entropy


class BatchedPopulation:
    """Batched population of feedforward MLPs for efficient GPU-parallel neuroevolution.

    Supports both Simple GA (hard selection) and Simple ES (soft selection) algorithms.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        pop_size: int,
        adaptive_sigma: bool = True,
        sigma_init: float = 1e-3,
        sigma_noise: float = 1e-2,
    ) -> None:
        """Initialize batched feedforward population.

        Args:
            input_size: Input dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension (number of actions)
            pop_size: Population size
            adaptive_sigma: Whether to use adaptive sigma mutation
            sigma_init: Initial mutation strength
            sigma_noise: Noise level for sigma adaptation
        """
        self.pop_size: int = pop_size
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size
        self.adaptive_sigma: bool = adaptive_sigma
        self.sigma_init: float = sigma_init
        self.sigma_noise: float = sigma_noise

        # Initialize batched parameters [pop_size, ...] using Xavier initialization
        fc1_std: float = (1.0 / input_size) ** 0.5
        fc2_std: float = (1.0 / hidden_size) ** 0.5

        self.fc1_weight: Float[Tensor, "pop_size hidden_size input_size"] = (
            torch.randn(pop_size, hidden_size, input_size, device=DEVICE)
            * fc1_std
        )
        self.fc1_bias: Float[Tensor, "pop_size hidden_size"] = (
            torch.randn(pop_size, hidden_size, device=DEVICE) * fc1_std
        )
        self.fc2_weight: Float[Tensor, "pop_size output_size hidden_size"] = (
            torch.randn(pop_size, output_size, hidden_size, device=DEVICE)
            * fc2_std
        )
        self.fc2_bias: Float[Tensor, "pop_size output_size"] = (
            torch.randn(pop_size, output_size, device=DEVICE) * fc2_std
        )

        # Initialize adaptive sigmas if needed
        if adaptive_sigma:
            self.fc1_weight_sigma: Float[
                Tensor, "pop_size hidden_size input_size"
            ] = torch.full_like(self.fc1_weight, sigma_init)
            self.fc1_bias_sigma: Float[Tensor, "pop_size hidden_size"] = (
                torch.full_like(self.fc1_bias, sigma_init)
            )
            self.fc2_weight_sigma: Float[
                Tensor, "pop_size output_size hidden_size"
            ] = torch.full_like(self.fc2_weight, sigma_init)
            self.fc2_bias_sigma: Float[Tensor, "pop_size output_size"] = (
                torch.full_like(self.fc2_bias, sigma_init)
            )

    def forward_batch(
        self, x: Float[Tensor, "N input_size"]
    ) -> Float[Tensor, "pop_size N output_size"]:
        """Batched forward pass for all networks in parallel.

        Args:
            x: Input observations [N, input_size]

        Returns:
            Logits for all networks [pop_size, N, output_size]
        """
        # x: [N, input_size] -> expand to [pop_size, N, input_size]
        x_expanded: Float[Tensor, "pop_size N input_size"] = x.unsqueeze(
            0
        ).expand(self.pop_size, -1, -1)

        # First layer: [pop_size, N, input_size] @ [pop_size, input_size, hidden_size]
        h: Float[Tensor, "pop_size N hidden_size"] = torch.bmm(
            x_expanded, self.fc1_weight.transpose(-1, -2)
        )
        # Add bias: [pop_size, N, hidden_size] + [pop_size, 1, hidden_size]
        h = h + self.fc1_bias.unsqueeze(1)
        # Activation
        h = torch.tanh(h)

        # Second layer: [pop_size, N, hidden_size] @ [pop_size, hidden_size, output_size]
        logits: Float[Tensor, "pop_size N output_size"] = torch.bmm(
            h, self.fc2_weight.transpose(-1, -2)
        )
        # Add bias: [pop_size, N, output_size] + [pop_size, 1, output_size]
        logits = logits + self.fc2_bias.unsqueeze(1)

        return logits

    def mutate(self) -> None:
        """Apply mutations to all networks in parallel using adaptive or fixed sigma."""
        if self.adaptive_sigma:
            # Adaptive sigma mutation - update sigmas then apply noise
            # Update fc1_weight sigma
            xi: Float[Tensor, "pop_size hidden_size input_size"] = (
                torch.randn_like(self.fc1_weight_sigma) * self.sigma_noise
            )
            self.fc1_weight_sigma = self.fc1_weight_sigma * (1 + xi)
            eps: Float[Tensor, "pop_size hidden_size input_size"] = (
                torch.randn_like(self.fc1_weight) * self.fc1_weight_sigma
            )
            self.fc1_weight = self.fc1_weight + eps

            # Update fc1_bias sigma
            xi = torch.randn_like(self.fc1_bias_sigma) * self.sigma_noise
            self.fc1_bias_sigma = self.fc1_bias_sigma * (1 + xi)
            eps = torch.randn_like(self.fc1_bias) * self.fc1_bias_sigma
            self.fc1_bias = self.fc1_bias + eps

            # Update fc2_weight sigma
            xi = torch.randn_like(self.fc2_weight_sigma) * self.sigma_noise
            self.fc2_weight_sigma = self.fc2_weight_sigma * (1 + xi)
            eps = torch.randn_like(self.fc2_weight) * self.fc2_weight_sigma
            self.fc2_weight = self.fc2_weight + eps

            # Update fc2_bias sigma
            xi = torch.randn_like(self.fc2_bias_sigma) * self.sigma_noise
            self.fc2_bias_sigma = self.fc2_bias_sigma * (1 + xi)
            eps = torch.randn_like(self.fc2_bias) * self.fc2_bias_sigma
            self.fc2_bias = self.fc2_bias + eps
        else:
            # Fixed sigma mutation
            self.fc1_weight = (
                self.fc1_weight
                + torch.randn_like(self.fc1_weight) * self.sigma_init
            )
            self.fc1_bias = (
                self.fc1_bias
                + torch.randn_like(self.fc1_bias) * self.sigma_init
            )
            self.fc2_weight = (
                self.fc2_weight
                + torch.randn_like(self.fc2_weight) * self.sigma_init
            )
            self.fc2_bias = (
                self.fc2_bias
                + torch.randn_like(self.fc2_bias) * self.sigma_init
            )

    def evaluate_episodes(
        self, episodes: list[dict[str, Float[Tensor, "..."]]]
    ) -> Float[Tensor, " pop_size"]:
        """Evaluate fitness of all networks on episodes (cross-entropy loss).

        Args:
            episodes: List of episode dictionaries with 'observations' and 'actions'

        Returns:
            Fitness tensor [pop_size] (lower is better)
        """
        with torch.no_grad():
            total_ce: Float[Tensor, " pop_size"] = torch.zeros(
                self.pop_size, device=DEVICE
            )
            total_samples: int = 0

            for episode in episodes:
                obs: Float[Tensor, "N input_size"] = episode["observations"]
                act: Int[Tensor, " N"] = episode["actions"]
                N: int = obs.shape[0]

                # Get logits for all networks: [pop_size, N, output_size]
                all_logits: Float[Tensor, "pop_size N output_size"] = (
                    self.forward_batch(obs)
                )

                # Expand actions to [pop_size, N]
                actions_expanded: Int[Tensor, "pop_size N"] = (
                    act.unsqueeze(0).expand(self.pop_size, -1)
                )

                # Reshape for cross_entropy: [pop_size * N, output_size] and [pop_size * N]
                flat_logits: Float[Tensor, "pop_sizexN output_size"] = (
                    all_logits.view(-1, self.output_size)
                )
                flat_actions: Int[Tensor, " pop_sizexN"] = (
                    actions_expanded.reshape(-1)
                )

                # Compute per-sample CE then reshape and sum per network
                per_sample_ce: Float[Tensor, " pop_sizexN"] = F.cross_entropy(
                    flat_logits, flat_actions, reduction="none"
                )
                per_network_ce: Float[Tensor, "pop_size N"] = (
                    per_sample_ce.view(self.pop_size, -1)
                )
                total_ce += per_network_ce.sum(dim=1)
                total_samples += N

            # Return mean CE loss per network
            fitness: Float[Tensor, " pop_size"] = total_ce / total_samples

        return fitness

    def select_simple_ga(
        self, fitness: Float[Tensor, " pop_size"], minimize: bool = True
    ) -> None:
        """Simple GA selection: top 50% survive and duplicate (hard selection).

        Args:
            fitness: Fitness values for each network [pop_size]
            minimize: Whether lower fitness is better (True for loss)
        """
        # Sort by fitness
        sorted_indices: Int[Tensor, " pop_size"] = torch.argsort(
            fitness, descending=not minimize
        )

        # Top 50% survive
        num_survivors: int = self.pop_size // 2
        survivor_indices: Int[Tensor, " num_survivors"] = sorted_indices[
            :num_survivors
        ]

        # Create mapping: each loser gets replaced by a survivor
        num_losers: int = self.pop_size - num_survivors
        replacement_indices: Int[Tensor, " num_losers"] = survivor_indices[
            torch.arange(num_losers, device=DEVICE) % num_survivors
        ]

        # Full new indices: survivors keep their params, losers get survivor params
        new_indices: Int[Tensor, " pop_size"] = torch.cat(
            [survivor_indices, replacement_indices]
        )

        # Reorder parameters using advanced indexing (creates copies)
        self.fc1_weight = self.fc1_weight[new_indices].clone()
        self.fc1_bias = self.fc1_bias[new_indices].clone()
        self.fc2_weight = self.fc2_weight[new_indices].clone()
        self.fc2_bias = self.fc2_bias[new_indices].clone()

        if self.adaptive_sigma:
            self.fc1_weight_sigma = self.fc1_weight_sigma[new_indices].clone()
            self.fc1_bias_sigma = self.fc1_bias_sigma[new_indices].clone()
            self.fc2_weight_sigma = self.fc2_weight_sigma[new_indices].clone()
            self.fc2_bias_sigma = self.fc2_bias_sigma[new_indices].clone()

    def select_simple_es(
        self, fitness: Float[Tensor, " pop_size"], minimize: bool = True
    ) -> None:
        """Simple ES selection: weighted combination of all networks (soft selection).

        Args:
            fitness: Fitness values for each network [pop_size]
            minimize: Whether lower fitness is better (True for loss)
        """
        # Standardize fitness
        if minimize:
            fitness_std: Float[Tensor, " pop_size"] = (
                -fitness - (-fitness).mean()
            ) / (fitness.std() + 1e-8)
        else:
            fitness_std = (fitness - fitness.mean()) / (fitness.std() + 1e-8)
        weights: Float[Tensor, " pop_size"] = F.softmax(fitness_std, dim=0)

        # Compute weighted average for each parameter tensor
        # Reshape weights for broadcasting
        w_fc1: Float[Tensor, "pop_size 1 1"] = weights.view(-1, 1, 1)
        avg_fc1_weight: Float[Tensor, "hidden_size input_size"] = (
            w_fc1 * self.fc1_weight
        ).sum(dim=0)
        self.fc1_weight = (
            avg_fc1_weight.unsqueeze(0).expand(self.pop_size, -1, -1).clone()
        )

        w_fc1_bias: Float[Tensor, "pop_size 1"] = weights.view(-1, 1)
        avg_fc1_bias: Float[Tensor, " hidden_size"] = (
            w_fc1_bias * self.fc1_bias
        ).sum(dim=0)
        self.fc1_bias = (
            avg_fc1_bias.unsqueeze(0).expand(self.pop_size, -1).clone()
        )

        w_fc2: Float[Tensor, "pop_size 1 1"] = weights.view(-1, 1, 1)
        avg_fc2_weight: Float[Tensor, "output_size hidden_size"] = (
            w_fc2 * self.fc2_weight
        ).sum(dim=0)
        self.fc2_weight = (
            avg_fc2_weight.unsqueeze(0).expand(self.pop_size, -1, -1).clone()
        )

        w_fc2_bias: Float[Tensor, "pop_size 1"] = weights.view(-1, 1)
        avg_fc2_bias: Float[Tensor, " output_size"] = (
            w_fc2_bias * self.fc2_bias
        ).sum(dim=0)
        self.fc2_bias = (
            avg_fc2_bias.unsqueeze(0).expand(self.pop_size, -1).clone()
        )

        if self.adaptive_sigma:
            avg_fc1_weight_sigma: Float[Tensor, "hidden_size input_size"] = (
                w_fc1 * self.fc1_weight_sigma
            ).sum(dim=0)
            self.fc1_weight_sigma = (
                avg_fc1_weight_sigma.unsqueeze(0)
                .expand(self.pop_size, -1, -1)
                .clone()
            )

            avg_fc1_bias_sigma: Float[Tensor, " hidden_size"] = (
                w_fc1_bias * self.fc1_bias_sigma
            ).sum(dim=0)
            self.fc1_bias_sigma = (
                avg_fc1_bias_sigma.unsqueeze(0)
                .expand(self.pop_size, -1)
                .clone()
            )

            avg_fc2_weight_sigma: Float[Tensor, "output_size hidden_size"] = (
                w_fc2 * self.fc2_weight_sigma
            ).sum(dim=0)
            self.fc2_weight_sigma = (
                avg_fc2_weight_sigma.unsqueeze(0)
                .expand(self.pop_size, -1, -1)
                .clone()
            )

            avg_fc2_bias_sigma: Float[Tensor, " output_size"] = (
                w_fc2_bias * self.fc2_bias_sigma
            ).sum(dim=0)
            self.fc2_bias_sigma = (
                avg_fc2_bias_sigma.unsqueeze(0)
                .expand(self.pop_size, -1)
                .clone()
            )

    def get_best_network_state(
        self, fitness: Float[Tensor, " pop_size"], minimize: bool = True
    ) -> tuple[
        Float[Tensor, "hidden_size input_size"],
        Float[Tensor, " hidden_size"],
        Float[Tensor, "output_size hidden_size"],
        Float[Tensor, " output_size"],
    ]:
        """Get the parameters of the best performing network.

        Args:
            fitness: Fitness values for each network [pop_size]
            minimize: Whether lower fitness is better (True for loss)

        Returns:
            Tuple of (fc1_weight, fc1_bias, fc2_weight, fc2_bias)
        """
        if minimize:
            best_idx: int = torch.argmin(fitness).item()
        else:
            best_idx: int = torch.argmax(fitness).item()
        return (
            self.fc1_weight[best_idx],
            self.fc1_bias[best_idx],
            self.fc2_weight[best_idx],
            self.fc2_bias[best_idx],
        )

    def create_best_model(
        self, fitness: Float[Tensor, " pop_size"], minimize: bool = True
    ) -> MLP:
        """Create an MLP from the best network's parameters.

        Args:
            fitness: Fitness values for each network [pop_size]
            minimize: Whether lower fitness is better (True for loss)

        Returns:
            MLP model with best network's parameters
        """
        fc1_w, fc1_b, fc2_w, fc2_b = self.get_best_network_state(
            fitness, minimize
        )
        mlp: MLP = MLP(self.input_size, self.hidden_size, self.output_size).to(
            DEVICE
        )
        mlp.fc1.weight.data = fc1_w
        mlp.fc1.bias.data = fc1_b
        mlp.fc2.weight.data = fc2_w
        mlp.fc2.bias.data = fc2_b
        return mlp

    def get_state_dict(self) -> dict[str, Tensor]:
        """Get state dict for checkpointing.

        Returns:
            Dictionary containing all parameters and sigma values
        """
        state: dict[str, Tensor] = {
            "fc1_weight": self.fc1_weight,
            "fc1_bias": self.fc1_bias,
            "fc2_weight": self.fc2_weight,
            "fc2_bias": self.fc2_bias,
        }
        if self.adaptive_sigma:
            state["fc1_weight_sigma"] = self.fc1_weight_sigma
            state["fc1_bias_sigma"] = self.fc1_bias_sigma
            state["fc2_weight_sigma"] = self.fc2_weight_sigma
            state["fc2_bias_sigma"] = self.fc2_bias_sigma
        return state

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Load state dict from checkpoint.

        Args:
            state: Dictionary containing saved parameters and sigma values
        """
        self.fc1_weight = state["fc1_weight"]
        self.fc1_bias = state["fc1_bias"]
        self.fc2_weight = state["fc2_weight"]
        self.fc2_bias = state["fc2_bias"]
        if self.adaptive_sigma and "fc1_weight_sigma" in state:
            self.fc1_weight_sigma = state["fc1_weight_sigma"]
            self.fc1_bias_sigma = state["fc1_bias_sigma"]
            self.fc2_weight_sigma = state["fc2_weight_sigma"]
            self.fc2_bias_sigma = state["fc2_bias_sigma"]


def _optimize_neuroevolution(
    algorithm: str,
    input_size: int,
    hidden_size: int,
    output_size: int,
    optim_obs: Float[Tensor, "optim_size input_size"],
    optim_act: Int[Tensor, " optim_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    metadata: dict,
    checkpoint_path: Path,
    max_optim_time: int = 36000,
    population_size: int = 50,
    adaptive_sigma: bool = True,
    sigma_init: float = 1e-3,
    sigma_noise: float = 1e-2,
    loss_eval_interval_seconds: int = 60,
    logger=None,
) -> tuple[list[float], list[float]]:
    """Shared optimization loop for Simple GA and Simple ES on feedforward models.

    Args:
        algorithm: 'ga' (hard selection) or 'es' (soft selection)
        input_size: Input dimension
        hidden_size: Hidden dimension
        output_size: Output dimension (number of actions)
        optim_obs: Optimization observations
        optim_act: Optimization actions
        test_obs: Test observations
        test_act: Test actions
        metadata: Metadata dict with episode_boundaries
        checkpoint_path: Path to save checkpoints
        max_optim_time: Maximum optimization time in seconds
        population_size: Population size
        adaptive_sigma: Whether to use adaptive sigma mutation
        sigma_init: Initial mutation strength
        sigma_noise: Noise level for sigma adaptation
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
    population = BatchedPopulation(
        input_size,
        hidden_size,
        output_size,
        population_size,
        adaptive_sigma=adaptive_sigma,
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

    print(f"  Optimizing with {algorithm.upper()} for {max_optim_time}s ({max_optim_time/60:.1f} min)...")

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

        # Selection (GA or ES)
        if algorithm == 'ga':
            population.select_simple_ga(fitness, minimize=True)
        elif algorithm == 'es':
            population.select_simple_es(fitness, minimize=True)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

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
                "algorithm": algorithm,
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
        "algorithm": algorithm,
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"  Final checkpoint saved to {checkpoint_path}")

    return fitness_history, test_loss_history


def optimize_ga_feedforward(
    input_size: int,
    hidden_size: int,
    output_size: int,
    optim_obs: Float[Tensor, "optim_size input_size"],
    optim_act: Int[Tensor, " optim_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    metadata: dict,
    checkpoint_path: Path,
    max_optim_time: int = 36000,
    population_size: int = 50,
    adaptive_sigma: bool = True,
    sigma_init: float = 1e-3,
    sigma_noise: float = 1e-2,
    loss_eval_interval_seconds: int = 60,
    logger=None,
) -> tuple[list[float], list[float]]:
    """Optimize feedforward MLP using Simple Genetic Algorithm (hard selection).

    Uses truncation selection where top 50% survive and bottom 50% are replaced
    by copies of survivors.

    Args:
        input_size: Input dimension
        hidden_size: Hidden dimension
        output_size: Output dimension (number of actions)
        optim_obs: Optimization observations
        optim_act: Optimization actions
        test_obs: Test observations
        test_act: Test actions
        metadata: Metadata dict with episode_boundaries
        checkpoint_path: Path to save checkpoints
        max_optim_time: Maximum optimization time in seconds
        population_size: Population size
        adaptive_sigma: Whether to use adaptive sigma mutation
        sigma_init: Initial mutation strength
        sigma_noise: Noise level for sigma adaptation
        loss_eval_interval_seconds: Evaluation interval in seconds
        logger: Optional ExperimentLogger for database logging

    Returns:
        Tuple of (fitness_history, test_loss_history)
    """
    return _optimize_neuroevolution(
        algorithm='ga',
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        optim_obs=optim_obs,
        optim_act=optim_act,
        test_obs=test_obs,
        test_act=test_act,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        max_optim_time=max_optim_time,
        population_size=population_size,
        adaptive_sigma=adaptive_sigma,
        sigma_init=sigma_init,
        sigma_noise=sigma_noise,
        loss_eval_interval_seconds=loss_eval_interval_seconds,
        logger=logger,
    )


def optimize_es_feedforward(
    input_size: int,
    hidden_size: int,
    output_size: int,
    optim_obs: Float[Tensor, "optim_size input_size"],
    optim_act: Int[Tensor, " optim_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    metadata: dict,
    checkpoint_path: Path,
    max_optim_time: int = 36000,
    population_size: int = 50,
    adaptive_sigma: bool = True,
    sigma_init: float = 1e-3,
    sigma_noise: float = 1e-2,
    loss_eval_interval_seconds: int = 60,
    logger=None,
) -> tuple[list[float], list[float]]:
    """Optimize feedforward MLP using Simple Evolution Strategies (soft selection).

    Uses fitness-weighted combination of all individuals where better networks
    contribute more to the next generation parameters.

    Args:
        input_size: Input dimension
        hidden_size: Hidden dimension
        output_size: Output dimension (number of actions)
        optim_obs: Optimization observations
        optim_act: Optimization actions
        test_obs: Test observations
        test_act: Test actions
        metadata: Metadata dict with episode_boundaries
        checkpoint_path: Path to save checkpoints
        max_optim_time: Maximum optimization time in seconds
        population_size: Population size
        adaptive_sigma: Whether to use adaptive sigma mutation
        sigma_init: Initial mutation strength
        sigma_noise: Noise level for sigma adaptation
        loss_eval_interval_seconds: Evaluation interval in seconds
        logger: Optional ExperimentLogger for database logging

    Returns:
        Tuple of (fitness_history, test_loss_history)
    """
    return _optimize_neuroevolution(
        algorithm='es',
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        optim_obs=optim_obs,
        optim_act=optim_act,
        test_obs=test_obs,
        test_act=test_act,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        max_optim_time=max_optim_time,
        population_size=population_size,
        adaptive_sigma=adaptive_sigma,
        sigma_init=sigma_init,
        sigma_noise=sigma_noise,
        loss_eval_interval_seconds=loss_eval_interval_seconds,
        logger=logger,
    )


class CMAESPopulation:
    """Diagonal CMA-ES population for feedforward MLPs.

    Implements separable/diagonal CMA-ES which adapts only the diagonal of the
    covariance matrix. This is more practical for neural networks with large
    parameter spaces than full CMA-ES.

    Based on Hansen & Ostermeier (2001) "Completely Derandomized Self-Adaptation"
    with diagonal approximation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        pop_size: int,
        sigma_init: float = 1e-1,
    ) -> None:
        """Initialize diagonal CMA-ES population.

        Args:
            input_size: Input dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension (number of actions)
            pop_size: Population size (lambda)
            sigma_init: Initial global step size
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pop_size = pop_size  # lambda
        self.mu = pop_size // 2  # Number of parents for recombination

        # Calculate parameter dimensions
        self.fc1_weight_size = hidden_size * input_size
        self.fc1_bias_size = hidden_size
        self.fc2_weight_size = output_size * hidden_size
        self.fc2_bias_size = output_size
        self.n_params = (
            self.fc1_weight_size
            + self.fc1_bias_size
            + self.fc2_weight_size
            + self.fc2_bias_size
        )

        # Initialize mean (m) - flattened parameter vector
        self.mean = torch.randn(self.n_params, device=DEVICE) * 0.01

        # Global step size (sigma)
        self.sigma = sigma_init

        # Diagonal covariance (diagonal of C)
        self.C_diag = torch.ones(self.n_params, device=DEVICE)

        # Evolution paths
        self.p_sigma = torch.zeros(self.n_params, device=DEVICE)
        self.p_c = torch.zeros(self.n_params, device=DEVICE)

        # Strategy parameters (standard CMA-ES settings)
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
        self.offspring = torch.zeros(
            self.pop_size, self.n_params, device=DEVICE
        )

    def _compute_mu_eff(self) -> float:
        """Compute variance effectiveness of weighted recombination."""
        weights_prime = torch.log(torch.tensor(self.mu + 0.5, device=DEVICE)) - torch.log(
            torch.arange(1, self.mu + 1, dtype=torch.float32, device=DEVICE)
        )
        weights_prime = weights_prime / weights_prime.sum()
        mu_eff = 1.0 / (weights_prime ** 2).sum()
        return mu_eff.item()

    def _compute_weights(self) -> Float[Tensor, " mu"]:
        """Compute recombination weights for top mu individuals."""
        weights = torch.log(torch.tensor(self.mu + 0.5, device=DEVICE)) - torch.log(
            torch.arange(1, self.mu + 1, dtype=torch.float32, device=DEVICE)
        )
        weights = weights / weights.sum()
        return weights

    def _flatten_params(
        self,
        fc1_w: Float[Tensor, "hidden_size input_size"],
        fc1_b: Float[Tensor, " hidden_size"],
        fc2_w: Float[Tensor, "output_size hidden_size"],
        fc2_b: Float[Tensor, " output_size"],
    ) -> Float[Tensor, " n_params"]:
        """Flatten MLP parameters into a single vector."""
        return torch.cat([
            fc1_w.reshape(-1),
            fc1_b.reshape(-1),
            fc2_w.reshape(-1),
            fc2_b.reshape(-1),
        ])

    def _unflatten_params(
        self, flat: Float[Tensor, " n_params"]
    ) -> tuple[
        Float[Tensor, "hidden_size input_size"],
        Float[Tensor, " hidden_size"],
        Float[Tensor, "output_size hidden_size"],
        Float[Tensor, " output_size"],
    ]:
        """Unflatten parameter vector back to MLP parameters."""
        idx = 0
        fc1_w = flat[idx : idx + self.fc1_weight_size].reshape(
            self.hidden_size, self.input_size
        )
        idx += self.fc1_weight_size

        fc1_b = flat[idx : idx + self.fc1_bias_size]
        idx += self.fc1_bias_size

        fc2_w = flat[idx : idx + self.fc2_weight_size].reshape(
            self.output_size, self.hidden_size
        )
        idx += self.fc2_weight_size

        fc2_b = flat[idx : idx + self.fc2_bias_size]

        return fc1_w, fc1_b, fc2_w, fc2_b

    def sample_offspring(self) -> None:
        """Sample lambda offspring from current distribution N(m, sigma^2 * diag(C))."""
        for i in range(self.pop_size):
            # Sample from N(0, I)
            z = torch.randn(self.n_params, device=DEVICE)
            # Scale by sqrt(C_diag) and sigma, then add mean
            self.offspring[i] = self.mean + self.sigma * torch.sqrt(self.C_diag) * z

    def evaluate_episodes(
        self, episodes: list[dict[str, Float[Tensor, "..."]]]
    ) -> Float[Tensor, " pop_size"]:
        """Evaluate fitness of all offspring on episodes (cross-entropy loss).

        Args:
            episodes: List of episode dictionaries with 'observations' and 'actions'

        Returns:
            Fitness tensor [pop_size] (lower is better)
        """
        with torch.no_grad():
            fitness = torch.zeros(self.pop_size, device=DEVICE)

            for i in range(self.pop_size):
                # Unflatten parameters for this offspring
                fc1_w, fc1_b, fc2_w, fc2_b = self._unflatten_params(self.offspring[i])

                total_ce = 0.0
                total_samples = 0

                for episode in episodes:
                    obs = episode["observations"]
                    act = episode["actions"]
                    N = obs.shape[0]

                    # Forward pass
                    h = torch.tanh(obs @ fc1_w.T + fc1_b)
                    logits = h @ fc2_w.T + fc2_b

                    # Cross-entropy loss
                    ce = F.cross_entropy(logits, act, reduction="sum")
                    total_ce += ce.item()
                    total_samples += N

                fitness[i] = total_ce / total_samples

        return fitness

    def update(self, fitness: Float[Tensor, " pop_size"]) -> None:
        """Update distribution parameters using CMA-ES update rules.

        Args:
            fitness: Fitness values for all offspring [pop_size] (minimization)
        """
        # Select top mu offspring
        sorted_indices = torch.argsort(fitness)
        elite_indices = sorted_indices[: self.mu]

        # Weighted recombination to get new mean
        old_mean = self.mean.clone()
        self.mean = torch.zeros_like(self.mean)
        for i, idx in enumerate(elite_indices):
            self.mean += self.weights[i] * self.offspring[idx]

        # Mean shift
        mean_shift = self.mean - old_mean

        # Update evolution path for sigma (cumulation)
        # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^(-1/2) * (m - m_old) / sigma
        # For diagonal C, C^(-1/2) = 1/sqrt(C_diag)
        c_sigma_factor = (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) ** 0.5
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + c_sigma_factor * mean_shift / (
            self.sigma * torch.sqrt(self.C_diag)
        )

        # Update global step size sigma
        p_sigma_norm = torch.norm(self.p_sigma).item()
        self.sigma = self.sigma * math.exp(
            (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)
        )

        # Update evolution path for C (cumulation)
        h_sig = 1.0 if p_sigma_norm / ((1.0 - (1.0 - self.c_sigma) ** (2.0 * self.mu_eff)) ** 0.5) < 1.4 + 2.0 / (self.n_params + 1.0) else 0.0
        c_c_factor = (self.c_c * (2.0 - self.c_c) * self.mu_eff) ** 0.5
        self.p_c = (1.0 - self.c_c) * self.p_c + h_sig * c_c_factor * mean_shift / self.sigma

        # Update diagonal covariance matrix
        # C = (1 - c_1 - c_mu) * C + c_1 * p_c * p_c^T + c_mu * sum(w_i * y_i * y_i^T)
        # For diagonal, this becomes element-wise operations
        self.C_diag = (
            (1.0 - self.c_1 - self.c_mu) * self.C_diag
            + self.c_1 * self.p_c ** 2
        )

        # Rank-mu update (diagonal version)
        for i, idx in enumerate(elite_indices):
            y_i = (self.offspring[idx] - old_mean) / self.sigma
            self.C_diag += self.c_mu * self.weights[i] * y_i ** 2

        # Ensure C_diag stays positive
        self.C_diag = torch.clamp(self.C_diag, min=1e-10)

    def create_best_model(self) -> MLP:
        """Create an MLP from the current mean parameters.

        Returns:
            MLP model with mean parameters
        """
        fc1_w, fc1_b, fc2_w, fc2_b = self._unflatten_params(self.mean)
        mlp = MLP(self.input_size, self.hidden_size, self.output_size).to(
            DEVICE
        )
        mlp.fc1.weight.data = fc1_w
        mlp.fc1.bias.data = fc1_b
        mlp.fc2.weight.data = fc2_w
        mlp.fc2.bias.data = fc2_b
        return mlp

    def get_state_dict(self) -> dict[str, Tensor]:
        """Get state dict for checkpointing.

        Returns:
            Dictionary containing all CMA-ES state
        """
        return {
            "mean": self.mean,
            "sigma": torch.tensor(self.sigma, device=DEVICE),
            "C_diag": self.C_diag,
            "p_sigma": self.p_sigma,
            "p_c": self.p_c,
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Load state dict from checkpoint.

        Args:
            state: Dictionary containing saved CMA-ES state
        """
        self.mean = state["mean"]
        self.sigma = state["sigma"].item()
        self.C_diag = state["C_diag"]
        self.p_sigma = state["p_sigma"]
        self.p_c = state["p_c"]


def optimize_cmaes_feedforward(
    input_size: int,
    hidden_size: int,
    output_size: int,
    optim_obs: Float[Tensor, "optim_size input_size"],
    optim_act: Int[Tensor, " optim_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    metadata: dict,
    checkpoint_path: Path,
    max_optim_time: int = 36000,
    population_size: int = 50,
    sigma_init: float = 1e-1,
    loss_eval_interval_seconds: int = 60,
    logger=None,
) -> tuple[list[float], list[float]]:
    """Optimize feedforward MLP using Diagonal CMA-ES.

    Uses Covariance Matrix Adaptation Evolution Strategy with diagonal approximation
    for adaptive step size and coordinate-wise variance adaptation.

    Args:
        input_size: Input dimension
        hidden_size: Hidden dimension
        output_size: Output dimension (number of actions)
        optim_obs: Optimization observations
        optim_act: Optimization actions
        test_obs: Test observations
        test_act: Test actions
        metadata: Metadata dict with episode_boundaries
        checkpoint_path: Path to save checkpoints
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
    population = CMAESPopulation(
        input_size,
        hidden_size,
        output_size,
        population_size,
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
