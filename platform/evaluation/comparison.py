"""Behavioral comparison functions for evaluating models against human behavior.

This module provides functions for comparing model behavior to human behavior
by running models on matched environment episodes and computing statistics.
"""

import gymnasium as gym
import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

from platform import config


def evaluate_progression_recurrent(
    model,
    episode_details: list[dict],
    env: gym.Env,
    max_steps: int = 1000,
    use_cl_features: bool = False,
) -> tuple[float, float, list[float]]:
    """Quick evaluation to track progression during optimization (for recurrent models).

    Runs recurrent model on matched human episodes and returns statistics.

    Args:
        model: Optimized recurrent policy (RecurrentMLPReservoir or RecurrentMLPTrainable)
        episode_details: List of dicts with episode info (seed, return, norm_session, norm_run)
        env: Gym environment
        max_steps: Maximum steps per episode
        use_cl_features: Whether the model expects CL features in input

    Returns:
        Tuple of (mean_pct_diff, std_pct_diff, model_returns)
            mean_pct_diff: Mean percentage difference from human returns
            std_pct_diff: Std percentage difference from human returns
            model_returns: List of model returns for each episode
    """
    model.eval()
    pct_differences: list[float] = []
    model_returns: list[float] = []

    with torch.no_grad():
        for ep_detail in episode_details:
            seed: int = ep_detail["seed"]
            human_return: float = ep_detail["return"]

            # Reset with seed
            obs, _ = env.reset(seed=seed)

            # Initialize hidden state
            h_t: Float[Tensor, "1 hidden_size"] = torch.zeros(
                1, model.hidden_size, device=config.DEVICE
            )

            # Append CL features if provided
            if use_cl_features:
                obs_with_cl: np.ndarray = np.concatenate(
                    [obs, [ep_detail["norm_session"], ep_detail["norm_run"]]]
                )
            else:
                obs_with_cl = obs

            obs_tensor: Float[Tensor, "1 input_size"] = (
                torch.from_numpy(obs_with_cl).float().unsqueeze(0).to(config.DEVICE)
            )

            total_return: float = 0.0
            step: int = 0
            terminated: bool = False
            truncated: bool = False

            while step < max_steps and not (terminated or truncated):
                # Get action from model (with hidden state)
                probs: Float[Tensor, "1 output_size"]
                probs, h_t = model.get_probs(obs_tensor, h_t)
                action: int = torch.multinomial(probs.squeeze(0), num_samples=1).item()

                # Step environment
                obs, reward, terminated, truncated, _ = env.step(action)

                # Prepare next observation
                if use_cl_features:
                    obs_with_cl = np.concatenate(
                        [obs, [ep_detail["norm_session"], ep_detail["norm_run"]]]
                    )
                else:
                    obs_with_cl = obs

                obs_tensor = (
                    torch.from_numpy(obs_with_cl).float().unsqueeze(0).to(config.DEVICE)
                )

                total_return += reward
                step += 1

            # Store model return
            model_returns.append(total_return)

            # Calculate percentage difference
            if human_return != 0:
                pct_diff: float = (
                    (total_return - human_return) / abs(human_return)
                ) * 100
            else:
                pct_diff = total_return * 100

            pct_differences.append(pct_diff)

    mean_pct_diff: float = float(np.mean(pct_differences))
    std_pct_diff: float = float(np.std(pct_differences))
    return mean_pct_diff, std_pct_diff, model_returns


def create_episode_list(
    observations: Float[Tensor, "N input_size"],
    actions: Int[Tensor, " N"],
    episode_boundaries: list[tuple[int, int]],
) -> list[dict]:
    """Convert flat data with episode boundaries into list of episode dicts.

    Args:
        observations: Flat tensor of all observations
        actions: Flat tensor of all actions
        episode_boundaries: List of (start_idx, length) tuples

    Returns:
        List of dicts with 'observations' and 'actions' tensors
    """
    episodes: list[dict] = []
    for start, length in episode_boundaries:
        episodes.append(
            {
                "observations": observations[start : start + length],
                "actions": actions[start : start + length],
            }
        )
    return episodes
