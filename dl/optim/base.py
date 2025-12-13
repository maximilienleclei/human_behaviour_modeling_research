"""Shared utilities for optimizers.

This module contains common functions used by both SGD and GA optimizers.
"""

from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor


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


def save_checkpoint(
    checkpoint_path: Path,
    checkpoint_data: dict,
) -> None:
    """Save checkpoint to disk.

    Args:
        checkpoint_path: Path to save checkpoint
        checkpoint_data: Dictionary of checkpoint data
    """
    torch.save(checkpoint_data, checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
) -> dict | None:
    """Load checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Checkpoint dictionary or None if not found
    """
    if checkpoint_path.exists():
        return torch.load(checkpoint_path, weights_only=False)
    return None
