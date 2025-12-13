"""Data preprocessing functions for continual learning features and episode handling.

This module provides functions for computing session/run IDs from timestamps,
normalizing continual learning features, and creating episode-based datasets
for recurrent model training.
"""

from datetime import datetime

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import Dataset


def compute_session_run_ids(
    timestamps: list[str],
) -> tuple[list[int], list[int]]:
    """Compute session and run IDs from episode timestamps.

    A new session begins if >= 30 minutes have passed since the previous episode.
    Within a session, runs are numbered sequentially starting from 0.

    Args:
        timestamps: List of ISO format timestamp strings

    Returns:
        Tuple of (session_ids, run_ids) - both are lists of integers
    """
    if len(timestamps) == 0:
        return [], []

    # Parse all timestamps
    dt_list: list[datetime] = [datetime.fromisoformat(ts) for ts in timestamps]

    session_ids: list[int] = []
    run_ids: list[int] = []

    current_session: int = 0
    current_run: int = 0

    session_ids.append(current_session)
    run_ids.append(current_run)

    # Threshold: 30 minutes = 1800 seconds
    session_threshold_seconds: float = 30 * 60

    for i in range(1, len(dt_list)):
        time_diff: float = (dt_list[i] - dt_list[i - 1]).total_seconds()

        if time_diff >= session_threshold_seconds:
            # New session
            current_session += 1
            current_run = 0
        else:
            # Same session, new run
            current_run += 1

        session_ids.append(current_session)
        run_ids.append(current_run)

    return session_ids, run_ids


def normalize_session_run_features(
    session_ids: list[int], run_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize session and run IDs to [-1, 1] range.

    Sessions are mapped with equal spacing across all data.
    Runs are mapped with equal spacing within each session.

    Args:
        session_ids: List of session IDs (integers)
        run_ids: List of run IDs (integers)

    Returns:
        Tuple of (normalized_sessions, normalized_runs) as numpy arrays
    """
    session_arr: np.ndarray = np.array(session_ids, dtype=np.int64)
    run_arr: np.ndarray = np.array(run_ids, dtype=np.int64)

    # Normalize sessions globally
    unique_sessions: np.ndarray = np.unique(session_arr)
    num_sessions: int = len(unique_sessions)

    if num_sessions == 1:
        normalized_sessions: np.ndarray = np.zeros(
            len(session_arr), dtype=np.float32
        )
    else:
        # Map sessions to [-1, 1] with equal spacing
        session_to_normalized: dict[int, float] = {
            s: -1.0 + 2.0 * i / (num_sessions - 1)
            for i, s in enumerate(unique_sessions)
        }
        normalized_sessions = np.array(
            [session_to_normalized[s] for s in session_arr], dtype=np.float32
        )

    # Normalize runs within each session
    normalized_runs: np.ndarray = np.zeros(len(run_arr), dtype=np.float32)

    for session_id in unique_sessions:
        mask: np.ndarray = session_arr == session_id
        runs_in_session: np.ndarray = run_arr[mask]
        unique_runs: np.ndarray = np.unique(runs_in_session)
        num_runs: int = len(unique_runs)

        if num_runs == 1:
            normalized_runs[mask] = 0.0
        else:
            # Map runs to [-1, 1] with equal spacing
            run_to_normalized: dict[int, float] = {
                r: -1.0 + 2.0 * i / (num_runs - 1)
                for i, r in enumerate(unique_runs)
            }
            for idx in np.where(mask)[0]:
                normalized_runs[idx] = run_to_normalized[run_arr[idx]]

    return normalized_sessions, normalized_runs


class EpisodeDataset(Dataset):
    """Dataset that returns complete episodes instead of individual steps.

    This dataset is used for training recurrent models where the entire
    episode sequence needs to be processed together.
    """

    def __init__(
        self,
        observations: Float[Tensor, "N input_size"],
        actions: Int[Tensor, " N"],
        episode_boundaries: list[tuple[int, int]],
    ) -> None:
        """Initialize episode dataset.

        Args:
            observations: Flat tensor of all observations
            actions: Flat tensor of all actions
            episode_boundaries: List of (start_idx, length) tuples for each episode
        """
        self.observations: Float[Tensor, "N input_size"] = observations
        self.actions: Int[Tensor, " N"] = actions
        self.episode_boundaries: list[tuple[int, int]] = episode_boundaries

    def __len__(self) -> int:
        return len(self.episode_boundaries)

    def __getitem__(self, idx: int) -> dict:
        start, length = self.episode_boundaries[idx]
        return {
            "observations": self.observations[start : start + length],
            "actions": self.actions[start : start + length],
            "length": length,
        }


def episode_collate_fn(batch: list[dict]) -> dict:
    """Collate episodes with padding to handle variable lengths.

    Args:
        batch: List of dicts from EpisodeDataset

    Returns:
        Dict with padded observations, actions, mask, and lengths
    """
    max_len: int = max(item["length"] for item in batch)
    batch_size: int = len(batch)
    input_size: int = batch[0]["observations"].shape[1]

    # Initialize padded tensors
    obs_padded: Float[Tensor, "BS max_len input_size"] = torch.zeros(
        batch_size, max_len, input_size
    )
    act_padded: Int[Tensor, "BS max_len"] = torch.zeros(
        batch_size, max_len, dtype=torch.long
    )
    mask: Float[Tensor, "BS max_len"] = torch.zeros(
        batch_size, max_len, dtype=torch.bool
    )

    for i, item in enumerate(batch):
        length: int = item["length"]
        obs_padded[i, :length] = item["observations"]
        act_padded[i, :length] = item["actions"]
        mask[i, :length] = True

    return {
        "observations": obs_padded,
        "actions": act_padded,
        "mask": mask,
        "lengths": torch.tensor([item["length"] for item in batch]),
    }
