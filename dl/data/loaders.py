"""Data loading functions for HuggingFace and human behavioral datasets.

This module handles loading data from various sources including HuggingFace
datasets (CartPole-v1, LunarLander-v2) and local JSON files containing human
behavioral data.
"""

import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from jaxtyping import Float, Int
from torch import Tensor

from config.environments import ENV_CONFIGS, get_data_file
from config.paths import DATA_DIR
from dl.data.preprocessing import (
    compute_session_run_ids,
    normalize_session_run_features,
)


def load_cartpole_data() -> tuple[
    Float[Tensor, "train_size 4"],
    Int[Tensor, " train_size"],
    Float[Tensor, "test_size 4"],
    Int[Tensor, " test_size"],
]:
    """Load CartPole-v1 dataset from HuggingFace."""
    dataset = load_dataset("NathanGavenski/CartPole-v1")

    print("  Converting observations to numpy...")
    obs_np: np.ndarray = np.array(dataset["train"]["obs"], dtype=np.float32)
    print("  Converting actions to numpy...")
    act_np: np.ndarray = np.array(dataset["train"]["actions"], dtype=np.int64)

    print("  Converting to tensors...")
    obs_tensor: Float[Tensor, "N 4"] = torch.from_numpy(obs_np)
    act_tensor: Int[Tensor, " N"] = torch.from_numpy(act_np)

    # Shuffle
    print("  Shuffling...")
    num_samples: int = obs_tensor.shape[0]
    perm: Int[Tensor, " N"] = torch.randperm(num_samples)
    obs_tensor = obs_tensor[perm]
    act_tensor = act_tensor[perm]

    # Split
    train_size: int = int(num_samples * 0.9)
    train_obs: Float[Tensor, "train_size 4"] = obs_tensor[:train_size]
    train_act: Int[Tensor, " train_size"] = act_tensor[:train_size]
    test_obs: Float[Tensor, "test_size 4"] = obs_tensor[train_size:]
    test_act: Int[Tensor, " test_size"] = act_tensor[train_size:]

    print(
        f"  Done: {train_obs.shape[0]} train, {test_obs.shape[0]} test samples"
    )
    return train_obs, train_act, test_obs, test_act


def load_lunarlander_data() -> tuple[
    Float[Tensor, "train_size 8"],
    Int[Tensor, " train_size"],
    Float[Tensor, "test_size 8"],
    Int[Tensor, " test_size"],
]:
    """Load LunarLander-v2 dataset from HuggingFace."""
    dataset = load_dataset("NathanGavenski/LunarLander-v2")

    print("  Converting observations to numpy...")
    obs_np: np.ndarray = np.array(dataset["train"]["obs"], dtype=np.float32)
    print("  Converting actions to numpy...")
    act_np: np.ndarray = np.array(dataset["train"]["actions"], dtype=np.int64)

    print("  Converting to tensors...")
    obs_tensor: Float[Tensor, "N 8"] = torch.from_numpy(obs_np)
    act_tensor: Int[Tensor, " N"] = torch.from_numpy(act_np)

    # Shuffle
    print("  Shuffling...")
    num_samples: int = obs_tensor.shape[0]
    perm: Int[Tensor, " N"] = torch.randperm(num_samples)
    obs_tensor = obs_tensor[perm]
    act_tensor = act_tensor[perm]

    # Split
    train_size: int = int(num_samples * 0.9)
    train_obs: Float[Tensor, "train_size 8"] = obs_tensor[:train_size]
    train_act: Int[Tensor, " train_size"] = act_tensor[:train_size]
    test_obs: Float[Tensor, "test_size 8"] = obs_tensor[train_size:]
    test_act: Int[Tensor, " test_size"] = act_tensor[train_size:]

    print(
        f"  Done: {train_obs.shape[0]} train, {test_obs.shape[0]} test samples"
    )
    return train_obs, train_act, test_obs, test_act


def load_human_data(
    env_name: str,
    use_cl_info: bool,
    subject: str = "sub01",
    holdout_pct: float = 0.1,
) -> tuple[
    Float[Tensor, "optim_size input_size"],
    Int[Tensor, " optim_size"],
    Float[Tensor, "test_size input_size"],
    Int[Tensor, " test_size"],
    dict,
]:
    """Load human behavior data from JSON files with random run holdout.

    Args:
        env_name: Environment name (cartpole, mountaincar, acrobot, lunarlander)
        use_cl_info: Whether to include session/run features in observations
        subject: Subject identifier (sub01, sub02)
        holdout_pct: Percentage of runs to randomly hold out for testing

    Returns:
        Tuple of (optim_obs, optim_act, test_obs, test_act, metadata)
        metadata contains:
            - test_episode_info: list of dicts with episode-level info for test set
            - session_ids: full list of session IDs per episode
            - run_ids: full list of run IDs per episode
            - num_optim_episodes: number of episodes in optimization set
            - num_test_episodes: number of episodes in test set
            - optim_episode_boundaries: list of (start_idx, length) tuples for optim
            - test_episode_boundaries: list of (start_idx, length) tuples for test
    """
    env_config: dict = ENV_CONFIGS[env_name]
    data_filename: str = get_data_file(env_name, subject)
    data_file: Path = DATA_DIR / data_filename

    print(f"  Loading {subject}'s data from {data_file}...")

    # Load JSON
    with open(data_file, "r") as f:
        episodes: list[dict] = json.load(f)

    print(f"  Loaded {len(episodes)} episodes")

    # Compute session and run IDs from episode timestamps
    episode_timestamps: list[str] = [ep["timestamp"] for ep in episodes]
    session_ids, run_ids = compute_session_run_ids(episode_timestamps)

    num_episodes: int = len(episodes)
    num_sessions: int = len(set(session_ids))
    print(f"  Found {num_sessions} sessions")

    # Per-session random split: within each session, hold out 10% of runs (min 1)
    np.random.seed(42)
    test_episode_indices: set[int] = set()
    optim_episode_indices: set[int] = set()

    for session_id in sorted(set(session_ids)):
        # Get all episode indices for this session
        session_ep_indices: list[int] = [
            i for i in range(len(session_ids)) if session_ids[i] == session_id
        ]
        num_runs_in_session: int = len(session_ep_indices)

        # Determine how many to hold out: 10% or minimum 1
        if num_runs_in_session < 10:
            num_test_runs: int = 1
        else:
            num_test_runs = max(1, int(num_runs_in_session * holdout_pct))

        # Randomly select test runs from this session
        shuffled: list[int] = session_ep_indices.copy()
        np.random.shuffle(shuffled)

        test_episode_indices.update(shuffled[:num_test_runs])
        optim_episode_indices.update(shuffled[num_test_runs:])

        print(
            f"    Session {session_id}: {num_runs_in_session} runs "
            f"→ {num_runs_in_session - num_test_runs} optim, {num_test_runs} test"
        )

    num_optim_episodes: int = len(optim_episode_indices)
    num_test_episodes: int = len(test_episode_indices)

    print(
        f"  Total: {num_optim_episodes} optim episodes, {num_test_episodes} test episodes "
        f"across {num_sessions} sessions"
    )

    # Normalize session/run features using ALL data (to preserve [-1, 1] range)
    # Expand to step level for normalization
    all_session_ids: list[int] = []
    all_run_ids: list[int] = []
    for ep_idx in range(num_episodes):
        num_steps: int = len(episodes[ep_idx]["steps"])
        all_session_ids.extend([session_ids[ep_idx]] * num_steps)
        all_run_ids.extend([run_ids[ep_idx]] * num_steps)

    norm_sessions, norm_runs = normalize_session_run_features(
        all_session_ids, all_run_ids
    )

    # Extract steps for optim and test separately
    optim_observations: list[list[float]] = []
    optim_actions: list[int] = []
    optim_norm_sessions: list[float] = []
    optim_norm_runs: list[float] = []
    optim_episode_boundaries: list[tuple[int, int]] = []  # (start_idx, length)

    test_observations: list[list[float]] = []
    test_actions: list[int] = []
    test_norm_sessions: list[float] = []
    test_norm_runs: list[float] = []
    test_episode_boundaries: list[tuple[int, int]] = []  # (start_idx, length)
    test_episode_info: list[dict] = []

    step_idx: int = 0
    optim_step_idx: int = 0
    test_step_idx: int = 0

    for ep_idx, episode in enumerate(episodes):
        steps: list[dict] = episode["steps"]
        num_steps: int = len(steps)

        # Extract observations and actions
        obs_list: list[list[float]] = [s["observation"] for s in steps]
        act_list: list[int] = [s["action"] for s in steps]

        # Get normalized CL features for this episode
        ep_norm_sessions: list[float] = norm_sessions[
            step_idx : step_idx + num_steps
        ].tolist()
        ep_norm_runs: list[float] = norm_runs[
            step_idx : step_idx + num_steps
        ].tolist()

        if ep_idx in optim_episode_indices:
            # Optimization episode
            optim_episode_boundaries.append((optim_step_idx, num_steps))
            optim_observations.extend(obs_list)
            optim_actions.extend(act_list)
            optim_norm_sessions.extend(ep_norm_sessions)
            optim_norm_runs.extend(ep_norm_runs)
            optim_step_idx += num_steps
        else:
            # Test episode
            test_episode_boundaries.append((test_step_idx, num_steps))
            test_observations.extend(obs_list)
            test_actions.extend(act_list)
            test_norm_sessions.extend(ep_norm_sessions)
            test_norm_runs.extend(ep_norm_runs)
            test_step_idx += num_steps

            # Compute episode return
            episode_return: float = sum(s["reward"] for s in steps)

            # Store episode info for evaluation
            test_episode_info.append(
                {
                    "episode_idx": ep_idx,
                    "seed": episode["seed_used"],
                    "return": episode_return,
                    "session_id": session_ids[ep_idx],
                    "run_id": run_ids[ep_idx],
                    "norm_session": ep_norm_sessions[
                        0
                    ],  # Same for all steps in episode
                    "norm_run": ep_norm_runs[0],
                    "num_steps": num_steps,
                    "timestamp": episode["timestamp"],
                }
            )

        step_idx += num_steps

    print(
        f"  Optim steps: {len(optim_observations)}, Test steps: {len(test_observations)}"
    )
    print(
        f"  Optim episodes: {len(optim_episode_boundaries)}, "
        f"Test episodes: {len(test_episode_boundaries)}"
    )

    # Convert to numpy arrays
    optim_obs_np: np.ndarray = np.array(optim_observations, dtype=np.float32)
    optim_act_np: np.ndarray = np.array(optim_actions, dtype=np.int64)
    test_obs_np: np.ndarray = np.array(test_observations, dtype=np.float32)
    test_act_np: np.ndarray = np.array(test_actions, dtype=np.int64)

    # Optionally concatenate CL features
    if use_cl_info:
        optim_cl: np.ndarray = np.stack(
            [
                np.array(optim_norm_sessions, dtype=np.float32),
                np.array(optim_norm_runs, dtype=np.float32),
            ],
            axis=1,
        )
        test_cl: np.ndarray = np.stack(
            [
                np.array(test_norm_sessions, dtype=np.float32),
                np.array(test_norm_runs, dtype=np.float32),
            ],
            axis=1,
        )

        optim_obs_np = np.concatenate([optim_obs_np, optim_cl], axis=1)
        test_obs_np = np.concatenate([test_obs_np, test_cl], axis=1)

        print(
            f"  Added CL features: input size {env_config['obs_dim']} -> {optim_obs_np.shape[1]}"
        )

    # Convert to tensors
    optim_obs: Float[Tensor, "optim_size input_size"] = torch.from_numpy(
        optim_obs_np
    )
    optim_act: Int[Tensor, " optim_size"] = torch.from_numpy(optim_act_np)
    test_obs: Float[Tensor, "test_size input_size"] = torch.from_numpy(
        test_obs_np
    )
    test_act: Int[Tensor, " test_size"] = torch.from_numpy(test_act_np)

    # Create metadata dictionary
    metadata: dict = {
        "test_episode_info": test_episode_info,
        "session_ids": session_ids,
        "run_ids": run_ids,
        "num_optim_episodes": num_optim_episodes,
        "num_test_episodes": num_test_episodes,
        "num_sessions": len(set(session_ids)),
        "optim_episode_boundaries": optim_episode_boundaries,
        "test_episode_boundaries": test_episode_boundaries,
    }

    return optim_obs, optim_act, test_obs, test_act, metadata
