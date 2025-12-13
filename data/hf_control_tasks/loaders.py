"""HuggingFace dataset loaders for control tasks.

Loads pre-trained agent trajectories from HuggingFace datasets
for CartPole-v1 and LunarLander-v2 environments.
"""

import numpy as np
import torch
from datasets import load_dataset
from jaxtyping import Float, Int
from torch import Tensor


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
