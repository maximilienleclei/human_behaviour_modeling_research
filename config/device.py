"""Device configuration for PyTorch models.

Manages the global DEVICE variable used across dl/ and ne/ modules.
"""

import torch


# Device configuration (will be set via set_device() based on --gpu argument)
DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_device(gpu_index: int | None = None) -> None:
    """Set the global DEVICE variable.

    Args:
        gpu_index: GPU index to use, or None for CPU
    """
    global DEVICE
    if gpu_index is None:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(f"cuda:{gpu_index}")
