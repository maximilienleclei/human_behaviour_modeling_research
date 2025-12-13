"""SGD (stochastic gradient descent) optimizer for neural networks.

This module provides gradient-based optimization using backpropagation.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from config.device import DEVICE
from dl.data.preprocessing import EpisodeDataset, episode_collate_fn
from dl.optim.base import create_episode_list


def optimize_sgd(
    model: torch.nn.Module,
    optim_obs: Float[Tensor, "optim_size input_size"],
    optim_act: Int[Tensor, " optim_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    output_size: int,
    metadata: dict,
    checkpoint_path: Path,
    max_optim_time: int = 36000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    loss_eval_interval_seconds: int = 60,
    logger=None,
) -> tuple[list[float], list[float]]:
    """Optimize model using SGD with backpropagation.

    Args:
        model: Neural network model to optimize
        optim_obs: Optimization observations
        optim_act: Optimization actions
        test_obs: Test observations
        test_act: Test actions
        output_size: Output dimension
        metadata: Metadata dict with episode_boundaries
        checkpoint_path: Path to save checkpoints
        max_optim_time: Maximum optimization time in seconds
        batch_size: Batch size for training
        learning_rate: Learning rate for SGD
        loss_eval_interval_seconds: Evaluation interval in seconds
        logger: Optional ExperimentLogger for database logging

    Returns:
        Tuple of (loss_history, f1_history)
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Create episode dataset for recurrent training
    if "optim_episode_boundaries" not in metadata:
        raise ValueError("metadata with optim_episode_boundaries required")

    optim_dataset = EpisodeDataset(
        optim_obs, optim_act, metadata["optim_episode_boundaries"]
    )
    optim_loader = DataLoader(
        optim_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=episode_collate_fn,
    )

    # Create test episodes
    test_episodes = create_episode_list(
        test_obs, test_act, metadata["test_episode_boundaries"]
    )

    loss_history = []
    test_loss_history = []

    # Try to resume from checkpoint
    start_epoch = 0
    if checkpoint_path.exists():
        print(f"  Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        loss_history = checkpoint["loss_history"]
        test_loss_history = checkpoint.get("test_loss_history", [])
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"  Resumed at epoch {start_epoch}")

    epoch = start_epoch
    start_time = time.time()
    last_eval_time = -loss_eval_interval_seconds

    print(f"  Optimizing for {max_optim_time}s ({max_optim_time/60:.1f} min)...")

    while True:
        # Check time limit
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_optim_time:
            print(f"  Time limit reached ({elapsed_time:.1f}s)")
            break

        model.train()
        epoch_losses = []

        for batch in optim_loader:
            obs = batch["observations"].to(config.DEVICE)
            act = batch["actions"].to(config.DEVICE)
            mask = batch["mask"].to(config.DEVICE)

            optimizer.zero_grad()

            # Forward pass
            logits, _ = model(obs)

            # Masked cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, output_size),
                act.view(-1),
                reduction="none",
            )
            loss = (loss * mask.view(-1)).sum() / mask.sum()

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        loss_history.append(avg_loss)

        # Periodic evaluation
        elapsed = time.time() - start_time
        if elapsed - last_eval_time >= loss_eval_interval_seconds:
            model.eval()
            with torch.no_grad():
                # Evaluate on test episodes (limit to 100)
                test_losses = []
                for ep in test_episodes[:100]:
                    obs_ep = ep["observations"].to(config.DEVICE)
                    act_ep = ep["actions"].to(config.DEVICE)

                    logits_ep, _ = model(obs_ep.unsqueeze(0))
                    loss_ep = F.cross_entropy(
                        logits_ep.squeeze(0), act_ep
                    ).item()
                    test_losses.append(loss_ep)

                test_loss = float(np.mean(test_losses))

            test_loss_history.append(test_loss)
            last_eval_time = elapsed
            remaining = max_optim_time - elapsed
            print(
                f"  Epoch {epoch} [{elapsed:.0f}s/{max_optim_time}s, {remaining:.0f}s left]: "
                f"Train={avg_loss:.4f}, Test={test_loss:.4f}"
            )

            # Log to database
            if logger is not None:
                logger.log_progress(
                    epoch=epoch,
                    train_loss=avg_loss,
                    test_loss=test_loss,
                )

        # Save checkpoint periodically (every 300s)
        if elapsed % 300 < (time.time() - start_time) % 300:
            checkpoint_data = {
                "epoch": epoch,
                "loss_history": loss_history,
                "test_loss_history": test_loss_history,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "optim_time": elapsed,
            }
            torch.save(checkpoint_data, checkpoint_path)

        epoch += 1

    # Final checkpoint
    total_time = time.time() - start_time
    print(f"  Complete: {epoch} epochs in {total_time:.1f}s ({total_time/60:.1f} min)")
    
    checkpoint_data = {
        "epoch": epoch,
        "loss_history": loss_history,
        "test_loss_history": test_loss_history,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "optim_time": total_time,
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"  Final checkpoint saved to {checkpoint_path}")

    return loss_history, test_loss_history
