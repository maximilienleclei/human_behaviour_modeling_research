"""Optimization functions for Experiment 4 - Recurrent and Dynamic Networks."""

import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from . import config
from .config import RESULTS_DIR, ExperimentConfig
from .data import EpisodeDataset, episode_collate_fn
from .metrics import compute_cross_entropy, compute_macro_f1
from .models import (
    BatchedRecurrentPopulation,
    DynamicNetPopulation,
    RecurrentMLPReservoir,
    RecurrentMLPTrainable,
)
from .utils import save_results


def evaluate_progression_recurrent(
    model: RecurrentMLPReservoir | RecurrentMLPTrainable,
    episode_details: list[dict],
    env: gym.Env,
    max_steps: int = 1000,
    use_cl_features: bool = False,
) -> tuple[float, float, list[float]]:
    """Quick evaluation to track progression during optimization (for recurrent models).

    Runs recurrent model on matched human episodes and returns statistics.

    Args:
        model: Optimized recurrent policy
        episode_details: List of dicts with episode info
        env: Gym environment
        max_steps: Maximum steps per episode
        use_cl_features: Whether the model expects CL features in input

    Returns:
        Tuple of (mean_pct_diff, std_pct_diff, model_returns)
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


def deeplearn_recurrent(
    optim_obs: Float[Tensor, "optim_size input_size"],
    optim_act: Int[Tensor, " optim_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    input_size: int,
    output_size: int,
    exp_config: ExperimentConfig,
    env_name: str,
    method_name: str,
    model_class: type,  # RecurrentMLPReservoir or RecurrentMLPTrainable
    subject: str = "sub01",
    metadata: dict | None = None,
    track_progression: bool = True,
    ckpt_and_behav_eval_interval_seconds: int | None = None,
) -> tuple[list[float], list[float]]:
    """Deep Learning (SGD) optimization for recurrent models."""
    model = model_class(input_size, exp_config.hidden_size, output_size).to(
        config.DEVICE
    )
    optimizer: torch.optim.SGD = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Create episode dataset
    if metadata is None or "optim_episode_boundaries" not in metadata:
        raise ValueError("metadata with optim_episode_boundaries required for recurrent training")

    optim_dataset: EpisodeDataset = EpisodeDataset(
        optim_obs, optim_act, metadata["optim_episode_boundaries"]
    )
    optim_loader: DataLoader = DataLoader(
        optim_dataset,
        batch_size=exp_config.batch_size,
        shuffle=True,
        collate_fn=episode_collate_fn,
    )

    # Create test episodes for evaluation
    test_episodes: list[dict] = create_episode_list(
        test_obs, test_act, metadata["test_episode_boundaries"]
    )

    loss_history: list[float] = []
    test_loss_history: list[float] = []
    f1_history: list[float] = []
    progression_history: list[dict] = []

    # Use config default if not specified
    if ckpt_and_behav_eval_interval_seconds is None:
        ckpt_and_behav_eval_interval_seconds = (
            exp_config.ckpt_and_behav_eval_interval_seconds
        )

    # Setup progression tracking if enabled
    episode_details: list[dict] | None = None
    eval_env: gym.Env | None = None
    max_steps: int = 1000
    last_ckpt_eval_time: float = -ckpt_and_behav_eval_interval_seconds
    use_cl_features_flag: bool = False

    if track_progression:
        from .environment import make_env, get_max_episode_steps
        from .evaluation import load_human_episode_details

        print(
            f"  Checkpoint & behaviour evaluation enabled (every {ckpt_and_behav_eval_interval_seconds}s)"
        )
        episode_details = load_human_episode_details(env_name, subject)
        eval_env = make_env(env_name)
        max_steps = get_max_episode_steps(env_name)

        # Determine if model uses CL features
        env_obs_dim: int = eval_env.observation_space.shape[0]
        use_cl_features_flag = input_size > env_obs_dim

    # Checkpointing paths
    checkpoint_path: Path = (
        RESULTS_DIR / f"{env_name}_{method_name}_{subject}_checkpoint.pt"
    )

    # Try to resume from checkpoint
    start_epoch: int = 0
    if checkpoint_path.exists():
        print(f"  Resuming from checkpoint...")
        checkpoint: dict = torch.load(checkpoint_path, weights_only=False)
        loss_history = checkpoint["loss_history"]
        test_loss_history = checkpoint.get("test_loss_history", [])
        f1_history = checkpoint["f1_history"]
        progression_history = checkpoint.get("progression_history", [])
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        if progression_history and track_progression:
            last_elapsed: float = progression_history[-1].get("elapsed_time", 0.0)
            last_ckpt_eval_time = last_elapsed

        print(f"  Resumed at epoch {start_epoch}")

    epoch: int = start_epoch
    start_time: float = time.time()

    # Initialize time tracking
    last_eval_time: float = -exp_config.loss_eval_interval_seconds
    print(
        f"  Optimizing for {exp_config.max_optim_time} seconds ({exp_config.max_optim_time/60:.1f} minutes)..."
    )

    while True:
        # Check time limit
        elapsed_time: float = time.time() - start_time
        if elapsed_time >= exp_config.max_optim_time:
            print(f"  Time limit reached ({elapsed_time:.1f}s)")
            break

        model.train()
        epoch_losses: list[float] = []

        for batch in optim_loader:
            obs: Float[Tensor, "BS max_len input_size"] = batch["observations"].to(
                config.DEVICE
            )
            act: Int[Tensor, "BS max_len"] = batch["actions"].to(config.DEVICE)
            mask: Float[Tensor, "BS max_len"] = batch["mask"].to(config.DEVICE)

            optimizer.zero_grad()

            # Forward pass (h_0 automatically initialized to zeros)
            logits: Float[Tensor, "BS max_len output_size"]
            logits, _ = model(obs)

            # Masked cross-entropy loss
            loss: Float[Tensor, ""] = F.cross_entropy(
                logits.view(-1, output_size),
                act.view(-1),
                reduction="none",
            )
            loss = (loss * mask.view(-1)).sum() / mask.sum()

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss: float = float(np.mean(epoch_losses))
        loss_history.append(avg_loss)

        # Check if it's time for evaluation (time-based)
        elapsed: float = time.time() - start_time
        if elapsed - last_eval_time >= exp_config.loss_eval_interval_seconds:
            model.eval()
            with torch.no_grad():
                # Evaluate on test episodes
                test_losses: list[float] = []
                for ep in test_episodes[:100]:  # Limit to first 100 episodes
                    obs_ep: Float[Tensor, "seq_len input_size"] = ep[
                        "observations"
                    ].to(config.DEVICE)
                    act_ep: Int[Tensor, " seq_len"] = ep["actions"].to(config.DEVICE)

                    logits_ep, _ = model(
                        obs_ep.unsqueeze(0)
                    )  # Add batch dim
                    loss_ep: float = F.cross_entropy(
                        logits_ep.squeeze(0), act_ep
                    ).item()
                    test_losses.append(loss_ep)

                test_loss: float = float(np.mean(test_losses))

                # Compute F1 on test episodes
                # For simplicity, use first few episodes
                all_preds: list[int] = []
                all_targets: list[int] = []
                for ep in test_episodes[:50]:
                    obs_ep = ep["observations"].to(config.DEVICE)
                    act_ep = ep["actions"].to(config.DEVICE)

                    # Sample actions from model
                    for _ in range(exp_config.num_f1_samples):
                        h_t: Float[Tensor, "1 hidden_size"] = torch.zeros(
                            1, model.hidden_size, device=config.DEVICE
                        )
                        for t in range(len(obs_ep)):
                            probs, h_t = model.get_probs(
                                obs_ep[t].unsqueeze(0), h_t
                            )
                            action: int = torch.multinomial(
                                probs.squeeze(0), num_samples=1
                            ).item()
                            all_preds.append(action)
                            all_targets.append(act_ep[t].item())

                # Compute macro F1
                from sklearn.metrics import f1_score

                f1: float = f1_score(
                    all_targets,
                    all_preds,
                    average="macro",
                    labels=list(range(output_size)),
                    zero_division=0,
                )

            test_loss_history.append(test_loss)
            f1_history.append(f1)
            last_eval_time = elapsed
            remaining: float = exp_config.max_optim_time - elapsed
            print(
                f"  DL Epoch {epoch} [{elapsed:.0f}s/{exp_config.max_optim_time}s, {remaining:.0f}s left]: "
                f"Optim Loss={avg_loss:.4f}, Test Loss={test_loss:.4f}, F1={f1:.4f}"
            )

            # Save results
            save_results(
                env_name,
                method_name,
                {
                    "loss": loss_history,
                    "test_loss": test_loss_history,
                    "f1": f1_history,
                },
                subject,
            )

        # Checkpoint and behaviour evaluation combined
        if elapsed - last_ckpt_eval_time >= ckpt_and_behav_eval_interval_seconds:
            # Run behaviour evaluation if enabled
            if (
                track_progression
                and episode_details is not None
                and eval_env is not None
            ):
                print(f"    Running behaviour evaluation at {elapsed:.0f}s...")
                mean_pct_diff, std_pct_diff, model_returns = (
                    evaluate_progression_recurrent(
                        model,
                        episode_details,
                        eval_env,
                        max_steps,
                        use_cl_features_flag,
                    )
                )
                progression_entry: dict = {
                    "elapsed_time": elapsed,
                    "epoch": epoch,
                    "mean_pct_diff": mean_pct_diff,
                    "std_pct_diff": std_pct_diff,
                    "model_returns": model_returns,
                    "test_loss": test_loss,
                    "f1": f1,
                }
                progression_history.append(progression_entry)
                print(
                    f"    Mean % diff from human: {mean_pct_diff:+.2f}% ± {std_pct_diff:.2f}%"
                )

            # Save checkpoint
            checkpoint_data: dict = {
                "epoch": epoch,
                "loss_history": loss_history,
                "test_loss_history": test_loss_history,
                "f1_history": f1_history,
                "progression_history": progression_history,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "optim_time": elapsed,
            }
            torch.save(checkpoint_data, checkpoint_path)
            last_ckpt_eval_time = elapsed

        epoch += 1

    # Final checkpoint save
    total_time: float = time.time() - start_time
    print(
        f"  Optimization complete: {epoch} epochs in {total_time:.1f}s ({total_time/60:.1f} minutes)"
    )
    checkpoint_data = {
        "epoch": epoch,
        "loss_history": loss_history,
        "test_loss_history": test_loss_history,
        "f1_history": f1_history,
        "progression_history": progression_history,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "optim_time": total_time,
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"  Final checkpoint saved to {checkpoint_path}")

    # Close environment if used
    if eval_env is not None:
        eval_env.close()

    return loss_history, f1_history


def neuroevolve_recurrent(
    optim_obs: Float[Tensor, "optim_size input_size"],
    optim_act: Int[Tensor, " optim_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    input_size: int,
    output_size: int,
    exp_config: ExperimentConfig,
    env_name: str,
    method_name: str,
    model_type: str,  # 'reservoir' or 'trainable'
    subject: str = "sub01",
    metadata: dict | None = None,
    track_progression: bool = True,
    ckpt_and_behav_eval_interval_seconds: int | None = None,
) -> tuple[list[float], list[float]]:
    """Neuroevolution optimization for recurrent models with batched GPU operations."""
    if metadata is None or "optim_episode_boundaries" not in metadata:
        raise ValueError("metadata with episode boundaries required for recurrent training")

    # Create episode lists
    optim_episodes: list[dict] = create_episode_list(
        optim_obs, optim_act, metadata["optim_episode_boundaries"]
    )
    test_episodes: list[dict] = create_episode_list(
        test_obs, test_act, metadata["test_episode_boundaries"]
    )

    # Initialize population
    population: BatchedRecurrentPopulation = BatchedRecurrentPopulation(
        input_size,
        exp_config.hidden_size,
        output_size,
        exp_config.population_size,
        model_type=model_type,
        sigma_init=exp_config.adaptive_sigma_init,
        sigma_noise=exp_config.adaptive_sigma_noise,
    )

    fitness_history: list[float] = []
    test_loss_history: list[float] = []
    f1_history: list[float] = []
    progression_history: list[dict] = []

    # Use config default if not specified
    if ckpt_and_behav_eval_interval_seconds is None:
        ckpt_and_behav_eval_interval_seconds = (
            exp_config.ckpt_and_behav_eval_interval_seconds
        )

    # Setup progression tracking if enabled
    episode_details: list[dict] | None = None
    eval_env: gym.Env | None = None
    max_steps: int = 1000
    last_ckpt_eval_time: float = -ckpt_and_behav_eval_interval_seconds
    use_cl_features_flag: bool = False

    if track_progression:
        from .environment import make_env, get_max_episode_steps
        from .evaluation import load_human_episode_details

        print(
            f"  Checkpoint & behaviour evaluation enabled (every {ckpt_and_behav_eval_interval_seconds}s)"
        )
        episode_details = load_human_episode_details(env_name, subject)
        eval_env = make_env(env_name)
        max_steps = get_max_episode_steps(env_name)

        # Determine if model uses CL features
        env_obs_dim: int = eval_env.observation_space.shape[0]
        use_cl_features_flag = input_size > env_obs_dim

    # Checkpointing paths
    checkpoint_path: Path = (
        RESULTS_DIR / f"{env_name}_{method_name}_{subject}_checkpoint.pt"
    )

    # Try to resume from checkpoint
    start_gen: int = 0
    if checkpoint_path.exists():
        print(f"  Resuming from checkpoint...")
        checkpoint: dict = torch.load(checkpoint_path, weights_only=False)
        fitness_history = checkpoint["fitness_history"]
        test_loss_history = checkpoint.get("test_loss_history", [])
        f1_history = checkpoint["f1_history"]
        progression_history = checkpoint.get("progression_history", [])
        start_gen = checkpoint["generation"] + 1
        population.load_state_dict(checkpoint["population_state"])

        if progression_history and track_progression:
            last_elapsed: float = progression_history[-1].get("elapsed_time", 0.0)
            last_ckpt_eval_time = last_elapsed

        print(f"  Resumed at generation {start_gen}")

    generation: int = start_gen
    start_time: float = time.time()

    # Initialize time tracking
    last_eval_time: float = -exp_config.loss_eval_interval_seconds
    print(
        f"  Optimizing for {exp_config.max_optim_time} seconds ({exp_config.max_optim_time/60:.1f} minutes)..."
    )

    # Determine number of episodes to sample for fitness evaluation
    num_eval_episodes: int = min(32, len(optim_episodes))

    while True:
        # Check time limit
        elapsed_time: float = time.time() - start_time
        if elapsed_time >= exp_config.max_optim_time:
            print(f"  Time limit reached ({elapsed_time:.1f}s)")
            break

        # Sample episodes for fitness evaluation
        sampled_episodes: list[dict] = random.sample(optim_episodes, k=num_eval_episodes)

        # Evaluate fitness
        fitness: Float[Tensor, " pop_size"] = population.evaluate_episodes(
            sampled_episodes
        )
        best_fitness: float = fitness.min().item()
        fitness_history.append(best_fitness)

        # Selection
        population.select_simple_ga(fitness)

        # Mutation
        population.mutate()

        # Check if it's time for evaluation
        elapsed: float = time.time() - start_time
        if elapsed - last_eval_time >= exp_config.loss_eval_interval_seconds:
            # Evaluate on test episodes
            test_sampled: list[dict] = random.sample(
                test_episodes, k=min(num_eval_episodes, len(test_episodes))
            )
            test_fitness: Float[Tensor, " pop_size"] = population.evaluate_episodes(
                test_sampled
            )
            test_loss: float = test_fitness.min().item()

            # Compute F1 (simplified)
            # For GA, F1 computation is expensive, so we skip it or compute less frequently
            f1: float = 0.0  # Placeholder

            test_loss_history.append(test_loss)
            f1_history.append(f1)
            last_eval_time = elapsed
            remaining: float = exp_config.max_optim_time - elapsed
            print(
                f"  GA Gen {generation} [{elapsed:.0f}s/{exp_config.max_optim_time}s, {remaining:.0f}s left]: "
                f"Best Fitness={best_fitness:.4f}, Test Loss={test_loss:.4f}"
            )

            # Save results
            save_results(
                env_name,
                method_name,
                {
                    "fitness": fitness_history,
                    "test_loss": test_loss_history,
                    "f1": f1_history,
                },
                subject,
            )

        # Checkpoint and behaviour evaluation
        if elapsed - last_ckpt_eval_time >= ckpt_and_behav_eval_interval_seconds:
            # Create best model for evaluation
            best_model = population.create_best_model(fitness)

            # Run behaviour evaluation if enabled
            if (
                track_progression
                and episode_details is not None
                and eval_env is not None
            ):
                print(f"    Running behaviour evaluation at {elapsed:.0f}s...")
                mean_pct_diff, std_pct_diff, model_returns = (
                    evaluate_progression_recurrent(
                        best_model,
                        episode_details,
                        eval_env,
                        max_steps,
                        use_cl_features_flag,
                    )
                )
                progression_entry: dict = {
                    "elapsed_time": elapsed,
                    "generation": generation,
                    "mean_pct_diff": mean_pct_diff,
                    "std_pct_diff": std_pct_diff,
                    "model_returns": model_returns,
                    "test_loss": test_loss,
                    "f1": f1,
                }
                progression_history.append(progression_entry)
                print(
                    f"    Mean % diff from human: {mean_pct_diff:+.2f}% ± {std_pct_diff:.2f}%"
                )

            # Save checkpoint
            checkpoint_data: dict = {
                "generation": generation,
                "fitness_history": fitness_history,
                "test_loss_history": test_loss_history,
                "f1_history": f1_history,
                "progression_history": progression_history,
                "population_state": population.get_state_dict(),
                "optim_time": elapsed,
            }
            torch.save(checkpoint_data, checkpoint_path)
            last_ckpt_eval_time = elapsed

        generation += 1

    # Final checkpoint save
    total_time: float = time.time() - start_time
    print(
        f"  Optimization complete: {generation} generations in {total_time:.1f}s ({total_time/60:.1f} minutes)"
    )
    checkpoint_data = {
        "generation": generation,
        "fitness_history": fitness_history,
        "test_loss_history": test_loss_history,
        "f1_history": f1_history,
        "progression_history": progression_history,
        "population_state": population.get_state_dict(),
        "optim_time": total_time,
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"  Final checkpoint saved to {checkpoint_path}")

    # Close environment if used
    if eval_env is not None:
        eval_env.close()

    return fitness_history, f1_history


def neuroevolve_dynamic(
    optim_obs: Float[Tensor, "optim_size input_size"],
    optim_act: Int[Tensor, " optim_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    input_size: int,
    output_size: int,
    exp_config: ExperimentConfig,
    env_name: str,
    method_name: str,
    subject: str = "sub01",
    track_progression: bool = True,
    ckpt_and_behav_eval_interval_seconds: int | None = None,
) -> tuple[list[float], list[float]]:
    """Neuroevolution optimization for dynamic complexity networks.

    Dynamic networks are feedforward (not recurrent), so they can use
    flat step batches like experiment 3.
    """
    optim_obs_gpu: Float[Tensor, "optim_size input_size"] = optim_obs.to(
        config.DEVICE
    )
    optim_act_gpu: Int[Tensor, " optim_size"] = optim_act.to(config.DEVICE)
    test_obs_gpu: Float[Tensor, "test_size input_size"] = test_obs.to(
        config.DEVICE
    )
    test_act_gpu: Int[Tensor, " test_size"] = test_act.to(config.DEVICE)

    # Sample a subset for fitness evaluation
    num_optim: int = optim_obs_gpu.shape[0]
    eval_batch_size: int = min(exp_config.batch_size * 100, num_optim)

    # Initialize dynamic network population
    population: DynamicNetPopulation = DynamicNetPopulation(
        input_size, output_size, exp_config.population_size
    )

    fitness_history: list[float] = []
    test_loss_history: list[float] = []
    f1_history: list[float] = []
    progression_history: list[dict] = []

    # Use config default if not specified
    if ckpt_and_behav_eval_interval_seconds is None:
        ckpt_and_behav_eval_interval_seconds = (
            exp_config.ckpt_and_behav_eval_interval_seconds
        )

    # Note: Progression tracking for dynamic nets would require
    # implementing a proper evaluation function, which is complex
    # For now, we'll just track fitness

    # Checkpointing paths
    checkpoint_path: Path = (
        RESULTS_DIR / f"{env_name}_{method_name}_{subject}_checkpoint.pt"
    )

    generation: int = 0
    start_time: float = time.time()

    # Initialize time tracking
    last_eval_time: float = -exp_config.loss_eval_interval_seconds
    print(
        f"  Optimizing for {exp_config.max_optim_time} seconds ({exp_config.max_optim_time/60:.1f} minutes)..."
    )

    while True:
        # Check time limit
        elapsed_time: float = time.time() - start_time
        if elapsed_time >= exp_config.max_optim_time:
            print(f"  Time limit reached ({elapsed_time:.1f}s)")
            break

        # Sample random batch for fitness evaluation
        indices: Int[Tensor, " eval_batch_size"] = torch.randint(
            0, num_optim, (eval_batch_size,), device=config.DEVICE
        )

        # Evaluate fitness (cross-entropy)
        fitness: Float[Tensor, " pop_size"] = population.evaluate(
            optim_obs_gpu[indices], optim_act_gpu[indices]
        )
        best_fitness: float = fitness.min().item()
        fitness_history.append(best_fitness)

        # Selection
        population.select_simple_ga(fitness)

        # Mutation (grows/shrinks architectures)
        population.mutate()

        # Check if it's time for evaluation
        elapsed: float = time.time() - start_time
        if elapsed - last_eval_time >= exp_config.loss_eval_interval_seconds:
            # Evaluate on test set
            test_indices: Int[Tensor, " eval_batch_size"] = torch.randint(
                0, len(test_obs_gpu), (min(eval_batch_size, len(test_obs_gpu)),), device=config.DEVICE
            )
            test_fitness: Float[Tensor, " pop_size"] = population.evaluate(
                test_obs_gpu[test_indices], test_act_gpu[test_indices]
            )
            test_loss: float = test_fitness.min().item()

            test_loss_history.append(test_loss)
            f1_history.append(0.0)  # Placeholder
            last_eval_time = elapsed
            remaining: float = exp_config.max_optim_time - elapsed
            print(
                f"  Dynamic GA Gen {generation} [{elapsed:.0f}s/{exp_config.max_optim_time}s, {remaining:.0f}s left]: "
                f"Best Fitness={best_fitness:.4f}, Test Loss={test_loss:.4f}"
            )

            # Save results
            save_results(
                env_name,
                method_name,
                {
                    "fitness": fitness_history,
                    "test_loss": test_loss_history,
                    "f1": f1_history,
                },
                subject,
            )

        generation += 1

    # Final save
    total_time: float = time.time() - start_time
    print(
        f"  Optimization complete: {generation} generations in {total_time:.1f}s ({total_time/60:.1f} minutes)"
    )

    return fitness_history, f1_history
