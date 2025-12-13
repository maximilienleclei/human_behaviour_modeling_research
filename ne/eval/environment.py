"""Environment-based evaluation orchestration for TorchRL.

Creates fitness evaluators for episode rollouts in TorchRL environments.
Provides high-level training interface similar to supervised learning.
"""

from pathlib import Path
from typing import Callable

from jaxtyping import Float
from torch import Tensor


def create_env_fitness_evaluator(
    population,
    env,
    num_episodes: int,
    max_steps_per_episode: int,
    metric: str = "return",
    state_config=None,
) -> Callable[[], Float[Tensor, "num_nets"]]:
    """Create fitness evaluator for environment rollouts.

    Returns closure that captures environment and rollout parameters.
    Optimizer only sees fitness values, never the environment or episode data.

    Args:
        population: Population wrapper
        env: TorchRL EnvBase environment
        num_episodes: Number of episodes per evaluation
        max_steps_per_episode: Maximum steps per episode
        metric: Fitness metric - "return" (episode return) or "cross_entropy"
        state_config: Optional state persistence configuration

    Returns:
        fitness_fn: Callable[[], Tensor] that returns fitness [num_nets]
    """
    from ne.eval.env import evaluate_env_episodes

    def fitness_fn() -> Float[Tensor, "num_nets"]:
        """Evaluate current population on environment episodes.

        Closure captures environment and rollout parameters.
        Uses population.curr_gen to track generation number.
        """
        return evaluate_env_episodes(
            population, env, num_episodes, max_steps_per_episode, metric,
            state_config=state_config, curr_gen=population.curr_gen
        )

    return fitness_fn


def train_environment(
    population,
    train_env,
    test_env,
    num_episodes: int,
    max_steps_per_episode: int,
    metric: str = "return",
    optimizer: str = "ga",
    max_time: int = 3600,
    eval_interval: int = 60,
    checkpoint_path: Path | None = None,
    logger=None,
    state_config=None,
) -> dict:
    """High-level environment-based training.

    Orchestrates episode rollouts and optimization with clean separation.
    Optimizer never sees environment or episode data, only fitness values.

    Args:
        population: Population wrapper
        train_env: TorchRL environment for training
        test_env: TorchRL environment for testing
        num_episodes: Episodes per evaluation
        max_steps_per_episode: Max steps per episode
        metric: Fitness metric - "return" or "cross_entropy"
        optimizer: Which algorithm to use - "ga", "es", or "cmaes"
        max_time: Maximum training time in seconds
        eval_interval: Seconds between test evaluations
        checkpoint_path: Optional path to save/load checkpoints
        logger: Optional logger for tracking
        state_config: Optional state persistence/transfer configuration

    Returns:
        Training results dict with fitness_history, test_loss_history, final_generation

    Raises:
        ValueError: If optimizer is not "ga", "es", or "cmaes"
        TypeError: If using ES/CMA-ES with DynamicNetPopulation
    """
    # Create fitness evaluators (closures that capture environments)
    fitness_fn = create_env_fitness_evaluator(
        population, train_env, num_episodes, max_steps_per_episode, metric, state_config
    )
    test_fitness_fn = create_env_fitness_evaluator(
        population, test_env, num_episodes, max_steps_per_episode, metric, state_config
    )

    # Select and run optimizer
    if optimizer == "ga":
        from ne.optim.ga import optimize_ga

        return optimize_ga(
            population=population,
            fitness_fn=fitness_fn,
            test_fitness_fn=test_fitness_fn,
            max_time=max_time,
            eval_interval=eval_interval,
            checkpoint_path=checkpoint_path,
            logger=logger,
            state_config=state_config,
        )

    elif optimizer == "es":
        from ne.optim.es import optimize_es

        return optimize_es(
            population=population,
            fitness_fn=fitness_fn,
            test_fitness_fn=test_fitness_fn,
            max_time=max_time,
            eval_interval=eval_interval,
            checkpoint_path=checkpoint_path,
            logger=logger,
            state_config=state_config,
        )

    elif optimizer == "cmaes":
        from ne.optim.cmaes import optimize_cmaes

        return optimize_cmaes(
            population=population,
            fitness_fn=fitness_fn,
            test_fitness_fn=test_fitness_fn,
            max_time=max_time,
            eval_interval=eval_interval,
            checkpoint_path=checkpoint_path,
            logger=logger,
            state_config=state_config,
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer}. Must be 'ga', 'es', or 'cmaes'"
        )
