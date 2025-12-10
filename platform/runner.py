"""Main experiment runner for the unified platform.

This module provides the main entry point for running experiments with the platform.
Supports both SGD and GA optimization on various model architectures.
"""

import argparse
import sys
from pathlib import Path

import torch

from platform import config
from platform.config import ENV_CONFIGS, RESULTS_DIR, set_device
from platform.data.loaders import load_human_data, load_cartpole_data, load_lunarlander_data
from platform.models import RecurrentMLPReservoir, RecurrentMLPTrainable
from platform.optimizers.sgd import optimize_sgd
from platform.optimizers.genetic import optimize_ga


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_data(dataset: str, use_cl_info: bool, subject: str = "sub01", holdout_pct: float = 0.1):
    """Load dataset based on configuration.
    
    Args:
        dataset: Dataset name (cartpole, mountaincar, acrobot, lunarlander, or HF:CartPole-v1, HF:LunarLander-v2)
        use_cl_info: Whether to include CL features (only for human data)
        subject: Subject identifier for human data
        holdout_pct: Percentage for test split
        
    Returns:
        Tuple of (optim_obs, optim_act, test_obs, test_act, metadata, input_size, output_size)
    """
    # HuggingFace datasets
    if dataset.startswith("HF:"):
        hf_name = dataset[3:]  # Remove "HF:" prefix
        if hf_name == "CartPole-v1":
            train_obs, train_act, test_obs, test_act = load_cartpole_data()
            metadata = {"num_train": len(train_obs), "num_test": len(test_obs)}
            input_size = 4
            output_size = 2
        elif hf_name == "LunarLander-v2":
            train_obs, train_act, test_obs, test_act = load_lunarlander_data()
            metadata = {"num_train": len(train_obs), "num_test": len(test_obs)}
            input_size = 8
            output_size = 4
        else:
            raise ValueError(f"Unknown HuggingFace dataset: {hf_name}")
        
        return train_obs, train_act, test_obs, test_act, metadata, input_size, output_size
    
    # Human behavioral data
    else:
        if dataset not in ENV_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        env_config = ENV_CONFIGS[dataset]
        print(f"\nLoading {env_config['name']} human behavior data...")
        
        optim_obs, optim_act, test_obs, test_act, metadata = load_human_data(
            dataset, use_cl_info, subject, holdout_pct
        )
        
        input_size = optim_obs.shape[1]
        output_size = env_config["action_dim"]
        
        return optim_obs, optim_act, test_obs, test_act, metadata, input_size, output_size


def create_model(model_type: str, input_size: int, hidden_size: int, output_size: int):
    """Create model based on type.
    
    Args:
        model_type: Model type (reservoir, trainable)
        input_size: Input dimension
        hidden_size: Hidden dimension
        output_size: Output dimension
        
    Returns:
        Model instance
    """
    if model_type == "reservoir":
        return RecurrentMLPReservoir(input_size, hidden_size, output_size)
    elif model_type == "trainable":
        return RecurrentMLPTrainable(input_size, hidden_size, output_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_experiment(
    dataset: str,
    method: str,
    model_type: str,
    optimizer_type: str,
    use_cl_info: bool = False,
    subject: str = "sub01",
    seed: int = 42,
    gpu: int = 0,
    hidden_size: int = 50,
    max_optim_time: int = 36000,
    batch_size: int = 32,
    population_size: int = 50,
    learning_rate: float = 1e-3,
    experiment_number: int = 5,
    logger=None,
) -> None:
    """Run a single experiment.
    
    Args:
        dataset: Dataset name
        method: Method name (for checkpoint naming)
        model_type: Model type (reservoir, trainable)
        optimizer_type: Optimizer type (sgd, ga)
        use_cl_info: Whether to use CL features
        subject: Subject identifier
        seed: Random seed
        gpu: GPU index
        hidden_size: Hidden dimension
        max_optim_time: Maximum optimization time in seconds
        batch_size: Batch size for SGD
        population_size: Population size for GA
        learning_rate: Learning rate for SGD
        experiment_number: Experiment number for tracking
        logger: Optional ExperimentLogger
    """
    # Set device and seeds
    set_device(gpu)
    set_random_seeds(seed)
    
    print(f"\n{'='*60}")
    print(f"Running {method} on {dataset}")
    print(f"{'='*60}")
    print(f"Device: {config.DEVICE}")
    print(f"Random seed: {seed}")
    print(f"Subject: {subject}")
    print(f"CL features: {use_cl_info}")
    
    # Load data
    optim_obs, optim_act, test_obs, test_act, metadata, input_size, output_size = load_data(
        dataset, use_cl_info, subject
    )
    
    print(f"Input size: {input_size}, Output size: {output_size}")
    print(f"Optim size: {optim_obs.shape[0]}, Test size: {test_obs.shape[0]}")
    
    # Create checkpoint path
    cl_suffix = "with_cl" if use_cl_info else "no_cl"
    method_full = f"{method}_{cl_suffix}"
    checkpoint_path = RESULTS_DIR / f"{dataset}_{method_full}_{subject}_checkpoint.pt"
    
    # Run optimizer
    if optimizer_type == "sgd":
        # Create model
        model = create_model(model_type, input_size, hidden_size, output_size)
        
        # Run SGD optimization
        loss_history, test_loss_history = optimize_sgd(
            model=model,
            optim_obs=optim_obs,
            optim_act=optim_act,
            test_obs=test_obs,
            test_act=test_act,
            output_size=output_size,
            metadata=metadata,
            checkpoint_path=checkpoint_path,
            max_optim_time=max_optim_time,
            batch_size=batch_size,
            learning_rate=learning_rate,
            logger=logger,
        )
        
    elif optimizer_type == "ga":
        # Run GA optimization
        fitness_history, test_loss_history = optimize_ga(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            optim_obs=optim_obs,
            optim_act=optim_act,
            test_obs=test_obs,
            test_act=test_act,
            metadata=metadata,
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            max_optim_time=max_optim_time,
            population_size=population_size,
            logger=logger,
        )
        
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    print(f"\n{'='*60}")
    print(f"{method} Complete!")
    print(f"{'='*60}")
    print(f"Results saved to {RESULTS_DIR}/")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Platform for Human Behavior Modeling"
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset: cartpole, mountaincar, acrobot, lunarlander, HF:CartPole-v1, HF:LunarLander-v2",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Method name (e.g., SGD_reservoir, GA_trainable)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["reservoir", "trainable"],
        required=True,
        help="Model type: reservoir (frozen) or trainable (rank-1)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "ga"],
        required=True,
        help="Optimizer: sgd or ga",
    )
    
    # Optional arguments
    parser.add_argument(
        "--use-cl-info",
        action="store_true",
        help="Include continual learning (session/run) features",
    )
    parser.add_argument(
        "--subject",
        type=str,
        choices=["sub01", "sub02"],
        default="sub01",
        help="Subject for human data (default: sub01)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index (default: 0)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=50,
        help="Hidden layer size (default: 50)",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=36000,
        help="Max optimization time in seconds (default: 36000 = 10 hours)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for SGD (default: 32)",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="Population size for GA (default: 50)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for SGD (default: 1e-3)",
    )
    parser.add_argument(
        "--exp-number",
        type=int,
        default=5,
        help="Experiment number for tracking (default: 5)",
    )
    parser.add_argument(
        "--no-logger",
        action="store_true",
        help="Disable database logging",
    )
    
    args = parser.parse_args()
    
    # Setup logger if enabled
    logger = None
    if not args.no_logger:
        try:
            # Try to import tracking system
            TRACKING_DIR = Path(__file__).parent.parent / "experiments" / "tracking"
            sys.path.insert(0, str(TRACKING_DIR))
            from database import ExperimentDB
            from logger import ExperimentLogger
            
            db_path = Path(__file__).parent.parent / "tracking.db"
            db = ExperimentDB(db_path)
            
            logger = ExperimentLogger(
                db=db,
                experiment_number=args.exp_number,
                dataset=args.dataset,
                method=args.method,
                subject=args.subject,
                use_cl_info=args.use_cl_info,
                seed=args.seed,
                config={"hidden_size": args.hidden_size},  # Simplified config
                gpu_id=args.gpu,
            )
            print("Database logging enabled")
        except Exception as e:
            print(f"Warning: Could not initialize logger: {e}")
            print("Continuing without database logging...")
    
    # Run experiment with logger context if available
    if logger is not None:
        with logger:
            run_experiment(
                dataset=args.dataset,
                method=args.method,
                model_type=args.model,
                optimizer_type=args.optimizer,
                use_cl_info=args.use_cl_info,
                subject=args.subject,
                seed=args.seed,
                gpu=args.gpu,
                hidden_size=args.hidden_size,
                max_optim_time=args.max_time,
                batch_size=args.batch_size,
                population_size=args.population_size,
                learning_rate=args.learning_rate,
                experiment_number=args.exp_number,
                logger=logger,
            )
    else:
        run_experiment(
            dataset=args.dataset,
            method=args.method,
            model_type=args.model,
            optimizer_type=args.optimizer,
            use_cl_info=args.use_cl_info,
            subject=args.subject,
            seed=args.seed,
            gpu=args.gpu,
            hidden_size=args.hidden_size,
            max_optim_time=args.max_time,
            batch_size=args.batch_size,
            population_size=args.population_size,
            learning_rate=args.learning_rate,
            experiment_number=args.exp_number,
            logger=None,
        )


if __name__ == "__main__":
    main()
