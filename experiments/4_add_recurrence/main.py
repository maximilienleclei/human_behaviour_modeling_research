"""Main script for Experiment 4: Add Recurrence and Dynamic Complexity."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

from src import config
from src.config import (
    ENV_CONFIGS,
    RESULTS_DIR,
    ExperimentConfig,
    set_device,
)
from src.data import load_human_data
from src.models import RecurrentMLPReservoir, RecurrentMLPTrainable
from src.optim import deeplearn_recurrent, neuroevolve_recurrent, neuroevolve_dynamic
from src.utils import set_random_seeds


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def get_all_methods() -> list[tuple[str, dict]]:
    """Get all method configurations for experiment 4."""
    return [
        ("SGD_reservoir", {"type": "dl", "model": "reservoir"}),
        ("SGD_trainable", {"type": "dl", "model": "trainable"}),
        ("adaptive_ga_reservoir", {"type": "ne", "model": "reservoir"}),
        ("adaptive_ga_trainable", {"type": "ne", "model": "trainable"}),
        ("adaptive_ga_dynamic", {"type": "ne_dynamic", "model": "dynamic"}),
    ]


def run_single_method(
    env_name: str,
    method_name: str,
    method_config: dict,
    use_cl_info: bool,
    optim_obs: Float[Tensor, "optim_size input_size"],
    optim_act: Int[Tensor, " optim_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    input_size: int,
    output_size: int,
    exp_config: ExperimentConfig,
    metadata: dict,
    subject: str = "sub01",
) -> None:
    """Run a single optimization method."""
    # Append CL variant to method name
    cl_suffix: str = "with_cl" if use_cl_info else "no_cl"
    method_name_full: str = f"{method_name}_{cl_suffix}"

    print(f"\n{'='*60}")
    print(f"Running {method_name_full} for {env_name} ({subject}'s data)")
    print(f"{'='*60}")
    print(f"Optim size: {optim_obs.shape[0]}, Test size: {test_obs.shape[0]}")
    print(f"Input size: {input_size}, Output size: {output_size}")

    if method_config["type"] == "dl":
        # Deep learning (SGD) with recurrent models
        if method_config["model"] == "reservoir":
            model_class = RecurrentMLPReservoir
        elif method_config["model"] == "trainable":
            model_class = RecurrentMLPTrainable
        else:
            raise ValueError(f"Unknown model type: {method_config['model']}")

        deeplearn_recurrent(
            optim_obs,
            optim_act,
            test_obs,
            test_act,
            input_size,
            output_size,
            exp_config,
            env_name,
            method_name_full,
            model_class,
            subject,
            metadata,
        )
    elif method_config["type"] == "ne":
        # Neuroevolution with recurrent models
        model_type: str = method_config["model"]  # 'reservoir' or 'trainable'

        neuroevolve_recurrent(
            optim_obs,
            optim_act,
            test_obs,
            test_act,
            input_size,
            output_size,
            exp_config,
            env_name,
            method_name_full,
            model_type,
            subject,
            metadata,
        )
    elif method_config["type"] == "ne_dynamic":
        # Neuroevolution with dynamic complexity networks
        neuroevolve_dynamic(
            optim_obs,
            optim_act,
            test_obs,
            test_act,
            input_size,
            output_size,
            exp_config,
            env_name,
            method_name_full,
            subject,
        )
    else:
        raise ValueError(f"Unknown method type: {method_config['type']}")


def main() -> None:
    """Main function to run Experiment 4."""
    parser = argparse.ArgumentParser(
        description="Experiment 4: Add Recurrence and Dynamic Complexity"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cartpole", "mountaincar", "acrobot", "lunarlander"],
        required=True,
        help="Environment to use",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "SGD_reservoir",
            "SGD_trainable",
            "adaptive_ga_reservoir",
            "adaptive_ga_trainable",
            "adaptive_ga_dynamic",
        ],
        help="Method to run. Use --list-methods to see all options.",
    )
    parser.add_argument(
        "--use-cl-info",
        action="store_true",
        help="Include continual learning (session/run) information as input features",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List all available methods",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (default: 0)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        choices=["sub01", "sub02"],
        default="sub01",
        help="Subject whose data to use (default: sub01)",
    )

    args = parser.parse_args()

    # Set global DEVICE based on --gpu argument
    set_device(args.gpu)
    print(f"Using device: {config.DEVICE}")

    all_methods: list[tuple[str, dict]] = get_all_methods()
    method_dict: dict[str, dict] = {name: cfg for name, cfg in all_methods}

    if args.list_methods:
        print("Available methods:")
        for name, cfg in all_methods:
            model_type: str = cfg.get("model", "unknown")
            opt_type: str = "SGD" if cfg["type"] == "dl" else "GA"
            print(f"  - {name:<30} ({opt_type}, {model_type})")
        return

    # Check method
    if not args.method:
        print("Error: --method is required unless using --list-methods")
        return

    if args.method not in method_dict:
        print(f"Error: Unknown method '{args.method}'")
        print("Use --list-methods to see available options")
        return

    # Setup environment
    env_config: dict = ENV_CONFIGS[args.dataset]
    env_name: str = args.dataset
    action_dim: int = env_config["action_dim"]

    exp_config: ExperimentConfig = ExperimentConfig(seed=args.seed)

    # Set random seeds for reproducibility
    set_random_seeds(exp_config.seed)
    print(f"Random seed: {exp_config.seed}")
    print(f"Subject: {args.subject}")
    print(f"Continual learning info: {args.use_cl_info}")

    # Load data
    print(f"\nLoading {env_config['name']} human behavior data...")
    optim_obs, optim_act, test_obs, test_act, metadata = load_human_data(
        env_name, args.use_cl_info, args.subject, exp_config.holdout_pct
    )

    # Save metadata for potential evaluation use
    metadata_path: Path = RESULTS_DIR / f"metadata_{env_name}_{args.subject}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)
    print(f"  Metadata saved to {metadata_path}")

    # Determine actual input size (may include CL features)
    input_size: int = optim_obs.shape[1]
    output_size: int = action_dim

    # Run single method
    run_single_method(
        env_name,
        args.method,
        method_dict[args.method],
        args.use_cl_info,
        optim_obs,
        optim_act,
        test_obs,
        test_act,
        input_size,
        output_size,
        exp_config,
        metadata,
        args.subject,
    )

    print("\n" + "=" * 60)
    print(f"{args.method} Complete!")
    print("=" * 60)
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    exit_code: int = 0
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        # Ensure cleanup even on error
        plt.close("all")
        torch.cuda.empty_cache()
        sys.exit(exit_code)
