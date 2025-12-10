#!/usr/bin/env python
"""Submit SLURM jobs for experiments.

Examples:
    # Submit single job
    python cli/submit_jobs.py --exp 4 --dataset cartpole --method SGD_reservoir

    # Submit full sweep
    python cli/submit_jobs.py --exp 4 --sweep-all

    # Submit with custom resources
    python cli/submit_jobs.py --exp 4 --dataset cartpole --method SGD_reservoir \
        --time 01:00:00 --mem 30G
"""

import argparse
import sys
from pathlib import Path

# Add experiments directory to path
EXPERIMENTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from tracking.database import ExperimentDB
from tracking.slurm_manager import SlurmManager


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for experiments")

    # Experiment parameters
    parser.add_argument("--exp", type=int, required=True, help="Experiment number")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--method", type=str, help="Method name")
    parser.add_argument("--subject", type=str, default="sub01", help="Subject (default: sub01)")
    parser.add_argument(
        "--use-cl-info",
        action="store_true",
        help="Use continual learning info",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Sweep options
    parser.add_argument(
        "--sweep-all",
        action="store_true",
        help="Submit full parameter sweep for experiment",
    )

    # SLURM options
    parser.add_argument("--time", type=str, default="00:30:00", help="Time limit (default: 00:30:00)")
    parser.add_argument(
        "--gpu",
        type=str,
        default="h100_1g.10gb:1",
        help="GPU type (default: h100_1g.10gb:1)",
    )
    parser.add_argument("--account", type=str, default="rrg-pbellec", help="SLURM account (default: rrg-pbellec)")
    parser.add_argument("--mem", type=str, default="15G", help="Memory limit (default: 15G)")
    parser.add_argument("--cpus", type=int, default=2, help="CPUs per task (default: 2)")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent jobs (default: 10)")

    args = parser.parse_args()

    # Initialize database and manager
    db_path = EXPERIMENTS_DIR / "tracking.db"
    template_dir = EXPERIMENTS_DIR / "slurm" / "templates"
    log_dir = EXPERIMENTS_DIR / "slurm" / "logs"
    config_dir = EXPERIMENTS_DIR / "slurm" / "configs"
    project_root = EXPERIMENTS_DIR.parent

    db = ExperimentDB(db_path)
    manager = SlurmManager(
        db=db,
        template_dir=template_dir,
        log_dir=log_dir,
        config_dir=config_dir,
        project_root=project_root,
    )

    if args.sweep_all:
        # Submit full parameter sweep
        print(f"Submitting full sweep for experiment {args.exp}...")

        # Define sweep parameters based on experiment
        if args.exp == 4:
            datasets = ["cartpole", "mountaincar", "acrobot", "lunarlander"]
            methods = [
                "SGD_reservoir",
                "SGD_trainable",
                "adaptive_ga_reservoir",
                "adaptive_ga_trainable",
                "adaptive_ga_dynamic",
            ]
            subjects = ["sub01"]
            use_cl_info_values = [False, True]
            seeds = [42]
        else:
            print(f"Error: Sweep configuration not defined for experiment {args.exp}")
            sys.exit(1)

        print(f"  Datasets: {datasets}")
        print(f"  Methods: {methods}")
        print(f"  Subjects: {subjects}")
        print(f"  CL Info: {use_cl_info_values}")
        print(f"  Seeds: {seeds}")

        total_jobs = (
            len(datasets)
            * len(methods)
            * len(subjects)
            * len(use_cl_info_values)
            * len(seeds)
        )
        print(f"  Total jobs: {total_jobs}")

        job_id = manager.submit_sweep(
            experiment_number=args.exp,
            datasets=datasets,
            methods=methods,
            subjects=subjects,
            use_cl_info_values=use_cl_info_values,
            seeds=seeds,
            time_limit=args.time,
            gpu_type=args.gpu,
            account=args.account,
            mem=args.mem,
            cpus=args.cpus,
            max_concurrent=args.max_concurrent,
        )

        print(f"\nSubmitted job array: {job_id}")
        print(f"Monitor with: python cli/monitor_jobs.py --exp {args.exp}")

    else:
        # Submit single job
        if not args.dataset or not args.method:
            print("Error: --dataset and --method are required for single job submission")
            print("Or use --sweep-all for full sweep")
            sys.exit(1)

        print(f"Submitting single job for experiment {args.exp}...")
        print(f"  Dataset: {args.dataset}")
        print(f"  Method: {args.method}")
        print(f"  Subject: {args.subject}")
        print(f"  CL Info: {args.use_cl_info}")
        print(f"  Seed: {args.seed}")

        job_id = manager.submit_single_job(
            experiment_number=args.exp,
            dataset=args.dataset,
            method=args.method,
            subject=args.subject,
            use_cl_info=args.use_cl_info,
            seed=args.seed,
            time_limit=args.time,
            gpu_type=args.gpu,
            account=args.account,
            mem=args.mem,
            cpus=args.cpus,
        )

        print(f"\nSubmitted job: {job_id}")


if __name__ == "__main__":
    main()
