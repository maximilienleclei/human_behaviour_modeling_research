#!/usr/bin/env python
"""Query experiment results (Claude-friendly interface).

Examples:
    # Overall summary
    python cli/query_results.py --summary

    # Experiment 4 details
    python cli/query_results.py --exp 4 --summary

    # Best runs
    python cli/query_results.py --exp 4 --best --metric test_loss --top 10

    # Compare methods on cartpole
    python cli/query_results.py --exp 4 --compare --dataset cartpole

    # Failed runs (for debugging)
    python cli/query_results.py --exp 4 --failed

    # Export to CSV
    python cli/query_results.py --exp 4 --export results_exp4.csv
"""

import argparse
import sys
from pathlib import Path

# Add experiments directory to path
EXPERIMENTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from tracking.query import ExperimentQuery


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Query experiment results")

    # Query options
    parser.add_argument("--exp", type=int, help="Experiment number")
    parser.add_argument("--summary", action="store_true", help="Show summary")
    parser.add_argument("--best", action="store_true", help="Show best runs")
    parser.add_argument("--compare", action="store_true", help="Compare methods on dataset")
    parser.add_argument("--failed", action="store_true", help="Show failed runs")
    parser.add_argument("--export", type=str, help="Export to CSV file")

    # Parameters
    parser.add_argument("--dataset", type=str, help="Dataset name (for --compare)")
    parser.add_argument("--metric", type=str, default="test_loss", help="Metric name (default: test_loss)")
    parser.add_argument("--top", type=int, default=5, help="Number of top runs (default: 5)")

    args = parser.parse_args()

    # Initialize query interface
    db_path = EXPERIMENTS_DIR / "tracking.db"
    query = ExperimentQuery(db_path)

    if args.summary:
        if args.exp:
            print(query.summarize_experiment(args.exp))
        else:
            print(query.summarize_all_experiments())

    elif args.best:
        if not args.exp:
            print("Error: --exp required for --best")
            sys.exit(1)

        runs = query.get_best_runs(args.exp, args.metric, args.top)

        if not runs:
            print(f"No completed runs found for experiment {args.exp}")
            sys.exit(0)

        print(f"Top {args.top} runs by {args.metric} (Experiment {args.exp})")
        print("=" * 80)
        print(f"{'Rank':<6} {'Dataset':<12} {'Method':<30} {args.metric:<12}")
        print("-" * 80)

        for i, run in enumerate(runs, 1):
            metric_value = run.get(args.metric, "N/A")
            if isinstance(metric_value, float):
                metric_str = f"{metric_value:.4f}"
            else:
                metric_str = str(metric_value)

            print(f"{i:<6} {run['dataset']:<12} {run['method']:<30} {metric_str:<12}")

    elif args.compare:
        if not args.exp or not args.dataset:
            print("Error: --exp and --dataset required for --compare")
            sys.exit(1)

        print(query.compare_methods(args.exp, args.dataset, args.metric))

    elif args.failed:
        if not args.exp:
            print("Error: --exp required for --failed")
            sys.exit(1)

        failed_runs = query.get_failed_runs(args.exp)

        if not failed_runs:
            print(f"No failed runs found for experiment {args.exp}")
            sys.exit(0)

        print(f"Failed Runs (Experiment {args.exp})")
        print("=" * 80)

        for run in failed_runs:
            print(f"\nRun {run['run_id']}: {run['method']} on {run['dataset']}")
            print(f"  Status: {run['status']}")
            if run.get("slurm_job_id"):
                job_id = run["slurm_job_id"]
                if run.get("slurm_array_task_id") is not None:
                    job_id = f"{job_id}_{run['slurm_array_task_id']}"
                print(f"  SLURM Job: {job_id}")
            if run.get("error_type"):
                print(f"  Error Type: {run['error_type']}")
            if run.get("error_message"):
                print(f"  Error Message: {run['error_message'][:200]}")
                if len(run.get("error_message", "")) > 200:
                    print("    ... (truncated)")

    elif args.export:
        if not args.exp:
            print("Error: --exp required for --export")
            sys.exit(1)

        output_path = Path(args.export)
        query.export_results_csv(args.exp, output_path)
        print(f"Results exported to {output_path}")

    else:
        print("Error: Must specify one of --summary, --best, --compare, --failed, or --export")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
