#!/usr/bin/env python
"""Monitor SLURM jobs and experiment status.

Examples:
    # Monitor all jobs
    python cli/monitor_jobs.py

    # Monitor specific experiment
    python cli/monitor_jobs.py --exp 4

    # Continuous monitoring (refresh every 30s)
    python cli/monitor_jobs.py --exp 4 --watch --interval 30
"""

import argparse
import sys
import time
from pathlib import Path

# Add experiments directory to path
EXPERIMENTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from tracking.database import ExperimentDB
from tracking.slurm_manager import SlurmManager


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor SLURM jobs and experiments")
    parser.add_argument("--exp", type=int, help="Filter by experiment number")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds (default: 30)")

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

    try:
        while True:
            # Clear screen (optional, comment out if not desired)
            if args.watch:
                print("\033[2J\033[H", end="")  # Clear screen and move cursor to top

            # Get status summary
            status = manager.monitor_jobs(experiment_number=args.exp)

            # Print summary
            print("=" * 60)
            if args.exp:
                print(f"Experiment {args.exp} Status Summary")
            else:
                print("All Experiments Status Summary")
            print("=" * 60)
            print(f"PENDING:   {status['pending']:4d}")
            print(f"RUNNING:   {status['running']:4d}")
            print(f"COMPLETED: {status['completed']:4d}")
            print(f"FAILED:    {status['failed']:4d}")
            print(f"TIMEOUT:   {status['timeout']:4d}")
            print("=" * 60)

            # Get detailed info for running and recent jobs
            runs = db.get_all_runs(experiment_number=args.exp)
            running_runs = [r for r in runs if r["status"] == "running"]

            if running_runs:
                print("\nCurrently Running:")
                print(f"{'Run ID':<8} {'Dataset':<12} {'Method':<30} {'SLURM Job':<12}")
                print("-" * 60)
                for run in running_runs[:20]:  # Show up to 20 running jobs
                    slurm_job = str(run["slurm_job_id"]) if run["slurm_job_id"] else "N/A"
                    if run["slurm_array_task_id"] is not None:
                        slurm_job += f"_{run['slurm_array_task_id']}"
                    print(
                        f"{run['run_id']:<8} {run['dataset']:<12} {run['method']:<30} {slurm_job:<12}"
                    )

            # Recent failures
            failed_runs = [r for r in runs if r["status"] in ["failed", "timeout"]]
            recent_failures = sorted(failed_runs, key=lambda x: x["updated_at"], reverse=True)[:5]

            if recent_failures:
                print("\nRecent Failures:")
                print(f"{'Run ID':<8} {'Dataset':<12} {'Method':<30} {'Status':<10}")
                print("-" * 60)
                for run in recent_failures:
                    print(
                        f"{run['run_id']:<8} {run['dataset']:<12} {run['method']:<30} {run['status']:<10}"
                    )

            if args.watch:
                print(f"\nRefreshing in {args.interval} seconds... (Ctrl+C to stop)")
                time.sleep(args.interval)
            else:
                break

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
