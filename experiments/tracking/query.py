"""Query interface for experiment results (context-efficient)."""

import csv
from collections import defaultdict
from pathlib import Path

from experiments.tracking.database import ExperimentDB


class ExperimentQuery:
    """High-level query interface optimized for minimal context usage."""

    def __init__(self, db_path: Path) -> None:
        """Initialize query interface.

        Args:
            db_path: Path to SQLite database
        """
        self.db = ExperimentDB(db_path)

    def summarize_all_experiments(self) -> str:
        """Return high-level summary of all experiments.

        Returns:
            Summary string (< 1KB)
        """
        all_runs = self.db.get_all_runs()

        # Group by experiment number
        by_experiment: dict[int, list] = defaultdict(list)
        for run in all_runs:
            by_experiment[run["experiment_number"]].append(run)

        lines = ["Experiment Summary", "=" * 60]

        for exp_num in sorted(by_experiment.keys()):
            runs = by_experiment[exp_num]
            total = len(runs)

            # Count by status
            status_counts = defaultdict(int)
            for run in runs:
                status_counts[run["status"]] += 1

            pending = status_counts.get("pending", 0)
            running = status_counts.get("running", 0)
            completed = status_counts.get("completed", 0)
            failed = status_counts.get("failed", 0)
            timeout = status_counts.get("timeout", 0)

            lines.append(
                f"Exp {exp_num}: {total} runs "
                f"({completed} completed, {running} running, "
                f"{pending} pending, {failed} failed, {timeout} timeout)"
            )

        return "\n".join(lines)

    def summarize_experiment(self, experiment_number: int) -> str:
        """Summarize single experiment.

        Args:
            experiment_number: Experiment number

        Returns:
            Summary string (< 10KB)
        """
        runs = self.db.get_all_runs(experiment_number=experiment_number)

        if not runs:
            return f"No runs found for experiment {experiment_number}"

        lines = [
            f"Experiment {experiment_number} Summary",
            "=" * 80,
            f"Total runs: {len(runs)}",
            "",
        ]

        # Status breakdown
        status_counts = defaultdict(int)
        for run in runs:
            status_counts[run["status"]] += 1

        lines.append("Status Breakdown:")
        for status, count in sorted(status_counts.items()):
            lines.append(f"  {status}: {count}")

        lines.append("")

        # Show completed runs in table format
        completed_runs = [r for r in runs if r["status"] == "completed"]

        if completed_runs:
            lines.append(f"Completed Runs ({len(completed_runs)}):")
            lines.append("")
            lines.append(
                f"{'Run ID':<8} {'Dataset':<12} {'Method':<30} {'Subject':<8} {'CL Info':<8}"
            )
            lines.append("-" * 80)

            for run in completed_runs[:50]:  # Limit to 50 runs
                cl_info = "Yes" if run["use_cl_info"] else "No"
                lines.append(
                    f"{run['run_id']:<8} {run['dataset']:<12} {run['method']:<30} "
                    f"{run['subject']:<8} {cl_info:<8}"
                )

            if len(completed_runs) > 50:
                lines.append(f"... and {len(completed_runs) - 50} more")

        return "\n".join(lines)

    def get_best_runs(
        self,
        experiment_number: int,
        metric: str = "test_loss",
        top_k: int = 5,
    ) -> list[dict]:
        """Get top K runs by metric.

        Args:
            experiment_number: Experiment number
            metric: Metric name (test_loss, f1_score, etc.)
            top_k: Number of top runs to return

        Returns:
            List of run dictionaries with metrics
        """
        runs = self.db.get_all_runs(experiment_number=experiment_number, status="completed")

        # Get latest metrics for each run
        runs_with_metrics = []
        for run in runs:
            metrics = self.db.get_run_metrics(run["run_id"])
            if metrics:
                latest = metrics[-1]
                run_data = {
                    "run_id": run["run_id"],
                    "dataset": run["dataset"],
                    "method": run["method"],
                    "subject": run["subject"],
                    "use_cl_info": run["use_cl_info"],
                    "seed": run["seed"],
                }
                # Add all metrics from latest
                run_data.update(latest)
                runs_with_metrics.append(run_data)

        # Sort by metric
        if metric in ["test_loss", "train_loss"]:
            # Lower is better
            runs_with_metrics.sort(key=lambda x: x.get(metric, float("inf")))
        else:
            # Higher is better (f1_score, etc.)
            runs_with_metrics.sort(key=lambda x: x.get(metric, float("-inf")), reverse=True)

        return runs_with_metrics[:top_k]

    def compare_methods(
        self,
        experiment_number: int,
        dataset: str,
        metric: str = "test_loss",
    ) -> str:
        """Compare all methods on a dataset.

        Args:
            experiment_number: Experiment number
            dataset: Dataset name
            metric: Metric to compare

        Returns:
            Formatted comparison table
        """
        runs = self.db.get_all_runs(experiment_number=experiment_number, status="completed")

        # Filter by dataset
        runs = [r for r in runs if r["dataset"] == dataset]

        if not runs:
            return f"No completed runs found for experiment {experiment_number} on {dataset}"

        # Get latest metrics for each run
        method_metrics: dict[str, list] = defaultdict(list)

        for run in runs:
            metrics = self.db.get_run_metrics(run["run_id"])
            if metrics:
                latest = metrics[-1]
                method_key = f"{run['method']}_{run['use_cl_info']}"
                method_metrics[method_key].append(latest.get(metric))

        # Compute averages
        method_avg = {}
        for method, values in method_metrics.items():
            valid_values = [v for v in values if v is not None]
            if valid_values:
                method_avg[method] = sum(valid_values) / len(valid_values)

        # Format table
        lines = [
            f"Method Comparison on {dataset} (Experiment {experiment_number})",
            "=" * 60,
            f"Metric: {metric}",
            "",
            f"{'Method':<40} {'Avg {metric}':<15}",
            "-" * 60,
        ]

        for method in sorted(method_avg.keys()):
            lines.append(f"{method:<40} {method_avg[method]:<15.4f}")

        return "\n".join(lines)

    def get_run_progress(self, run_id: int, max_points: int = 50) -> dict:
        """Get training curve with downsampling.

        Args:
            run_id: Run identifier
            max_points: Maximum number of points to return

        Returns:
            Dictionary with downsampled metrics
        """
        metrics = self.db.get_run_metrics(run_id, max_points=max_points)

        return {
            "run_id": run_id,
            "num_points": len(metrics),
            "metrics": metrics,
        }

    def get_failed_runs(self, experiment_number: int | None = None) -> list[dict]:
        """Get all failed runs with error info.

        Args:
            experiment_number: Filter by experiment number (optional)

        Returns:
            List of failed run dictionaries with error messages
        """
        cursor = self.db.conn.cursor()

        query = """
            SELECT r.*, e.error_type, e.error_message, e.traceback
            FROM experiment_runs r
            LEFT JOIN run_errors e ON r.run_id = e.run_id
            WHERE r.status IN ('failed', 'timeout')
        """

        params = []
        if experiment_number is not None:
            query += " AND r.experiment_number = ?"
            params.append(experiment_number)

        query += " ORDER BY r.updated_at DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def export_results_csv(
        self,
        experiment_number: int,
        output_path: Path,
    ) -> None:
        """Export results to CSV.

        Args:
            experiment_number: Experiment number
            output_path: Output CSV path
        """
        runs = self.db.get_all_runs(experiment_number=experiment_number, status="completed")

        with open(output_path, "w", newline="") as f:
            if not runs:
                return

            # Get all metrics for first run to determine columns
            sample_metrics = self.db.get_run_metrics(runs[0]["run_id"])
            if not sample_metrics:
                return

            # Define columns
            run_cols = ["run_id", "dataset", "method", "subject", "use_cl_info", "seed"]
            metric_cols = list(sample_metrics[0].keys())
            all_cols = run_cols + metric_cols

            writer = csv.DictWriter(f, fieldnames=all_cols)
            writer.writeheader()

            # Write rows
            for run in runs:
                metrics = self.db.get_run_metrics(run["run_id"])
                if metrics:
                    latest = metrics[-1]
                    row = {
                        "run_id": run["run_id"],
                        "dataset": run["dataset"],
                        "method": run["method"],
                        "subject": run["subject"],
                        "use_cl_info": run["use_cl_info"],
                        "seed": run["seed"],
                    }
                    row.update(latest)
                    writer.writerow(row)
