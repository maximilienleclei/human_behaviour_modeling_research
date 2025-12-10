"""Database interface for experiment tracking."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class ExperimentDB:
    """SQLite database interface for tracking experiments."""

    def __init__(self, db_path: Path) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dicts
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Core experiment runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_number INTEGER NOT NULL,
                dataset TEXT NOT NULL,
                method TEXT NOT NULL,
                subject TEXT NOT NULL,
                use_cl_info BOOLEAN NOT NULL,
                seed INTEGER NOT NULL,

                slurm_job_id INTEGER,
                slurm_array_task_id INTEGER,
                slurm_node TEXT,
                slurm_submit_time TIMESTAMP,
                slurm_start_time TIMESTAMP,
                slurm_end_time TIMESTAMP,

                status TEXT NOT NULL DEFAULT 'pending',
                exit_code INTEGER,
                gpu_id INTEGER,

                config_json TEXT,

                checkpoint_path TEXT,
                results_json_path TEXT,
                log_path TEXT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(experiment_number, dataset, method, subject, use_cl_info, seed)
            )
        """)

        # Summary statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,

                elapsed_time REAL NOT NULL,
                epoch_or_generation INTEGER,

                train_loss REAL,
                test_loss REAL,
                f1_score REAL,
                best_fitness REAL,

                mean_pct_diff REAL,
                std_pct_diff REAL,

                gpu_memory_mb REAL,

                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id)
            )
        """)

        # Aggregated final results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_results (
                run_id INTEGER PRIMARY KEY,

                final_train_loss REAL,
                final_test_loss REAL,
                final_f1_score REAL,
                final_mean_pct_diff REAL,
                final_std_pct_diff REAL,

                min_test_loss REAL,
                min_test_loss_epoch INTEGER,
                best_f1_score REAL,
                best_f1_epoch INTEGER,

                total_epochs INTEGER,
                total_time_seconds REAL,
                converged BOOLEAN,

                final_num_parameters INTEGER,
                final_num_layers INTEGER,

                FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id)
            )
        """)

        # Error tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_errors (
                error_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                error_type TEXT NOT NULL,
                error_message TEXT,
                traceback TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id)
            )
        """)

        # SLURM job tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS slurm_jobs (
                slurm_job_id INTEGER PRIMARY KEY,
                job_name TEXT,
                partition TEXT,
                account TEXT,
                time_limit TEXT,
                num_tasks INTEGER,
                nodes_allocated TEXT,
                submit_command TEXT,
                submit_dir TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_run_status ON experiment_runs(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_run_experiment ON experiment_runs(experiment_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run ON run_metrics(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_time ON run_metrics(elapsed_time)")

        self.conn.commit()

    def create_run(
        self,
        experiment_number: int,
        dataset: str,
        method: str,
        subject: str,
        use_cl_info: bool,
        seed: int,
        config: Any,
        slurm_job_id: int | None = None,
        slurm_array_task_id: int | None = None,
        gpu_id: int | None = None,
    ) -> int:
        """Create new experiment run entry.

        Args:
            experiment_number: Experiment number (e.g., 4, 5, 6)
            dataset: Dataset name
            method: Method name
            subject: Subject identifier
            use_cl_info: Whether continual learning info is used
            seed: Random seed
            config: ExperimentConfig object
            slurm_job_id: SLURM job ID (if running on SLURM)
            slurm_array_task_id: SLURM array task ID
            gpu_id: GPU index

        Returns:
            run_id: Unique run identifier
        """
        cursor = self.conn.cursor()

        # Serialize config to JSON
        if hasattr(config, "__dict__"):
            config_json = json.dumps(config.__dict__)
        else:
            config_json = json.dumps(config)

        # Check if run already exists
        existing_run = self.get_run_by_params(
            experiment_number, dataset, method, subject, use_cl_info, seed
        )

        if existing_run:
            return existing_run["run_id"]

        cursor.execute(
            """
            INSERT INTO experiment_runs (
                experiment_number, dataset, method, subject, use_cl_info, seed,
                slurm_job_id, slurm_array_task_id, gpu_id, config_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                experiment_number,
                dataset,
                method,
                subject,
                use_cl_info,
                seed,
                slurm_job_id,
                slurm_array_task_id,
                gpu_id,
                config_json,
            ),
        )

        self.conn.commit()
        return cursor.lastrowid

    def update_run_status(
        self,
        run_id: int,
        status: str,
        exit_code: int | None = None,
    ) -> None:
        """Update run status.

        Args:
            run_id: Run identifier
            status: New status (pending/running/completed/failed/timeout)
            exit_code: Exit code if completed
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            UPDATE experiment_runs
            SET status = ?, exit_code = ?, updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
            """,
            (status, exit_code, run_id),
        )

        self.conn.commit()

    def log_metrics(
        self,
        run_id: int,
        elapsed_time: float,
        epoch_or_generation: int,
        train_loss: float | None = None,
        test_loss: float | None = None,
        f1_score: float | None = None,
        best_fitness: float | None = None,
        mean_pct_diff: float | None = None,
        std_pct_diff: float | None = None,
        gpu_memory_mb: float | None = None,
    ) -> None:
        """Log training metrics.

        Args:
            run_id: Run identifier
            elapsed_time: Seconds since start
            epoch_or_generation: Current iteration number
            train_loss: Training loss
            test_loss: Test loss
            f1_score: F1 score
            best_fitness: Best fitness (for GA)
            mean_pct_diff: Mean percentage difference from human
            std_pct_diff: Std percentage difference from human
            gpu_memory_mb: GPU memory usage in MB
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO run_metrics (
                run_id, elapsed_time, epoch_or_generation,
                train_loss, test_loss, f1_score, best_fitness,
                mean_pct_diff, std_pct_diff, gpu_memory_mb
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                elapsed_time,
                epoch_or_generation,
                train_loss,
                test_loss,
                f1_score,
                best_fitness,
                mean_pct_diff,
                std_pct_diff,
                gpu_memory_mb,
            ),
        )

        self.conn.commit()

    def finalize_run(
        self,
        run_id: int,
        final_metrics: dict[str, Any],
    ) -> None:
        """Store final aggregated results.

        Args:
            run_id: Run identifier
            final_metrics: Dictionary of final metrics
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO run_results (
                run_id, final_train_loss, final_test_loss, final_f1_score,
                final_mean_pct_diff, final_std_pct_diff,
                min_test_loss, min_test_loss_epoch,
                best_f1_score, best_f1_epoch,
                total_epochs, total_time_seconds, converged,
                final_num_parameters, final_num_layers
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                final_metrics.get("final_train_loss"),
                final_metrics.get("final_test_loss"),
                final_metrics.get("final_f1_score"),
                final_metrics.get("final_mean_pct_diff"),
                final_metrics.get("final_std_pct_diff"),
                final_metrics.get("min_test_loss"),
                final_metrics.get("min_test_loss_epoch"),
                final_metrics.get("best_f1_score"),
                final_metrics.get("best_f1_epoch"),
                final_metrics.get("total_epochs"),
                final_metrics.get("total_time_seconds"),
                final_metrics.get("converged"),
                final_metrics.get("final_num_parameters"),
                final_metrics.get("final_num_layers"),
            ),
        )

        self.conn.commit()

    def log_error(
        self,
        run_id: int,
        error_type: str,
        error_message: str,
        traceback: str | None = None,
    ) -> None:
        """Log error for debugging.

        Args:
            run_id: Run identifier
            error_type: Type of error
            error_message: Error message
            traceback: Full traceback
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO run_errors (run_id, error_type, error_message, traceback)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, error_type, error_message, traceback),
        )

        self.conn.commit()

    def get_run_by_id(self, run_id: int) -> dict | None:
        """Get run metadata by ID.

        Args:
            run_id: Run identifier

        Returns:
            Run metadata as dictionary, or None if not found
        """
        cursor = self.conn.cursor()

        cursor.execute("SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    def get_run_by_params(
        self,
        experiment_number: int,
        dataset: str,
        method: str,
        subject: str,
        use_cl_info: bool,
        seed: int,
    ) -> dict | None:
        """Find existing run by parameters.

        Args:
            experiment_number: Experiment number
            dataset: Dataset name
            method: Method name
            subject: Subject identifier
            use_cl_info: Whether CL info is used
            seed: Random seed

        Returns:
            Run metadata as dictionary, or None if not found
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT * FROM experiment_runs
            WHERE experiment_number = ?
                AND dataset = ?
                AND method = ?
                AND subject = ?
                AND use_cl_info = ?
                AND seed = ?
            """,
            (experiment_number, dataset, method, subject, use_cl_info, seed),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    def get_all_runs(
        self,
        experiment_number: int | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """Get all runs, optionally filtered.

        Args:
            experiment_number: Filter by experiment number
            status: Filter by status

        Returns:
            List of run metadata dictionaries
        """
        cursor = self.conn.cursor()

        query = "SELECT * FROM experiment_runs WHERE 1=1"
        params = []

        if experiment_number is not None:
            query += " AND experiment_number = ?"
            params.append(experiment_number)

        if status is not None:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_run_metrics(
        self,
        run_id: int,
        max_points: int | None = None,
    ) -> list[dict]:
        """Get metrics for a run.

        Args:
            run_id: Run identifier
            max_points: Maximum number of points to return (for downsampling)

        Returns:
            List of metric dictionaries
        """
        cursor = self.conn.cursor()

        if max_points is None:
            cursor.execute(
                "SELECT * FROM run_metrics WHERE run_id = ? ORDER BY elapsed_time",
                (run_id,),
            )
        else:
            # Downsample by selecting evenly spaced points
            cursor.execute(
                """
                SELECT * FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (ORDER BY elapsed_time) as rn,
                           COUNT(*) OVER () as total
                    FROM run_metrics
                    WHERE run_id = ?
                )
                WHERE (rn - 1) % MAX(1, total / ?) = 0
                ORDER BY elapsed_time
                """,
                (run_id, max_points),
            )

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
