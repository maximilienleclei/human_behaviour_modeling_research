"""Experiment logger for tracking runs in the database."""

import os
import time
import traceback as tb
from typing import Any

from experiments.tracking.database import ExperimentDB


class ExperimentLogger:
    """Context manager for experiment logging."""

    def __init__(
        self,
        db: ExperimentDB,
        experiment_number: int,
        dataset: str,
        method: str,
        subject: str,
        use_cl_info: bool,
        seed: int,
        config: Any,
        gpu_id: int | None = None,
    ) -> None:
        """Initialize experiment logger.

        Args:
            db: Database interface
            experiment_number: Experiment number (e.g., 4, 5, 6)
            dataset: Dataset name
            method: Method name
            subject: Subject identifier
            use_cl_info: Whether continual learning info is used
            seed: Random seed
            config: ExperimentConfig object
            gpu_id: GPU index
        """
        self.db = db
        self.experiment_number = experiment_number
        self.dataset = dataset
        self.method = method
        self.subject = subject
        self.use_cl_info = use_cl_info
        self.seed = seed
        self.config = config
        self.gpu_id = gpu_id

        self.run_id: int | None = None
        self.start_time: float | None = None

    def __enter__(self) -> "ExperimentLogger":
        """Create run entry and start timer."""
        # Get SLURM environment variables if running on SLURM
        slurm_job_id = os.getenv("SLURM_JOB_ID")
        slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")

        # Create run entry
        self.run_id = self.db.create_run(
            experiment_number=self.experiment_number,
            dataset=self.dataset,
            method=self.method,
            subject=self.subject,
            use_cl_info=self.use_cl_info,
            seed=self.seed,
            config=self.config,
            slurm_job_id=int(slurm_job_id) if slurm_job_id else None,
            slurm_array_task_id=int(slurm_array_task_id) if slurm_array_task_id else None,
            gpu_id=self.gpu_id,
        )

        # Update status to running
        self.db.update_run_status(self.run_id, "running")

        # Start timer
        self.start_time = time.time()

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Finalize run on exit."""
        if self.run_id is None:
            return False

        if exc_type is None:
            # Success
            self.db.update_run_status(self.run_id, "completed", exit_code=0)
        else:
            # Failure
            self.db.update_run_status(self.run_id, "failed", exit_code=1)

            # Log error
            error_message = str(exc_val)
            traceback = "".join(tb.format_tb(exc_tb)) if exc_tb else None

            self.db.log_error(
                self.run_id,
                error_type="python_exception",
                error_message=error_message,
                traceback=traceback,
            )

        return False  # Don't suppress exceptions

    def log_progress(
        self,
        epoch: int,
        train_loss: float | None = None,
        test_loss: float | None = None,
        f1_score: float | None = None,
        best_fitness: float | None = None,
        mean_pct_diff: float | None = None,
        std_pct_diff: float | None = None,
        gpu_memory_mb: float | None = None,
    ) -> None:
        """Log metrics during training.

        Args:
            epoch: Current epoch/generation number
            train_loss: Training loss
            test_loss: Test loss
            f1_score: F1 score
            best_fitness: Best fitness (for GA)
            mean_pct_diff: Mean percentage difference from human
            std_pct_diff: Std percentage difference from human
            gpu_memory_mb: GPU memory usage in MB
        """
        if self.run_id is None or self.start_time is None:
            return

        elapsed = time.time() - self.start_time

        self.db.log_metrics(
            run_id=self.run_id,
            elapsed_time=elapsed,
            epoch_or_generation=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            f1_score=f1_score,
            best_fitness=best_fitness,
            mean_pct_diff=mean_pct_diff,
            std_pct_diff=std_pct_diff,
            gpu_memory_mb=gpu_memory_mb,
        )
