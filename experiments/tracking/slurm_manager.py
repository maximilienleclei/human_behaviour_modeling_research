"""SLURM job management for experiments."""

import json
import subprocess
from datetime import datetime
from itertools import product
from pathlib import Path

from experiments.tracking.database import ExperimentDB


class SlurmManager:
    """Manage SLURM job submission and monitoring."""

    def __init__(
        self,
        db: ExperimentDB,
        template_dir: Path,
        log_dir: Path,
        config_dir: Path,
        project_root: Path,
    ) -> None:
        """Initialize SLURM manager.

        Args:
            db: Database interface
            template_dir: Directory containing SLURM templates
            log_dir: Directory for SLURM logs
            config_dir: Directory for job array configs
            project_root: Project root directory
        """
        self.db = db
        self.template_dir = template_dir
        self.log_dir = log_dir
        self.config_dir = config_dir
        self.project_root = project_root

        # Ensure directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def submit_single_job(
        self,
        experiment_number: int,
        dataset: str,
        method: str,
        subject: str,
        use_cl_info: bool,
        seed: int,
        time_limit: str = "00:30:00",
        gpu_type: str = "h100_1g.10gb:1",
        account: str = "rrg-pbellec",
        mem: str = "15G",
        cpus: int = 2,
    ) -> int:
        """Submit single SLURM job.

        Args:
            experiment_number: Experiment number
            dataset: Dataset name
            method: Method name
            subject: Subject identifier
            use_cl_info: Whether to use CL info
            seed: Random seed
            time_limit: SLURM time limit
            gpu_type: GPU type specification
            account: SLURM account
            mem: Memory limit
            cpus: Number of CPUs

        Returns:
            SLURM job ID
        """
        # Read template
        template_path = self.template_dir / "job_single.sh"
        with open(template_path) as f:
            template = f.read()

        # Prepare substitutions
        job_name = f"exp{experiment_number}_{dataset}_{method}"
        experiment_dir = self.project_root / "experiments" / f"{experiment_number}_*"

        # Find the actual experiment directory
        experiment_dirs = list(self.project_root.glob(f"experiments/{experiment_number}_*"))
        if not experiment_dirs:
            raise ValueError(f"No experiment directory found for experiment {experiment_number}")
        experiment_dir = experiment_dirs[0]

        use_cl_flag = "--use-cl-info" if use_cl_info else ""

        # Create job-specific log directory
        job_log_dir = self.log_dir / job_name
        job_log_dir.mkdir(parents=True, exist_ok=True)

        # Substitute template variables
        script_content = template.format(
            job_name=job_name,
            log_dir=str(job_log_dir),
            time_limit=time_limit,
            gpu_type=gpu_type,
            account=account,
            mem=mem,
            cpus=cpus,
            experiment_dir=str(experiment_dir),
            dataset=dataset,
            method=method,
            subject=subject,
            use_cl_flag=use_cl_flag,
            seed=seed,
        )

        # Write temporary script file
        script_path = self.config_dir / f"job_{job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sh"
        with open(script_path, "w") as f:
            f.write(script_content)

        # Submit job
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")

        # Parse job ID from output
        # Output format: "Submitted batch job 12345"
        job_id = int(result.stdout.strip().split()[-1])

        return job_id

    def submit_job_array(
        self,
        experiment_number: int,
        run_configs: list[dict],
        time_limit: str = "00:30:00",
        gpu_type: str = "h100_1g.10gb:1",
        account: str = "rrg-pbellec",
        mem: str = "15G",
        cpus: int = 2,
        max_concurrent: int = 10,
    ) -> int:
        """Submit SLURM job array.

        Args:
            experiment_number: Experiment number
            run_configs: List of run configuration dictionaries
            time_limit: SLURM time limit
            gpu_type: GPU type specification
            account: SLURM account
            mem: Memory limit
            cpus: Number of CPUs
            max_concurrent: Maximum concurrent jobs

        Returns:
            SLURM job ID
        """
        if not run_configs:
            raise ValueError("run_configs cannot be empty")

        # Read template
        template_path = self.template_dir / "job_array.sh"
        with open(template_path) as f:
            template = f.read()

        # Find experiment directory
        experiment_dirs = list(self.project_root.glob(f"experiments/{experiment_number}_*"))
        if not experiment_dirs:
            raise ValueError(f"No experiment directory found for experiment {experiment_number}")
        experiment_dir = experiment_dirs[0]
        experiment_main = experiment_dir / "main.py"

        # Create config file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = self.config_dir / f"array_config_{experiment_number}_{timestamp}.json"

        with open(config_file, "w") as f:
            json.dump(run_configs, f, indent=2)

        # Prepare substitutions
        job_name = f"exp{experiment_number}_array"
        max_task_id = len(run_configs) - 1

        # Create job-specific log directory
        job_log_dir = self.log_dir / f"{job_name}_{timestamp}"
        job_log_dir.mkdir(parents=True, exist_ok=True)

        # Substitute template variables
        script_content = template.format(
            job_name=job_name,
            log_dir=str(job_log_dir),
            max_task_id=max_task_id,
            max_concurrent=max_concurrent,
            time_limit=time_limit,
            gpu_type=gpu_type,
            account=account,
            mem=mem,
            cpus=cpus,
            project_root=str(self.project_root),
            config_file=str(config_file),
            experiment_main=str(experiment_main),
        )

        # Write temporary script file
        script_path = self.config_dir / f"job_array_{experiment_number}_{timestamp}.sh"
        with open(script_path, "w") as f:
            f.write(script_content)

        # Submit job
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")

        # Parse job ID
        job_id = int(result.stdout.strip().split()[-1])

        return job_id

    def submit_sweep(
        self,
        experiment_number: int,
        datasets: list[str],
        methods: list[str],
        subjects: list[str],
        use_cl_info_values: list[bool],
        seeds: list[int],
        time_limit: str = "00:30:00",
        gpu_type: str = "h100_1g.10gb:1",
        account: str = "rrg-pbellec",
        mem: str = "15G",
        cpus: int = 2,
        max_concurrent: int = 10,
    ) -> int:
        """Submit full parameter sweep as job array.

        Args:
            experiment_number: Experiment number
            datasets: List of datasets
            methods: List of methods
            subjects: List of subjects
            use_cl_info_values: List of CL info values
            seeds: List of seeds
            time_limit: SLURM time limit
            gpu_type: GPU type specification
            account: SLURM account
            mem: Memory limit
            cpus: Number of CPUs
            max_concurrent: Maximum concurrent jobs

        Returns:
            SLURM job ID
        """
        # Generate all combinations
        combinations = product(datasets, methods, subjects, use_cl_info_values, seeds)

        run_configs = []
        for dataset, method, subject, use_cl_info, seed in combinations:
            run_configs.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "subject": subject,
                    "use_cl_info": use_cl_info,
                    "seed": seed,
                }
            )

        # Submit as job array
        return self.submit_job_array(
            experiment_number=experiment_number,
            run_configs=run_configs,
            time_limit=time_limit,
            gpu_type=gpu_type,
            account=account,
            mem=mem,
            cpus=cpus,
            max_concurrent=max_concurrent,
        )

    def get_job_status(self, job_id: int) -> dict:
        """Query SLURM for job status.

        Args:
            job_id: SLURM job ID

        Returns:
            Job status dictionary
        """
        result = subprocess.run(
            ["sacct", "-j", str(job_id), "--format=JobID,State,ExitCode", "--parsable2"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return {"job_id": job_id, "status": "unknown", "error": result.stderr}

        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return {"job_id": job_id, "status": "not_found"}

        # Parse output (skip header)
        parts = lines[1].split("|")
        return {
            "job_id": job_id,
            "status": parts[1] if len(parts) > 1 else "unknown",
            "exit_code": parts[2] if len(parts) > 2 else None,
        }

    def cancel_job(self, job_id: int) -> None:
        """Cancel SLURM job.

        Args:
            job_id: SLURM job ID
        """
        subprocess.run(["scancel", str(job_id)], check=True)

    def monitor_jobs(self, experiment_number: int | None = None) -> dict:
        """Get status of all tracked jobs.

        Args:
            experiment_number: Filter by experiment number

        Returns:
            Status summary dictionary
        """
        runs = self.db.get_all_runs(experiment_number=experiment_number)

        status_counts = {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "timeout": 0,
        }

        for run in runs:
            status = run.get("status", "unknown")
            if status in status_counts:
                status_counts[status] += 1

        return status_counts
