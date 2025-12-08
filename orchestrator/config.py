"""Environment configurations for the Experiment Orchestrator."""

from .models import Environment, SSHConfig, SLURMConfig


def get_default_environments():
    """Get the default environment configurations."""

    # Local environment
    local = Environment(
        name="local",
        concurrency_limit=2,
        activation_cmd="source /home/maximilienleclei/venvs/hbmr/bin/activate",
        remote_dropbox="/home/maximilienleclei/Dropbox"
    )

    # Ginkgo environment (SSH with ProxyJump)
    ginkgo = Environment(
        name="ginkgo",
        concurrency_limit=2,
        activation_cmd="source /scratch/mleclei/venv/bin/activate",
        remote_dropbox="/scratch/mleclei/Dropbox",
        ssh_config=SSHConfig(
            host="ginkgo.criugm.qc.ca",
            user="mleclei",
            port=22,
            proxy_jump="mleclei@elm.criugm.qc.ca"
        )
    )

    # Rorqual environment (SLURM HPC)
    rorqual = Environment(
        name="rorqual",
        concurrency_limit=10,
        activation_cmd="module load gcc arrow python/3.12 && source ~/venv/bin/activate",
        remote_dropbox="/scratch/mleclei/Dropbox",
        ssh_config=SSHConfig(
            host="rorqual1.alliancecan.ca",
            user="mleclei",
            port=22
        ),
        slurm_config=SLURMConfig(
            partition="",  # Use default partition
            time_limit="3:00:00",
            extra_flags="--gpus=h100:1 --account=rrg-pbellec --mem=124G --cpus-per-task=16"
        )
    )

    return {
        'local': local,
        'ginkgo': ginkgo,
        'rorqual': rorqual
    }
