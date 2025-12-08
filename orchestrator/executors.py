"""Execution layer for running tasks in different environments."""

import subprocess
import paramiko
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from datetime import datetime
from .models import Task, Environment


class BaseExecutor(ABC):
    """Abstract base class for task executors."""

    def __init__(self, environment: Environment):
        self.environment = environment

    @abstractmethod
    def execute(self, task: Task) -> bool:
        """
        Start executing a task.

        Args:
            task: The task to execute

        Returns:
            True if execution started successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_output(self, task: Task) -> List[str]:
        """
        Get new output lines from a running task.

        Args:
            task: The task to get output from

        Returns:
            List of new output lines
        """
        pass

    @abstractmethod
    def check_status(self, task: Task) -> Tuple[bool, Optional[int]]:
        """
        Check if a task is still running.

        Args:
            task: The task to check

        Returns:
            Tuple of (is_running, exit_code)
        """
        pass

    @abstractmethod
    def kill(self, task: Task) -> bool:
        """
        Kill a running task.

        Args:
            task: The task to kill

        Returns:
            True if killed successfully, False otherwise
        """
        pass

    @abstractmethod
    def session_exists(self, session_name: str) -> bool:
        """
        Check if a tmux session exists.

        Args:
            session_name: Name of the tmux session

        Returns:
            True if session exists, False otherwise
        """
        pass


class LocalExecutor(BaseExecutor):
    """Executor for local machine tasks."""

    def execute(self, task: Task) -> bool:
        """Start executing a task in a local tmux session."""
        try:
            # Generate tmux session name
            session_name = task.get_tmux_session_name()
            task.tmux_session = session_name

            # Create tmux session and run command
            # First create a detached session
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session_name],
                check=True
            )

            # Send activation command
            subprocess.run(
                ["tmux", "send-keys", "-t", session_name,
                 self.environment.activation_cmd, "Enter"],
                check=True
            )

            # Send the actual command
            subprocess.run(
                ["tmux", "send-keys", "-t", session_name,
                 task.command, "Enter"],
                check=True
            )

            # Record start time
            task.start_time = datetime.now().isoformat()

            return True

        except subprocess.CalledProcessError as e:
            print(f"Error starting local task {task.id[:8]}: {e}")
            return False

    def get_output(self, task: Task) -> List[str]:
        """Get output from tmux session."""
        if not task.tmux_session:
            return []

        try:
            # Capture pane content
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", task.tmux_session, "-p"],
                capture_output=True,
                text=True,
                check=True
            )

            # Split into lines and return
            lines = result.stdout.split('\n')
            return lines

        except subprocess.CalledProcessError:
            return []

    def check_status(self, task: Task) -> Tuple[bool, Optional[int]]:
        """Check if the tmux session is still running."""
        if not task.tmux_session:
            return False, None

        # Check if session exists
        if not self.session_exists(task.tmux_session):
            return False, 0  # Session ended, assume success (we can't get exit code)

        # Session exists, task is still running
        return True, None

    def kill(self, task: Task) -> bool:
        """Kill the tmux session."""
        if not task.tmux_session:
            return False

        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", task.tmux_session],
                check=True
            )
            return True

        except subprocess.CalledProcessError as e:
            print(f"Error killing task {task.id[:8]}: {e}")
            return False

    def session_exists(self, session_name: str) -> bool:
        """Check if a tmux session exists."""
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", session_name],
                capture_output=True,
                check=False
            )
            return result.returncode == 0

        except Exception:
            return False


class SSHExecutor(BaseExecutor):
    """Executor for remote SSH tasks."""

    def __init__(self, environment: Environment):
        super().__init__(environment)
        self.ssh_client: Optional[paramiko.SSHClient] = None
        self._connect()

    def _connect(self):
        """Establish SSH connection."""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            ssh_config = self.environment.ssh_config
            if not ssh_config:
                return

            # Build connection parameters
            connect_kwargs = {
                'hostname': ssh_config.host,
                'port': ssh_config.port,
                'username': ssh_config.user
            }

            # Handle ProxyJump if specified
            if ssh_config.proxy_jump:
                # Parse proxy jump (format: user@host)
                proxy_parts = ssh_config.proxy_jump.split('@')
                if len(proxy_parts) == 2:
                    proxy_user, proxy_host = proxy_parts

                    # Create proxy client
                    proxy_client = paramiko.SSHClient()
                    proxy_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    proxy_client.connect(hostname=proxy_host, username=proxy_user)

                    # Create transport through proxy
                    proxy_transport = proxy_client.get_transport()
                    dest_addr = (ssh_config.host, ssh_config.port)
                    local_addr = ('127.0.0.1', 0)
                    proxy_channel = proxy_transport.open_channel("direct-tcpip", dest_addr, local_addr)

                    connect_kwargs['sock'] = proxy_channel

            self.ssh_client.connect(**connect_kwargs)

        except Exception as e:
            print(f"Error connecting to {self.environment.name}: {e}")
            self.ssh_client = None

    def _exec_command(self, command: str) -> Tuple[str, str, int]:
        """Execute a command over SSH."""
        if not self.ssh_client:
            return "", "SSH not connected", 1

        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            exit_code = stdout.channel.recv_exit_status()
            return stdout.read().decode(), stderr.read().decode(), exit_code

        except Exception as e:
            return "", str(e), 1

    def execute(self, task: Task) -> bool:
        """Start executing a task in a remote tmux session."""
        try:
            # Generate tmux session name
            session_name = task.get_tmux_session_name()
            task.tmux_session = session_name

            # Create tmux session remotely
            stdout, stderr, code = self._exec_command(
                f"tmux new-session -d -s {session_name}"
            )
            if code != 0:
                print(f"Error creating tmux session: {stderr}")
                return False

            # Send activation command
            self._exec_command(
                f"tmux send-keys -t {session_name} '{self.environment.activation_cmd}' Enter"
            )

            # Send the actual command
            # Escape single quotes in the command
            escaped_cmd = task.command.replace("'", "'\"'\"'")
            self._exec_command(
                f"tmux send-keys -t {session_name} '{escaped_cmd}' Enter"
            )

            # Record start time
            task.start_time = datetime.now().isoformat()

            return True

        except Exception as e:
            print(f"Error starting remote task {task.id[:8]}: {e}")
            return False

    def get_output(self, task: Task) -> List[str]:
        """Get output from remote tmux session."""
        if not task.tmux_session:
            return []

        try:
            stdout, stderr, code = self._exec_command(
                f"tmux capture-pane -t {task.tmux_session} -p"
            )

            if code == 0:
                return stdout.split('\n')
            else:
                return []

        except Exception:
            return []

    def check_status(self, task: Task) -> Tuple[bool, Optional[int]]:
        """Check if the remote tmux session is still running."""
        if not task.tmux_session:
            return False, None

        # Check if session exists
        if not self.session_exists(task.tmux_session):
            return False, 0

        return True, None

    def kill(self, task: Task) -> bool:
        """Kill the remote tmux session."""
        if not task.tmux_session:
            return False

        try:
            stdout, stderr, code = self._exec_command(
                f"tmux kill-session -t {task.tmux_session}"
            )
            return code == 0

        except Exception as e:
            print(f"Error killing remote task {task.id[:8]}: {e}")
            return False

    def session_exists(self, session_name: str) -> bool:
        """Check if a remote tmux session exists."""
        try:
            stdout, stderr, code = self._exec_command(
                f"tmux has-session -t {session_name}"
            )
            return code == 0

        except Exception:
            return False

    def __del__(self):
        """Close SSH connection on cleanup."""
        if self.ssh_client:
            self.ssh_client.close()


class SLURMExecutor(SSHExecutor):
    """Executor for SLURM-based HPC tasks."""

    def execute(self, task: Task) -> bool:
        """Submit a SLURM job that runs the task in tmux."""
        try:
            # Generate tmux session name
            session_name = task.get_tmux_session_name()
            task.tmux_session = session_name

            # Create sbatch script
            slurm = self.environment.slurm_config
            sbatch_script = self._generate_sbatch_script(task, session_name)

            # Write script to remote temp file
            script_path = f"/tmp/{session_name}.sbatch"
            # Escape the script content for echo command
            escaped_script = sbatch_script.replace("'", "'\"'\"'").replace("\n", "\\n")

            # Write script using echo
            self._exec_command(f"cat > {script_path} << 'EOFSCRIPT'\n{sbatch_script}\nEOFSCRIPT")

            # Submit the job
            stdout, stderr, code = self._exec_command(f"sbatch {script_path}")

            if code != 0:
                print(f"Error submitting SLURM job: {stderr}")
                return False

            # Parse job ID from output (format: "Submitted batch job 12345")
            job_id = stdout.strip().split()[-1]
            task.slurm_job_id = job_id

            # Record start time
            task.start_time = datetime.now().isoformat()

            return True

        except Exception as e:
            print(f"Error starting SLURM task {task.id[:8]}: {e}")
            return False

    def _generate_sbatch_script(self, task: Task, session_name: str) -> str:
        """Generate SLURM sbatch script."""
        slurm = self.environment.slurm_config

        script = "#!/bin/bash\n"

        # Add SLURM directives
        if slurm.partition:
            script += f"#SBATCH --partition={slurm.partition}\n"
        script += f"#SBATCH --time={slurm.time_limit}\n"
        script += f"#SBATCH --job-name={session_name}\n"

        # Add extra flags
        if slurm.extra_flags:
            for flag in slurm.extra_flags.split():
                script += f"#SBATCH {flag}\n"

        script += "\n"

        # Create tmux session and run command
        script += f"tmux new-session -d -s {session_name}\n"
        script += f"tmux send-keys -t {session_name} '{self.environment.activation_cmd}' Enter\n"

        # Escape single quotes in command
        escaped_cmd = task.command.replace("'", "'\"'\"'")
        script += f"tmux send-keys -t {session_name} '{escaped_cmd}' Enter\n"

        # Wait for tmux session to finish
        script += f"while tmux has-session -t {session_name} 2>/dev/null; do sleep 10; done\n"

        return script

    def check_status(self, task: Task) -> Tuple[bool, Optional[int]]:
        """Check SLURM job status."""
        if not task.slurm_job_id:
            return False, None

        try:
            # Check job status using squeue
            stdout, stderr, code = self._exec_command(
                f"squeue -j {task.slurm_job_id} --noheader"
            )

            # If squeue returns nothing, job is done
            if not stdout.strip():
                # Job finished, check if tmux session still exists
                if self.session_exists(task.tmux_session):
                    return True, None
                else:
                    return False, 0

            # Job is still in queue or running
            return True, None

        except Exception:
            return False, None
