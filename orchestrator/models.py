"""Data models for the Experiment Orchestrator."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List
from datetime import datetime
import uuid


class TaskState(Enum):
    """Possible states for a task."""
    UNAPPROVED = "Unapproved"
    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    KILLED = "Killed"


@dataclass
class Task:
    """Represents a computational task."""
    id: str
    command: str
    environment: str
    state: TaskState = TaskState.UNAPPROVED
    tmux_session: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    output_buffer: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None
    slurm_job_id: Optional[str] = None  # For SLURM tasks

    @staticmethod
    def create(command: str, environment: str) -> 'Task':
        """Factory method to create a new task with a unique ID."""
        return Task(
            id=str(uuid.uuid4()),
            command=command,
            environment=environment
        )

    def get_display_name(self) -> str:
        """Get a display-friendly name for the task."""
        # Truncate command if too long
        cmd_preview = self.command[:50] + "..." if len(self.command) > 50 else self.command
        return f"[{self.state.value}] {cmd_preview}"

    def get_tmux_session_name(self) -> str:
        """Generate tmux session name for this task."""
        # Use first 8 characters of UUID for brevity
        short_id = self.id[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"hbmr_{self.environment}_{short_id}_{timestamp}"

    def append_output(self, lines: List[str]):
        """Append output lines, keeping only last 500 lines."""
        self.output_buffer.extend(lines)
        if len(self.output_buffer) > 500:
            self.output_buffer = self.output_buffer[-500:]

    def to_dict(self) -> dict:
        """Serialize task to dictionary."""
        return {
            'id': self.id,
            'command': self.command,
            'environment': self.environment,
            'state': self.state.value,
            'tmux_session': self.tmux_session,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'output_buffer': self.output_buffer,
            'exit_code': self.exit_code,
            'slurm_job_id': self.slurm_job_id
        }

    @staticmethod
    def from_dict(data: dict) -> 'Task':
        """Deserialize task from dictionary."""
        return Task(
            id=data['id'],
            command=data['command'],
            environment=data['environment'],
            state=TaskState(data['state']),
            tmux_session=data.get('tmux_session'),
            start_time=data.get('start_time'),
            end_time=data.get('end_time'),
            output_buffer=data.get('output_buffer', []),
            exit_code=data.get('exit_code'),
            slurm_job_id=data.get('slurm_job_id')
        )


@dataclass
class SLURMConfig:
    """Configuration for SLURM execution."""
    partition: str = ""
    time_limit: str = "3:00:00"
    extra_flags: str = ""


@dataclass
class SSHConfig:
    """Configuration for SSH connection."""
    host: str
    user: str
    port: int = 22
    proxy_jump: Optional[str] = None


@dataclass
class Environment:
    """Represents an execution environment."""
    name: str
    concurrency_limit: int
    activation_cmd: str
    remote_dropbox: str
    ssh_config: Optional[SSHConfig] = None
    slurm_config: Optional[SLURMConfig] = None

    def is_local(self) -> bool:
        """Check if this is a local environment."""
        return self.ssh_config is None

    def is_slurm(self) -> bool:
        """Check if this environment uses SLURM."""
        return self.slurm_config is not None

    def to_dict(self) -> dict:
        """Serialize environment to dictionary."""
        data = {
            'name': self.name,
            'concurrency_limit': self.concurrency_limit,
            'activation_cmd': self.activation_cmd,
            'remote_dropbox': self.remote_dropbox
        }
        if self.ssh_config:
            data['ssh_config'] = {
                'host': self.ssh_config.host,
                'port': self.ssh_config.port,
                'user': self.ssh_config.user,
                'proxy_jump': self.ssh_config.proxy_jump
            }
        if self.slurm_config:
            data['slurm_config'] = {
                'partition': self.slurm_config.partition,
                'time_limit': self.slurm_config.time_limit,
                'extra_flags': self.slurm_config.extra_flags
            }
        return data

    @staticmethod
    def from_dict(data: dict) -> 'Environment':
        """Deserialize environment from dictionary."""
        ssh_config = None
        if 'ssh_config' in data:
            ssh_config = SSHConfig(**data['ssh_config'])

        slurm_config = None
        if 'slurm_config' in data:
            slurm_config = SLURMConfig(**data['slurm_config'])

        return Environment(
            name=data['name'],
            concurrency_limit=data['concurrency_limit'],
            activation_cmd=data['activation_cmd'],
            remote_dropbox=data['remote_dropbox'],
            ssh_config=ssh_config,
            slurm_config=slurm_config
        )


class AppState:
    """Manages the application state including tasks and environments."""

    def __init__(self):
        self.tasks: Dict[str, List[Task]] = {
            'local': [],
            'ginkgo': [],
            'rorqual': []
        }
        self.environments: Dict[str, Environment] = {}

    def add_task(self, task: Task):
        """Add a task to the appropriate environment queue."""
        if task.environment in self.tasks:
            self.tasks[task.environment].append(task)

    def get_tasks(self, environment: str) -> List[Task]:
        """Get all tasks for an environment."""
        return self.tasks.get(environment, [])

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Find a task by its ID across all environments."""
        for env_tasks in self.tasks.values():
            for task in env_tasks:
                if task.id == task_id:
                    return task
        return None

    def remove_task(self, task: Task):
        """Remove a task from its environment queue."""
        if task.environment in self.tasks:
            try:
                self.tasks[task.environment].remove(task)
            except ValueError:
                pass

    def get_running_tasks(self, environment: str) -> List[Task]:
        """Get all running tasks for an environment."""
        return [t for t in self.get_tasks(environment) if t.state == TaskState.RUNNING]

    def get_pending_tasks(self, environment: str) -> List[Task]:
        """Get all pending tasks for an environment."""
        return [t for t in self.get_tasks(environment) if t.state == TaskState.PENDING]

    def to_dict(self) -> dict:
        """Serialize application state to dictionary."""
        return {
            'tasks': {
                env: [task.to_dict() for task in tasks]
                for env, tasks in self.tasks.items()
            },
            'environments': {
                name: env.to_dict()
                for name, env in self.environments.items()
            }
        }

    @staticmethod
    def from_dict(data: dict) -> 'AppState':
        """Deserialize application state from dictionary."""
        state = AppState()

        # Restore tasks
        if 'tasks' in data:
            for env, task_list in data['tasks'].items():
                state.tasks[env] = [Task.from_dict(t) for t in task_list]

        # Restore environments
        if 'environments' in data:
            for name, env_data in data['environments'].items():
                state.environments[name] = Environment.from_dict(env_data)

        return state
