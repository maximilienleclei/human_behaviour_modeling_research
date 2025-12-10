"""Core API for orchestrator operations.

This module provides a unified API for both TUI and CLI interfaces,
ensuring consistent behavior across all interfaces.
"""

from typing import List, Optional
from .models import AppState, Task, TaskState


class OrchestratorAPI:
    """Core API for orchestrator operations."""

    def __init__(self, app_state: AppState, scheduler):
        """
        Initialize the API.

        Args:
            app_state: The application state
            scheduler: The task scheduler instance
        """
        self.app_state = app_state
        self.scheduler = scheduler

    def load_tasks(self, environment: str, filename: str) -> int:
        """
        Load tasks from a file.

        Args:
            environment: The target environment
            filename: Path to file containing tasks (one per line)

        Returns:
            Number of tasks loaded

        Raises:
            Exception: If file cannot be read
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Create tasks from lines
        count = 0
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                task = Task.create(command=line, environment=environment)
                self.app_state.add_task(task)
                count += 1

        return count

    def approve_tasks(self, task_ids: List[str]) -> int:
        """
        Approve tasks by ID.

        Args:
            task_ids: List of task IDs to approve

        Returns:
            Number of tasks approved
        """
        count = 0
        for task_id in task_ids:
            task = self.app_state.get_task_by_id(task_id)
            if task and task.state == TaskState.UNAPPROVED:
                task.state = TaskState.PENDING
                count += 1
        return count

    def approve_task_objects(self, tasks: List[Task]) -> int:
        """
        Approve tasks (objects).

        Args:
            tasks: List of Task objects to approve

        Returns:
            Number of tasks approved
        """
        count = 0
        for task in tasks:
            if task.state == TaskState.UNAPPROVED:
                task.state = TaskState.PENDING
                count += 1
        return count

    def clear_unapproved(self, environment: str) -> int:
        """
        Clear all unapproved tasks in an environment.

        Args:
            environment: The target environment

        Returns:
            Number of tasks removed
        """
        tasks = self.app_state.get_tasks(environment)
        to_remove = [t for t in tasks if t.state == TaskState.UNAPPROVED]

        for task in to_remove:
            self.app_state.remove_task(task)

        return len(to_remove)

    def remove_tasks(self, task_ids: List[str]) -> int:
        """
        Remove tasks by ID.

        Args:
            task_ids: List of task IDs to remove

        Returns:
            Number of tasks removed
        """
        count = 0
        for task_id in task_ids:
            task = self.app_state.get_task_by_id(task_id)
            if task:
                # Kill if running
                if task.state == TaskState.RUNNING:
                    self.kill_task(task_id)
                self.app_state.remove_task(task)
                count += 1
        return count

    def remove_task_objects(self, tasks: List[Task]) -> int:
        """
        Remove tasks (objects).

        Args:
            tasks: List of Task objects to remove

        Returns:
            Number of tasks removed
        """
        for task in tasks:
            # Kill if running
            if task.state == TaskState.RUNNING:
                self.scheduler.kill_task(task)
            self.app_state.remove_task(task)
        return len(tasks)

    def remove_all_done(self, environment: str) -> int:
        """
        Remove all completed/failed/killed tasks.

        Args:
            environment: The target environment

        Returns:
            Number of tasks removed
        """
        tasks = self.app_state.get_tasks(environment)
        done_states = [TaskState.COMPLETED, TaskState.FAILED, TaskState.KILLED]
        to_remove = [t for t in tasks if t.state in done_states]

        for task in to_remove:
            self.app_state.remove_task(task)

        return len(to_remove)

    def rerun_tasks(self, task_ids: List[str]) -> int:
        """
        Reset tasks to pending state for re-running.

        Args:
            task_ids: List of task IDs to rerun

        Returns:
            Number of tasks reset
        """
        count = 0
        for task_id in task_ids:
            task = self.app_state.get_task_by_id(task_id)
            if task and task.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.KILLED]:
                task.state = TaskState.PENDING
                task.tmux_session = None
                task.start_time = None
                task.end_time = None
                task.exit_code = None
                task.output_buffer = []
                count += 1
        return count

    def rerun_task_objects(self, tasks: List[Task]) -> int:
        """
        Reset tasks (objects) to pending state for re-running.

        Args:
            tasks: List of Task objects to rerun

        Returns:
            Number of tasks reset
        """
        count = 0
        for task in tasks:
            if task.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.KILLED]:
                task.state = TaskState.PENDING
                task.tmux_session = None
                task.start_time = None
                task.end_time = None
                task.exit_code = None
                task.output_buffer = []
                count += 1
        return count

    def set_concurrency(self, environment: str, limit: int) -> bool:
        """
        Set concurrency limit for an environment.

        Args:
            environment: The target environment
            limit: New concurrency limit (must be positive)

        Returns:
            True if successful, False otherwise
        """
        if limit <= 0:
            return False

        env = self.app_state.environments.get(environment)
        if env:
            env.concurrency_limit = limit
            return True
        return False

    def kill_task(self, task_id: str) -> bool:
        """
        Kill a running task.

        Args:
            task_id: ID of task to kill

        Returns:
            True if task was killed, False otherwise
        """
        task = self.app_state.get_task_by_id(task_id)
        if task and task.state == TaskState.RUNNING:
            self.scheduler.kill_task(task)
            return True
        return False

    def get_tasks(self, environment: str) -> List[Task]:
        """
        Get all tasks for an environment.

        Args:
            environment: The target environment

        Returns:
            List of tasks
        """
        return self.app_state.get_tasks(environment)

    def get_task_status(self, task_id: str) -> Optional[Task]:
        """
        Get task details by ID.

        Args:
            task_id: Task ID to look up

        Returns:
            Task object if found, None otherwise
        """
        return self.app_state.get_task_by_id(task_id)

    def get_environments(self) -> List[str]:
        """
        Get list of available environment names.

        Returns:
            List of environment names
        """
        return list(self.app_state.environments.keys())

    def get_concurrency_limit(self, environment: str) -> Optional[int]:
        """
        Get concurrency limit for an environment.

        Args:
            environment: The target environment

        Returns:
            Concurrency limit if environment exists, None otherwise
        """
        env = self.app_state.environments.get(environment)
        return env.concurrency_limit if env else None
