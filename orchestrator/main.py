"""Main entry point for the Experiment Orchestrator."""

import queue
import signal
import sys
from pathlib import Path
from .models import AppState, Task, TaskState
from .config import get_default_environments
from .persistence import StateManager
from .scheduler import TaskScheduler
from .gui import OrchestratorGUI


class Orchestrator:
    """Main orchestrator application."""

    def __init__(self):
        # Create message queue for thread communication
        self.message_queue = queue.Queue()

        # Initialize state manager
        self.state_manager = StateManager()

        # Load or create application state
        self.app_state = self.state_manager.load()

        if self.app_state is None:
            # Create new state with default environments
            self.app_state = AppState()
            self.app_state.environments = get_default_environments()
        else:
            # Merge with default environments (in case config changed)
            default_envs = get_default_environments()
            for name, env in default_envs.items():
                if name not in self.app_state.environments:
                    self.app_state.environments[name] = env

        # Create and start scheduler
        self.scheduler = TaskScheduler(self.app_state, self.message_queue)
        self.scheduler.start()

        # Reconnect to running tasks
        self.scheduler.reconnect_running_tasks()

        # Create GUI
        self.gui = OrchestratorGUI(
            app_state=self.app_state,
            message_queue=self.message_queue,
            load_tasks_callback=self.load_tasks,
            approve_callback=self.approve_tasks,
            clear_unapproved_callback=self.clear_unapproved,
            remove_selected_callback=self.remove_tasks,
            remove_all_done_callback=self.remove_all_done,
            rerun_callback=self.rerun_tasks,
            set_concurrency_callback=self.set_concurrency,
            kill_task_callback=self.kill_task
        )

        # Register cleanup handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def load_tasks(self, environment: str, filename: str):
        """Load tasks from a file."""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()

            # Create tasks from lines
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    task = Task.create(command=line, environment=environment)
                    self.app_state.add_task(task)

            print(f"Loaded {len(lines)} tasks from {filename}")

        except Exception as e:
            print(f"Error loading tasks: {e}")

    def approve_tasks(self, tasks):
        """Approve selected tasks."""
        for task in tasks:
            if task.state == TaskState.UNAPPROVED:
                task.state = TaskState.PENDING

    def clear_unapproved(self, environment: str):
        """Clear all unapproved tasks."""
        tasks = self.app_state.get_tasks(environment)
        to_remove = [t for t in tasks if t.state == TaskState.UNAPPROVED]

        for task in to_remove:
            self.app_state.remove_task(task)

    def remove_tasks(self, tasks):
        """Remove selected tasks."""
        for task in tasks:
            self.app_state.remove_task(task)

    def remove_all_done(self, environment: str):
        """Remove all completed/failed/killed tasks."""
        tasks = self.app_state.get_tasks(environment)
        done_states = [TaskState.COMPLETED, TaskState.FAILED, TaskState.KILLED]
        to_remove = [t for t in tasks if t.state in done_states]

        for task in to_remove:
            self.app_state.remove_task(task)

    def rerun_tasks(self, tasks):
        """Reset tasks to pending state."""
        for task in tasks:
            if task.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.KILLED]:
                task.state = TaskState.PENDING
                task.tmux_session = None
                task.start_time = None
                task.end_time = None
                task.exit_code = None
                task.output_buffer = []

    def set_concurrency(self, environment: str, limit: int):
        """Set concurrency limit for an environment."""
        env = self.app_state.environments.get(environment)
        if env:
            env.concurrency_limit = limit

    def kill_task(self, task: Task):
        """Kill a running task."""
        self.scheduler.kill_task(task)

    def run(self):
        """Run the orchestrator."""
        try:
            self.gui.run()
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup on exit."""
        print("Saving state...")
        self.state_manager.save(self.app_state)

        print("Stopping scheduler...")
        self.scheduler.stop()

        print("Shutdown complete")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)


def main():
    """Main entry point."""
    orchestrator = Orchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
