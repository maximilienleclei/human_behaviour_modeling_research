"""Main entry point for the Experiment Orchestrator."""

import queue
import signal
import sys
from pathlib import Path
from .models import AppState, Task, TaskState
from .config import get_default_environments
from .persistence import StateManager
from .scheduler import TaskScheduler
from .tui import OrchestratorTUI
from .api import OrchestratorAPI


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

        # Create API
        self.api = OrchestratorAPI(self.app_state, self.scheduler)

        # Create TUI
        self.tui = OrchestratorTUI(
            app_state=self.app_state,
            message_queue=self.message_queue,
            api=self.api
        )

        # Register cleanup handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def run(self):
        """Run the orchestrator."""
        try:
            self.tui.run()
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
