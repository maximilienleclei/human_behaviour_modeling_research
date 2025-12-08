"""State persistence for the Experiment Orchestrator."""

import json
import os
from pathlib import Path
from typing import Optional
from .models import AppState, TaskState


class StateManager:
    """Manages saving and loading application state."""

    def __init__(self, state_file: str = "orchestrator/state.json"):
        self.state_file = Path(state_file)

    def save(self, state: AppState) -> None:
        """Save application state to JSON file."""
        try:
            # Create directory if it doesn't exist
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            # Serialize state
            data = state.to_dict()

            # Write to file with pretty printing
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error saving state: {e}")

    def load(self) -> Optional[AppState]:
        """Load application state from JSON file."""
        try:
            # Check if file exists
            if not self.state_file.exists():
                return None

            # Read from file
            with open(self.state_file, 'r') as f:
                data = json.load(f)

            # Deserialize state
            state = AppState.from_dict(data)
            return state

        except Exception as e:
            print(f"Error loading state: {e}")
            return None

    def reconnect_tasks(self, state: AppState, check_session_fn) -> None:
        """
        Reconnect to running tmux sessions after restart.

        Args:
            state: The application state
            check_session_fn: Function to check if a tmux session exists
                             Should take (environment, session_name) and return bool
        """
        for env_name, tasks in state.tasks.items():
            for task in tasks:
                # Only try to reconnect to tasks that were running
                if task.state == TaskState.RUNNING and task.tmux_session:
                    # Check if the tmux session still exists
                    if check_session_fn(env_name, task.tmux_session):
                        print(f"Reconnected to task {task.id[:8]}: {task.tmux_session}")
                    else:
                        # Session no longer exists, mark as failed
                        print(f"Task {task.id[:8]} session not found, marking as failed")
                        task.state = TaskState.FAILED
                        task.tmux_session = None
