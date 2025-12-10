"""TUI components for the Experiment Orchestrator using Textual."""

import asyncio
import queue
from typing import Optional, List
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Button, DataTable, Static, Input,
    RadioSet, RadioButton, Label, RichLog
)
from textual.reactive import reactive
from textual import work
from textual.binding import Binding

from .models import AppState, Task, TaskState
from .api import OrchestratorAPI


class OrchestratorTUI(App):
    """Main TUI application for the orchestrator."""

    CSS_PATH = "tui.tcss"
    TITLE = "Experiment Orchestrator"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("question_mark", "help", "Help", key_display="?"),
        Binding("a", "approve", "Approve"),
        Binding("k", "kill", "Kill"),
        Binding("d", "delete", "Delete"),
        Binding("r", "rerun", "Re-run"),
        Binding("1", "env_local", "Local"),
        Binding("2", "env_ginkgo", "Ginkgo"),
        Binding("3", "env_rorqual", "Rorqual"),
        Binding("ctrl+a", "select_all", "Select All"),
        Binding("ctrl+l", "load_tasks", "Load Tasks"),
    ]

    # Reactive variables
    current_environment = reactive("local")
    task_list_version = reactive(0)
    status_message = reactive("Ready")
    selected_task_id = reactive(None, init=False)

    def __init__(self, app_state: AppState, message_queue: queue.Queue, api: OrchestratorAPI):
        """
        Initialize the TUI.

        Args:
            app_state: The application state
            message_queue: Queue for receiving updates from background threads
            api: The orchestrator API instance
        """
        super().__init__()
        self.app_state = app_state
        self.message_queue = message_queue
        self.api = api
        self.stop_flag = False

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()

        # Main content area
        with Container(id="app-grid"):
            # Left panel
            with Vertical(id="left-panel"):
                # Environment selector
                with Container(id="env-container"):
                    yield Label("Environment:", id="env-label")
                    with RadioSet(id="env-radio"):
                        yield RadioButton("Local", value=True, id="radio-local")
                        yield RadioButton("Ginkgo", id="radio-ginkgo")
                        yield RadioButton("Rorqual", id="radio-rorqual")

                # Concurrency control
                with Horizontal(id="concurrency-container"):
                    yield Label("Concurrency:", id="conc-label")
                    yield Input(value="2", placeholder="N", id="concurrency-input")
                    yield Button("Set", id="concurrency-button", variant="primary")

                # Load tasks button
                yield Button("Load Tasks", id="load-tasks-button", variant="success")

                # Task queue label
                yield Label("Task Queue:", id="task-queue-label")

                # Task list (DataTable)
                yield DataTable(id="task-table", zebra_stripes=True, cursor_type="row")

                # Action buttons
                with Container(id="action-buttons"):
                    with Horizontal(classes="button-row"):
                        yield Button("Approve", id="btn-approve", variant="success")
                        yield Button("Clear Unapproved", id="btn-clear")
                    with Horizontal(classes="button-row"):
                        yield Button("Remove Selected", id="btn-remove", variant="error")
                        yield Button("Remove All Done", id="btn-remove-done")
                    with Horizontal(classes="button-row"):
                        yield Button("Re-run", id="btn-rerun", variant="warning")

            # Right panel
            with Vertical(id="right-panel"):
                yield Label("Output:", id="output-label")
                yield RichLog(id="output-viewer", wrap=True, markup=True)

        # Footer with status
        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        # Set up the task table
        table = self.query_one("#task-table", DataTable)
        table.add_columns("State", "ID", "Command")
        table.cursor_type = "row"

        # Start message queue processing worker
        self.process_message_queue_worker()

        # Initial refresh
        self._refresh_task_list()
        self._update_concurrency_display()
        self._update_status("Ready")

    @work(exclusive=False, thread=True)
    def process_message_queue_worker(self) -> None:
        """Worker that polls the message queue and updates the UI."""
        while not self.stop_flag:
            try:
                message = self.message_queue.get(timeout=0.1)
                self.call_from_thread(self._handle_message, message)
            except queue.Empty:
                pass

    def _handle_message(self, message: dict) -> None:
        """
        Handle messages from background threads.

        Args:
            message: Message dict with 'type' and other fields
        """
        msg_type = message.get('type')

        if msg_type == 'task_started':
            self.task_list_version += 1
            task_id = message.get('task_id', '')[:8]
            self._update_status(f"Task started: {task_id}")

        elif msg_type == 'task_completed':
            self.task_list_version += 1
            task_id = message.get('task_id', '')[:8]
            state = message.get('state', '')
            state_str = state.value if hasattr(state, 'value') else str(state)
            self._update_status(f"Task {task_id} {state_str}")

        elif msg_type == 'task_failed':
            self.task_list_version += 1
            task_id = message.get('task_id', '')[:8]
            self._update_status(f"Task failed: {task_id}")

        elif msg_type == 'task_killed':
            self.task_list_version += 1
            task_id = message.get('task_id', '')[:8]
            self._update_status(f"Task killed: {task_id}")

        elif msg_type == 'output_update':
            # Refresh output if this task is currently selected
            task_id = message.get('task_id')
            if self.selected_task_id == task_id:
                self._display_task_output_by_id(task_id)

    def watch_task_list_version(self, new_version: int) -> None:
        """Called when task_list_version changes."""
        self._refresh_task_list()

    def watch_current_environment(self, new_env: str) -> None:
        """Called when current_environment changes."""
        self._refresh_task_list()
        self._update_concurrency_display()
        self._clear_output()

    def _refresh_task_list(self) -> None:
        """Refresh the task list display."""
        table = self.query_one("#task-table", DataTable)
        table.clear()

        tasks = self.api.get_tasks(self.current_environment)
        for task in tasks:
            # Get state display with color
            state_str = self._get_state_display(task.state)

            # Truncate command if too long
            cmd_preview = task.command[:60] + "..." if len(task.command) > 60 else task.command

            # Add row with task ID as key
            table.add_row(state_str, task.id[:8], cmd_preview, key=task.id)

    def _get_state_display(self, state: TaskState) -> str:
        """Get colored display string for task state."""
        color_map = {
            TaskState.RUNNING: "yellow",
            TaskState.COMPLETED: "green",
            TaskState.FAILED: "red",
            TaskState.KILLED: "bright_black",
            TaskState.PENDING: "blue",
            TaskState.UNAPPROVED: "white",
        }
        color = color_map.get(state, "white")
        return f"[{color}]{state.value}[/{color}]"

    def _update_concurrency_display(self) -> None:
        """Update concurrency display for current environment."""
        limit = self.api.get_concurrency_limit(self.current_environment)
        if limit is not None:
            input_widget = self.query_one("#concurrency-input", Input)
            input_widget.value = str(limit)

    def _update_status(self, message: str) -> None:
        """Update status bar message."""
        self.status_message = message
        # Also update the subtitle on the header
        try:
            header = self.query_one(Header)
            header.sub_title = message
        except Exception:
            pass

    def _get_selected_tasks(self) -> List[Task]:
        """Get currently selected tasks."""
        table = self.query_one("#task-table", DataTable)
        if table.cursor_row is not None:
            row_key = table.get_row_at(table.cursor_row)
            if row_key:
                task_id = row_key[0]  # Row key is the task ID
                task = self.api.get_task_status(task_id)
                return [task] if task else []
        return []

    def _display_task_output(self, task: Task) -> None:
        """Display output for a task."""
        output = self.query_one("#output-viewer", RichLog)
        output.clear()

        # Show task info
        output.write(f"[bold]Task:[/bold] {task.id[:8]}")
        output.write(f"[bold]Command:[/bold] {task.command}")
        output.write(f"[bold]State:[/bold] {self._get_state_display(task.state)}")
        if task.tmux_session:
            output.write(f"[bold]Session:[/bold] {task.tmux_session}")
        output.write("")
        output.write("=" * 80)
        output.write("")

        # Show output
        for line in task.output_buffer:
            output.write(line)

        # Track currently displayed task
        self.selected_task_id = task.id

    def _display_task_output_by_id(self, task_id: str) -> None:
        """Display output for a task by ID."""
        task = self.api.get_task_status(task_id)
        if task:
            self._display_task_output(task)

    def _clear_output(self) -> None:
        """Clear the output display."""
        output = self.query_one("#output-viewer", RichLog)
        output.clear()
        self.selected_task_id = None

    # Event handlers for widgets

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle environment radio button changes."""
        if event.radio_set.id == "env-radio":
            if event.pressed.id == "radio-local":
                self.current_environment = "local"
            elif event.pressed.id == "radio-ginkgo":
                self.current_environment = "ginkgo"
            elif event.pressed.id == "radio-rorqual":
                self.current_environment = "rorqual"

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle task selection in the table."""
        if event.row_key:
            task_id = event.row_key.value
            self._display_task_output_by_id(task_id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "concurrency-button":
            self._on_set_concurrency()
        elif button_id == "load-tasks-button":
            self._on_load_tasks()
        elif button_id == "btn-approve":
            self.action_approve()
        elif button_id == "btn-clear":
            self._on_clear_unapproved()
        elif button_id == "btn-remove":
            self.action_delete()
        elif button_id == "btn-remove-done":
            self._on_remove_all_done()
        elif button_id == "btn-rerun":
            self.action_rerun()

    def _on_set_concurrency(self) -> None:
        """Handle concurrency set button."""
        input_widget = self.query_one("#concurrency-input", Input)
        try:
            limit = int(input_widget.value)
            if self.api.set_concurrency(self.current_environment, limit):
                self._update_status(f"Concurrency set to {limit}")
            else:
                self._update_status("Error: Invalid concurrency value")
        except ValueError:
            self._update_status("Error: Invalid concurrency value")

    def _on_load_tasks(self) -> None:
        """Handle load tasks button."""
        # For now, prompt user to enter filename
        # In a more advanced implementation, we could use a file picker modal
        self._update_status("Load tasks: Use CLI mode for now (python -m orchestrator.cli load <file> --env <env>)")

    def _on_clear_unapproved(self) -> None:
        """Handle clear unapproved button."""
        count = self.api.clear_unapproved(self.current_environment)
        self.task_list_version += 1
        self._update_status(f"Cleared {count} unapproved task(s)")

    def _on_remove_all_done(self) -> None:
        """Handle remove all done button."""
        count = self.api.remove_all_done(self.current_environment)
        self.task_list_version += 1
        self._update_status(f"Removed {count} completed task(s)")

    # Actions (keyboard shortcuts)

    def action_approve(self) -> None:
        """Approve selected tasks."""
        selected = self._get_selected_tasks()
        if selected:
            count = self.api.approve_task_objects(selected)
            self.task_list_version += 1
            self._update_status(f"Approved {count} task(s)")
        else:
            self._update_status("No task selected")

    def action_kill(self) -> None:
        """Kill selected tasks."""
        selected = self._get_selected_tasks()
        if selected:
            count = 0
            for task in selected:
                if self.api.kill_task(task.id):
                    count += 1
            self._update_status(f"Killed {count} task(s)")
        else:
            self._update_status("No task selected")

    def action_delete(self) -> None:
        """Delete selected tasks."""
        selected = self._get_selected_tasks()
        if selected:
            count = self.api.remove_task_objects(selected)
            self.task_list_version += 1
            self._clear_output()
            self._update_status(f"Removed {count} task(s)")
        else:
            self._update_status("No task selected")

    def action_rerun(self) -> None:
        """Re-run selected tasks."""
        selected = self._get_selected_tasks()
        if selected:
            count = self.api.rerun_task_objects(selected)
            self.task_list_version += 1
            self._update_status(f"Reset {count} task(s) to pending")
        else:
            self._update_status("No task selected")

    def action_env_local(self) -> None:
        """Switch to local environment."""
        radio_set = self.query_one("#env-radio", RadioSet)
        radio_set.pressed_button = self.query_one("#radio-local", RadioButton)

    def action_env_ginkgo(self) -> None:
        """Switch to ginkgo environment."""
        radio_set = self.query_one("#env-radio", RadioSet)
        radio_set.pressed_button = self.query_one("#radio-ginkgo", RadioButton)

    def action_env_rorqual(self) -> None:
        """Switch to rorqual environment."""
        radio_set = self.query_one("#env-radio", RadioSet)
        radio_set.pressed_button = self.query_one("#radio-rorqual", RadioButton)

    def action_select_all(self) -> None:
        """Select all tasks."""
        # DataTable doesn't support multi-select out of the box
        # This would require custom implementation
        self._update_status("Select all: Not yet implemented")

    def action_load_tasks(self) -> None:
        """Load tasks (placeholder)."""
        self._on_load_tasks()

    def action_help(self) -> None:
        """Show help overlay."""
        help_text = """
[bold]Keyboard Shortcuts:[/bold]

[yellow]Navigation:[/yellow]
  ↑/↓       Navigate task list
  Enter     Show task output

[yellow]Environment:[/yellow]
  1         Switch to Local
  2         Switch to Ginkgo
  3         Switch to Rorqual

[yellow]Task Actions:[/yellow]
  a         Approve selected task(s)
  k         Kill selected task(s)
  d         Delete selected task(s)
  r         Re-run selected task(s)

[yellow]Other:[/yellow]
  Ctrl+L    Load tasks
  Ctrl+A    Select all
  ?         Show this help
  q         Quit application
"""
        output = self.query_one("#output-viewer", RichLog)
        output.clear()
        output.write(help_text)

    def on_unmount(self) -> None:
        """Called when app is unmounted."""
        self.stop_flag = True
