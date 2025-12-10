"""CLI interface for the Experiment Orchestrator.

This module provides command-line commands for AI agents and scripts
to interact with the orchestrator programmatically.
"""

import json
import sys
import time
from pathlib import Path

import click

from .models import AppState, TaskState
from .config import get_default_environments
from .persistence import StateManager
from .api import OrchestratorAPI


def load_state() -> tuple[AppState, StateManager]:
    """Load the application state."""
    state_manager = StateManager()
    app_state = state_manager.load()

    if app_state is None:
        # Create new state with default environments
        app_state = AppState()
        app_state.environments = get_default_environments()
    else:
        # Merge with default environments
        default_envs = get_default_environments()
        for name, env in default_envs.items():
            if name not in app_state.environments:
                app_state.environments[name] = env

    return app_state, state_manager


def save_state(state_manager: StateManager, app_state: AppState) -> None:
    """Save the application state."""
    state_manager.save(app_state)


@click.group()
def cli():
    """Experiment Orchestrator CLI - Control the orchestrator programmatically."""
    pass


@cli.command()
@click.option('--env', default='local', help='Environment to list tasks from')
@click.option('--format', 'output_format', type=click.Choice(['json', 'table']), default='table',
              help='Output format')
@click.option('--state', 'filter_state', help='Filter by state (UNAPPROVED, PENDING, RUNNING, etc.)')
def list(env: str, output_format: str, filter_state: str):
    """List all tasks in an environment."""
    app_state, _ = load_state()

    # Validate environment
    if env not in app_state.environments:
        click.echo(f"Error: Unknown environment '{env}'", err=True)
        sys.exit(1)

    tasks = app_state.get_tasks(env)

    # Filter by state if specified
    if filter_state:
        try:
            state_enum = TaskState(filter_state)
            tasks = [t for t in tasks if t.state == state_enum]
        except ValueError:
            click.echo(f"Error: Invalid state '{filter_state}'", err=True)
            sys.exit(1)

    if output_format == 'json':
        # JSON output
        tasks_data = {
            "environment": env,
            "tasks": [
                {
                    "id": t.id,
                    "command": t.command,
                    "state": t.state.value,
                    "environment": t.environment,
                    "start_time": t.start_time,
                    "end_time": t.end_time,
                    "tmux_session": t.tmux_session,
                    "exit_code": t.exit_code,
                    "output_lines": len(t.output_buffer)
                }
                for t in tasks
            ]
        }
        click.echo(json.dumps(tasks_data, indent=2))
    else:
        # Table output
        if not tasks:
            click.echo("No tasks found.")
            return

        click.echo(f"\nTasks in {env} environment:\n")
        click.echo(f"{'State':<12} {'ID':<10} {'Command':<60}")
        click.echo("-" * 82)

        for task in tasks:
            cmd_preview = task.command[:57] + "..." if len(task.command) > 60 else task.command
            click.echo(f"{task.state.value:<12} {task.id[:8]:<10} {cmd_preview:<60}")

        click.echo(f"\nTotal: {len(tasks)} task(s)")


@cli.command()
@click.argument('filename', type=click.Path(exists=True))
@click.option('--env', default='local', help='Target environment')
def load(filename: str, env: str):
    """Load tasks from a file (one command per line)."""
    app_state, state_manager = load_state()

    # Validate environment
    if env not in app_state.environments:
        click.echo(f"Error: Unknown environment '{env}'", err=True)
        sys.exit(1)

    # Create API (no scheduler needed for this operation)
    api = OrchestratorAPI(app_state, None)

    try:
        count = api.load_tasks(env, filename)
        save_state(state_manager, app_state)
        click.echo(f"Loaded {count} task(s) into {env} environment")
    except Exception as e:
        click.echo(f"Error loading tasks: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('task_ids', nargs=-1, required=True)
def approve(task_ids: tuple):
    """Approve tasks by ID."""
    app_state, state_manager = load_state()
    api = OrchestratorAPI(app_state, None)

    count = api.approve_tasks(list(task_ids))
    save_state(state_manager, app_state)

    click.echo(f"Approved {count} task(s)")


@cli.command()
@click.argument('task_ids', nargs=-1, required=True)
def kill(task_ids: tuple):
    """Kill running tasks by ID."""
    click.echo("Error: Kill operation requires the scheduler to be running.", err=True)
    click.echo("Use the TUI mode to kill tasks.", err=True)
    sys.exit(1)


@cli.command()
@click.argument('task_ids', nargs=-1, required=True)
def remove(task_ids: tuple):
    """Remove tasks by ID."""
    app_state, state_manager = load_state()
    api = OrchestratorAPI(app_state, None)

    count = api.remove_tasks(list(task_ids))
    save_state(state_manager, app_state)

    click.echo(f"Removed {count} task(s)")


@cli.command()
@click.argument('task_id')
@click.option('--format', 'output_format', type=click.Choice(['json', 'text']), default='text',
              help='Output format')
def status(task_id: str, output_format: str):
    """Get detailed status of a task."""
    app_state, _ = load_state()
    api = OrchestratorAPI(app_state, None)

    task = api.get_task_status(task_id)

    if not task:
        click.echo(f"Error: Task '{task_id}' not found", err=True)
        sys.exit(1)

    if output_format == 'json':
        task_data = {
            "id": task.id,
            "command": task.command,
            "state": task.state.value,
            "environment": task.environment,
            "start_time": task.start_time,
            "end_time": task.end_time,
            "tmux_session": task.tmux_session,
            "exit_code": task.exit_code,
            "output": task.output_buffer
        }
        click.echo(json.dumps(task_data, indent=2))
    else:
        click.echo(f"\nTask: {task.id}")
        click.echo(f"Command: {task.command}")
        click.echo(f"State: {task.state.value}")
        click.echo(f"Environment: {task.environment}")
        click.echo(f"Start Time: {task.start_time or 'N/A'}")
        click.echo(f"End Time: {task.end_time or 'N/A'}")
        click.echo(f"Exit Code: {task.exit_code if task.exit_code is not None else 'N/A'}")

        if task.tmux_session:
            click.echo(f"Tmux Session: {task.tmux_session}")

        if task.output_buffer:
            click.echo(f"\nOutput ({len(task.output_buffer)} lines):")
            click.echo("-" * 80)
            for line in task.output_buffer[-50:]:  # Show last 50 lines
                click.echo(line)


@cli.command()
@click.argument('task_id')
@click.option('--interval', default=2, help='Polling interval in seconds')
def watch(task_id: str, interval: int):
    """Watch task output in real-time (like tail -f)."""
    app_state, _ = load_state()
    api = OrchestratorAPI(app_state, None)

    task = api.get_task_status(task_id)

    if not task:
        click.echo(f"Error: Task '{task_id}' not found", err=True)
        sys.exit(1)

    click.echo(f"Watching task {task_id[:8]} (State: {task.state.value})")
    click.echo("Press Ctrl+C to stop\n")

    last_line_count = 0

    try:
        while True:
            # Reload state to get latest output
            app_state, _ = load_state()
            task = api.get_task_status(task_id)

            if not task:
                click.echo("\nTask was removed.")
                break

            # Show new output lines
            if len(task.output_buffer) > last_line_count:
                new_lines = task.output_buffer[last_line_count:]
                for line in new_lines:
                    click.echo(line)
                last_line_count = len(task.output_buffer)

            # Check if task completed
            if task.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.KILLED]:
                click.echo(f"\nTask finished with state: {task.state.value}")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        click.echo("\n\nWatch stopped.")


@cli.command()
@click.option('--env', required=True, help='Target environment')
@click.option('--limit', required=True, type=int, help='Concurrency limit')
def set_concurrency(env: str, limit: int):
    """Set concurrency limit for an environment."""
    app_state, state_manager = load_state()
    api = OrchestratorAPI(app_state, None)

    if api.set_concurrency(env, limit):
        save_state(state_manager, app_state)
        click.echo(f"Set concurrency limit for {env} to {limit}")
    else:
        click.echo(f"Error: Could not set concurrency limit", err=True)
        sys.exit(1)


@cli.command()
@click.option('--env', default='local', help='Environment to clear')
def clear_unapproved(env: str):
    """Clear all unapproved tasks in an environment."""
    app_state, state_manager = load_state()
    api = OrchestratorAPI(app_state, None)

    count = api.clear_unapproved(env)
    save_state(state_manager, app_state)

    click.echo(f"Cleared {count} unapproved task(s) from {env}")


@cli.command()
@click.option('--env', default='local', help='Environment to clean')
def remove_done(env: str):
    """Remove all completed/failed/killed tasks."""
    app_state, state_manager = load_state()
    api = OrchestratorAPI(app_state, None)

    count = api.remove_all_done(env)
    save_state(state_manager, app_state)

    click.echo(f"Removed {count} completed task(s) from {env}")


@cli.command()
@click.argument('task_ids', nargs=-1, required=True)
def rerun(task_ids: tuple):
    """Reset tasks to pending state for re-running."""
    app_state, state_manager = load_state()
    api = OrchestratorAPI(app_state, None)

    count = api.rerun_tasks(list(task_ids))
    save_state(state_manager, app_state)

    click.echo(f"Reset {count} task(s) to pending")


if __name__ == '__main__':
    cli()
