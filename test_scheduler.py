#!/usr/bin/env python3
"""Test script to verify orchestrator scheduler can start and run tasks."""

import sys
sys.path.insert(0, '/home/maximilienleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research')

import time
import queue
from orchestrator.models import AppState, TaskState
from orchestrator.persistence import StateManager
from orchestrator.scheduler import TaskScheduler
from orchestrator.config import get_default_environments

# Load state
state_manager = StateManager()
app_state = state_manager.load()

if app_state is None:
    print("No state found!")
    sys.exit(1)

# Merge with default environments
default_envs = get_default_environments()
for name, env in default_envs.items():
    if name not in app_state.environments:
        app_state.environments[name] = env

# Create message queue
message_queue = queue.Queue()

# Create and start scheduler
print("Starting scheduler...")
scheduler = TaskScheduler(app_state, message_queue)
scheduler.start()

# Reconnect to running tasks
scheduler.reconnect_running_tasks()

# Get pending tasks
tasks = app_state.get_tasks("local")
pending = [t for t in tasks if t.state == TaskState.PENDING]
print(f"\nFound {len(pending)} pending task(s):")
for task in pending:
    print(f"  - {task.id[:8]}: {task.command[:60]}...")

# Wait a bit for scheduler to start tasks
print("\nWaiting 10 seconds for scheduler to start tasks...")
for i in range(10):
    time.sleep(1)
    running = [t for t in tasks if t.state == TaskState.RUNNING]
    completed = [t for t in tasks if t.state == TaskState.COMPLETED]
    failed = [t for t in tasks if t.state == TaskState.FAILED]
    print(f"  {i+1}s: Running={len(running)}, Completed={len(completed)}, Failed={len(failed)}, Pending={len([t for t in tasks if t.state == TaskState.PENDING])}")

# Final status
print(f"\nFinal status:")
running = [t for t in tasks if t.state == TaskState.RUNNING]
completed = [t for t in tasks if t.state == TaskState.COMPLETED]
failed = [t for t in tasks if t.state == TaskState.FAILED]

if running:
    print(f"\nRunning tasks ({len(running)}):")
    for task in running:
        print(f"  - {task.id[:8]}: {task.command[:60]}...")
        if task.tmux_session:
            print(f"    tmux session: {task.tmux_session}")

if completed:
    print(f"\nCompleted tasks ({len(completed)}):")
    for task in completed:
        print(f"  - {task.id[:8]}: {task.command[:60]}...")

if failed:
    print(f"\nFailed tasks ({len(failed)}):")
    for task in failed:
        print(f"  - {task.id[:8]}: {task.command[:60]}...")
        if task.output_buffer:
            print(f"    Last output: {task.output_buffer[-1][:100]}...")

# Save state
state_manager.save(app_state)

# Stop scheduler
print("\nStopping scheduler...")
scheduler.stop()

print("\n✓ Orchestrator scheduler test complete!")
if running or completed or failed:
    print("  Tasks were successfully started by the scheduler!")
print("\nTo monitor tasks in the TUI, run:")
print("  source $HOME/venvs/orchestrator/bin/activate && python -m orchestrator.main")
