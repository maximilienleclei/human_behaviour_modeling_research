#!/usr/bin/env python3
"""Quick script to approve all unapproved tasks."""

from orchestrator.models import AppState, TaskState
from orchestrator.persistence import StateManager
from orchestrator.config import get_default_environments

# Load state
state_manager = StateManager()
app_state = state_manager.load()

if app_state is None:
    print("No state found!")
    exit(1)

# Get unapproved tasks
tasks = app_state.get_tasks("local")
unapproved = [t for t in tasks if t.state == TaskState.UNAPPROVED]

print(f"Found {len(unapproved)} unapproved task(s):")
for task in unapproved:
    print(f"  - {task.id[:8]}: {task.command[:60]}...")

# Approve them
for task in unapproved:
    task.state = TaskState.PENDING
    print(f"Approved: {task.id[:8]}")

# Save state
state_manager.save(app_state)
print(f"\nApproved {len(unapproved)} task(s). They will start when you launch the orchestrator.")
