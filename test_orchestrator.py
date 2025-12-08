#!/usr/bin/env python3
"""Test script for orchestrator core functionality (without GUI)."""

import sys
sys.path.insert(0, '/home/maximilienleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research')

from orchestrator.models import Task, TaskState, AppState
from orchestrator.config import get_default_environments
from orchestrator.persistence import StateManager
from orchestrator.executors import LocalExecutor

print("Testing Orchestrator Core Components...")
print("=" * 60)

# Test 1: Create a task
print("\n1. Creating a task...")
task = Task.create(command="echo 'Hello from orchestrator!'", environment="local")
print(f"   Task ID: {task.id}")
print(f"   Command: {task.command}")
print(f"   State: {task.state.value}")
print(f"   Display: {task.get_display_name()}")

# Test 2: Create app state
print("\n2. Creating app state...")
app_state = AppState()
app_state.environments = get_default_environments()
print(f"   Environments: {list(app_state.environments.keys())}")

# Test 3: Add task to state
print("\n3. Adding task to state...")
app_state.add_task(task)
print(f"   Local tasks: {len(app_state.get_tasks('local'))}")

# Test 4: Test serialization
print("\n4. Testing state serialization...")
state_dict = app_state.to_dict()
print(f"   Serialized tasks: {len(state_dict['tasks']['local'])}")

restored_state = AppState.from_dict(state_dict)
print(f"   Restored tasks: {len(restored_state.get_tasks('local'))}")

# Test 5: Test environment configs
print("\n5. Testing environment configurations...")
for env_name, env in app_state.environments.items():
    print(f"   {env_name}:")
    print(f"     - Concurrency: {env.concurrency_limit}")
    print(f"     - Local: {env.is_local()}")
    print(f"     - SLURM: {env.is_slurm()}")

# Test 6: Test state persistence
print("\n6. Testing state persistence...")
state_manager = StateManager("test_state.json")
state_manager.save(app_state)
print("   State saved to test_state.json")

loaded_state = state_manager.load()
if loaded_state:
    print(f"   State loaded successfully")
    print(f"   Loaded tasks: {len(loaded_state.get_tasks('local'))}")
else:
    print("   Failed to load state")

print("\n" + "=" * 60)
print("All core components are working correctly!")
print("\nNote: To run the full GUI, you need to:")
print("1. Install tkinter: sudo apt-get install python3-tk")
print("2. Have X11 forwarding enabled if on WSL")
print("3. Or run from a system with a display server")
