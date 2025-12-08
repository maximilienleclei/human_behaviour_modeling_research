# Experiment Orchestrator

A Tkinter-based GUI for queuing, executing, and monitoring computational tasks across multiple environments.

## Features

- Execute tasks on local machine, remote SSH servers, and SLURM HPC clusters
- Per-environment task queues with independent concurrency limits
- tmux-based execution for resilience (tasks survive GUI crashes)
- Dark-themed GUI with real-time output monitoring
- State persistence across restarts

## Installation

1. Install dependencies:
```bash
pip install -r orchestrator/requirements.txt
```

2. Ensure tmux is installed on all target systems:
```bash
# On Ubuntu/Debian
sudo apt-get install tmux

# On macOS
brew install tmux
```

## Usage

Run the orchestrator:
```bash
python -m orchestrator.main
```

Or use the convenience script:
```bash
python run_orchestrator.py
```

## Workflow

1. Select an environment (local, ginkgo, or rorqual)
2. Load tasks from a text file (one command per line)
3. Approve tasks to change them from Unapproved to Pending
4. Tasks will automatically start when slots are available
5. Select a single task to view its output
6. Use action buttons to manage tasks:
   - Approve: Move unapproved tasks to pending
   - Clear Unapproved: Remove all unapproved tasks
   - Remove Selected: Kill and remove selected tasks
   - Remove All Done: Clean up completed/failed tasks
   - Re-run: Reset completed/failed tasks to pending

## Keyboard Shortcuts

- Ctrl+A: Select all tasks
- Ctrl+Click: Add/remove task from selection
- Shift+Up/Down: Extend selection

## Configuration

Environment settings are defined in `orchestrator/config.py`. Each environment has:
- Concurrency limit
- Activation command (virtualenv/module load)
- Remote Dropbox path
- SSH configuration (for remote environments)
- SLURM configuration (for HPC environments)

## State Persistence

Application state is saved to `orchestrator/state.json` on exit and restored on startup. This includes:
- All tasks and their states
- Environment settings
- Running tmux sessions (automatically reconnected)
