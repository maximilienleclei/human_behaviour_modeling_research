

## 1. Overview & Architecture

### Purpose

The Experiment Orchestrator is a Tkinter-based dark-theme GUI that enables me to queue, execute, and monitor computational tasks across three execution environments:
- **local**: My local machine
- **ginkgo**: My lab's remote machine via SSH
- **rorqual**: A HPC with SLURM job scheduling I have access to

### Core Architecture Principles

**Per-Environment Task Queues**: Each environment maintains its own independent task queue and concurrency limit. Tasks never move between environments.

**tmux as Resilience Layer**: All task execution happens inside tmux sessions. This ensures:
- Tasks survive GUI crashes or closures
- Output remains available for reconnection
- Sessions can be inspected independently

**Thread-Based Concurrency**: Three types of daemon threads:
1. **Scheduler thread**: Monitors all environments, starts pending tasks when slots available
2. **Monitor threads**: One per running task, polls output and detects completion
3. **GUI update loop**: Processes thread-safe queue messages to update the interface

**State Persistence**: All tasks and settings save to JSON on exit. On restart, the GUI reconnects to running tasks and restores state.

## 2. User Interface Layout

### Two-Panel Layout

**Left Panel (50% width)**: Control center
- Environment selector (3 preset buttons)
- Concurrency control (text field + Set button)
- Load Tasks button (opens file browser)
- Task Queue (with scrollbar)
- Action buttons (Approve, Clear Unapproved, Remove Selected, etc.)
- Status bar (shows current operation status)

**Right Panel (50% width)**: Output viewer
- Scrollable output text area (last 500 lines)

### Task Queue

**Multi-Selection Keyboard Shortcuts**:
Ctrl+A: Select all tasks
Ctrl+Click: Add/remove single task
Shift+Up: Extend selection upward
Shift+Down: Extend selection downward

### Action Buttons

Located below Task Queue:
- **Approve**: Change selected Unapproved tasks to Pending
- **Clear Unapproved**: Remove all Unapproved tasks from current environment
- **Remove Selected**: Kill (if running) and remove selected tasks
- **Remove All Done**: Remove all Completed/Failed/Killed tasks
- **Re-run**: Reset selected Completed/Failed/Killed tasks to Pending

## 3. Environment specifics

**local**
activation_cmd: "source /home/maximilienleclei/venvs/hbmr/bin/activate"  # No activation needed
remote_dropbox: "/home/maximilienleclei/Dropbox"

**ginkgo**
ssh_host: "ginkgo.criugm.qc.ca"
ssh_port: 22
ssh_user: "mleclei"
ssh_proxy_jump: "mleclei@elm.criugm.qc.ca"  # ProxyJump through elm
activation_cmd: "source /scratch/mleclei/venv/bin/activate"
remote_dropbox: "/scratch/mleclei/Dropbox"

**rorqual**:
ssh_host: "rorqual1.alliancecan.ca"
ssh_port: 22
ssh_user: "mleclei"
activation_cmd: "module load gcc arrow python/3.12 && source ~/venv/bin/activate"
remote_dropbox: "/scratch/mleclei/Dropbox"
slurm_partition: ""  # Use default
slurm_time_limit: "3:00:00"  # 3 hours
slurm_extra_flags: "--gpus=h100:1 --account=rrg-pbellec --mem=124G --cpus-per-task=16"

