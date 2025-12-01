#- [Experiment Orchestrator Specification](#experiment-orchestrator-specification)
- [1. Overview \& Architecture](#1-overview--architecture)
  - [Purpose](#purpose)
  - [Core Architecture Principles](#core-architecture-principles)
  - [Key Data Structures](#key-data-structures)
  - [Dark Theme Color Scheme](#dark-theme-color-scheme)
- [2. User Interface Layout](#2-user-interface-layout)
  - [Window Structure](#window-structure)
  - [Two-Panel Layout](#two-panel-layout)
  - [Task Queue Treeview](#task-queue-treeview)
  - [Action Buttons](#action-buttons)
  - [Output Viewer Controls](#output-viewer-controls)
- [3. Environment Management](#3-environment-management)
  - [Backend Types](#backend-types)
  - [Three Hard-Coded Presets](#three-hard-coded-presets)
  - [SSH Key Auto-Detection](#ssh-key-auto-detection)
  - [Dropbox Path Mapping](#dropbox-path-mapping)
- [4. Task Lifecycle \& State Machine](#4-task-lifecycle--state-machine)
  - [Task States](#task-states)
  - [State Transitions](#state-transitions)
  - [Task Data Structure](#task-data-structure)
  - [Task Counter](#task-counter)
  - [Duplicate Detection](#duplicate-detection)
- [5. Execution Backend Architecture](#5-execution-backend-architecture)
  - [Abstract Interface](#abstract-interface)
  - [LocalBackend](#localbackend)
  - [SSHBackend](#sshbackend)
  - [SLURMBackend](#slurmbackend)
- [6. tmux Session Management](#6-tmux-session-management)
  - [Global Configuration](#global-configuration)
  - [LocalBackend tmux Sequence](#localbackend-tmux-sequence)
  - [SSHBackend tmux Sequence](#sshbackend-tmux-sequence)
  - [SLURMBackend tmux Sequence](#slurmbackend-tmux-sequence)
- [7. Threading \& Concurrency Model](#7-threading--concurrency-model)
  - [Three Thread Types](#three-thread-types)
  - [Scheduler Thread Logic](#scheduler-thread-logic)
  - [Monitor Thread Logic](#monitor-thread-logic)
  - [GUI Update Loop](#gui-update-loop)
  - [Shutdown Handling](#shutdown-handling)
- [8. Output Polling \& Display](#8-output-polling--display)
  - [Full Buffer Strategy](#full-buffer-strategy)
  - [Backend-Specific Polling](#backend-specific-polling)
  - [Change Detection](#change-detection)
  - [Display Rendering](#display-rendering)
  - [Exit Code Detection](#exit-code-detection)
- [9. State Persistence \& Recovery](#9-state-persistence--recovery)
  - [State File](#state-file)
  - [Save Trigger](#save-trigger)
  - [Load Process](#load-process)
  - [Reconnection Logic](#reconnection-logic)
- [10. Task Loading \& File Management](#10-task-loading--file-management)
  - [Task File Format](#task-file-format)
  - [File Selection Dialog](#file-selection-dialog)
  - [Working Directory Computation](#working-directory-computation)
  - [Task Selection Dialog](#task-selection-dialog)
- [11. Task Management Operations](#11-task-management-operations)
  - [Approve](#approve)
  - [Remove Selected](#remove-selected)
  - [Remove All Done](#remove-all-done)
  - [Clear Unapproved](#clear-unapproved)
  - [Re-run](#re-run)
  - [Tree Update](#tree-update)
- [12. Platform-Specific Details (Windows/itmux)](#12-platform-specific-details-windowsitmux)
  - [itmux Binary Paths](#itmux-binary-paths)
  - [Command Substitution](#command-substitution)
  - [Path Handling](#path-handling)
  - [Monitor Positioning](#monitor-positioning)
  - [SSH Key Detection](#ssh-key-detection)
  - [Bash Script Line Endings](#bash-script-line-endings)
- [13. Error Handling \& Recovery](#13-error-handling--recovery)
  - [Backend Start Failures](#backend-start-failures)
  - [Monitor Thread Exceptions](#monitor-thread-exceptions)
  - [Session Loss Detection](#session-loss-detection)
  - [Reconnection Failures](#reconnection-failures)
  - [Shutdown Handling](#shutdown-handling-1)
  - [GUI Exception Handling](#gui-exception-handling)
- [Summary](#summary)

This document specifies a GUI application for managing parallel experiment runs across multiple execution environments.

---

## 1. Overview & Architecture

### Purpose
The Experiment Orchestrator is a Tkinter-based dark-theme GUI that enables users to queue, execute, and monitor computational tasks across three execution environments:
- **local**: Direct execution on the local machine
- **ginkgo**: Execution on remote machines via SSH (e.g., lab workstations)
- **SLURM Cluster**: Execution on HPC clusters with SLURM job scheduling

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

### Key Data Structures

```python
# Main task storage (per-environment queues)
tasks: dict[str, list[Task]] = {
    "local": [],
    "ginkgo": [],
    "rorqual": []
}

# Per-environment concurrency limits
max_concurrent: dict[str, int] = {
    "local": 1,
    "ginkgo": 1,
    "rorqual": 1
}

# Thread-safe communication between worker threads and GUI
update_queue: queue.Queue = queue.Queue()
```

### Dark Theme Color Scheme

```python
bg_color = "#1e1e1e"        # Main background
fg_color = "#d4d4d4"        # Main text
accent_color = "#3c3c3c"    # Frame backgrounds
entry_bg = "#2d2d2d"        # Input fields
button_bg = "#404040"       # Buttons
select_bg = "#264f78"       # Selection highlight
```

---

## 2. User Interface Layout

### Window Structure
- Title: "Experiment Orchestrator"
- Size: 1400x800 (positioned on monitor where cursor is located)
- Theme: Dark (using ttk.Style with "clam" base theme)

### Two-Panel Layout

**Left Panel (60% width)**: Control center
- Environment selector (3 preset buttons)
- Concurrency control (text field + Set button)
- Load Tasks button (opens file browser)
- Task Queue (treeview with scrollbar)
- Action buttons (Approve, Clear Unapproved, Remove Selected, etc.)
- Status bar (shows current operation status)

**Right Panel (40% width)**: Output viewer
- Poll interval control
- Scrollable output text area (last 500 lines)
- Selected task indicator

### Task Queue Treeview

**Columns**:
- ID: 40px, center-aligned (integer task ID)
- Status: 80px, center-aligned (Unapproved, Pending, Running, Completed, Failed, Killed)
- Command: 400px, left-aligned (full command string)

**Selection Mode**: Extended (multiple selection supported)

**Multi-Selection Keyboard Shortcuts**:
```python
Ctrl+A: Select all tasks
    → tree.selection_set(tree.get_children())

Ctrl+Click: Add/remove single task (default ttk behavior)

Shift+Up: Extend selection upward
    → Find topmost selected item index
    → Add item at (index - 1) to selection
    → Move focus to new item

Shift+Down: Extend selection downward
    → Find bottommost selected item index
    → Add item at (index + 1) to selection
    → Move focus to new item
```

**Status Color Coding**:
```python
# Using tree.tag_configure() for each status
"unapproved": background="#2d2d3d", foreground="#9999cc"  # Purple tint
"pending":    background="#2d2d2d", foreground="#888888"  # Gray
"running":    background="#3d3d00", foreground="#ffd700"  # Gold/yellow
"completed":  background="#1e3d1e", foreground="#90ee90"  # Light green
"failed":     background="#3d1e1e", foreground="#ff6b6b"  # Light red
"killed":     background="#3d2d1e", foreground="#ffa07a"  # Light salmon
```

### Action Buttons

Located below Task Queue:
- **Approve**: Change selected Unapproved tasks to Pending
- **Clear Unapproved**: Remove all Unapproved tasks from current environment
- **Remove Selected**: Kill (if running) and remove selected tasks
- **Remove All Done**: Remove all Completed/Failed/Killed tasks
- **Re-run**: Reset selected Completed/Failed/Killed tasks to Pending

### Output Viewer Controls

- **Poll Interval (s)**: Text field (default: 5) + Set button
- **Clear Output**: Button to clear the output text area
- **Output Text**: ScrolledText widget (disabled for editing, dark theme)
- **Selected Task Label**: Shows "Selected: Task #X" or "Selected: N tasks"

---

## 3. Environment Management

### Backend Types

```python
class BackendType(Enum):
    LOCAL = "Local"
    SSH = "SSH Remote"
    SLURM = "SLURM Cluster"
```

### Three Hard-Coded Presets

**Local Preset**:
```python
backend_type: BackendType.LOCAL
activation_cmd: ""  # No activation needed
remote_dropbox: ""  # Not applicable
```

**Ginkgo Preset** (Lab machine via SSH):
```python
backend_type: BackendType.SSH
ssh_host: "ginkgo.criugm.qc.ca"
ssh_port: 22
ssh_user: "mleclei"
ssh_proxy_jump: "mleclei@elm.criugm.qc.ca"  # ProxyJump through elm
ssh_key_file: <auto-detected from ~/.ssh/>
activation_cmd: "source /scratch/mleclei/venv/bin/activate"
remote_dropbox: "/scratch/mleclei/Dropbox"
```

**Rorqual Preset** (SLURM cluster):
```python
backend_type: BackendType.SLURM
ssh_host: "rorqual1.alliancecan.ca"
ssh_port: 22
ssh_user: "mleclei"
ssh_proxy_jump: ""  # Direct connection
ssh_key_file: <auto-detected from ~/.ssh/>
activation_cmd: "module load gcc arrow python/3.12 && source ~/venv/bin/activate"
remote_dropbox: "/scratch/mleclei/Dropbox"

# SLURM-specific settings
slurm_partition: ""  # Use default
slurm_time_limit: "3:00:00"  # 3 hours
slurm_extra_flags: "--gpus=h100:1 --account=rrg-pbellec --mem=124G --cpus-per-task=16"
```

### SSH Key Auto-Detection

Search `~/.ssh/` in priority order:
1. `id_ed25519`
2. `id_rsa`
3. `id_ecdsa`
4. `id_dsa`

Return first existing file, or empty string if none found.

### Dropbox Path Mapping

When loading tasks for SSH/SLURM environments, automatically compute remote working directory:

```python
# If task file is inside local Dropbox folder
local_dropbox = Path.home() / "Dropbox"
task_file_dir = Path(task_file_path).parent.resolve()

try:
    relative_path = task_file_dir.relative_to(local_dropbox)
    remote_workdir = f"{remote_dropbox}/{relative_path.as_posix()}"
except ValueError:
    # Task file not in Dropbox, must specify manually
    pass
```

Example:
- Local: `C:\Users\Max\Dropbox\repos\project\experiment_1\`
- Remote: `/scratch/mleclei/Dropbox/repos/project/experiment_1/`

---

## 4. Task Lifecycle & State Machine

### Task States

```python
class TaskStatus(Enum):
    UNAPPROVED = "Unapproved"  # Loaded from file, not yet approved
    PENDING = "Pending"        # Approved, waiting for execution slot
    RUNNING = "Running"        # Currently executing
    COMPLETED = "Completed"    # Finished with exit code 0
    FAILED = "Failed"          # Finished with non-zero exit code or error
    KILLED = "Killed"          # Manually terminated by user
```

### State Transitions

```
UNAPPROVED ──[User clicks "Approve"]──→ PENDING

PENDING ──[Scheduler finds available slot]──→ RUNNING

RUNNING ──[Exit code 0 detected]──→ COMPLETED
        ──[Exit code ≠0 detected]──→ FAILED
        ──[Exception in backend]───→ FAILED
        ──[User kills task]────────→ KILLED

COMPLETED ──[User clicks "Re-run"]──→ PENDING
FAILED    ──[User clicks "Re-run"]──→ PENDING
KILLED    ──[User clicks "Re-run"]──→ PENDING
```

### Task Data Structure

```python
@dataclass
class Task:
    id: int                                  # Unique across ALL environments
    command: str                             # Shell command to execute
    status: TaskStatus                       # Current state
    backend_type: BackendType                # LOCAL, SSH, or SLURM
    activation_cmd: str = ""                 # Environment setup (venv, modules)
    ssh_config: Optional[SSHConfig] = None   # For SSH/SLURM backends
    slurm_config: Optional[SLURMConfig] = None  # For SLURM backend
    remote_workdir: str = ""                 # Working directory (local or remote)
    backend: Optional[ExecutionBackend] = None  # Runtime backend instance
    output_buffer: list[str] = field(default_factory=list)  # Captured output
    last_poll_time: float = 0.0              # Timestamp of last poll
    tmux_session: Optional[str] = None       # Session name (for reconnection)
    log_file_path: Optional[str] = None      # Log file path (local tasks only)
```

### Task Counter

Global counter increments for each new task (across all environments). This ensures unique IDs even when tasks are distributed across environments.

```python
task_counter: int = 0  # Initialized from state file or 0

# When adding new task:
task_counter += 1
task = Task(id=task_counter, ...)
```

### Duplicate Detection

When loading tasks from file, check if command already exists in ANY environment:

```python
existing_commands = {
    t.command
    for task_list in tasks.values()
    for t in task_list
}

if command in existing_commands:
    continue  # Skip duplicate
```

---

## 5. Execution Backend Architecture

### Abstract Interface

```python
class ExecutionBackend(ABC):
    @abstractmethod
    def start(self, task: Task) -> None:
        """Initialize and start task execution."""
        pass

    @abstractmethod
    def poll_output(self) -> list[str]:
        """Return complete output buffer (not incremental)."""
        pass

    @abstractmethod
    def get_status(self) -> TaskStatus:
        """Check current execution status."""
        pass

    @abstractmethod
    def kill(self) -> None:
        """Terminate the running task."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources (sessions, files)."""
        pass
```

### LocalBackend

**Strategy**: Single tmux session running a bash script, with output redirected to a log file.

**Session Naming**: `orch_local_task_{task_id}`

**Implementation**:
```python
class LocalBackend(ExecutionBackend):
    tmux_session: str = f"orch_local_task_{task_id}"
    _script_path: Optional[str]  # Temporary bash script
    _log_path: Optional[str]     # Output log file
    _output_buffer: list[str] = []
    _status: TaskStatus = TaskStatus.PENDING
```

**Execution Flow**:
1. Create temporary bash script with output redirection
2. Create tmux session running the script
3. Poll by reading log file
4. Detect completion via exit code marker

**Exit Code Marker**: `[TASK_EXIT_CODE:{code}]`

### SSHBackend

**Strategy**: Local tmux session that SSHs to remote host and creates a remote tmux session.

**Session Naming**:
- Local: `orch_ssh_{task_id}`
- Remote: `orch_task_{task_id}`

**Implementation**:
```python
class SSHBackend(ExecutionBackend):
    local_tmux_session: str = f"orch_ssh_{task_id}"
    remote_tmux_session: str = f"orch_task_{task_id}"
    tmux_session: str = remote_tmux_session  # For compatibility
    ssh_config: SSHConfig
    _output_buffer: list[str] = []
    _status: TaskStatus = TaskStatus.PENDING
```

**Execution Flow**:
1. Create local tmux session with large history buffer
2. Send SSH command to local session
3. Configure remote tmux with `remain-on-exit`
4. Send command to create remote tmux session
5. Poll by capturing local tmux pane
6. Detect completion via exit code marker

### SLURMBackend

**Strategy**: Similar to SSH, but wraps command in `salloc` for cluster allocation.

**Session Naming**:
- Local: `orch_slurm_{task_id}`
- Remote: `orch_task_{task_id}`

**Implementation**:
```python
class SLURMBackend(ExecutionBackend):
    local_tmux_session: str = f"orch_slurm_{task_id}"
    remote_tmux_session: str = f"orch_task_{task_id}"
    tmux_session: str = remote_tmux_session  # For compatibility
    slurm_config: SLURMConfig
    job_id: Optional[str] = None  # SLURM job ID
    _output_buffer: list[str] = []
    _status: TaskStatus = TaskStatus.PENDING
```

**Execution Flow**:
1. Create local tmux session
2. Send SSH command
3. Handle MFA prompt (wait 5s, send "1", wait 5s)
4. Configure remote tmux
5. Send salloc command wrapping task command
6. Extract job ID from "Granted job allocation" message
7. Poll and detect completion like SSH

---

## 6. tmux Session Management

### Global Configuration

Set `remain-on-exit on` globally to allow output capture after commands finish:

```bash
tmux set-option -g remain-on-exit on
```

This is configured once when the first LocalBackend is created.

### LocalBackend tmux Sequence

**1. Cleanup Previous Session**:
```bash
tmux kill-session -t orch_local_task_{task_id}
```

**2. Create Temporary Files**:
```python
# Log file for output
log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix="orch_output_")

# Bash script with output redirection
script_content = f"""#!/bin/bash
exec > "{log_path}" 2>&1
echo "--- Start of Task (Local) ---"
echo "Intended Dir: {workdir}"
echo "Command: {command}"
echo "--- Output ---"
cd "{workdir}" && {command}
exit_code=$?
echo "[TASK_EXIT_CODE:$exit_code]"
exit $exit_code
"""
```

**3. Create tmux Session**:
```bash
tmux new-session -d -s orch_local_task_{task_id} \
    -x 200 -y 50 \  # Set window size
    "bash {script_path}"
```

**4. Poll Output**:
```python
# Read entire log file
with open(log_path, 'r') as f:
    all_lines = f.read().splitlines()
```

**5. Detect Completion**:
```python
for line in all_lines:
    if "[TASK_EXIT_CODE:" in line:
        code = int(line.split(":")[1].rstrip("]"))
        status = COMPLETED if code == 0 else FAILED
```

**6. Cleanup**:
```bash
tmux kill-session -t orch_local_task_{task_id}
rm {script_path}
rm {log_path}
```

### SSHBackend tmux Sequence

**1. Create Local Session**:
```bash
tmux new-session -d -s orch_ssh_{task_id}
tmux set-option -t orch_ssh_{task_id} history-limit 50000
```

**2. Send SSH Command**:
```bash
# Build SSH command
ssh -i '{key_file}' -J {proxy_jump} -p {port} {user}@{host}

# Send to tmux
tmux send-keys -t orch_ssh_{task_id} "{ssh_command}" C-m
```

**3. Configure Remote tmux**:
```bash
tmux send-keys -t orch_ssh_{task_id} \
    "tmux start-server ; tmux set-option -g remain-on-exit on" C-m
```

**4. Create Remote Session with Command**:
```bash
# Build full command
full_command = f"""
echo '--- Start of Task (SSH) ---'
echo 'Intended Dir: {workdir}'
echo 'Command: {command}'
echo '--- Output ---'
{activation_cmd if present}
cd '{workdir}' && {command}
echo '[TASK_EXIT_CODE:$?]'
exec bash
"""

# Create remote tmux session
tmux send-keys -t orch_ssh_{task_id} \
    "tmux new-session -A -s orch_task_{task_id} '{escaped_command}'" C-m
```

**5. Poll Output**:
```bash
tmux capture-pane -t orch_ssh_{task_id} -p -S -
# -p: Print to stdout
# -S -: Start from beginning of history
```

**6. Kill Sequence**:
```bash
# 1. Interrupt running command
tmux send-keys -t orch_ssh_{task_id} C-c

# 2. Detach from remote tmux
tmux send-keys -t orch_ssh_{task_id} C-b d

# 3. Kill remote session
tmux send-keys -t orch_ssh_{task_id} \
    "tmux kill-session -t orch_task_{task_id}" C-m

# 4. Kill local session
tmux kill-session -t orch_ssh_{task_id}
```

### SLURMBackend tmux Sequence

**Similar to SSH, with these differences**:

**1. MFA Handling** (after SSH connection):
```python
time.sleep(5)  # Wait for MFA prompt
tmux send-keys -t orch_slurm_{task_id} "1" C-m  # Select option 1
time.sleep(5)  # Wait for connection
```

**2. salloc Command Wrapper**:
```bash
# Build salloc command
salloc_cmd = f"""salloc \
    --partition={partition} \
    --time={time_limit} \
    {extra_flags} \
    bash -c "{inner_command}; echo '[TASK_EXIT_CODE:$?]'; exec bash"
"""
```

**3. Job ID Extraction**:
```python
for line in output_lines:
    if "Granted job allocation" in line:
        parts = line.split()
        if len(parts) >= 4:
            job_id = parts[3]
```

---

## 7. Threading & Concurrency Model

### Three Thread Types

**Scheduler Thread** (1 daemon thread):
- Monitors all environments for available execution slots
- Starts pending tasks when slots become available
- Runs continuously with 1-second sleep between checks

**Monitor Threads** (N daemon threads, one per running task):
- Poll task output every 2 seconds
- Detect task completion via exit code markers
- Signal GUI to update via thread-safe queue
- Perform cleanup after task finishes

**GUI Update Loop** (tkinter.after callback):
- Processes messages from update_queue every 500ms
- Updates treeview and output display
- Runs on main GUI thread

### Scheduler Thread Logic

```python
def scheduler_loop():
    while True:
        # Check each environment independently
        for env_name, task_list in tasks.items():
            running_count = sum(
                1 for t in task_list
                if t.status == TaskStatus.RUNNING
            )
            max_for_env = max_concurrent.get(env_name, 1)

            if running_count < max_for_env:
                # Find next pending task
                for task in task_list:
                    if task.status == TaskStatus.PENDING:
                        _start_task(task)
                        break  # Start one at a time

        # Sleep before next check
        threading.Event().wait(1.0)
```

Started as daemon thread in `__init__`:
```python
thread = threading.Thread(target=scheduler_loop, daemon=True)
thread.start()
```

### Monitor Thread Logic

```python
def _monitor_task(task: Task):
    backend = task.backend
    try:
        # Poll until completion
        while backend.get_status() == TaskStatus.RUNNING \
              and not _shutting_down:

            # Get complete output buffer
            full_buffer = backend.poll_output()

            # Update if changed
            if full_buffer and full_buffer != task.output_buffer:
                task.output_buffer = full_buffer
                update_queue.put("output")

            time.sleep(2.0)

        # Final status update
        if not _shutting_down:
            task.output_buffer = backend.poll_output()
            task.status = backend.get_status()

    except Exception as e:
        if not _shutting_down:
            task.status = TaskStatus.FAILED
            task.output_buffer.append(f"[Monitor Error: {e}]")

    finally:
        # Cleanup only if not shutting down
        if backend and not _shutting_down:
            backend.cleanup()
        if not _shutting_down:
            update_queue.put("update")
```

Started in `_start_task()` after backend initialization:
```python
thread = threading.Thread(target=_monitor_task, args=(task,), daemon=True)
thread.start()
```

### GUI Update Loop

```python
def check_updates():
    try:
        while True:
            msg = update_queue.get_nowait()
            if msg == "update":
                _update_tree()
            elif msg == "output":
                _incremental_update_output_view()
    except queue.Empty:
        pass

    # Schedule next check
    root.after(500, check_updates)

# Start loop
root.after(500, check_updates)
```

### Shutdown Handling

Global flag prevents cleanup during GUI exit:
```python
_shutting_down: bool = False

def _on_closing():
    _shutting_down = True  # Signal to monitor threads
    _save_state()
    root.destroy()
```

Monitor threads check this flag before cleanup and updates.

---

## 8. Output Polling & Display

### Full Buffer Strategy

**Design Decision**: All backends return the complete output buffer on each poll, not incremental changes. This simplifies the implementation because:
- SSH/SLURM backends use `tmux capture-pane` which always returns full pane
- LocalBackend reads entire log file each time
- Change detection uses hashing to avoid unnecessary UI updates

### Backend-Specific Polling

**LocalBackend**:
```python
def poll_output(self) -> list[str]:
    if self._log_path:
        with open(self._log_path, 'r') as f:
            all_lines = f.read().splitlines()
    else:
        # Fallback to tmux capture (for reconnected sessions)
        ret_code, stdout, _ = _run_command([
            "tmux", "capture-pane", "-t", self.tmux_session,
            "-p", "-S", "-"
        ])
        all_lines = stdout.splitlines() if ret_code == 0 else []

    return all_lines
```

**SSH/SLURMBackend**:
```python
def poll_output(self) -> list[str]:
    ret_code, stdout, _ = _run_command([
        "tmux", "capture-pane", "-t", self.local_tmux_session,
        "-p",  # Print to stdout
        "-S", "-"  # Start from beginning of history
    ])

    if ret_code != 0:
        return []

    return stdout.splitlines()
```

### Change Detection

Track both line count AND content hash to detect changes:

```python
# In Orchestrator class
displayed_output_task_id: Optional[int] = None
displayed_output_line_count: int = 0
displayed_output_hash: int = 0

def _incremental_update_output_view(self):
    if displayed_output_task_id is None:
        return

    # Find the task
    found_task = <find task by ID>

    # Check if output changed
    current_hash = hash(tuple(found_task.output_buffer))
    if (len(found_task.output_buffer) != displayed_output_line_count or
        current_hash != displayed_output_hash):

        # Refresh display
        _full_refresh_output_view(found_task)
        displayed_output_line_count = len(found_task.output_buffer)
        displayed_output_hash = current_hash
```

**Why hash-based?**: tmux capture-pane may return a constant number of lines (buffer size) but with changing content. Line count alone doesn't detect this.

### Display Rendering

```python
def _full_refresh_output_view(self, task: Task) -> None:
    # Enable editing
    output_text.configure(state=tk.NORMAL)

    # Clear existing content
    output_text.delete("1.0", tk.END)

    # Show only last 500 lines (prevent UI slowdown)
    if task.output_buffer:
        lines_to_show = task.output_buffer[-500:]
        text_content = "\n".join(lines_to_show) + "\n"
        output_text.insert(tk.END, text_content)
    else:
        output_text.insert(tk.END, "[No output yet]\n")

    # Scroll to bottom
    output_text.see(tk.END)

    # Disable editing
    output_text.configure(state=tk.DISABLED)

    # Force visual update
    output_text.update_idletasks()
```

### Exit Code Detection

Search for marker in reversed buffer (to find most recent):

```python
for line in reversed(all_lines):
    if "[TASK_EXIT_CODE:" in line:
        try:
            code = int(line.split(":")[-1].strip("[]"))
            status = TaskStatus.COMPLETED if code == 0 else TaskStatus.FAILED
        except (ValueError, IndexError):
            pass
        break  # Found the latest exit code
```

---

## 9. State Persistence & Recovery

### State File

**Location**: `orchestrator_state.json` (in current working directory)

**Structure**:
```json
{
  "task_counter": 123,
  "last_task_file_path": "/path/to/last/tasks.txt",
  "max_concurrent": {
    "local": 1,
    "ginkgo": 4,
    "rorqual": 2
  },
  "tasks": [
    {
      "id": 42,
      "command": "python train.py --model A",
      "status": "Running",
      "backend_type": "Local",
      "activation_cmd": "",
      "remote_workdir": "/path/to/workdir",
      "tmux_session": "orch_local_task_42",
      "log_file_path": "/tmp/orch_output_xyz.log",
      "output_buffer": [
        "--- Start of Task (Local) ---",
        "Epoch 1/10...",
        "..."
      ],
      "ssh_config": null,
      "slurm_config": null
    },
    {
      "id": 43,
      "command": "python train.py --model B",
      "status": "Pending",
      "backend_type": "SSH Remote",
      "activation_cmd": "source /scratch/mleclei/venv/bin/activate",
      "remote_workdir": "/scratch/mleclei/Dropbox/repos/project",
      "tmux_session": "orch_task_43",
      "log_file_path": null,
      "output_buffer": [],
      "ssh_config": {
        "host": "ginkgo.criugm.qc.ca",
        "port": 22,
        "username": "mleclei",
        "key_file": "/home/user/.ssh/id_ed25519",
        "password": "",
        "proxy_jump": "mleclei@elm.criugm.qc.ca"
      },
      "slurm_config": null
    }
  ]
}
```

### Save Trigger

Automatically save on window close:

```python
def _on_closing(self):
    _shutting_down = True
    _save_state()
    root.destroy()
```

### Load Process

Called during `__init__`:

```python
def _load_state(self):
    if not os.path.exists(state_file):
        return

    with open(state_file, 'r') as f:
        state = json.load(f)

    # Restore global state
    task_counter = state.get("task_counter", 0)
    last_task_file_path = state.get("last_task_file_path")
    max_concurrent = state.get("max_concurrent", {})

    # Reconstruct Task objects
    for task_data in state.get("tasks", []):
        task = Task(
            id=task_data["id"],
            command=task_data["command"],
            status=TaskStatus(task_data["status"]),
            backend_type=BackendType(task_data["backend_type"]),
            activation_cmd=task_data.get("activation_cmd", ""),
            remote_workdir=task_data.get("remote_workdir", ""),
            tmux_session=task_data.get("tmux_session"),
            log_file_path=task_data.get("log_file_path"),
            output_buffer=task_data.get("output_buffer", [])
        )

        # Restore SSH config if present
        if "ssh_config" in task_data:
            task.ssh_config = SSHConfig(**task_data["ssh_config"])

        # Restore SLURM config if present
        if "slurm_config" in task_data:
            slurm_data = task_data["slurm_config"]
            slurm_data["ssh_config"] = SSHConfig(**slurm_data["ssh_config"])
            task.slurm_config = SLURMConfig(**slurm_data)

        # Add to appropriate environment
        env_name = _get_env_name_from_backend_type(task.backend_type)
        tasks[env_name].append(task)

    # Reconnect to running tasks
    _reconnect_running_tasks()
```

### Reconnection Logic

```python
def _reconnect_running_tasks(self):
    for task_list in tasks.values():
        for task in task_list:
            if task.status != TaskStatus.RUNNING:
                continue

            try:
                backend = None
                session_exists = False

                # Create appropriate backend
                if task.backend_type == BackendType.LOCAL:
                    backend = LocalBackend(task.id)
                    backend.tmux_session = task.tmux_session
                    backend._log_path = task.log_file_path

                    # Check if session exists
                    ret_code, _, _ = backend._run_command([
                        "tmux", "has-session", "-t", backend.tmux_session
                    ])
                    session_exists = (ret_code == 0)

                elif task.backend_type == BackendType.SSH:
                    backend = SSHBackend(task.ssh_config, task.id)
                    backend.tmux_session = task.tmux_session
                    session_exists = True  # Trust saved state

                elif task.backend_type == BackendType.SLURM:
                    backend = SLURMBackend(task.slurm_config, task.id)
                    backend.tmux_session = task.tmux_session
                    session_exists = True  # Trust saved state

                if backend and session_exists:
                    # Set backend to running state
                    backend._status = TaskStatus.RUNNING
                    task.backend = backend
                    task.output_buffer.append(
                        f"[Reconnected to session: {backend.tmux_session}]"
                    )

                    # Start monitoring thread
                    thread = threading.Thread(
                        target=_monitor_task,
                        args=(task,),
                        daemon=True
                    )
                    thread.start()
                else:
                    # Session lost
                    task.status = TaskStatus.FAILED
                    task.output_buffer.append(
                        f"[Session {task.tmux_session} no longer exists]"
                    )

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.output_buffer.append(f"[Reconnect failed: {e}]")
```

---

## 10. Task Loading & File Management

### Task File Format

**Syntax**:
- One command per line
- Lines starting with `#` are comments (ignored)
- Empty lines are ignored
- No special escaping needed

**Example**:
```bash
# Training runs for model A
python train.py --model A --epochs 10
python train.py --model A --epochs 20

# Training runs for model B
python train.py --model B --epochs 10

# Evaluation
python evaluate.py --all
```

### File Selection Dialog

```python
def _load_tasks_from_file(self):
    # Determine initial directory
    initial_dir = dropbox_path
    if last_task_file_path:
        initial_dir = str(Path(last_task_file_path).parent)

    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select tasks.txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*")],
        initialdir=initial_dir
    )

    if file_path:
        last_task_file_path = file_path
        _load_tasks_file(file_path)
```

### Working Directory Computation

```python
def _load_tasks_file(self, file_path: str):
    # Parse file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Local working directory
    local_workdir = str(Path(file_path).parent.resolve())

    # Auto-compute remote working directory
    if current_backend_type in (BackendType.SSH, BackendType.SLURM):
        if current_remote_dropbox:
            try:
                local_dropbox = Path(dropbox_path).resolve()
                local_task_dir = Path(local_workdir).resolve()
                relative_path = local_task_dir.relative_to(local_dropbox)
                remote_workdir = f"{current_remote_dropbox}/{relative_path.as_posix()}"
            except ValueError:
                # Not in Dropbox, must specify manually
                pass

    # Parse commands
    commands = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        commands.append(line)

    # Show selection dialog
    _show_task_selection_dialog(commands, local_workdir)
```

### Task Selection Dialog

**Layout**:
- Modal window (800x500)
- Scrollable canvas with checkboxes
- Each command shows as: `{index}. {command_display}`
- Command truncation: `cmd[:97] + "..."` if len > 100
- All tasks selected by default

**Controls**:
- **Select All** button: Check all checkboxes
- **Deselect All** button: Uncheck all checkboxes
- **Add Selected** button: Add checked tasks to queue
- **Cancel** button: Close without adding

**Implementation**:
```python
def _show_task_selection_dialog(self, commands: list[str], local_workdir: str):
    dialog = tk.Toplevel(root)
    dialog.title("Select Tasks to Add")
    dialog.geometry("800x500")
    dialog.configure(bg="#1e1e1e")
    dialog.transient(root)
    dialog.grab_set()

    # Create scrollable frame with checkboxes
    check_vars: list[tk.BooleanVar] = []
    for i, cmd in enumerate(commands):
        var = tk.BooleanVar(value=True)  # Default selected
        check_vars.append(var)

        display_cmd = cmd if len(cmd) <= 100 else cmd[:97] + "..."
        cb = ttk.Checkbutton(
            scrollable_frame,
            text=f"{i+1}. {display_cmd}",
            variable=var
        )
        cb.pack(anchor="w", pady=1)

    def add_selected():
        # Get selected commands
        selected = [cmd for cmd, var in zip(commands, check_vars) if var.get()]

        # Check for duplicates
        existing = {t.command for all tasks in tasks.values() for t in all tasks}

        # Add new tasks
        for cmd in selected:
            if cmd in existing:
                continue

            task_counter += 1
            task = Task(
                id=task_counter,
                command=cmd,
                status=TaskStatus.UNAPPROVED,
                backend_type=current_backend_type,
                activation_cmd=activation_cmd_var.get().strip()
            )

            # Set working directory and backend config
            if current_backend_type == BackendType.LOCAL:
                task.remote_workdir = local_workdir
            elif current_backend_type == BackendType.SSH:
                task.ssh_config = _get_current_ssh_config()
                task.remote_workdir = remote_workdir_var.get().strip()
            elif current_backend_type == BackendType.SLURM:
                task.slurm_config = _get_current_slurm_config()
                task.remote_workdir = remote_workdir_var.get().strip()

            tasks[current_env].append(task)

        _update_tree()
        dialog.destroy()
```

---

## 11. Task Management Operations

### Approve

**Purpose**: Move tasks from Unapproved to Pending state (ready for execution).

**Behavior**:
- If no selection: Show warning
- For each selected task in current environment:
  - If status == UNAPPROVED: status = PENDING
- Update tree
- Show count of approved tasks

```python
def _approve_selected(self):
    selection = tree.selection()
    if not selection:
        messagebox.showwarning("No Selection", "Please select task(s) to approve.")
        return

    task_ids = [int(tree.item(s)["values"][0]) for s in selection]
    count = 0

    current_env = _get_env_name()
    for task in tasks.get(current_env, []):
        if task.id in task_ids and task.status == TaskStatus.UNAPPROVED:
            task.status = TaskStatus.PENDING
            count += 1

    if count > 0:
        _update_tree()
        status_var.set(f"Approved {count} task(s)")
```

### Remove Selected

**Purpose**: Kill (if running) and remove selected tasks from queue.

**Behavior**:
- If no selection: Show warning
- For each selected task:
  - If status == RUNNING: backend.kill()
  - Remove task from list
- Update tree

```python
def _remove_selected(self):
    selection = tree.selection()
    if not selection:
        messagebox.showwarning("No Selection", "Please select task(s) to remove.")
        return

    task_ids = set(int(tree.item(s)["values"][0]) for s in selection)
    current_env = _get_env_name()
    task_list = tasks.get(current_env, [])

    # Kill running tasks first
    for task in task_list:
        if task.id in task_ids and task.status == TaskStatus.RUNNING:
            if task.backend:
                task.backend.kill()

    # Remove all selected tasks
    original_count = len(task_list)
    tasks[current_env] = [t for t in task_list if t.id not in task_ids]
    count = original_count - len(tasks[current_env])

    _update_tree()
    status_var.set(f"Removed {count} task(s)")
```

### Remove All Done

**Purpose**: Clear completed, failed, and killed tasks.

**Behavior**:
- Filter out tasks with status in {COMPLETED, FAILED, KILLED}
- Update tree

```python
def _remove_completed(self):
    current_env = _get_env_name()
    tasks[current_env] = [
        t for t in tasks.get(current_env, [])
        if t.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED)
    ]
    _update_tree()
    status_var.set("Removed completed/failed/killed tasks")
```

### Clear Unapproved

**Purpose**: Remove all unapproved tasks from current environment.

**Behavior**:
- Filter out tasks with status == UNAPPROVED
- Update tree

```python
def _clear_unapproved(self):
    current_env = _get_env_name()
    tasks[current_env] = [
        t for t in tasks.get(current_env, [])
        if t.status != TaskStatus.UNAPPROVED
    ]
    _update_tree()
    status_var.set("Cleared unapproved tasks")
```

### Re-run

**Purpose**: Reset failed/killed/completed tasks to pending state.

**Behavior**:
- **If selection exists**: Re-run selected tasks
  - For selected with status in {COMPLETED, FAILED, KILLED}:
    - status = PENDING, backend = None, tmux_session = None, output_buffer = []
- **If no selection**: Re-run all failed/killed tasks in current environment

```python
def _rerun_tasks(self):
    selection = tree.selection()
    current_env = _get_env_name()
    task_list = tasks.get(current_env, [])

    if selection:
        # Re-run selected
        task_ids = [int(tree.item(s)["values"][0]) for s in selection]
        count = 0

        for task in task_list:
            if task.id in task_ids and \
               task.status in (TaskStatus.FAILED, TaskStatus.KILLED, TaskStatus.COMPLETED):
                task.status = TaskStatus.PENDING
                task.backend = None
                task.tmux_session = None
                task.output_buffer = []
                count += 1

        if count > 0:
            _update_tree()
            status_var.set(f"Re-queued {count} task(s)")
        else:
            messagebox.showinfo("Cannot Re-run",
                "No completed/failed/killed tasks in selection.")
    else:
        # Re-run all failed/killed
        count = 0
        for task in task_list:
            if task.status in (TaskStatus.FAILED, TaskStatus.KILLED):
                task.status = TaskStatus.PENDING
                task.backend = None
                task.tmux_session = None
                task.output_buffer = []
                count += 1

        _update_tree()
        status_var.set(f"Re-queued {count} failed/killed tasks")
```

### Tree Update

Updates the treeview to reflect current task state:

```python
def _update_tree(self):
    # Clear tree
    for item in tree.get_children():
        tree.delete(item)

    # Re-populate with tasks from current environment
    current_env = _get_env_name()
    for task in tasks.get(current_env, []):
        tag = task.status.value.lower()
        tree.insert("", "end",
            values=(task.id, task.status.value, task.command),
            tags=(tag,))

    # Update running count
    env_running = sum(1 for t in tasks[current_env]
                      if t.status == TaskStatus.RUNNING)
    total_running = sum(1 for tl in tasks.values() for t in tl
                        if t.status == TaskStatus.RUNNING)
    running_label.config(text=f"Running: {env_running} ({total_running} total)")
```

---

## 12. Platform-Specific Details (Windows/itmux)

### itmux Binary Paths

Hard-coded paths to itmux installation:

```python
class LocalBackend(ExecutionBackend):
    _tmux_path = r"C:\Users\Max\Documents\itmux\bin\tmux.exe"
    _bash_path = r"C:\Users\Max\Documents\itmux\bin\bash.exe"

class SSHBackend(ExecutionBackend):
    _tmux_path = r"C:\Users\Max\Documents\itmux\bin\tmux.exe"

class SLURMBackend(ExecutionBackend):
    _tmux_path = r"C:\Users\Max\Documents\itmux\bin\tmux.exe"
```

### Command Substitution

Replace "tmux" with full path before executing:

```python
def _run_command(self, command: list[str]) -> tuple[int, str, str]:
    if command and command[0] == "tmux":
        command = [self._tmux_path] + command[1:]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False
    )
    return process.returncode, process.stdout, process.stderr
```

### Path Handling

itmux bash understands Windows paths with forward slashes:

```python
# Convert backslashes to forward slashes
win_workdir = working_dir.replace("\\", "/")
win_script_path = script_path.replace("\\", "/")
bash_win_path = self._bash_path.replace("\\", "/")

# Use directly in commands
full_command = f'"{bash_win_path}" "{win_script_path}"'
```

**No need for** `/c/path` conversion (MSYS2 style). Direct Windows paths work.

### Monitor Positioning

Position window on the monitor where the cursor is located (Windows-specific):

```python
def _position_on_cursor_monitor(self):
    try:
        import ctypes

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long)
            ]

        class MONITORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_ulong),
                ("rcMonitor", RECT),
                ("rcWork", RECT),
                ("dwFlags", ctypes.c_ulong)
            ]

        user32 = ctypes.windll.user32

        # Get cursor position
        cursor = POINT()
        user32.GetCursorPos(ctypes.byref(cursor))

        # Get monitor from cursor position
        monitor_handle = user32.MonitorFromPoint(cursor, 2)

        # Get monitor info
        monitor_info = MONITORINFO()
        monitor_info.cbSize = ctypes.sizeof(MONITORINFO)
        user32.GetMonitorInfoW(monitor_handle, ctypes.byref(monitor_info))

        # Position window on this monitor
        win_x = monitor_info.rcWork.left + 50
        win_y = monitor_info.rcWork.top + 50
        root.geometry(f"1400x800+{win_x}+{win_y}")

    except Exception as e:
        # Fallback to default
        root.geometry("1400x800")
```

### SSH Key Detection

Search `~/.ssh/` directory in priority order:

```python
def _find_ssh_key(self) -> str:
    ssh_dir = Path.home() / ".ssh"
    key_names = ["id_ed25519", "id_rsa", "id_ecdsa", "id_dsa"]

    for key_name in key_names:
        key_path = ssh_dir / key_name
        if key_path.exists():
            return str(key_path)

    return ""  # No key found
```

### Bash Script Line Endings

**Critical**: Bash scripts must use LF (`\n`) line endings, not CRLF (`\r\n`).

```python
# Construct script with \n only
script_lines = [
    "#!/bin/bash",
    f'exec > "{log_path}" 2>&1',
    'echo "--- Start of Task ---"',
    # ...
]
script_content = "\n".join(script_lines) + "\n"

# Write with UTF-8 encoding (preserves \n)
os.write(script_fd, script_content.encode("utf-8"))
```

---

## 13. Error Handling & Recovery

### Backend Start Failures

**Scenario**: Exception during `backend.start()`.

**Handling**:
```python
def _start_task(self, task: Task):
    task.status = TaskStatus.RUNNING
    update_queue.put("update")

    def run_task():
        try:
            # Create and start backend
            backend = create_backend(task)
            task.backend = backend
            backend.start(task)

            # Monitor task
            _monitor_task(task)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.output_buffer.append(f"[Start Error: {str(e)}]")
            update_queue.put("update")

    thread = threading.Thread(target=run_task, daemon=True)
    thread.start()
```

**Result**: Task marked as FAILED with error message in output.

### Monitor Thread Exceptions

**Scenario**: Exception during output polling or status checking.

**Handling**:
```python
def _monitor_task(task: Task):
    try:
        # Monitoring loop
        while backend.get_status() == TaskStatus.RUNNING and not _shutting_down:
            full_buffer = backend.poll_output()
            # ...
            time.sleep(2.0)

        # Final status update
        if not _shutting_down:
            task.status = backend.get_status()

    except Exception as e:
        if not _shutting_down:
            task.status = TaskStatus.FAILED
            task.output_buffer.append(f"[Monitor Error: {str(e)}]")

    finally:
        if backend and not _shutting_down:
            backend.cleanup()
        if not _shutting_down:
            update_queue.put("update")
```

**Result**: Task marked as FAILED, cleanup performed, GUI updated.

### Session Loss Detection

**Scenario**: tmux session ends unexpectedly during monitoring.

**Handling** (in `backend.get_status()`):
```python
def get_status(self) -> TaskStatus:
    if self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED):
        return self._status

    try:
        ret_code, _, _ = _run_command([
            "tmux", "has-session", "-t", self.tmux_session
        ])

        if ret_code != 0:
            # Session is gone
            if self._status == TaskStatus.RUNNING:
                self._status = TaskStatus.COMPLETED
                self._output_buffer.append("[tmux session ended]")

    except Exception:
        pass  # Keep current status

    return self._status
```

**Result**: Task marked as COMPLETED when session ends (assumes successful exit).

### Reconnection Failures

**Scenario**: Loading state file with RUNNING tasks, but sessions no longer exist.

**Handling**:
```python
def _reconnect_running_tasks(self):
    for task in all_running_tasks:
        try:
            backend = create_backend(task)

            # Check if session exists
            ret_code, _, _ = backend._run_command([
                "tmux", "has-session", "-t", task.tmux_session
            ])

            if ret_code == 0:
                # Session exists, reconnect
                backend._status = TaskStatus.RUNNING
                task.backend = backend
                start_monitor_thread(task)
            else:
                # Session lost
                task.status = TaskStatus.FAILED
                task.output_buffer.append(
                    f"[Session {task.tmux_session} no longer exists]"
                )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.output_buffer.append(f"[Reconnect failed: {e}]")
```

**Result**: Lost sessions marked as FAILED with explanation.

### Shutdown Handling

**Scenario**: User closes GUI while tasks are running.

**Handling**:
```python
_shutting_down: bool = False

def _on_closing(self):
    _shutting_down = True  # Signal to all threads
    _save_state()  # Persist current state
    root.destroy()

# In monitor threads:
if not _shutting_down:
    backend.cleanup()  # Only cleanup if not shutting down
```

**Result**: Tasks keep running in tmux, state saved for reconnection on next launch.

### GUI Exception Handling

**Status Bar Messages**: Non-critical errors shown in status bar.
```python
status_var.set(f"Error loading state: {e}")
```

**Message Boxes**: Critical errors shown in modal dialogs.
```python
messagebox.showerror("Load State Error",
    f"Failed to load state from {state_file}:\n{e}")
```

**Try/Except in Callbacks**: All GUI event handlers wrapped in try/except to prevent crash.
```python
def _load_tasks_from_file(self):
    try:
        # File loading logic
        _load_tasks_file(file_path)
    except Exception as e:
        messagebox.showerror("Load Error",
            f"Failed to load tasks:\n{str(e)}")
```

**Paramiko Availability**: Check for optional dependency.
```python
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False

# When using SSH/SLURM backends:
if not PARAMIKO_AVAILABLE:
    messagebox.showwarning("Missing Dependency",
        "paramiko not installed. Run: pip install paramiko")
```

---

## Summary

This specification provides a complete blueprint for implementing the Experiment Orchestrator GUI. Key implementation aspects:

- **3 execution backends** with tmux-based resilience
- **Thread-based architecture** for concurrent task management
- **State persistence** enabling reconnection after restart
- **Per-environment queuing** with independent concurrency limits
- **Dark-themed Tkinter GUI** with multi-selection and real-time output
- **Platform-specific handling** for Windows/itmux

The design prioritizes resilience (tasks survive GUI crashes), simplicity (full buffer polling), and user experience (dark theme, keyboard shortcuts, visual feedback).
