"""Threading layer for task scheduling and monitoring."""

import threading
import time
import queue
from typing import Dict, Callable
from datetime import datetime
from .models import AppState, Task, TaskState, Environment
from .executors import BaseExecutor, LocalExecutor, SSHExecutor, SLURMExecutor


class TaskMonitor(threading.Thread):
    """Monitor a single running task."""

    def __init__(self, task: Task, executor: BaseExecutor, message_queue: queue.Queue):
        super().__init__(daemon=True)
        self.task = task
        self.executor = executor
        self.message_queue = message_queue
        self.stop_flag = threading.Event()
        self._last_output_lines = 0

    def run(self):
        """Monitor task output and status."""
        while not self.stop_flag.is_set():
            try:
                # Get latest output
                output_lines = self.executor.get_output(self.task)

                # Check if there's new output
                if len(output_lines) > self._last_output_lines:
                    new_lines = output_lines[self._last_output_lines:]
                    self.task.append_output(new_lines)
                    self._last_output_lines = len(output_lines)

                    # Send output update message
                    self.message_queue.put({
                        'type': 'output_update',
                        'task_id': self.task.id
                    })

                # Check task status
                is_running, exit_code = self.executor.check_status(self.task)

                if not is_running:
                    # Task completed
                    self.task.end_time = datetime.now().isoformat()
                    self.task.exit_code = exit_code

                    # Determine final state based on exit code
                    if exit_code == 0:
                        self.task.state = TaskState.COMPLETED
                    else:
                        self.task.state = TaskState.FAILED

                    # Send completion message
                    self.message_queue.put({
                        'type': 'task_completed',
                        'task_id': self.task.id,
                        'state': self.task.state
                    })

                    break

                # Sleep before next poll
                time.sleep(2)

            except Exception as e:
                print(f"Error monitoring task {self.task.id[:8]}: {e}")
                time.sleep(5)

    def stop(self):
        """Stop monitoring this task."""
        self.stop_flag.set()


class TaskScheduler(threading.Thread):
    """Scheduler that manages task execution across all environments."""

    def __init__(self, app_state: AppState, message_queue: queue.Queue):
        super().__init__(daemon=True)
        self.app_state = app_state
        self.message_queue = message_queue
        self.stop_flag = threading.Event()

        # Create executors for each environment
        self.executors: Dict[str, BaseExecutor] = {}
        for env_name, env in app_state.environments.items():
            if env.is_slurm():
                self.executors[env_name] = SLURMExecutor(env)
            elif env.is_local():
                self.executors[env_name] = LocalExecutor(env)
            else:
                self.executors[env_name] = SSHExecutor(env)

        # Track active monitors
        self.monitors: Dict[str, TaskMonitor] = {}

    def run(self):
        """Main scheduler loop."""
        while not self.stop_flag.is_set():
            try:
                # Check each environment
                for env_name, env in self.app_state.environments.items():
                    self._process_environment(env_name, env)

                # Sleep before next iteration
                time.sleep(1)

            except Exception as e:
                print(f"Error in scheduler: {e}")
                time.sleep(5)

    def _process_environment(self, env_name: str, env: Environment):
        """Process pending tasks for an environment."""
        # Get running and pending tasks
        running_tasks = self.app_state.get_running_tasks(env_name)
        pending_tasks = self.app_state.get_pending_tasks(env_name)

        # Calculate available slots
        available_slots = env.concurrency_limit - len(running_tasks)

        # Start pending tasks if we have slots
        if available_slots > 0 and pending_tasks:
            tasks_to_start = pending_tasks[:available_slots]

            for task in tasks_to_start:
                self._start_task(task, env_name)

    def _start_task(self, task: Task, env_name: str):
        """Start executing a task."""
        try:
            executor = self.executors[env_name]

            # Execute the task
            if executor.execute(task):
                # Update task state
                task.state = TaskState.RUNNING

                # Create and start monitor
                monitor = TaskMonitor(task, executor, self.message_queue)
                monitor.start()
                self.monitors[task.id] = monitor

                # Send start message
                self.message_queue.put({
                    'type': 'task_started',
                    'task_id': task.id
                })

            else:
                # Failed to start
                task.state = TaskState.FAILED
                self.message_queue.put({
                    'type': 'task_failed',
                    'task_id': task.id
                })

        except Exception as e:
            print(f"Error starting task {task.id[:8]}: {e}")
            task.state = TaskState.FAILED

    def kill_task(self, task: Task):
        """Kill a running task."""
        try:
            # Stop the monitor
            if task.id in self.monitors:
                self.monitors[task.id].stop()
                del self.monitors[task.id]

            # Kill the task
            executor = self.executors[task.environment]
            if executor.kill(task):
                task.state = TaskState.KILLED
                task.end_time = datetime.now().isoformat()

                self.message_queue.put({
                    'type': 'task_killed',
                    'task_id': task.id
                })

        except Exception as e:
            print(f"Error killing task {task.id[:8]}: {e}")

    def reconnect_running_tasks(self):
        """Reconnect to running tasks after restart."""
        for env_name, env in self.app_state.environments.items():
            running_tasks = self.app_state.get_running_tasks(env_name)

            for task in running_tasks:
                if task.tmux_session:
                    # Start a monitor for this task
                    executor = self.executors[env_name]
                    monitor = TaskMonitor(task, executor, self.message_queue)
                    monitor.start()
                    self.monitors[task.id] = monitor

    def stop(self):
        """Stop the scheduler."""
        self.stop_flag.set()

        # Stop all monitors
        for monitor in self.monitors.values():
            monitor.stop()
