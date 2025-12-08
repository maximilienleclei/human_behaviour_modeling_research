"""GUI components for the Experiment Orchestrator."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import queue
from typing import Optional, List
from .models import AppState, Task, TaskState


class OrchestratorGUI:
    """Main GUI window for the orchestrator."""

    def __init__(self, app_state: AppState, message_queue: queue.Queue,
                 load_tasks_callback, approve_callback, clear_unapproved_callback,
                 remove_selected_callback, remove_all_done_callback, rerun_callback,
                 set_concurrency_callback, kill_task_callback):
        """
        Initialize the GUI.

        Args:
            app_state: The application state
            message_queue: Queue for receiving updates from background threads
            Various callbacks for actions
        """
        self.app_state = app_state
        self.message_queue = message_queue

        # Callbacks
        self.load_tasks_callback = load_tasks_callback
        self.approve_callback = approve_callback
        self.clear_unapproved_callback = clear_unapproved_callback
        self.remove_selected_callback = remove_selected_callback
        self.remove_all_done_callback = remove_all_done_callback
        self.rerun_callback = rerun_callback
        self.set_concurrency_callback = set_concurrency_callback
        self.kill_task_callback = kill_task_callback

        # Current environment
        self.current_environment = "local"

        # Create main window
        self.root = tk.Tk()
        self.root.title("Experiment Orchestrator")
        self.root.geometry("1400x800")

        # Apply dark theme
        self._apply_dark_theme()

        # Create UI components
        self._create_ui()

        # Start GUI update loop
        self._schedule_update()

    def _apply_dark_theme(self):
        """Apply dark theme to the GUI."""
        # Configure colors
        bg_color = "#2b2b2b"
        fg_color = "#ffffff"
        select_color = "#404040"

        # Configure root window
        self.root.configure(bg=bg_color)

        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')

        # Configure basic styles
        style.configure(".", background=bg_color, foreground=fg_color,
                       fieldbackground=bg_color, borderwidth=1)
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color)
        style.configure("TButton", background="#404040", foreground=fg_color,
                       borderwidth=1, focuscolor='none')
        style.map("TButton",
                 background=[('active', '#505050')])
        style.configure("TEntry", fieldbackground="#404040", foreground=fg_color,
                       insertcolor=fg_color)
        style.configure("TRadiobutton", background=bg_color, foreground=fg_color)

    def _create_ui(self):
        """Create the UI layout."""
        # Create main paned window (left/right split)
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)

        # Right panel
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)

        # Build left panel
        self._create_left_panel(left_frame)

        # Build right panel
        self._create_right_panel(right_frame)

    def _create_left_panel(self, parent):
        """Create left panel with controls."""
        # Environment selector
        env_frame = ttk.Frame(parent)
        env_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(env_frame, text="Environment:").pack(side=tk.LEFT, padx=5)

        self.env_var = tk.StringVar(value="local")
        for env_name in ["local", "ginkgo", "rorqual"]:
            rb = ttk.Radiobutton(env_frame, text=env_name.capitalize(),
                                variable=self.env_var, value=env_name,
                                command=self._on_environment_changed)
            rb.pack(side=tk.LEFT, padx=5)

        # Concurrency control
        conc_frame = ttk.Frame(parent)
        conc_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(conc_frame, text="Concurrency:").pack(side=tk.LEFT, padx=5)

        self.concurrency_var = tk.StringVar(value="2")
        conc_entry = ttk.Entry(conc_frame, textvariable=self.concurrency_var, width=5)
        conc_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(conc_frame, text="Set", command=self._on_set_concurrency).pack(side=tk.LEFT, padx=5)

        # Load tasks button
        load_frame = ttk.Frame(parent)
        load_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(load_frame, text="Load Tasks", command=self._on_load_tasks).pack(fill=tk.X)

        # Task queue
        queue_frame = ttk.Frame(parent)
        queue_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(queue_frame, text="Task Queue:").pack(anchor=tk.W)

        # Create listbox with scrollbar
        list_frame = ttk.Frame(queue_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.task_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED,
                                       yscrollcommand=scrollbar.set,
                                       bg="#2b2b2b", fg="#ffffff",
                                       selectbackground="#404040",
                                       highlightthickness=0)
        self.task_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.task_listbox.yview)

        # Bind selection event
        self.task_listbox.bind('<<ListboxSelect>>', self._on_task_selected)

        # Bind keyboard shortcuts
        self.task_listbox.bind('<Control-a>', self._on_select_all)
        self.task_listbox.bind('<Shift-Up>', self._on_shift_up)
        self.task_listbox.bind('<Shift-Down>', self._on_shift_down)

        # Action buttons
        actions_frame = ttk.Frame(parent)
        actions_frame.pack(fill=tk.X, padx=5, pady=5)

        buttons = [
            ("Approve", self._on_approve),
            ("Clear Unapproved", self._on_clear_unapproved),
            ("Remove Selected", self._on_remove_selected),
            ("Remove All Done", self._on_remove_all_done),
            ("Re-run", self._on_rerun)
        ]

        for i, (text, command) in enumerate(buttons):
            btn = ttk.Button(actions_frame, text=text, command=command)
            btn.grid(row=i // 2, column=i % 2, padx=2, pady=2, sticky=tk.EW)

        actions_frame.columnconfigure(0, weight=1)
        actions_frame.columnconfigure(1, weight=1)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(parent, textvariable=self.status_var,
                                relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X, padx=5, pady=5)

    def _create_right_panel(self, parent):
        """Create right panel with output viewer."""
        ttk.Label(parent, text="Output:").pack(anchor=tk.W, padx=5, pady=5)

        # Create text widget with scrollbar
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.output_text = tk.Text(text_frame, wrap=tk.NONE,
                                   yscrollcommand=scrollbar.set,
                                   bg="#2b2b2b", fg="#00ff00",
                                   insertbackground="#ffffff",
                                   state=tk.DISABLED,
                                   font=("Courier", 10))
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.output_text.yview)

    # Event handlers

    def _on_environment_changed(self):
        """Handle environment selection change."""
        self.current_environment = self.env_var.get()
        self._refresh_task_list()
        self._update_concurrency_display()
        self._clear_output()

    def _on_set_concurrency(self):
        """Handle concurrency limit change."""
        try:
            limit = int(self.concurrency_var.get())
            if limit <= 0:
                raise ValueError("Concurrency must be positive")

            self.set_concurrency_callback(self.current_environment, limit)
            self.status_var.set(f"Concurrency set to {limit}")

        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid concurrency value: {e}")

    def _on_load_tasks(self):
        """Handle load tasks button."""
        filename = filedialog.askopenfilename(
            title="Select Task File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            self.load_tasks_callback(self.current_environment, filename)
            self._refresh_task_list()

    def _on_approve(self):
        """Handle approve button."""
        selected = self._get_selected_tasks()
        if selected:
            self.approve_callback(selected)
            self._refresh_task_list()

    def _on_clear_unapproved(self):
        """Handle clear unapproved button."""
        self.clear_unapproved_callback(self.current_environment)
        self._refresh_task_list()

    def _on_remove_selected(self):
        """Handle remove selected button."""
        selected = self._get_selected_tasks()
        if selected:
            # Kill running tasks
            for task in selected:
                if task.state == TaskState.RUNNING:
                    self.kill_task_callback(task)

            self.remove_selected_callback(selected)
            self._refresh_task_list()

    def _on_remove_all_done(self):
        """Handle remove all done button."""
        self.remove_all_done_callback(self.current_environment)
        self._refresh_task_list()

    def _on_rerun(self):
        """Handle re-run button."""
        selected = self._get_selected_tasks()
        if selected:
            self.rerun_callback(selected)
            self._refresh_task_list()

    def _on_task_selected(self, event):
        """Handle task selection."""
        selected = self._get_selected_tasks()

        # Only show output if exactly one task is selected
        if len(selected) == 1:
            self._display_task_output(selected[0])
        else:
            self._clear_output()

    def _on_select_all(self, event):
        """Handle Ctrl+A to select all."""
        self.task_listbox.select_set(0, tk.END)
        return "break"

    def _on_shift_up(self, event):
        """Handle Shift+Up for extending selection."""
        # Let default behavior handle it
        pass

    def _on_shift_down(self, event):
        """Handle Shift+Down for extending selection."""
        # Let default behavior handle it
        pass

    # Helper methods

    def _get_selected_tasks(self) -> List[Task]:
        """Get currently selected tasks."""
        selected_indices = self.task_listbox.curselection()
        tasks = self.app_state.get_tasks(self.current_environment)
        return [tasks[i] for i in selected_indices if i < len(tasks)]

    def _refresh_task_list(self):
        """Refresh the task list display."""
        self.task_listbox.delete(0, tk.END)

        tasks = self.app_state.get_tasks(self.current_environment)
        for task in tasks:
            self.task_listbox.insert(tk.END, task.get_display_name())

    def _update_concurrency_display(self):
        """Update concurrency display for current environment."""
        env = self.app_state.environments.get(self.current_environment)
        if env:
            self.concurrency_var.set(str(env.concurrency_limit))

    def _display_task_output(self, task: Task):
        """Display output for a task."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)

        # Show task info
        self.output_text.insert(tk.END, f"Task: {task.id[:8]}\n", "header")
        self.output_text.insert(tk.END, f"Command: {task.command}\n", "header")
        self.output_text.insert(tk.END, f"State: {task.state.value}\n", "header")
        if task.tmux_session:
            self.output_text.insert(tk.END, f"Session: {task.tmux_session}\n", "header")
        self.output_text.insert(tk.END, "\n" + "="*80 + "\n\n")

        # Show output
        for line in task.output_buffer:
            self.output_text.insert(tk.END, line + "\n")

        self.output_text.config(state=tk.DISABLED)

        # Scroll to bottom
        self.output_text.see(tk.END)

    def _clear_output(self):
        """Clear the output display."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)

    def _schedule_update(self):
        """Schedule GUI update loop."""
        self._process_messages()
        self.root.after(100, self._schedule_update)

    def _process_messages(self):
        """Process messages from background threads."""
        try:
            while True:
                message = self.message_queue.get_nowait()

                msg_type = message.get('type')

                if msg_type == 'task_started':
                    self._refresh_task_list()
                    self.status_var.set(f"Task started: {message.get('task_id', '')[:8]}")

                elif msg_type == 'task_completed':
                    self._refresh_task_list()
                    task_id = message.get('task_id', '')[:8]
                    state = message.get('state', '')
                    self.status_var.set(f"Task {task_id} {state.value if hasattr(state, 'value') else state}")

                elif msg_type == 'task_failed':
                    self._refresh_task_list()
                    self.status_var.set(f"Task failed: {message.get('task_id', '')[:8]}")

                elif msg_type == 'task_killed':
                    self._refresh_task_list()
                    self.status_var.set(f"Task killed: {message.get('task_id', '')[:8]}")

                elif msg_type == 'output_update':
                    # Refresh output if this task is currently selected
                    selected = self._get_selected_tasks()
                    if len(selected) == 1 and selected[0].id == message.get('task_id'):
                        self._display_task_output(selected[0])

        except queue.Empty:
            pass

    def run(self):
        """Start the GUI main loop."""
        # Initial refresh
        self._refresh_task_list()
        self._update_concurrency_display()

        # Run main loop
        self.root.mainloop()

    def destroy(self):
        """Destroy the GUI window."""
        self.root.destroy()
