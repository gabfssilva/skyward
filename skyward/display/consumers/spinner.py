"""Spinner consumer that renders animated single-line status."""

from __future__ import annotations

import logging
import sys
import threading
from typing import TYPE_CHECKING

from skyward.events import (
    BootstrapCompleted,
    BootstrapProgress,
    BootstrapStarting,
    CostUpdate,
    Error,
    InfraCreated,
    InfraCreating,
    InstanceLaunching,
    InstanceProvisioned,
    InstanceStopping,
    LogLine,
    Metrics,
    PoolStarted,
    PoolStopping,
    SkywardEvent,
)

if TYPE_CHECKING:
    from skyward.pool import ComputePool


class SpinnerConsumer:
    """Consumer that renders animated spinner with status line.

    Uses carriage return for in-place updates. Falls back to
    simple line output in non-TTY environments.
    """

    SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    SPINNER_INTERVAL = 0.08  # seconds between frames

    CHECKMARK = "✓"
    CROSS = "✗"

    def __init__(self, pool: ComputePool) -> None:
        self._is_tty = sys.stdout.isatty()

        # State tracking
        self._phase: str = "init"
        self._message: str = "Initializing..."
        self._instance_counts: dict[str, int] = {
            "total": 0,
            "provisioned": 0,
            "ready": 0,
            "spot": 0,
            "ondemand": 0,
        }
        # Metrics tracking (node -> {cpu, gpu, mem, gpu_mem})
        self._node_metrics: dict[int, dict[str, float | None]] = {}
        # Instance info tracking
        self._instance_types: set[str] = set()
        # Cost and time tracking
        self._cost: float = 0.0
        self._elapsed_seconds: float = 0.0
        # Table rendering
        self._table_lines: int = 0

        # Threading
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._spinner_thread: threading.Thread | None = None
        self._frame_idx = 0
        self._last_line_len = 0

        # Logging suppression
        self._saved_root_level: int = logging.NOTSET
        self._saved_skyward_level: int = logging.NOTSET

        # Pause state
        self._paused = False

        # Register event handlers
        @pool.on(PoolStarted)
        def on_start(event: PoolStarted) -> None:
            self._do_start()

        @pool.on(PoolStopping)
        def on_stop(event: PoolStopping) -> None:
            self._do_stop()

        @pool.on()  # wildcard - all events
        def on_event(event: SkywardEvent) -> None:
            # Skip lifecycle events (handled separately)
            if isinstance(event, (PoolStarted, PoolStopping)):
                return
            self._handle(event)

    def _do_start(self) -> None:
        """Start the spinner animation thread."""
        # Suppress all logging during animation
        self._suppress_logging()

        if not self._is_tty:
            return

        self._stop_event.clear()
        self._spinner_thread = threading.Thread(
            target=self._animate,
            daemon=True,
            name="SpinnerConsumer-animation",
        )
        self._spinner_thread.start()

    def _do_stop(self) -> None:
        """Stop spinner and print final status."""
        if not self._paused:
            self._stop_event.set()

            if self._spinner_thread and self._spinner_thread.is_alive():
                self._spinner_thread.join(timeout=1.0)

            self._print_final()

        # Always restore logging at the end
        self._restore_logging()

    def _handle(self, event: SkywardEvent) -> None:
        """Process event and update state."""
        # Handle LogLine separately - print and return
        if isinstance(event, LogLine):
            self._print_log_line(event.node, event.line)
            return

        should_pause = False
        should_resume = False

        with self._lock:
            self._update_state(event)
            # Check if we should pause after cluster ready
            if self._phase == "done" and not self._paused:
                should_pause = True
            # Check if we should resume for metrics display
            if isinstance(event, Metrics) and self._paused:
                should_resume = True

        # Pause outside the lock to avoid deadlock
        if should_pause:
            self._pause()

        # Resume for metrics display
        if should_resume:
            self._resume()

        # Non-TTY: print each event (only if not paused)
        if not self._is_tty and not self._paused:
            event_name = type(event).__name__
            print(f"[{event_name}] {self._message}")

    def _update_state(self, event: SkywardEvent) -> None:
        """Update internal state based on event. Must hold lock."""
        match event:
            # === Provision Phase ===
            case InfraCreating():
                self._phase = "provision"
                self._message = "Provisioning infrastructure..."

            case InfraCreated(region=region):
                self._message = f"Infrastructure ready ({region})"

            case InstanceLaunching(count=count, instance_type=itype):
                if count:
                    self._instance_counts["total"] = count
                self._message = f"Launching {count} x {itype}..."

            case InstanceProvisioned(
                instance_id=instance_id, spot=is_spot, instance_type=itype
            ):
                self._instance_counts["provisioned"] += 1
                if is_spot:
                    self._instance_counts["spot"] += 1
                else:
                    self._instance_counts["ondemand"] += 1
                if itype:
                    self._instance_types.add(itype)
                p = self._instance_counts["provisioned"]
                t = self._instance_counts["total"] or "?"
                spot_label = "[spot]" if is_spot else "[on-demand]"
                self._message = f"Provisioned {self._short_id(instance_id)} [{p}/{t}] {spot_label}"

            # === Setup Phase ===
            case BootstrapStarting(instance_id=instance_id):
                self._phase = "setup"
                r = self._instance_counts["ready"]
                t = self._instance_counts["total"] or "?"
                self._message = f"Bootstrapping {self._short_id(instance_id)}... ({r}/{t})"

            case BootstrapProgress(instance_id=instance_id, step=step):
                r = self._instance_counts["ready"]
                t = self._instance_counts["total"] or "?"
                self._message = f"Bootstrapping {self._short_id(instance_id)} ({step}) ({r}/{t})"

            case BootstrapCompleted(instance_id=instance_id):
                self._instance_counts["ready"] += 1
                r = self._instance_counts["ready"]
                t = self._instance_counts["total"] or self._instance_counts["provisioned"]
                self._message = f"Instance {self._short_id(instance_id)} ready ({r}/{t})"

                # Check if all instances ready
                if r >= t > 0:
                    self._phase = "done"
                    s = self._instance_counts["spot"]
                    od = self._instance_counts["ondemand"]
                    if s > 0 and od > 0:
                        market_info = f" [{s} spot, {od} on-demand]"
                    elif s > 0:
                        market_info = " [spot]"
                    else:
                        market_info = " [on-demand]"
                    self._message = f"Cluster ready ({t}/{t}){market_info}"

            # === Execution Phase (Metrics) ===
            case Metrics(
                node=node,
                cpu_percent=cpu,
                gpu_utilization=gpu,
                memory_percent=mem,
                gpu_memory_used_mb=gpu_mem,
                gpu_memory_total_mb=gpu_mem_total,
            ):
                gpu_mem_pct = None
                if gpu_mem is not None and gpu_mem_total:
                    gpu_mem_pct = (gpu_mem / gpu_mem_total) * 100
                self._node_metrics[node] = {
                    "cpu": cpu,
                    "gpu": gpu,
                    "mem": mem,
                    "gpu_mem": gpu_mem_pct,
                }
                # Message will be built by _build_table() in _animate()

            # === Cost Update ===
            case CostUpdate(accumulated_cost=cost, elapsed_seconds=elapsed):
                self._cost = cost
                self._elapsed_seconds = elapsed

            # === Shutdown Phase ===
            case InstanceStopping(instance_id=instance_id):
                self._phase = "shutdown"
                self._message = f"Stopping {self._short_id(instance_id)}..."

            # === Error ===
            case Error(message=msg):
                self._phase = "error"
                self._message = f"Error: {msg[:50]}"

    def _build_table(self) -> str:
        """Build metrics table string for multi-line display."""
        # Header line: Nx instance_type [spot info] | uptime | cost
        types_str = "/".join(sorted(self._instance_types)) or "unknown"
        n = len(self._node_metrics) or self._instance_counts["total"]
        spot = self._instance_counts["spot"]

        mins, secs = divmod(int(self._elapsed_seconds), 60)
        time_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
        cost_str = f"${self._cost:.4f}" if self._cost > 0 else "$0.0000"

        spot_str = f"[{spot} spot]" if spot else "[on-demand]"
        header = f"{n}x {types_str} {spot_str} | {time_str} | {cost_str}"

        # Table rows
        lines = [header, ""]
        lines.append("Node   CPU    GPU    Mem    GPU Mem")

        for node in sorted(self._node_metrics.keys()):
            m = self._node_metrics[node]
            cpu = f"{m['cpu']:.0f}%" if m["cpu"] is not None else "N/A"
            gpu = f"{m['gpu']:.0f}%" if m["gpu"] is not None else "N/A"
            mem = f"{m['mem']:.0f}%" if m["mem"] is not None else "N/A"
            gpu_mem = f"{m['gpu_mem']:.0f}%" if m["gpu_mem"] is not None else "N/A"
            lines.append(f"  {node}    {cpu:>4}   {gpu:>4}   {mem:>4}     {gpu_mem:>4}")

        return "\n".join(lines)

    def _animate(self) -> None:
        """Background thread that updates spinner animation."""
        while not self._stop_event.is_set():
            with self._lock:
                frame = self.SPINNER_FRAMES[self._frame_idx]
                phase = self._phase
                node_metrics = self._node_metrics
                message = self._message

            if phase == "executing" and node_metrics:
                # Multi-line table mode
                with self._lock:
                    table = self._build_table()
                n_lines = table.count("\n") + 1
                # Move cursor up and clear previous lines
                if self._table_lines > 0:
                    sys.stdout.write(f"\033[{self._table_lines}A\033[J")
                sys.stdout.write(f"{frame} {table}\n")
                sys.stdout.flush()
                self._table_lines = n_lines
            else:
                # Single line mode
                line = f"\r{frame} {message}"
                padding = max(0, self._last_line_len - len(line))
                sys.stdout.write(line + " " * padding)
                sys.stdout.flush()
                self._last_line_len = len(line)

            self._frame_idx = (self._frame_idx + 1) % len(self.SPINNER_FRAMES)
            self._stop_event.wait(self.SPINNER_INTERVAL)

    def _print_final(self) -> None:
        """Print final status line (non-animated)."""
        with self._lock:
            symbol = self.CROSS if self._phase == "error" else self.CHECKMARK
            message = self._message

        if self._is_tty:
            padding = max(0, self._last_line_len - len(message) - 2)
            sys.stdout.write(f"\r{symbol} {message}" + " " * padding + "\n")
        else:
            print(f"{symbol} {message}")

        sys.stdout.flush()

    def _short_id(self, instance_id: str) -> str:
        """Shorten instance ID for display."""
        if len(instance_id) > 12:
            return instance_id[:12]
        return instance_id

    def _pause(self) -> None:
        """Pause spinner for execution phase (keep logging suppressed)."""
        if self._paused:
            return

        # Stop animation
        self._stop_event.set()
        if self._spinner_thread and self._spinner_thread.is_alive():
            self._spinner_thread.join(timeout=1.0)

        # Print final status for this phase
        if self._is_tty:
            padding = max(0, self._last_line_len - len(self._message) - 2)
            sys.stdout.write(f"\r{self.CHECKMARK} {self._message}" + " " * padding + "\n")
            sys.stdout.flush()

        # Keep logging suppressed - only restore in stop()
        self._paused = True

    def _resume(self) -> None:
        """Resume spinner for execution phase with metrics display."""
        if not self._paused or not self._is_tty:
            return

        self._paused = False
        self._phase = "executing"
        self._stop_event.clear()
        self._spinner_thread = threading.Thread(
            target=self._animate,
            daemon=True,
            name="SpinnerConsumer-animation",
        )
        self._spinner_thread.start()

    def _suppress_logging(self) -> None:
        """Suppress all logging during spinner animation."""
        # Suppress root logger
        root = logging.getLogger()
        self._saved_root_level = root.level
        root.setLevel(logging.CRITICAL + 1)

        # Suppress skyward logger (has its own handler)
        skyward_logger = logging.getLogger("skyward")
        self._saved_skyward_level = skyward_logger.level
        skyward_logger.setLevel(logging.CRITICAL + 1)


    def _restore_logging(self) -> None:
        """Restore logging after spinner stops."""
        root = logging.getLogger()
        root.setLevel(self._saved_root_level)

        skyward_logger = logging.getLogger("skyward")
        skyward_logger.setLevel(self._saved_skyward_level)

    def _print_log_line(self, node: int, line: str) -> None:
        """Print log line from remote instance above the spinner/table."""
        if line.strip() == "":
            return

        if not self._is_tty:
            print(f"[node {node}] {line}")
            return

        # Clear current display (table or single line)
        if self._table_lines > 0:
            # Move cursor up and clear all table lines
            sys.stdout.write(f"\033[{self._table_lines}A\033[J")
            self._table_lines = 0  # Reset so table redraws fresh
        else:
            sys.stdout.write("\r" + " " * self._last_line_len + "\r")

        # Print log line
        print(f"[node {node}] {line}")
        sys.stdout.flush()
