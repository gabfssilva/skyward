"""Spinner callback that renders animated terminal UI."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from skyward.events import (
    BootstrapCompleted,
    BootstrapProgress,
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
    from skyward.callback import Callback

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
SPINNER_INTERVAL = 0.08
CHECKMARK = "✓"
CROSS = "✗"


@dataclass
class _SpinnerState:
    """Internal state for spinner animation."""

    is_tty: bool = field(default_factory=lambda: sys.stdout.isatty())
    phase: str = "init"
    message: str = "Initializing..."

    # Instance tracking
    total: int = 0
    provisioned: int = 0
    ready: int = 0
    spot: int = 0
    ondemand: int = 0

    # Metrics
    node_metrics: dict[int, dict[str, float | None]] = field(default_factory=dict)
    instance_types: set[str] = field(default_factory=set)

    # Cost/time
    cost: float = 0.0
    elapsed_seconds: float = 0.0

    # Animation
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    frame_idx: int = 0
    last_line_len: int = 0
    table_lines: int = 0
    paused: bool = False

    def short_id(self, instance_id: str) -> str:
        return instance_id[:12] if len(instance_id) > 12 else instance_id


def _start_animation(state: _SpinnerState, lock: threading.Lock) -> None:
    """Start the spinner animation thread."""
    if not state.is_tty:
        return

    state.stop_event.clear()
    state.thread = threading.Thread(
        target=_animate,
        args=(state, lock),
        daemon=True,
        name="SpinnerCallback-animation",
    )
    state.thread.start()


def _stop_animation(state: _SpinnerState) -> None:
    """Stop the spinner animation thread."""
    state.stop_event.set()
    if state.thread and state.thread.is_alive():
        state.thread.join(timeout=1.0)


def _animate(state: _SpinnerState, lock: threading.Lock) -> None:
    """Background animation loop."""
    while not state.stop_event.is_set():
        with lock:
            frame = SPINNER_FRAMES[state.frame_idx]
            phase = state.phase
            message = state.message
            node_metrics = dict(state.node_metrics)

        if phase == "executing" and node_metrics:
            # Multi-line table mode
            with lock:
                table = _build_table(state)
            n_lines = table.count("\n") + 1
            if state.table_lines > 0:
                sys.stdout.write(f"\033[{state.table_lines}A\033[J")
            sys.stdout.write(f"{frame} {table}\n")
            sys.stdout.flush()
            state.table_lines = n_lines
        else:
            # Single line mode
            line = f"\r{frame} {message}"
            padding = max(0, state.last_line_len - len(line))
            sys.stdout.write(line + " " * padding)
            sys.stdout.flush()
            state.last_line_len = len(line)

        state.frame_idx = (state.frame_idx + 1) % len(SPINNER_FRAMES)
        state.stop_event.wait(SPINNER_INTERVAL)


def _build_table(state: _SpinnerState) -> str:
    """Build metrics table for display."""
    types_str = "/".join(sorted(state.instance_types)) or "unknown"
    n = len(state.node_metrics) or state.total

    mins, secs = divmod(int(state.elapsed_seconds), 60)
    time_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
    cost_str = f"${state.cost:.4f}" if state.cost > 0 else "$0.0000"

    spot_str = f"[{state.spot} spot]" if state.spot else "[on-demand]"
    header = f"{n}x {types_str} {spot_str} | {time_str} | {cost_str}"

    lines = [header, "", "Node   CPU    GPU    Mem    GPU Mem"]

    for node in sorted(state.node_metrics.keys()):
        m = state.node_metrics[node]
        cpu = f"{m['cpu']:.0f}%" if m["cpu"] is not None else "N/A"
        gpu = f"{m['gpu']:.0f}%" if m["gpu"] is not None else "N/A"
        mem = f"{m['mem']:.0f}%" if m["mem"] is not None else "N/A"
        gpu_mem = f"{m['gpu_mem']:.0f}%" if m["gpu_mem"] is not None else "N/A"
        lines.append(f"  {node}    {cpu:>4}   {gpu:>4}   {mem:>4}     {gpu_mem:>4}")

    return "\n".join(lines)


def _print_final(state: _SpinnerState) -> None:
    """Print final status line."""
    symbol = CROSS if state.phase == "error" else CHECKMARK

    if state.is_tty:
        padding = max(0, state.last_line_len - len(state.message) - 2)
        sys.stdout.write(f"\r{symbol} {state.message}" + " " * padding + "\n")
    else:
        print(f"{symbol} {state.message}")

    sys.stdout.flush()


def _print_log_line(state: _SpinnerState, node: int, line: str) -> None:
    """Print log line above the spinner."""
    if not line.strip():
        return

    if not state.is_tty:
        print(f"[node {node}] {line}")
        return

    # Clear current display
    if state.table_lines > 0:
        sys.stdout.write(f"\033[{state.table_lines}A\033[J")
        state.table_lines = 0
    else:
        sys.stdout.write("\r" + " " * state.last_line_len + "\r")

    print(f"[node {node}] {line}")
    sys.stdout.flush()


def _update_state(state: _SpinnerState, event: SkywardEvent) -> None:
    """Update spinner state based on event."""
    match event:
        case InfraCreating():
            state.phase = "provision"
            state.message = "Provisioning infrastructure..."

        case InfraCreated(region=region):
            state.message = f"Infrastructure ready ({region})"

        case InstanceLaunching(count=count, instance_type=itype):
            if count:
                state.total = count
            state.message = f"Launching {count}x {itype}..."

        case InstanceProvisioned(instance_id=iid, spot=is_spot, instance_type=itype):
            state.provisioned += 1
            if is_spot:
                state.spot += 1
            else:
                state.ondemand += 1
            if itype:
                state.instance_types.add(itype)
            spot_label = "[spot]" if is_spot else "[on-demand]"
            state.message = (
                f"Provisioned {state.short_id(iid)} "
                f"[{state.provisioned}/{state.total or '?'}] {spot_label}"
            )

        case BootstrapProgress(instance_id=iid, step=step):
            state.phase = "setup"
            state.message = (
                f"Bootstrapping {state.short_id(iid)} ({step}) "
                f"({state.ready}/{state.total or '?'})"
            )

        case BootstrapCompleted(instance_id=iid):
            state.ready += 1
            total = state.total or state.provisioned
            state.message = (
                f"Instance {state.short_id(iid)} ready "
                f"({state.ready}/{total})"
            )

            if state.ready >= total > 0:
                state.phase = "done"
                if state.spot > 0 and state.ondemand > 0:
                    market = f" [{state.spot} spot, {state.ondemand} on-demand]"
                elif state.spot > 0:
                    market = " [spot]"
                else:
                    market = " [on-demand]"
                state.message = f"Cluster ready ({total}/{total}){market}"

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
            state.node_metrics[node] = {
                "cpu": cpu,
                "gpu": gpu,
                "mem": mem,
                "gpu_mem": gpu_mem_pct,
            }

        case CostUpdate(accumulated_cost=cost, elapsed_seconds=elapsed):
            state.cost = cost
            state.elapsed_seconds = elapsed

        case InstanceStopping(instance_id=iid):
            state.phase = "shutdown"
            state.message = f"Stopping {state.short_id(iid)}..."

        case Error(message=msg):
            state.phase = "error"
            state.message = f"Error: {msg[:50]}"


def spinner() -> Callback:
    """Create a spinner callback for animated terminal UI.

    The spinner shows progress during provisioning and setup,
    then displays a metrics table during execution.

    Returns:
        A callback that renders animated spinner/progress.

    Example:
        callback = compose(cost_tracker(), spinner())
        with use_callback(callback):
            emit(PoolStarted())
            # ... spinner animates ...
            emit(PoolStopping())
    """
    state = _SpinnerState()
    lock = threading.Lock()

    def handle(event: SkywardEvent) -> None:
        # Handle LogLine separately (prints above spinner)
        if isinstance(event, LogLine):
            _print_log_line(state, event.node, event.line)
            return

        should_pause = False
        should_resume = False

        with lock:
            match event:
                case PoolStarted():
                    _start_animation(state, lock)
                    return

                case PoolStopping():
                    if not state.paused:
                        _stop_animation(state)
                        _print_final(state)
                    return

                case _:
                    _update_state(state, event)

                    # Check if we should pause after cluster ready
                    if state.phase == "done" and not state.paused:
                        should_pause = True

                    # Check if we should resume for metrics
                    if isinstance(event, Metrics) and state.paused:
                        should_resume = True

        # Pause/resume outside lock
        if should_pause:
            _stop_animation(state)
            if state.is_tty:
                padding = max(0, state.last_line_len - len(state.message) - 2)
                sys.stdout.write(
                    f"\r{CHECKMARK} {state.message}" + " " * padding + "\n"
                )
                sys.stdout.flush()
            state.paused = True

        if should_resume:
            state.paused = False
            state.phase = "executing"
            _start_animation(state, lock)

        # Non-TTY fallback
        if not state.is_tty and not state.paused:
            event_name = type(event).__name__
            print(f"[{event_name}] {state.message}")

    return handle
