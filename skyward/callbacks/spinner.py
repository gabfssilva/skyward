"""Spinner callback using yaspin for animated terminal UI."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import singledispatchmethod
from io import TextIOBase
from typing import TYPE_CHECKING, TextIO

from termcolor import colored
from yaspin import yaspin
from yaspin.core import Yaspin

from skyward.events import (
    BootstrapCompleted,
    BootstrapProgress,
    CostUpdate,
    Error,
    InstanceLaunching,
    InstanceProvisioned,
    InstanceReady,
    InstanceStopping,
    LogLine,
    Metrics,
    NetworkReady,
    PoolReady,
    PoolStarted,
    PoolStopping,
    ProvisioningStarted,
    SkywardEvent,
)

if TYPE_CHECKING:
    from skyward.callback import Callback

# Thresholds for color coding (low, high)
_CPU_THRESHOLDS = (60, 85)
_GPU_THRESHOLDS = (70, 90)
_MEM_THRESHOLDS = (70, 90)


class _SpinnerStream(TextIOBase):
    """Stream wrapper that redirects output through yaspin.write()."""

    def __init__(self, sp: Yaspin, original: TextIO) -> None:
        self._sp = sp
        self._original = original
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0

        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._sp.write(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._sp.write(self._buffer)
        self._buffer = ""

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return self._original.isatty()


@dataclass(slots=True)
class _Tracking:
    """Pure data for tracking cluster state."""

    phase: str = "init"
    total: int = 0
    provisioned: int = 0
    ready: int = 0
    spot: int = 0
    ondemand: int = 0
    node_metrics: dict[int, dict[str, float | None]] = field(default_factory=dict)
    instance_types: set[str] = field(default_factory=set)
    cost: float = 0.0
    elapsed_seconds: float = 0.0


def _short_id(instance_id: str) -> str:
    return instance_id[:12] if len(instance_id) > 12 else instance_id


def _color_metric(
    value: float | None,
    label: str,
    thresholds: tuple[float, float],
    *,
    invert: bool = False,
) -> str:
    """Color a metric based on thresholds. Green=good, green=warn, Red=bad."""
    if value is None:
        return f"{label} --"

    low, high = thresholds
    if invert:
        color = "green" if value >= high else ("green" if value >= low else "red")
    else:
        color = "green" if value < low else ("green" if value < high else "red")

    return colored(f"{label} {value:.0f}%", color)


def _avg_metrics(node_metrics: dict[int, dict[str, float | None]]) -> dict[str, float | None]:
    """Calculate average metrics across all nodes."""
    if not node_metrics:
        return {"cpu": None, "gpu": None, "mem": None, "gpu_mem": None}

    keys = ("cpu", "gpu", "mem", "gpu_mem")
    totals = dict.fromkeys(keys, 0.0)
    counts = dict.fromkeys(keys, 0)

    for m in node_metrics.values():
        for k in keys:
            if (v := m.get(k)) is not None:
                totals[k] += v
                counts[k] += 1

    return {k: (totals[k] / counts[k] if counts[k] > 0 else None) for k in keys}


def _format_metrics(tracking: _Tracking) -> str:
    """Build colored metrics text."""
    avg = _avg_metrics(tracking.node_metrics)
    parts: list[str] = []

    n = len(tracking.node_metrics) or tracking.total
    if n > 1:
        parts.append(colored(f"{n} nodes", "blue", attrs=["bold"]))

    if avg["cpu"] is not None:
        parts.append(_color_metric(avg["cpu"], "CPU", _CPU_THRESHOLDS))
    if avg["gpu"] is not None:
        parts.append(_color_metric(avg["gpu"], "GPU", _GPU_THRESHOLDS, invert=True))
    if avg["mem"] is not None:
        parts.append(_color_metric(avg["mem"], "Mem", _MEM_THRESHOLDS))

    if tracking.cost > 0:
        mins, secs = divmod(int(tracking.elapsed_seconds), 60)
        time_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
        parts.append(colored(f"${tracking.cost:.4f}", "green") + f"/{time_str}")

    return " │ ".join(parts) if parts else colored("Executing...", "blue")


class SpinnerController:
    """Controls the yaspin spinner lifecycle and display."""

    def __init__(self) -> None:
        self._sp: Yaspin | None = None
        self._tracking = _Tracking()
        self._is_tty = sys.stdout.isatty()
        self._original_stdout: TextIO | None = None
        self._original_stderr: TextIO | None = None

    def _set_text(self, text: str) -> None:
        if self._sp:
            self._sp.text = text

    def _redirect_streams(self) -> None:
        if not self._sp or not self._is_tty:
            return
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = _SpinnerStream(self._sp, self._original_stdout)  # type: ignore[assignment]
        sys.stderr = _SpinnerStream(self._sp, self._original_stderr)  # type: ignore[assignment]

    def _restore_streams(self) -> None:
        if self._original_stdout:
            sys.stdout = self._original_stdout
            self._original_stdout = None
        if self._original_stderr:
            sys.stderr = self._original_stderr
            self._original_stderr = None

    @singledispatchmethod
    def handle(self, event: SkywardEvent) -> None:
        """Handle unknown events - ignore."""

    @handle.register
    def _(self, event: PoolStarted) -> None:
        self._sp = yaspin(text="Initializing...", color="blue")
        self._sp.start()
        self._redirect_streams()

    @handle.register
    def _(self, event: PoolStopping) -> None:
        self._restore_streams()
        if self._sp:
            if self._tracking.phase == "error":
                self._sp.fail(colored("✗", "red"))
            else:
                self._sp.ok(colored("✓", "green"))
            self._sp = None

    @handle.register
    def _(self, event: LogLine) -> None:
        if event.line.strip() and self._sp:
            prefix = colored(f"[node {event.instance.node}]", "blue")
            self._sp.write(f"{prefix} {event.line}")

    @handle.register
    def _(self, event: ProvisioningStarted) -> None:
        self._tracking.phase = "provision"
        self._set_text(colored("Starting provisioning...", "blue"))

    @handle.register
    def _(self, event: NetworkReady) -> None:
        self._set_text(colored("Network ready ", "green") + f"({event.region})")

    @handle.register
    def _(self, event: InstanceLaunching) -> None:
        if event.count:
            self._tracking.total = event.count
        itypes = [c.name for c in event.candidates] if event.candidates else []

        main_types = ', '.join(itypes[:3])

        suffix = f"candidates are {main_types}"

        if len(itypes) > 5:
            suffix = f"{suffix} and other {len(itypes) - 3}"

        text = (
            colored(f"Attempting to launch {event.count} nodes ({suffix})", "blue")
            + f"({itypes[0]} +{len(itypes) - 1})"
        )

        self._set_text(text)

    @handle.register
    def _(self, event: InstanceProvisioned) -> None:
        t = self._tracking
        inst = event.instance
        t.provisioned += 1
        if inst.spot:
            t.spot += 1
        else:
            t.ondemand += 1
        if inst.spec:
            t.instance_types.add(inst.spec.name)

        spot_color = "green" if inst.spot else "blue"
        spot_label = "spot" if inst.spot else "on-demand"

        self._set_text(
            colored("Provisioned ", "green")
            + colored(_short_id(inst.instance_id), attrs=["bold"])
            + f" [{inst.spec.name} {t.provisioned}/{t.total or '?'}] "
            + colored(f"[{spot_label}]", spot_color)
        )

    @handle.register
    def _(self, event: BootstrapProgress) -> None:
        t = self._tracking
        t.phase = "setup"
        self._set_text(
            colored("Bootstrapping ", "blue")
            + colored(_short_id(event.instance.instance_id), attrs=["bold"])
            + colored(f" ({event.step}) ", "blue")
            + f"({t.ready}/{t.total or '?'})"
        )

    @handle.register
    def _(self, event: BootstrapCompleted) -> None:
        t = self._tracking
        t.ready += 1
        total = t.total or t.provisioned

        if t.ready >= total > 0:
            t.phase = "executing"
            if t.spot > 0 and t.ondemand > 0:
                market = (
                    colored(f"[{t.spot} spot", "green")
                    + ", "
                    + colored(f"{t.ondemand} on-demand]", "blue")
                )
            elif t.spot > 0:
                market = colored("[spot] ", "green")
            else:
                market = colored("[on-demand] ", "blue")
            self._set_text(
                colored("Cluster ready ", "green", attrs=["bold"])
                + f"({total}/{total}) "
                + market
                + colored(f"[{event.instance.spec.name}] ", "blue")
            )
        else:
            self._set_text(
                colored("Instance ", "green")
                + colored(_short_id(event.instance.instance_id), attrs=["bold"])
                + colored(" ready ", "green")
                + f"({t.ready}/{total})"
            )

    @handle.register
    def _(self, event: Metrics) -> None:
        t = self._tracking
        gpu_mem_pct = None
        if event.gpu_memory_used_mb is not None and event.gpu_memory_total_mb:
            gpu_mem_pct = (event.gpu_memory_used_mb / event.gpu_memory_total_mb) * 100
        t.node_metrics[event.instance.node] = {
            "cpu": event.cpu_percent,
            "gpu": event.gpu_utilization,
            "mem": event.memory_percent,
            "gpu_mem": gpu_mem_pct,
        }
        self._set_text(_format_metrics(t))

    @handle.register
    def _(self, event: CostUpdate) -> None:
        t = self._tracking
        t.cost = event.accumulated_cost
        t.elapsed_seconds = event.elapsed_seconds
        if t.node_metrics:
            self._set_text(_format_metrics(t))

    @handle.register
    def _(self, event: InstanceReady) -> None:
        inst = event.instance
        spot_label = "spot" if inst.spot else "on-demand"
        spot_color = "green" if inst.spot else "blue"
        self._set_text(
            colored("Instance ", "green")
            + colored(_short_id(inst.instance_id), attrs=["bold"])
            + colored(" ready ", "green")
            + colored(f"[{spot_label}]", spot_color)
        )

    @handle.register
    def _(self, event: PoolReady) -> None:
        t = self._tracking
        t.phase = "executing"
        spot_count = sum(1 for i in event.instances if i.spot)
        ondemand_count = len(event.instances) - spot_count
        if spot_count > 0 and ondemand_count > 0:
            market = (
                colored(f"[{spot_count} spot", "green")
                + ", "
                + colored(f"{ondemand_count} on-demand]", "blue")
            )
        elif spot_count > 0:
            market = colored("[spot]", "green")
        else:
            market = colored("[on-demand]", "blue")
        self._set_text(
            colored("Pool ready ", "green", attrs=["bold"])
            + f"({len(event.instances)} instances, {event.connections} connections) "
            + market
        )

    @handle.register
    def _(self, event: InstanceStopping) -> None:
        self._tracking.phase = "shutdown"
        self._set_text(
            colored("Stopping ", "green")
            + colored(_short_id(event.instance.instance_id), attrs=["bold"])
        )

    @handle.register
    def _(self, event: Error) -> None:
        self._tracking.phase = "error"
        self._set_text(
            colored("Error: ", "red", attrs=["bold"]) + colored(event.message[:50], "red")
        )


def spinner() -> Callback:
    """Create a spinner callback for animated terminal UI.

    The spinner shows progress during provisioning and setup,
    then displays aggregated metrics during execution.

    Features:
        - Color-coded metrics (green/green/red based on thresholds)
        - Redirects stdout/stderr to print above spinner
        - Handles log lines from remote nodes

    Returns:
        A callback that renders animated spinner/progress.

    Example:
        callback = compose(cost_tracker(), spinner())
        with use_callback(callback):
            emit(PoolStarted())
            # ... spinner animates ...
            emit(PoolStopping())
    """
    return SpinnerController().handle
