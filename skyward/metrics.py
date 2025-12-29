"""Metrics polling for instances."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from skyward.callback import emit
from skyward.events import Metrics

if TYPE_CHECKING:
    from skyward.types import Instance


class MetricsPoller:
    """Polls metrics from instances via instance.metrics().

    Uses a background thread to periodically collect metrics and emit
    events via the callback system.
    """

    def __init__(
        self,
        instance: Instance,
        interval: float = 2.0,
    ) -> None:
        self.instance = instance
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start polling in background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop polling."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 1)

    def _poll_loop(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            try:
                data = self.instance.metrics()
                self._emit_metrics(data)
            except Exception:
                pass  # Metrics collection failures are not critical
            self._stop_event.wait(self.interval)

    def _emit_metrics(self, data: dict[str, Any]) -> None:
        """Convert raw data to Metrics event and emit."""
        emit(
            Metrics(
                instance_id=self.instance.id,
                node=self.instance.node,
                cpu_percent=data.get("cpu_percent", 0.0),
                memory_percent=data.get("memory_percent", 0.0),
                memory_used_mb=data.get("memory_used_mb", 0.0),
                memory_total_mb=data.get("memory_total_mb", 0.0),
                gpu_utilization=data.get("gpu_utilization"),
                gpu_memory_used_mb=data.get("gpu_memory_used_mb"),
                gpu_memory_total_mb=data.get("gpu_memory_total_mb"),
                gpu_temperature=data.get("gpu_temperature"),
            )
        )


@contextlib.contextmanager
def metrics_polling(
    instances: tuple[Instance, ...],
) -> Iterator[None]:
    """Context manager that polls metrics during execution.

    Args:
        instances: Instances to poll metrics from.

    Example:
        with metrics_polling(instances):
            result = run_function_on_instances(...)
    """
    pollers: list[MetricsPoller] = []

    for instance in instances:
        poller = MetricsPoller(instance)
        poller.start()
        pollers.append(poller)

    try:
        yield
    finally:
        for poller in pollers:
            poller.stop()
