"""Metrics polling for instances."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from skyward.events import Metrics, ProviderName, ProvisionedInstance

if TYPE_CHECKING:
    from skyward.callback import Callback
    from skyward.types import Instance


class MetricsPoller:
    """Polls metrics from instances via instance.metrics().

    Uses a background thread to periodically collect metrics and emit
    events via the callback system.
    """

    def __init__(
        self,
        instance: Instance,
        callback: Callback | None = None,
        interval: float = 2.0,
        provider_name: ProviderName = ProviderName.AWS,
    ) -> None:
        self.instance = instance
        self.callback = callback
        self.interval = interval
        self.provider_name = provider_name
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Cache ProvisionedInstance
        self._provisioned = ProvisionedInstance(
            instance_id=instance.id,
            node=instance.node,
            provider=provider_name,
            spot=instance.spot,
            spec=None,
            ip=instance.private_ip,
        )

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
        if self.callback is None:
            return

        self.callback(
            Metrics(
                instance=self._provisioned,
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
