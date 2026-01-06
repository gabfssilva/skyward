"""Metrics streaming from instances via RPyC generator polling.

This module provides metrics streaming using RPyC generators.
The client polls the server at the configured interval.
"""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING, Any

from skyward.events import Metrics, ProviderName, ProvisionedInstance

if TYPE_CHECKING:
    from rpyc import Connection

    from skyward.callback import Callback
    from skyward.types import Instance


class MetricsStreamer:
    """Streams metrics from instance via RPyC generator polling.

    Example:
        streamer = MetricsStreamer(
            instance=instance,
            conn=rpyc_connection,
            callback=emit,
            interval=0.2,
        )
        streamer.start()
        # ... metrics flow to callback at ~0.2s intervals
        streamer.stop()
    """

    def __init__(
        self,
        instance: Instance,
        conn: Connection,
        callback: Callback | None = None,
        interval: float = 0.2,
        provider_name: ProviderName = ProviderName.AWS,
    ) -> None:
        """Initialize metrics streamer.

        Args:
            instance: Instance to stream metrics from.
            conn: RPyC connection to the instance (dedicated for metrics).
            callback: Callback to emit Metrics events.
            interval: Time between samples in seconds.
            provider_name: Provider name for event metadata.
        """
        self._instance = instance
        self._conn = conn
        self._callback = callback
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._provisioned = ProvisionedInstance(
            instance_id=instance.id,
            node=instance.node,
            provider=provider_name,
            spot=instance.spot,
            spec=None,
            ip=instance.private_ip,
        )

    @property
    def instance(self) -> Instance:
        """The instance being monitored."""
        return self._instance

    def start(self) -> None:
        """Start streaming in background thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop streaming."""
        from loguru import logger

        inst_id = self._instance.id
        logger.debug(f"[{inst_id}] Stopping metrics streamer")

        self._stop.set()

        # Close connection to unblock the generator
        with contextlib.suppress(Exception):
            self._conn.close()

        if self._thread:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning(f"[{inst_id}] Metrics thread didn't terminate in 2s")

        logger.debug(f"[{inst_id}] Metrics streamer stopped")

    def _stream_loop(self) -> None:
        """Poll metrics from server generator."""
        from loguru import logger

        inst_id = self._instance.id
        metrics_received = 0

        try:
            logger.info(f"[{inst_id}] Starting metrics polling (interval={self._interval}s)")

            for data in self._conn.root.stream_metrics(self._interval):
                if self._stop.is_set():
                    break
                metrics_received += 1
                self._emit(data)

            logger.info(f"[{inst_id}] Metrics polling ended (received {metrics_received} samples)")
        except EOFError:
            logger.debug(f"[{inst_id}] Metrics connection closed")
        except Exception as e:
            if not self._stop.is_set():
                logger.warning(f"[{inst_id}] Metrics streaming error: {e}")

    def _emit(self, data: dict[str, Any]) -> None:
        """Convert raw data to Metrics event and emit."""
        if not self._callback:
            return

        event = Metrics(
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
        self._callback(event)


# Backwards compatibility alias
MetricsPoller = MetricsStreamer
