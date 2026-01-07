"""Metrics streaming from instances via RPyC generator polling.

This module provides metrics streaming using RPyC generators.
The client polls the server at the configured interval.
"""

from __future__ import annotations

import threading
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import rpyc

from skyward.callback import emit
from skyward.constants import RPYC_PORT
from skyward.events import Metrics, ProviderName, ProvisionedInstance

if TYPE_CHECKING:
    from skyward.providers.ssh import SSHConnection
    from skyward.types import Instance


class MetricsStreamer:
    """Streams metrics from instance via RPyC generator polling.

    Manages its own SSH connection lifecycle - acquires on start, releases on stop.

    Example:
        streamer = MetricsStreamer(
            instance=instance,
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
        interval: float = 0.2,
        provider_name: ProviderName = ProviderName.AWS,
    ) -> None:
        """Initialize metrics streamer.

        Args:
            instance: Instance to stream metrics from.
            interval: Time between samples in seconds.
            provider_name: Provider name for event metadata.
        """
        self._instance = instance
        self._interval = interval
        self._provider_name = provider_name

        # Connection state (managed by start/stop)
        self._ssh_conn: SSHConnection | None = None
        self._rpyc_conn: rpyc.Connection | None = None

        # Thread state
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Event metadata
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
        """Start streaming in background thread.

        Acquires SSH connection and establishes RPyC connection.
        """
        from loguru import logger

        self._stop.clear()

        # Acquire SSH and setup RPyC
        logger.debug(f"[{self._instance.id}] Acquiring SSH for metrics")
        self._ssh_conn = self._instance.pool.acquire()
        channel = self._ssh_conn.open_tunnel(RPYC_PORT)
        self._rpyc_conn = rpyc.connect_stream(
            channel,
            config={
                "allow_pickle": True,
                "allow_public_attrs": True,
                "sync_request_timeout": 3600,
            },
        )

        # Start streaming thread
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop streaming and release SSH connection."""
        from loguru import logger

        inst_id = self._instance.id
        logger.debug(f"[{inst_id}] Stopping metrics streamer")

        self._stop.set()

        if self._thread:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning(f"[{inst_id}] Metrics thread didn't terminate in 2s")

        # Mark RPyC as closed to prevent __del__ from trying to send close message
        if self._rpyc_conn:
            with suppress(Exception):
                self._rpyc_conn._closed = True

        # Release SSH back to pool
        if self._ssh_conn:
            with suppress(Exception):
                self._instance.pool.release(self._ssh_conn)
            self._ssh_conn = None
        self._rpyc_conn = None

        logger.debug(f"[{inst_id}] Metrics streamer stopped")

    def _stream_loop(self) -> None:
        """Poll metrics from server generator."""
        from loguru import logger

        inst_id = self._instance.id
        metrics_received = 0

        try:
            logger.info(f"[{inst_id}] Starting metrics polling (interval={self._interval}s)")

            for data in self._rpyc_conn.root.stream_metrics(self._interval):
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
        emit(
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


# Backwards compatibility alias
MetricsPoller = MetricsStreamer
