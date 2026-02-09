"""Background monitors and event streaming.

Provides instance registry and event streaming for bootstrap monitoring.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from injector import Module, provider, singleton

from .app import component
from .bus import AsyncEventBus
from .events import (
    InstanceDestroyed,
    InstanceId,
    InstanceMetadata,
    InstanceProvisioned,
    ShutdownRequested,
)
from .transport import SSHTransport


# =============================================================================
# Instance Registry
# =============================================================================


@dataclass
class InstanceRegistry:
    """Tracks active instances for monitoring.

    Shared state between monitors and handlers.
    Registered as singleton via DI.
    """

    _instances: dict[InstanceId, InstanceMetadata] = field(default_factory=dict)

    def register(self, info: InstanceMetadata) -> None:
        """Register an active instance."""
        self._instances[info.id] = info

    def unregister(self, instance_id: InstanceId) -> None:
        """Remove an instance from tracking."""
        self._instances.pop(instance_id, None)

    @property
    def instances(self) -> list[InstanceMetadata]:
        """All tracked instances."""
        return list(self._instances.values())

    @property
    def spot_instances(self) -> list[InstanceMetadata]:
        """Only spot instances (preemption-eligible)."""
        return [i for i in self._instances.values() if i.spot]

    def get(self, instance_id: InstanceId) -> InstanceMetadata | None:
        """Get instance by ID."""
        return self._instances.get(instance_id)


# =============================================================================
# Monitor Module
# =============================================================================


class MonitorModule(Module):
    """DI module that provides monitors and registry.

    Usage:
        >>> async with app_context(AppModule(), MonitorModule()) as app:
        ...     pool = app.get(ComputePool)
        ...     await pool.start()
    """

    @singleton
    @provider
    def provide_registry(self) -> InstanceRegistry:
        """Provide singleton instance registry."""
        return InstanceRegistry()

    @singleton
    @provider
    def provide_ssh_credentials_registry(self) -> SSHCredentialsRegistry:
        """Provide singleton SSH credentials registry."""
        return SSHCredentialsRegistry()


# =============================================================================
# Preemption Monitor
# =============================================================================


# =============================================================================
# Event Streamer (Continuous Streaming)
# =============================================================================


@dataclass
class SSHCredentials:
    """SSH credentials for a cluster."""

    user: str
    key_path: str


@dataclass
class SSHCredentialsRegistry:
    """Registry for SSH credentials, shared via DI.

    Provider handlers register credentials when clusters are provisioned.
    EventStreamer looks up credentials when streaming starts.
    """

    _credentials: dict[str, SSHCredentials] = field(default_factory=dict)

    def register(self, cluster_id: str, user: str, key_path: str) -> None:
        """Register SSH credentials for a cluster."""
        from loguru import logger
        self._credentials[cluster_id] = SSHCredentials(user=user, key_path=key_path)
        logger.debug(f"SSHCredentialsRegistry: Registered {cluster_id} (registry id={id(self)}, total={len(self._credentials)})")

    def unregister(self, cluster_id: str) -> None:
        """Remove credentials for a cluster."""
        self._credentials.pop(cluster_id, None)

    def get(self, cluster_id: str) -> SSHCredentials | None:
        """Get credentials for a cluster."""
        return self._credentials.get(cluster_id)

    def get_any(self) -> tuple[str, SSHCredentials] | None:
        """Get any registered credentials (fallback)."""
        from loguru import logger
        logger.debug(f"SSHCredentialsRegistry: get_any() called (registry id={id(self)}, credentials={list(self._credentials.keys())})")
        if self._credentials:
            cluster_id = next(iter(self._credentials.keys()))
            return cluster_id, self._credentials[cluster_id]
        return None


@component
@dataclass
class EventStreamer:
    """Streams events from instances throughout their lifecycle.

    Listens for InstanceProvisioned and starts streaming from each instance.
    Continues streaming until instance is destroyed or pool shuts down.

    Dependencies are injected via DI:
    - bus: Event bus for emitting/subscribing to events
    - credentials_registry: Shared registry for SSH credentials
    """

    bus: AsyncEventBus
    credentials_registry: SSHCredentialsRegistry

    # Streaming state (private - excluded from __init__)
    _tasks: dict[InstanceId, "asyncio.Task[None]"] = field(default_factory=dict, init=False)
    _transports: dict[InstanceId, SSHTransport] = field(default_factory=dict, init=False)
    _instance_clusters: dict[InstanceId, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Subscribe to instance lifecycle events."""
        from loguru import logger

        logger.debug("EventStreamer: Initializing and subscribing to events")
        self.bus.connect(InstanceProvisioned, self._on_provisioned)
        self.bus.connect(InstanceDestroyed, self._on_destroyed)
        self.bus.connect(ShutdownRequested, self._on_shutdown)

    async def _on_provisioned(self, _: Any, event: InstanceProvisioned) -> None:
        """Start streaming from newly provisioned instance."""
        from loguru import logger

        info = event.instance

        if info.id in self._tasks:
            logger.debug(f"EventStreamer: Already streaming {info.id}")
            return

        # Find credentials - try cluster_id from request, then fallback to any
        cluster_id = event.request_id.split("-")[0] if "-" in event.request_id else ""
        creds = self.credentials_registry.get(cluster_id)

        if not creds:
            # Try to find any registered credentials
            result = self.credentials_registry.get_any()
            if result:
                cluster_id, creds = result
            else:
                raise RuntimeError(
                    f"EventStreamer: No SSH credentials registered for instance {info.id}. "
                    "Provider must call ssh_credentials.register() before emitting InstanceRunning."
                )

        logger.info(f"EventStreamer: Starting stream for {info.id}")
        self._instance_clusters[info.id] = cluster_id

        # Create transport
        transport = SSHTransport(
            host=info.ip,
            user=creds.user,
            key_path=creds.key_path,
            port=info.ssh_port,
        )
        self._transports[info.id] = transport

        # Start streaming task
        task = asyncio.create_task(
            self._stream_instance(transport, info),
            name=f"stream-{info.id}",
        )
        self._tasks[info.id] = task

    async def _on_destroyed(self, _: Any, event: InstanceDestroyed) -> None:
        """Stop streaming for destroyed instance."""
        await self._stop_instance(event.instance_id)

    async def _on_shutdown(self, _: Any, event: ShutdownRequested) -> None:
        """Stop all streaming for cluster."""
        from loguru import logger

        logger.debug(f"EventStreamer: Shutdown requested for {event.cluster_id}")

        # Find instances belonging to this cluster
        to_stop = [
            instance_id
            for instance_id, cluster_id in self._instance_clusters.items()
            if cluster_id == event.cluster_id or not event.cluster_id
        ]

        for instance_id in to_stop:
            await self._stop_instance(instance_id)

        # Clean up credentials
        self.credentials_registry.unregister(event.cluster_id)

    async def _stop_instance(self, instance_id: InstanceId) -> None:
        """Stop streaming for a specific instance."""
        from loguru import logger

        # Cancel task
        task = self._tasks.pop(instance_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Close transport
        transport = self._transports.pop(instance_id, None)
        if transport:
            await transport.close()

        # Clean up tracking
        self._instance_clusters.pop(instance_id, None)
        logger.debug(f"EventStreamer: Stopped streaming for {instance_id}")

    async def _stream_instance(
        self,
        transport: "SSHTransport",
        info: InstanceMetadata,
    ) -> None:
        """Stream events from instance with retry on connection loss."""
        from loguru import logger

        from skyward.events import (
            BootstrapCommand,
            BootstrapConsole,
            BootstrapFailed,
            BootstrapPhase,
            Log,
            Metric,
        )
        from skyward.retry import retry
        from skyward.transport import (
            RawBootstrapCommand,
            RawBootstrapConsole,
            RawBootstrapPhase,
            RawLogEvent,
            RawMetricEvent,
        )

        log_prefix = f"[{info.provider}:{info.node}] "

        @retry(
            on=Exception,
            max_attempts=5,
            base_delay=2.0,
            exponential_base=2.0,
            max_delay=30.0,
            jitter=True,
        )
        async def stream_with_retry() -> None:
            # Reconnect if needed
            if not transport.is_connected:
                await transport.connect()

            # Stream events
            async for raw_event in transport.stream_events(timeout=600.0):
                match raw_event:
                    case RawBootstrapConsole(content=content, stream=stream):
                        self.bus.emit(BootstrapConsole(
                            instance=info,
                            content=content,
                            stream=stream,
                        ))
                        display = content[:100] + "..." if len(content) > 100 else content
                        if stream == "stderr":
                            logger.warning(f"{log_prefix}[stderr] {display}")
                        else:
                            logger.debug(f"{log_prefix}[stdout] {display}")

                    case RawBootstrapPhase(event=event, phase=phase, elapsed=elapsed, error=error):
                        self.bus.emit(BootstrapPhase(
                            instance=info,
                            event=event,
                            phase=phase,
                            elapsed=elapsed,
                            error=error,
                        ))
                        if event == "started":
                            logger.info(f"{log_prefix}Phase '{phase}' started")
                        elif event == "completed":
                            elapsed_str = f" ({elapsed:.1f}s)" if elapsed else ""
                            logger.info(f"{log_prefix}Phase '{phase}' completed{elapsed_str}")
                        elif event == "failed":
                            logger.error(f"{log_prefix}Phase '{phase}' FAILED: {error}")
                            self.bus.emit(BootstrapFailed(
                                instance=info,
                                phase=phase,
                                error=error or "unknown",
                            ))

                    case RawBootstrapCommand(command=command):
                        self.bus.emit(BootstrapCommand(
                            instance=info,
                            command=command,
                        ))
                        display = command[:80] + "..." if len(command) > 80 else command
                        logger.debug(f"{log_prefix}Running: {display}")

                    case RawLogEvent(content=content, stream=stream):
                        self.bus.emit(Log(
                            instance=info,
                            line=content,
                            stream=stream,
                        ))

                    case RawMetricEvent(name=name, value=value, ts=ts):
                        self.bus.emit(Metric(
                            instance=info,
                            name=name,
                            value=value,
                            timestamp=ts,
                        ))
                        logger.trace(f"{log_prefix}metric {name}={value}")

        try:
            await stream_with_retry()
        except asyncio.CancelledError:
            logger.debug(f"{log_prefix}Streaming cancelled")
        except Exception as e:
            logger.error(f"{log_prefix}Streaming failed after retries: {e}")
            self.bus.emit(BootstrapFailed(
                instance=info,
                phase="streaming",
                error=f"Connection lost: {e}",
            ))
        finally:
            logger.debug(f"{log_prefix}Streaming ended")

    async def _wait_for_ssh(self, transport: "SSHTransport") -> None:
        """Wait for SSH to be available.

        Retry logic is built into SSHTransport.connect().
        """
        await transport.connect()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "InstanceRegistry",
    "MonitorModule",
    "EventStreamer",
    "SSHCredentials",
    "SSHCredentialsRegistry",
]
