"""Background monitors for preemption detection and health checks.

Uses the @monitor decorator for periodic background tasks with DI.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from injector import Module, provider, singleton

from .app import component, monitor
from .bus import AsyncEventBus
from .events import (
    InstanceDestroyed,
    InstanceId,
    InstanceInfo,
    InstancePreempted,
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

    _instances: dict[InstanceId, InstanceInfo] = field(default_factory=dict)

    def register(self, info: InstanceInfo) -> None:
        """Register an active instance."""
        self._instances[info.id] = info

    def unregister(self, instance_id: InstanceId) -> None:
        """Remove an instance from tracking."""
        self._instances.pop(instance_id, None)

    @property
    def instances(self) -> list[InstanceInfo]:
        """All tracked instances."""
        return list(self._instances.values())

    @property
    def spot_instances(self) -> list[InstanceInfo]:
        """Only spot instances (preemption-eligible)."""
        return [i for i in self._instances.values() if i.spot]

    def get(self, instance_id: InstanceId) -> InstanceInfo | None:
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


@monitor(interval=5.0, name="preemption_monitor")
async def check_preemption(
    registry: InstanceRegistry,
    bus: AsyncEventBus,
) -> None:
    """Check for spot instance preemptions.

    AWS spot instances get a 2-minute warning before termination.
    This monitor polls instance metadata to detect preemption early.
    """
    for info in registry.spot_instances:
        is_preempted, reason = await _check_instance_preemption(info)
        if is_preempted:
            bus.emit(
                InstancePreempted(
                    instance=info,
                    reason=reason or "spot-interruption",
                )
            )


async def _check_instance_preemption(info: InstanceInfo) -> tuple[bool, str | None]:
    """Check if instance was preempted via provider API.

    TODO: Implement actual preemption detection per provider:
    - AWS: Check instance-action metadata endpoint
    - GCP: Check preemption metadata
    - Vast.ai: Check instance status

    For now, returns False (not preempted).
    """
    # This would normally:
    # 1. SSH into instance and curl metadata endpoint
    # 2. Or use provider API to check status
    _ = info
    return False, None


# =============================================================================
# Health Monitor
# =============================================================================


@monitor(interval=30.0, name="health_monitor")
async def check_health(
    registry: InstanceRegistry,
    bus: AsyncEventBus,
) -> None:
    """Health check via SSH ping.

    Detects instances that have become unreachable (network issues,
    crashes, etc.) and emits preemption events to trigger replacement.
    """
    for info in registry.instances:
        if not await _ping_instance(info):
            bus.emit(
                InstancePreempted(
                    instance=info,
                    reason="unreachable",
                )
            )


async def _ping_instance(info: InstanceInfo) -> bool:
    """Check if instance is reachable via SSH.

    TODO: Implement actual SSH health check.
    """
    # This would normally try SSH connection
    _ = info
    return True


# =============================================================================
# AWS-Specific Preemption Detection
# =============================================================================


async def check_aws_spot_interruption(
    instance_id: str,
    region: str,
) -> tuple[bool, str | None]:
    """Check AWS instance for spot interruption via EC2 API.

    AWS marks instances as 'terminated' with a StateTransitionReason
    containing 'Spot' or 'capacity' when interrupted.
    """
    import aioboto3

    session = aioboto3.Session()
    async with session.client("ec2", region_name=region) as ec2:
        try:
            response = await ec2.describe_instances(InstanceIds=[instance_id])

            for reservation in response.get("Reservations", []):
                for instance in reservation.get("Instances", []):
                    state = instance["State"]["Name"]

                    if state in ("terminated", "shutting-down"):
                        reason = instance.get("StateTransitionReason", "")
                        if "spot" in reason.lower() or "capacity" in reason.lower():
                            return True, "spot-interruption"
                        return True, "terminated"

                    # Check instance status for degraded hardware
                    statuses = await ec2.describe_instance_status(
                        InstanceIds=[instance_id],
                        IncludeAllInstances=True,
                    )
                    for status in statuses.get("InstanceStatuses", []):
                        instance_status = status.get("InstanceStatus", {}).get("Status")
                        system_status = status.get("SystemStatus", {}).get("Status")

                        if instance_status == "impaired" or system_status == "impaired":
                            return True, "hardware-impaired"

            return False, None

        except Exception:
            # Instance might not exist anymore
            return True, "not-found"


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
                logger.warning(f"EventStreamer: No credentials for {info.id}, skipping streaming")
                return

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
        info: InstanceInfo,
    ) -> None:
        """Stream events from instance indefinitely."""
        from loguru import logger

        from skyward.events import (
            BootstrapCommand,
            BootstrapConsole,
            BootstrapFailed,
            BootstrapPhase,
            Log,
            Metric,
        )
        from skyward.transport import (
            RawBootstrapCommand,
            RawBootstrapConsole,
            RawBootstrapPhase,
            RawLogEvent,
            RawMetricEvent,
        )

        log_prefix = f"[{info.provider}:{info.node}] "

        try:
            # Wait for SSH
            await self._wait_for_ssh(transport, log_prefix)

            # Stream events indefinitely
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
                        logger.debug(f"{log_prefix}metric {name}={value}")

        except asyncio.CancelledError:
            logger.debug(f"{log_prefix}Streaming cancelled")
        except TimeoutError:
            logger.warning(f"{log_prefix}Streaming timeout")
        except Exception as e:
            logger.error(f"{log_prefix}Streaming error: {e}")
        finally:
            logger.debug(f"{log_prefix}Streaming ended")

    async def _wait_for_ssh(
        self,
        transport: "SSHTransport",
        log_prefix: str,  # noqa: ARG002 - kept for API compatibility
        timeout: float = 300.0,  # noqa: ARG002 - transport handles retry
    ) -> None:
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
    "check_preemption",
    "check_health",
    "check_aws_spot_interruption",
    "EventStreamer",
    "SSHCredentials",
    "SSHCredentialsRegistry",
]
