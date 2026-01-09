"""InstancePool - manages a pool of cloud instances with lifecycle operations."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from skyward.core.callback import emit
from skyward.core.events import (
    BootstrapCommand,
    BootstrapCompleted,
    BootstrapConsole,
    BootstrapPhase,
    BootstrapProgress,
    BootstrapStarting,
    InstanceStopping,
    LogLine,
    MetricValue,
    ProviderName,
    ProvisioningCompleted,
    ProvisioningStarted,
)
from skyward.core.monitor import Monitor, monitor
from skyward.pool.preemption import PreemptionHandler, preemption_check
from skyward.providers.common import BootstrapError, LogEvent, MetricEvent, make_provisioned
from skyward.utils.conc import for_each_async

if TYPE_CHECKING:
    from skyward.spec.preemption import Preemption
    from skyward.types import ComputeSpec, ExitedInstance, Instance, Provider

@dataclass
class InstancePool:
    """Manages a pool of cloud instances with lifecycle operations.

    InstancePool holds provider, compute spec, and instances.
    Use as a context manager for automatic cleanup.

    Attributes:
        provider: Cloud provider instance (AWS, DigitalOcean, etc.).
        compute: Compute specification (nodes, accelerator, image, etc.).
        instances: Tuple of provisioned instances (empty until provision()).
    """

    provider: Provider
    compute: ComputeSpec
    instances: tuple[Instance, ...] = ()

    # Preemption monitoring state
    _monitors: list[Monitor] = field(default_factory=list)
    _preemption_handler: PreemptionHandler | None = field(default=None)

    def __iter__(self) -> Iterator[Instance]:
        """Iterate over current instances."""
        return iter(self.instances)

    def __len__(self) -> int:
        """Number of instances."""
        return len(self.instances)

    def provision(self) -> None:
        """Provision instances and store internally.

        Calls provider.provision() and emits provisioning events:
        - ProvisioningStarted: Before provisioning begins
        - InstanceProvisioned: For each instance (emitted by provider)
        - ProvisioningCompleted: After all instances are provisioned

        Example:
            pool.provision()
            pool.setup()
        """
        emit(ProvisioningStarted())

        self.instances = self.provider.provision(self.compute)

        provider_name = _get_provider_name(self.provider)
        provisioned_instances = tuple(
            make_provisioned(inst, provider_name)
            for inst in self.instances
        )

        region = self.instances[0].get_meta("region", "unknown") if self.instances else "unknown"
        emit(
            ProvisioningCompleted(
                instances=provisioned_instances,
                provider=provider_name,
                region=region,
            )
        )

    def setup(
        self,
        timeout: int = 300,
        preemption: Preemption | None = None,
    ) -> None:
        """Bootstrap all instances in parallel with real-time streaming.

        Waits for SSH connectivity and streams bootstrap progress in real-time.
        Emits bootstrap events for progress tracking.

        Preemption monitoring starts BEFORE bootstrap begins, so preemption
        is detected even if an instance is terminated during bootstrap.

        Args:
            timeout: Maximum seconds to wait for bootstrap per instance.
            preemption: Preemption handling configuration. If provided, starts
                monitoring for spot instance preemption.

        Emits:
            BootstrapStarting: When bootstrap begins on each instance.
            BootstrapProgress: For each phase and console output.
            BootstrapCompleted: When bootstrap finishes on each instance.
        """
        import contextvars

        if not self.instances:
            return

        provider_name = _get_provider_name(self.provider)
        use_systemd = provider_name != ProviderName.VastAI

        # Setup preemption monitoring BEFORE bootstrap
        # This ensures we detect preemption even if it happens during bootstrap
        if preemption:
            self._setup_preemption_monitor(preemption)

        def bootstrap_instance(inst: Instance) -> None:
            # Capture context here - for_each_async already runs us in a copied context,
            # so we copy again for the streaming thread (ctx.run is not reentrant)
            ctx = contextvars.copy_context()

            provisioned = make_provisioned(inst, provider_name)
            emit(BootstrapStarting(instance=provisioned))

            # Wait for SSH port to be ready
            inst.wait_for_ssh(timeout)

            # Shared state between main thread and streaming thread
            duration: float | None = None
            bootstrap_done = threading.Event()
            stream_error: BaseException | None = None

            def stream_events_loop() -> None:
                """Stream events continuously in background thread."""
                nonlocal duration, stream_error
                logger.debug(f"[{inst.id}] Starting event streaming thread")
                try:
                    for event in inst.stream_events(timeout=timeout):
                        logger.debug(f"[{inst.id}] Received event: {event}")

                        match event:
                            case BootstrapCommand(command=cmd):
                                emit(BootstrapProgress(
                                    instance=provisioned, step="command", message=cmd
                                ))

                            case BootstrapConsole(content=content):
                                emit(BootstrapProgress(
                                    instance=provisioned, step="console", message=content
                                ))

                            case BootstrapPhase(event="started", phase=phase):
                                emit(BootstrapProgress(instance=provisioned, step=phase))

                            case BootstrapPhase(
                                event="completed", phase="bootstrap", elapsed=elapsed
                            ):
                                logger.debug(f"[{inst.id}] Bootstrap completed")
                                duration = elapsed
                                bootstrap_done.set()

                            case BootstrapPhase(event="completed", phase=phase):
                                emit(BootstrapProgress(instance=provisioned, step=f"{phase} ✓"))

                            case BootstrapPhase(event="failed", phase=phase, error=error):
                                logger.error(f"[{inst.id}] Phase '{phase}' failed: {error}")
                                stream_error = BootstrapError(f"Phase '{phase}' failed: {error}")
                                bootstrap_done.set()
                                return  # Exit thread on failure

                            case MetricEvent(name=name, value=value, ts=ts):
                                emit(MetricValue(
                                    instance=provisioned,
                                    name=name,
                                    value=value,
                                    timestamp=ts,
                                ))

                            case LogEvent(content=content, stream=stream):
                                emit(LogLine(
                                    instance=provisioned,
                                    line=content,
                                    timestamp=time.time(),
                                    stream=stream,
                                ))
                except Exception as e:
                    logger.error(f"[{inst.id}] Streaming error: {e}")
                    stream_error = e
                    bootstrap_done.set()
                finally:
                    logger.debug(f"[{inst.id}] Streaming thread ending")

            # Start streaming in background thread (daemon=True for auto-cleanup)
            # Use ctx.run() to propagate ContextVars (emit callback) to the thread
            stream_thread = threading.Thread(
                target=lambda: ctx.run(stream_events_loop), daemon=True
            )
            stream_thread.start()

            # Wait for bootstrap to complete
            if not bootstrap_done.wait(timeout=timeout):
                raise TimeoutError(f"Bootstrap did not complete within {timeout}s")

            # Check for errors from streaming thread
            if stream_error:
                raise stream_error

            # Install skyward wheel (skips if not "local" source)
            inst.install_skyward(self.compute, use_systemd=use_systemd)

            emit(BootstrapCompleted(instance=provisioned, duration=duration))

        for_each_async(bootstrap_instance, self.instances)

    def _setup_preemption_monitor(self, config: Preemption) -> None:
        """Setup preemption monitoring for spot/bid instances.

        Creates a PreemptionHandler (event consumer) and a Monitor (detection loop).
        The handler reacts to InstancePreempted events based on the configured policy.
        """
        from skyward.core.events import SkywardEvent

        # Create handler that reacts to preemption events
        handler = PreemptionHandler(
            config=config,
            provider=self.provider,
            get_instances=lambda: self.instances,
            set_instances=self._set_instances,
            compute_spec=self.compute,
        )
        self._preemption_handler = handler

        # Create composed emit: handler receives event, then normal emit broadcasts it
        def handler_emit(event: SkywardEvent) -> None:
            handler(event)  # Handler reacts (may replace instance)
            emit(event)  # Broadcast to other callbacks

        # Create monitor that detects preemption and emits events
        preemption_monitor = monitor(
            name="preemption",
            interval=config.monitor_interval,
            check=preemption_check(self.provider, lambda: self.instances),
            emit=handler_emit,
        )
        self._monitors.append(preemption_monitor)

        # Start monitoring
        preemption_monitor.start()
        logger.debug(f"Preemption monitoring started (interval={config.monitor_interval}s)")

    def _set_instances(self, instances: tuple[Instance, ...]) -> None:
        """Thread-safe instance update for preemption handler."""
        self.instances = instances

    def shutdown(self) -> tuple[ExitedInstance, ...]:
        """Shutdown all instances.

        Stops all monitors, then destroys each instance.

        Returns:
            Tuple of ExitedInstance objects with exit information.

        Emits:
            InstanceStopping: For each instance being stopped.
        """
        from skyward.types import ExitedInstance

        # Stop all monitors first
        for m in self._monitors:
            m.shutdown()
        self._monitors.clear()

        if not self.instances:
            return ()

        provider_name = _get_provider_name(self.provider)
        exited: list[ExitedInstance] = []

        for inst in self.instances:
            provisioned = make_provisioned(inst, provider_name)
            emit(InstanceStopping(instance=provisioned))
            inst.destroy()
            exited.append(
                ExitedInstance(
                    instance=inst,
                    exit_code=0,
                    exit_reason="shutdown",
                )
            )

        return tuple(exited)

    def __enter__(self) -> InstancePool:
        """Enter context manager.

        Provisions and sets up instances automatically.

        Returns:
            Self with instances provisioned and bootstrapped.
        """
        self.provision()
        self.setup()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager.

        Shuts down all instances, regardless of exceptions.
        """
        self.shutdown()


def _get_provider_name(provider: Provider) -> ProviderName:
    """Map provider.name to ProviderName enum."""
    match provider.name:
        case "aws":
            return ProviderName.AWS
        case "digitalocean":
            return ProviderName.DigitalOcean
        case "verda":
            return ProviderName.Verda
        case "vastai":
            return ProviderName.VastAI
        case _:
            return ProviderName.AWS  # Default fallback

__all__ = ["InstancePool"]
