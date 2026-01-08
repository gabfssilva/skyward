"""InstancePool - manages a pool of cloud instances with lifecycle operations."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from skyward.core.callback import emit
from skyward.core.events import (
    BootstrapCommand,
    BootstrapCompleted,
    BootstrapConsole,
    BootstrapMetrics,
    BootstrapPhase,
    BootstrapProgress,
    BootstrapStarting,
    InstanceStopping,
    LogLine,
    Metrics,
    ProviderName,
    ProvisioningCompleted,
    ProvisioningStarted,
)
from skyward.providers.common import BootstrapError, LogEvent, make_provisioned
from skyward.utils.conc import for_each_async

if TYPE_CHECKING:
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

    def setup(self, timeout: int = 300) -> None:
        """Bootstrap all instances in parallel with real-time streaming.

        Waits for SSH connectivity and streams bootstrap progress in real-time.
        Emits bootstrap events for progress tracking.

        Args:
            timeout: Maximum seconds to wait for bootstrap per instance.

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

                            case BootstrapMetrics() as m:
                                gpu_mem_used = (
                                    float(m.gpu_mem_used) if m.gpu_mem_used else None
                                )
                                gpu_mem_total = (
                                    float(m.gpu_mem_total) if m.gpu_mem_total else None
                                )
                                emit(Metrics(
                                    instance=provisioned,
                                    cpu_percent=m.cpu,
                                    memory_percent=m.mem,
                                    memory_used_mb=float(m.mem_used_mb),
                                    memory_total_mb=float(m.mem_total_mb),
                                    gpu_utilization=m.gpu_util,
                                    gpu_memory_used_mb=gpu_mem_used,
                                    gpu_memory_total_mb=gpu_mem_total,
                                    gpu_temperature=float(m.gpu_temp) if m.gpu_temp else None,
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

    def shutdown(self) -> tuple[ExitedInstance, ...]:
        """Shutdown all instances.

        Destroys each instance and returns ExitedInstance objects.

        Returns:
            Tuple of ExitedInstance objects with exit information.

        Emits:
            InstanceStopping: For each instance being stopped.
        """
        from skyward.types import ExitedInstance

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
