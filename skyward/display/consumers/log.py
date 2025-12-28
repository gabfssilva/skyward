"""Log consumer that emits events as traditional Python logs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from skyward.events import (
    BootstrapCompleted,
    BootstrapProgress,
    BootstrapStarting,
    CostFinal,
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
    from skyward.pool import ComputePool


class LogConsumer:
    """Consumer that emits events as traditional Python logs.

    Uses pattern matching on event types to format log messages.
    """

    def __init__(self, pool: ComputePool, logger_name: str = "skyward") -> None:
        self._logger = logging.getLogger(logger_name)

        # Register event handlers
        @pool.on(PoolStarted)
        def on_start(event: PoolStarted) -> None:
            pass  # LogConsumer has no initialization

        @pool.on(PoolStopping)
        def on_stop(event: PoolStopping) -> None:
            pass  # LogConsumer has no finalization

        @pool.on()  # wildcard - all events
        def on_event(event: SkywardEvent) -> None:
            # Skip lifecycle events (handled separately)
            if isinstance(event, (PoolStarted, PoolStopping)):
                return
            self._handle(event)

    def _handle(self, event: SkywardEvent) -> None:
        """Process an event by logging it."""
        match event:
            # Provision events
            case InfraCreating():
                self._logger.info("[PROVISION] Creating infrastructure...")

            case InfraCreated(region=region):
                self._logger.info(f"[PROVISION] Infrastructure ready ({region})")

            case InstanceLaunching(count=count, instance_type=itype):
                self._logger.info(f"[PROVISION] Launching {count} x {itype}")

            case InstanceProvisioned(instance_id=iid, spot=is_spot):
                label = "[spot]" if is_spot else "[on-demand]"
                self._logger.info(f"[{iid[:12]}] Provisioned {label}")

            # Setup events
            case BootstrapStarting(instance_id=iid):
                self._logger.info(f"[{iid[:12]}] Bootstrap starting")

            case BootstrapProgress(instance_id=iid, step=step):
                self._logger.info(f"[{iid[:12]}] Bootstrap: {step}")

            case BootstrapCompleted(instance_id=iid):
                self._logger.info(f"[{iid[:12]}] Bootstrap completed")

            # Execution events
            case Metrics(instance_id=iid, cpu_percent=cpu, gpu_utilization=gpu, memory_percent=mem, gpu_memory_used_mb=gpu_mem, gpu_memory_total_mb=gpu_mem_total):
                gpu_str = f"{gpu:.0f}%" if gpu is not None else "N/A"
                gpu_mem_str = (
                    f"{(gpu_mem / gpu_mem_total) * 100:.0f}% (of {gpu_mem_total}MB)"
                    if gpu_mem is not None and gpu_mem_total
                    else "N/A"
                )
                self._logger.info(f"[{iid[:12]}] Mem: {mem:.0f}% | CPU: {cpu:.0f}% | GPU: {gpu_str} | GPU mem: {gpu_mem_str}")

            case LogLine(node=node, line=line):
                if line.strip():
                    self._logger.info(f"[node {node}] {line.rstrip()}")

            # Shutdown events
            case InstanceStopping(instance_id=iid):
                self._logger.info(f"[{iid[:12]}] Stopping")

            # Cost events
            case CostUpdate(
                accumulated_cost=cost,
                elapsed_seconds=elapsed,
                hourly_rate=hourly,
                spot_count=spot,
                ondemand_count=ondemand,
            ):
                mins, secs = divmod(int(elapsed), 60)
                self._logger.info(
                    f"[COST] ${cost:.4f} ({mins}m{secs:02d}s) | "
                    f"${hourly:.2f}/hr | {spot} spot + {ondemand} on-demand"
                )

            case CostFinal(
                total_cost=cost,
                total_seconds=elapsed,
                hourly_rate=hourly,
                spot_count=spot,
                ondemand_count=ondemand,
                savings_vs_ondemand=savings,
            ):
                mins, secs = divmod(int(elapsed), 60)
                self._logger.info(
                    f"[COST] Final: ${cost:.4f} ({mins}m{secs:02d}s) | "
                    f"{spot} spot + {ondemand} on-demand | saved ${savings:.4f}"
                )

            # Errors
            case Error(message=msg, instance_id=iid):
                prefix = f"[{iid[:12]}] " if iid else ""
                self._logger.error(f"{prefix}Error: {msg}")
