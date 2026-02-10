"""Instance monitor actor - streams JSONL events from instances.

An instance monitor tells this story: connecting → streaming → stopped.

The monitor SSHes into an instance, reads the events.jsonl file
(tail -F), converts raw events to typed messages, and sends them
to pool_ref for observability (spy on pool captures them for the
panel). It runs for the entire instance lifetime — not just bootstrap.

Bootstrap completion is detected as a side effect: when the monitor
sees BootstrapPhase(phase="bootstrap", event="completed"), it signals
the provider via reply_to. Streaming continues after that for metrics,
logs, and other runtime events.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from casty import ActorContext, ActorRef, Behavior, Behaviors
from loguru import logger

from skyward.actors.provider import BootstrapDone
from skyward.messages import (
    BootstrapCommand,
    BootstrapConsole,
    BootstrapFailed,
    BootstrapPhase,
    Event,
    InstanceMetadata,
    Log,
    Metric,
)


# =============================================================================
# Messages
# =============================================================================


@dataclass(frozen=True, slots=True)
class StopMonitor:
    pass


@dataclass(frozen=True, slots=True)
class _StreamedEvent:
    event: Event


@dataclass(frozen=True, slots=True)
class _StreamEnded:
    error: str | None = None


type MonitorMsg = StopMonitor | _StreamedEvent | _StreamEnded


# =============================================================================
# Behavior
# =============================================================================


def instance_monitor(
    info: InstanceMetadata,
    ssh_user: str,
    ssh_key_path: str,
    pool_ref: ActorRef,
    reply_to: ActorRef[BootstrapDone],
) -> Behavior[MonitorMsg]:
    """An instance monitor tells this story: connecting → streaming → stopped."""

    resources: dict[str, Any] = {}

    async def _cleanup(ctx: ActorContext[MonitorMsg]) -> None:
        task = resources.get("task")
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        transport = resources.get("transport")
        if transport:
            await transport.close()

    async def _setup(ctx: ActorContext[MonitorMsg]) -> Behavior[MonitorMsg]:
        from skyward.transport.ssh import SSHTransport

        logger.info(f"Instance monitor connecting to {info.id} at {info.ip}:{info.ssh_port}...")

        transport = SSHTransport(
            host=info.ip,
            user=ssh_user,
            key_path=ssh_key_path,
            port=info.ssh_port,
            retry_max_attempts=60,
            retry_delay=5.0,
        )
        await transport.connect()
        resources["transport"] = transport

        logger.info(f"Instance monitor connected to {info.id}")

        async def _read_loop() -> None:
            try:
                async for raw_event in transport.stream_events(timeout=600.0):
                    event = _convert(raw_event, info)
                    if event is not None:
                        ctx.self.tell(_StreamedEvent(event=event))
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"Instance monitor stream error on {info.id}: {e}")
            ctx.self.tell(_StreamEnded())

        task = asyncio.create_task(_read_loop())
        resources["task"] = task

        return streaming(transport, task, bootstrap_signaled=False)

    def streaming(
        transport: Any,
        read_task: asyncio.Task[None],
        bootstrap_signaled: bool,
    ) -> Behavior[MonitorMsg]:
        async def receive(ctx: ActorContext[MonitorMsg], msg: MonitorMsg) -> Behavior[MonitorMsg]:
            match msg:
                case _StreamedEvent(event=event):
                    pool_ref.tell(event)

                    if not bootstrap_signaled:
                        match event:
                            case BootstrapPhase(phase="bootstrap", event="completed"):
                                reply_to.tell(BootstrapDone(instance=info, success=True))
                                return streaming(transport, read_task, bootstrap_signaled=True)
                            case BootstrapFailed(error=error):
                                reply_to.tell(BootstrapDone(instance=info, success=False, error=error))
                                return streaming(transport, read_task, bootstrap_signaled=True)

                    return Behaviors.same()

                case _StreamEnded():
                    if not bootstrap_signaled:
                        reply_to.tell(BootstrapDone(
                            instance=info,
                            success=False,
                            error="stream ended before bootstrap completed",
                        ))
                    return Behaviors.stopped()

                case StopMonitor():
                    return Behaviors.stopped()

        return Behaviors.receive(receive)

    return Behaviors.with_lifecycle(Behaviors.setup(_setup), post_stop=_cleanup)


# =============================================================================
# Event Conversion
# =============================================================================


def _convert(raw_event: object, info: InstanceMetadata) -> Event | None:
    from skyward.transport.ssh import (
        RawBootstrapCommand,
        RawBootstrapConsole,
        RawBootstrapPhase,
        RawLogEvent,
        RawMetricEvent,
    )

    match raw_event:
        case RawBootstrapConsole(content=content, stream=stream):
            return BootstrapConsole(instance=info, content=content, stream=stream)
        case RawBootstrapPhase(event=event, phase=phase, elapsed=elapsed, error=error):
            if event == "failed":
                return BootstrapFailed(instance=info, phase=phase, error=error or "unknown")
            return BootstrapPhase(instance=info, event=event, phase=phase, elapsed=elapsed, error=error)
        case RawBootstrapCommand(command=command):
            return BootstrapCommand(instance=info, command=command)
        case RawMetricEvent(name=name, value=value, ts=ts):
            try:
                return Metric(instance=info, name=name, value=float(value), timestamp=ts)
            except (ValueError, TypeError):
                return None
        case RawLogEvent(content=content, stream=stream):
            return Log(instance=info, line=content, stream=stream)
        case _:
            return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "StopMonitor",
    "instance_monitor",
]
