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

from collections.abc import AsyncIterator
from typing import Any

from casty import ActorContext, ActorRef, Behavior, Behaviors
from loguru import logger

from .messages import (
    BootstrapCommand,
    BootstrapConsole,
    BootstrapDone,
    BootstrapFailed,
    BootstrapPhase,
    Event,
    InstanceMetadata,
    Log,
    Metric,
    MonitorMsg,
    StopMonitor,
    _StreamedEvent,
    _StreamEnded,
)

# =============================================================================
# Behavior
# =============================================================================


async def _read_next(stream: AsyncIterator, info: InstanceMetadata) -> MonitorMsg:
    try:
        raw_event = await stream.__anext__()
        event = _convert(raw_event, info)
        match event:
            case None:
                return await _read_next(stream, info)
            case _:
                return _StreamedEvent(event=event)
    except StopAsyncIteration:
        return _StreamEnded()
    except Exception as e:
        logger.warning(f"Instance monitor stream error on {info.id}: {e}")
        return _StreamEnded(error=str(e))


def instance_monitor(
    info: InstanceMetadata,
    ssh_user: str,
    ssh_key_path: str,
    pool_ref: ActorRef,
    reply_to: ActorRef[BootstrapDone],
) -> Behavior[MonitorMsg]:
    """An instance monitor tells this story: connecting → streaming → stopped."""

    async def _setup(ctx: ActorContext[MonitorMsg]) -> Behavior[MonitorMsg]:
        from skyward.infra.ssh import SSHTransport

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

        logger.info(f"Instance monitor connected to {info.id}")

        stream = transport.stream_events(timeout=600.0).__aiter__()

        ctx.pipe_to_self(_read_next(stream, info), mapper=lambda msg: msg)

        async def _cleanup(ctx: ActorContext[MonitorMsg]) -> None:
            await transport.close()

        return Behaviors.with_lifecycle(
            streaming(transport, stream, bootstrap_signaled=False),
            post_stop=_cleanup,
        )

    def streaming(
        transport: Any,
        stream: AsyncIterator,
        bootstrap_signaled: bool,
    ) -> Behavior[MonitorMsg]:
        async def receive(ctx: ActorContext[MonitorMsg], msg: MonitorMsg) -> Behavior[MonitorMsg]:
            match msg:
                case _StreamedEvent(event=event):
                    pool_ref.tell(event)

                    ctx.pipe_to_self(_read_next(stream, info), mapper=lambda msg: msg)

                    if not bootstrap_signaled:
                        match event:
                            case BootstrapPhase(phase="bootstrap", event="completed"):
                                reply_to.tell(BootstrapDone(instance=info, success=True))
                                return streaming(transport, stream, bootstrap_signaled=True)
                            case BootstrapFailed(error=error):
                                reply_to.tell(BootstrapDone(
                                    instance=info,
                                    success=False,
                                    error=error,
                                ))
                                return streaming(transport, stream, bootstrap_signaled=True)

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

    return Behaviors.setup(_setup)


# =============================================================================
# Event Conversion
# =============================================================================


def _convert(raw_event: object, info: InstanceMetadata) -> Event | None:
    from skyward.infra.ssh import (
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
            return BootstrapPhase(
                instance=info, event=event, phase=phase,
                elapsed=elapsed, error=error,
            )
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
