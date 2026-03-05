"""Instance monitor actor — streams JSONL events from instances.

An instance monitor tells this story:
streaming → reconnecting → streaming → ... → stopped.

The monitor receives a shared SSHTransport, reads the events.jsonl file
(tail -F with offset tracking), converts raw events to typed messages,
and sends them to event_listener for observability. It runs for the
entire instance lifetime — not just bootstrap.

On stream failure after bootstrap, the monitor reconnects automatically
with exponential backoff, resuming from the last known line offset.
Before bootstrap, stream failure is fatal.

Bootstrap completion is detected as a side effect: when the monitor
sees BootstrapPhase(phase="bootstrap", event="completed"), it signals
via reply_to. Streaming continues after that for metrics, logs, and
other runtime events.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from casty import ActorContext, ActorRef, Behavior, Behaviors

if TYPE_CHECKING:
    from skyward.infra.ssh import SSHTransport
from skyward.observability.logger import logger

from .messages import (
    BootstrapCommand,
    BootstrapConsole,
    BootstrapDone,
    BootstrapFailed,
    BootstrapPhase,
    Event,
    Log,
    Metric,
    MonitorMsg,
    NodeInstance,
    StopMonitor,
    _Reconnect,
    _StreamedEvent,
    _StreamEnded,
)


async def _read_next(stream: AsyncIterator, info: NodeInstance) -> MonitorMsg:
    try:
        lines_read, raw_event = await stream.__anext__()
        logger.trace("Read {lines_read}: {raw_event}", lines_read=lines_read, raw_event=raw_event)
        event = _convert(raw_event, info)
        match event:
            case None:
                return await _read_next(stream, info)
            case _:
                return _StreamedEvent(event=event, lines_read=lines_read)
    except StopAsyncIteration:
        return _StreamEnded()
    except Exception as e:
        logger.warning("Instance monitor stream error on {iid}: {err}", iid=info.instance.id, err=e)
        return _StreamEnded(error=str(e))


def instance_monitor(
    info: NodeInstance,
    transport: SSHTransport,
    event_listener: ActorRef,
    reply_to: ActorRef[BootstrapDone],
    max_reconnect_attempts: int = 5,
    max_backoff: float = 30.0,
) -> Behavior[MonitorMsg]:
    """An instance monitor tells this story:
    streaming → reconnecting → streaming → ... → stopped.
    """

    def _open_stream(start_line: int) -> AsyncIterator:
        return transport.stream_events(timeout=600.0, start_line=start_line).__aiter__()

    async def _setup(ctx: ActorContext[MonitorMsg]) -> Behavior[MonitorMsg]:
        log = logger.bind(actor="monitor", instance_id=info.instance.id)
        log.info("Starting event stream on shared transport")
        stream = _open_stream(start_line=0)
        ctx.pipe_to_self(_read_next(stream, info), mapper=lambda msg: msg)
        return streaming(stream, bootstrap_signaled=False, lines_read=0)

    def streaming(
        stream: AsyncIterator,
        bootstrap_signaled: bool,
        lines_read: int,
        reconnect_attempts: int = 0,
    ) -> Behavior[MonitorMsg]:
        async def receive(ctx: ActorContext[MonitorMsg], msg: MonitorMsg) -> Behavior[MonitorMsg]:
            match msg:
                case _StreamedEvent(event=event, lines_read=new_lines_read):
                    log = logger.bind(actor="monitor", instance_id=info.instance.id)
                    match event:
                        case Metric():
                            log.trace("metric: {name}={value}", name=event.name, value=event.value)
                        case Log():
                            log.info("{line}", line=event.line.rstrip())
                        case BootstrapPhase():
                            log.info("Phase {phase}: {ev}", phase=event.phase, ev=event.event)
                        case BootstrapCommand():
                            log.info("Command: {cmd}", cmd=event.command)
                        case BootstrapConsole():
                            log.info("{content}", content=event.content.rstrip())
                        case BootstrapFailed():
                            log.error("Bootstrap failed: {err}", err=event.error)

                    event_listener.tell(event)

                    ctx.pipe_to_self(_read_next(stream, info), mapper=lambda msg: msg)

                    signaled = bootstrap_signaled
                    if not signaled:
                        match event:
                            case BootstrapPhase(phase="bootstrap", event="completed"):
                                mon_log = logger.bind(
                                    actor="monitor", instance_id=info.instance.id,
                                )
                                mon_log.info("Bootstrap completion signal received")
                                reply_to.tell(BootstrapDone(instance=info, success=True))
                                signaled = True
                            case BootstrapFailed(error=error):
                                reply_to.tell(BootstrapDone(
                                    instance=info,
                                    success=False,
                                    error=error,
                                ))
                                signaled = True

                    changed = (
                        new_lines_read != lines_read
                        or signaled != bootstrap_signaled
                        or reconnect_attempts > 0
                    )
                    if changed:
                        return streaming(stream, signaled, new_lines_read, reconnect_attempts=0)
                    return Behaviors.same()

                case _StreamEnded(error=error):
                    log = logger.bind(actor="monitor", instance_id=info.instance.id)
                    if error is None:
                        log.info("Stream ended cleanly")
                    else:
                        log.warning("Stream ended with error: {err}", err=error)

                    if not bootstrap_signaled:
                        reply_to.tell(BootstrapDone(
                            instance=info,
                            success=False,
                            error=error or "stream ended before bootstrap completed",
                        ))
                        return Behaviors.stopped()

                    return reconnecting(lines_read, attempts=reconnect_attempts)

                case StopMonitor():
                    return Behaviors.stopped()
            return Behaviors.same()

        return Behaviors.receive(receive)

    def reconnecting(lines_read: int, attempts: int) -> Behavior[MonitorMsg]:
        log = logger.bind(actor="monitor", instance_id=info.instance.id)

        if attempts >= max_reconnect_attempts:
            log.error("Giving up after {n} reconnect attempts", n=attempts)
            return Behaviors.stopped()

        delay = min(2.0 ** attempts, max_backoff)
        log.info(
            "Stream lost, reconnecting in {delay}s (attempt {n}/{max})",
            delay=delay, n=attempts + 1, max=max_reconnect_attempts,
        )

        async def _schedule_reconnect() -> _Reconnect:
            await asyncio.sleep(delay)
            return _Reconnect()

        async def receive(ctx: ActorContext[MonitorMsg], msg: MonitorMsg) -> Behavior[MonitorMsg]:
            match msg:
                case _Reconnect():
                    stream = _open_stream(start_line=lines_read)
                    log.info("Reopening stream at line {offset}", offset=lines_read)
                    ctx.pipe_to_self(_read_next(stream, info), mapper=lambda msg: msg)
                    return streaming(
                        stream, bootstrap_signaled=True,
                        lines_read=lines_read, reconnect_attempts=attempts + 1,
                    )

                case StopMonitor():
                    return Behaviors.stopped()
            return Behaviors.same()

        async def _enter(ctx: ActorContext[MonitorMsg]) -> Behavior[MonitorMsg]:
            ctx.pipe_to_self(_schedule_reconnect(), mapper=lambda msg: msg)
            return Behaviors.receive(receive)

        return Behaviors.setup(_enter)

    return Behaviors.setup(_setup)


def _convert(raw_event: object, info: NodeInstance) -> Event | None:
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
