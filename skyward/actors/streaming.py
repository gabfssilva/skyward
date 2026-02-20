"""Instance monitor actor - streams JSONL events from instances.

An instance monitor tells this story: connecting → streaming → stopped.

The monitor SSHes into an instance, reads the events.jsonl file
(tail -F with offset tracking), converts raw events to typed messages,
and sends them to pool_ref for observability. It runs for the entire
instance lifetime — not just bootstrap.

On SSH disconnection, the monitor reconnects from the last known line
offset, ensuring no events are lost between reconnections.

Bootstrap completion is detected as a side effect: when the monitor
sees BootstrapPhase(phase="bootstrap", event="completed"), it signals
the provider via reply_to. Streaming continues after that for metrics,
logs, and other runtime events.
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
    _StreamedEvent,
    _StreamEnded,
)

_MAX_RECONNECT_ATTEMPTS = 10
_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MAX_DELAY = 60.0

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


async def _reconnect(
    info: NodeInstance,
    ssh_user: str,
    ssh_key_path: str,
    lines_read: int,
    ctx: ActorContext[MonitorMsg],
) -> tuple[SSHTransport, AsyncIterator] | None:
    from skyward.infra.ssh import SSHTransport

    log = logger.bind(actor="monitor", instance_id=info.instance.id)

    for attempt in range(1, _MAX_RECONNECT_ATTEMPTS + 1):
        delay = min(_RECONNECT_BASE_DELAY * (2 ** (attempt - 1)), _RECONNECT_MAX_DELAY)
        log.info(
            "Reconnecting (attempt {attempt}/{max}, offset={offset}, delay={delay:.1f}s)",
            attempt=attempt, max=_MAX_RECONNECT_ATTEMPTS, offset=lines_read, delay=delay,
        )
        await asyncio.sleep(delay)

        try:
            transport = SSHTransport(
                host=info.instance.ip or "",
                user=ssh_user,
                key_path=ssh_key_path,
                port=info.instance.ssh_port,
                retry_max_attempts=10,
                retry_delay=3.0,
            )
            await transport.connect()
            log.info("Reconnected at line offset {offset}", offset=lines_read)
            stream = transport.stream_events(timeout=600.0, start_line=lines_read).__aiter__()
            ctx.pipe_to_self(_read_next(stream, info), mapper=lambda msg: msg)
            return transport, stream
        except Exception as e:
            log.warning("Reconnect attempt {attempt} failed: {err}", attempt=attempt, err=e)

    log.error("Exhausted reconnection attempts")
    return None


def instance_monitor(
    info: NodeInstance,
    ssh_user: str,
    ssh_key_path: str,
    event_listener: ActorRef,
    reply_to: ActorRef[BootstrapDone],
    ssh_timeout: float = 300.0,
    ssh_retry_interval: float = 5.0,
) -> Behavior[MonitorMsg]:
    """An instance monitor tells this story: connecting → streaming → stopped."""

    async def _setup(ctx: ActorContext[MonitorMsg]) -> Behavior[MonitorMsg]:
        from skyward.infra.ssh import SSHTransport

        log = logger.bind(actor="monitor", instance_id=info.instance.id)
        log.info("Connecting to {ip}:{port}", ip=info.instance.ip, port=info.instance.ssh_port)

        transport = SSHTransport(
            host=info.instance.ip or "",
            user=ssh_user,
            key_path=ssh_key_path,
            port=info.instance.ssh_port,
            retry_max_attempts=int(ssh_timeout / ssh_retry_interval) + 1,
            retry_delay=ssh_retry_interval,
        )
        await transport.connect()

        log.info("Connected")

        stream = transport.stream_events(timeout=600.0, start_line=0).__aiter__()

        ctx.pipe_to_self(_read_next(stream, info), mapper=lambda msg: msg)

        return Behaviors.with_lifecycle(
            streaming(transport, stream, bootstrap_signaled=False, lines_read=0),
            post_stop=lambda _: transport.close(),
        )

    def streaming(
        transport: SSHTransport,
        stream: AsyncIterator,
        bootstrap_signaled: bool,
        lines_read: int,
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

                    if new_lines_read != lines_read or signaled != bootstrap_signaled:
                        return streaming(transport, stream, signaled, new_lines_read)
                    return Behaviors.same()

                case _StreamEnded(error=error):
                    if error is None:
                        logger.bind(
                            actor="monitor", instance_id=info.instance.id,
                        ).info("Stream ended cleanly")
                    if error is not None:
                        logger.bind(actor="monitor", instance_id=info.instance.id).warning(
                            "Stream ended with error, attempting reconnect: {err}", err=error,
                        )
                        await transport.close()
                        result = await _reconnect(
                            info, ssh_user, ssh_key_path, lines_read, ctx,
                        )
                        if result is not None:
                            new_transport, new_stream = result
                            return Behaviors.with_lifecycle(
                                streaming(
                                    new_transport, new_stream, bootstrap_signaled, lines_read,
                                ),
                                post_stop=lambda _: new_transport.close(),
                            )

                    if not bootstrap_signaled:
                        reply_to.tell(BootstrapDone(
                            instance=info,
                            success=False,
                            error=error or "stream ended before bootstrap completed",
                        ))
                    return Behaviors.stopped()

                case StopMonitor():
                    return Behaviors.stopped()

        return Behaviors.receive(receive)

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
