"""Instance monitor — a task consuming transport events.

Consumes ``SshTransport.events()`` (JSONL events streamed from the node),
emits domain events (metrics, logs) and forwards bootstrap/console
messages to ``on_stream``. Bootstrap completion is signalled once via
``on_bootstrap_done``. Reconnection is handled entirely by the transport;
the loop ends when the transport closes or permanently fails.
"""
from __future__ import annotations

from collections.abc import Callable

from skyward.actors.messages import (
    BootstrapCommand,
    BootstrapFailed,
    BootstrapPhase,
    ConsoleOutput,
    Event,
    Log,
    Metric,
    NodeInstance,
)
from skyward.api import events as domain
from skyward.infra.ssh_transport import SshTransport
from skyward.observability.logger import logger

type StreamMsg = BootstrapPhase | BootstrapCommand | BootstrapFailed | ConsoleOutput


def _no_emit(_event: domain.SessionEvent) -> None:
    pass


async def monitor_instance(
    transport: SshTransport,
    info: NodeInstance,
    on_stream: Callable[[StreamMsg], None],
    on_bootstrap_done: Callable[[bool, str | None], None],
    emit: domain.Emit | None = None,
    pool_name: str = "",
) -> None:
    emit = emit if emit is not None else _no_emit
    log = logger.bind(component="monitor", instance_id=info.instance.id)
    log.info("Monitoring transport events")
    bootstrap_signaled = False

    async for stream_event in transport.events(start_line=0):
        event = _convert(stream_event.event, info)
        if event is None:
            continue

        match event:
            case Metric(name=name, value=value):
                log.trace("metric: {name}={value}", name=name, value=value)
                emit(domain.Metric.Sampled(pool_name, info.node, name, value))
            case Log(line=line, overwrite=ow):
                log.info("{line}", line=line.rstrip())
                if stripped := line.strip():
                    emit(domain.Log.Emitted(pool_name, info.node, stripped, overwrite=ow))
            case BootstrapPhase():
                log.info("Phase {phase}: {ev}", phase=event.phase, ev=event.event)
                on_stream(event)
            case BootstrapCommand():
                log.info("Command: {cmd}", cmd=event.command)
                on_stream(event)
            case ConsoleOutput():
                log.info("{content}", content=event.content.rstrip())
                on_stream(event)
            case BootstrapFailed():
                log.error("Bootstrap failed: {err}", err=event.error)
                on_stream(event)

        if not bootstrap_signaled:
            match event:
                case BootstrapPhase(phase="bootstrap", event="completed"):
                    log.info("Bootstrap completion signal received")
                    on_bootstrap_done(True, None)
                    bootstrap_signaled = True
                case BootstrapFailed(error=error):
                    on_bootstrap_done(False, error)
                    bootstrap_signaled = True


def _convert(raw_event: object, info: NodeInstance) -> Event | None:
    from skyward.infra.ssh import (
        RawBootstrapCommand,
        RawBootstrapPhase,
        RawConsoleOutput,
        RawLogEvent,
        RawMetricEvent,
    )

    match raw_event:
        case RawConsoleOutput(content=content, stream=stream, overwrite=ow):
            return ConsoleOutput(instance=info, content=content, stream=stream, overwrite=ow)
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
        case RawLogEvent(content=content, stream=stream, overwrite=ow):
            return Log(instance=info, line=content, stream=stream, overwrite=ow)
        case _:
            return None
