"""Instance monitor actor — subscribes to transport for JSONL events.

An instance monitor tells this story: streaming → stopped.

The monitor sends SubscribeEvents to the transport actor and receives
pushed StreamEvent messages. It converts raw events to typed messages
and forwards them to event_listener. Bootstrap completion is detected
as a side effect.

Reconnection is handled entirely by the transport actor.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from casty import ActorContext, ActorRef, Behavior, Behaviors

if TYPE_CHECKING:
    from skyward.infra.ssh_actor import StreamEvent

from skyward.actors.messages import (
    BootstrapCommand,
    BootstrapConsole,
    BootstrapDone,
    BootstrapFailed,
    BootstrapPhase,
    Event,
    Log,
    Metric,
    NodeInstance,
)
from skyward.observability.logger import logger

from .messages import (
    MonitorMsg,
    StopMonitor,
)
from .state import MonitorState


def instance_monitor(
    info: NodeInstance,
    transport: ActorRef,  # ActorRef[TransportMsg]
    event_listener: ActorRef,
    reply_to: ActorRef[BootstrapDone],
) -> Behavior[MonitorMsg | StreamEvent]:
    """streaming → stopped."""

    async def _setup(ctx: ActorContext[MonitorMsg | StreamEvent]) -> Behavior[MonitorMsg | StreamEvent]:
        from skyward.infra.ssh_actor import SubscribeEvents
        log = logger.bind(actor="monitor", instance_id=info.instance.id)
        log.info("Subscribing to transport for events")
        transport.tell(SubscribeEvents(start_line=0, subscriber=ctx.self))
        s = MonitorState(
            info=info, transport=transport,
            event_listener=event_listener, reply_to=reply_to,
        )
        return streaming(s)

    return Behaviors.setup(_setup)


def streaming(
    s: MonitorState,
) -> Behavior[MonitorMsg | StreamEvent]:
    async def receive(
        ctx: ActorContext[MonitorMsg | StreamEvent],
        msg: MonitorMsg | StreamEvent,
    ) -> Behavior[MonitorMsg | StreamEvent]:
        from skyward.infra.ssh_actor import StreamEvent

        match msg:
            case StreamEvent(event=raw_event):
                log = logger.bind(actor="monitor", instance_id=s.info.instance.id)
                event = _convert(raw_event, s.info)
                if event is None:
                    return Behaviors.same()

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

                s.event_listener.tell(event)

                signaled = s.bootstrap_signaled
                if not signaled:
                    match event:
                        case BootstrapPhase(phase="bootstrap", event="completed"):
                            log.info("Bootstrap completion signal received")
                            s.reply_to.tell(BootstrapDone(instance=s.info, success=True))
                            signaled = True
                        case BootstrapFailed(error=error):
                            s.reply_to.tell(BootstrapDone(
                                instance=s.info, success=False, error=error,
                            ))
                            signaled = True

                if signaled != s.bootstrap_signaled:
                    return streaming(replace(s, bootstrap_signaled=signaled))
                return Behaviors.same()

            case StopMonitor():
                return Behaviors.stopped()

        return Behaviors.same()

    return Behaviors.receive(receive)


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
