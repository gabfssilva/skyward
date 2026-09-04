from __future__ import annotations

from collections.abc import AsyncGenerator

from litestar import Controller, get
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.response import ServerSentEvent, ServerSentEventMessage

from skyward.server.application import ports
from skyward.shared.events import Event


class EventController(Controller):
    path = "/events"
    tags = ["events"]

    @get(
        summary="Event stream (SSE)",
        description=(
            "Each event's `id:` is the **global sequence**, monotonic. `Last-Event-ID` replays from the event log; the "
            "snapshot and the cursor are captured in the same logical order, so there is no gap between reading state "
            "and subscribing.\n\n"
            "There is no automatic event GC, so any valid cursor stays resumable.\n\n"
            "A slow consumer never blocks a commit: the adapter closes the connection when its local queue overflows and "
            "the client reconnects from its last id.\n\n"
            "Task stdout/stderr and node bootstrap output are events here. There is no `logs` resource with a second "
            "source of truth.\n\n"
            "The schema below is one message's `data:`, not the stream. Its `type` tag discriminates the union; the "
            "frame's `event:` field is what `types` filters on. For a compute the two are the same name, one per fact: "
            "`compute.created`, `compute.bound`, `compute.adopted`, `compute.provisioning`, `compute.ready`, "
            "`compute.degraded`, `compute.generation.created`, `compute.generation.applied`, `compute.lease.claimed`, "
            "`compute.lease.released`, `compute.abandoned`, `compute.deleting`, `compute.release_failed`, "
            "`compute.strays_terminated`, `compute.deleted`, `compute.cost`. Every compute state change is one of them: "
            "there is no way to move a compute's state without the stream saying so. For a node or a task the frame is "
            "finer than the tag:\n\n"
            "| `event:` | `data.type` |\n"
            "|---|---|\n"
            "| `node.{state}` — ten of them | `node.state` |\n"
            "| `node.progress` | `node.progress` |\n"
            "| `node.console` | `node.console` |\n"
            "| `node.phase` | `node.phase` |\n"
            "| `node.metrics` | `node.metrics` |\n"
            "| `task.started`, `task.succeeded`, `task.failed`, `task.indeterminate` | `task.state` |\n\n"
            "`compute.cost`, `node.metrics` and `node.progress` are published rather than recorded: they ride the live "
            "feed, carry the last sequence seen rather than one of their own, and never replay."
        ),
        responses={
            200: ResponseSpec(
                Event,
                media_type="text/event-stream",
                description="One `data:` payload per message, framed as Server-Sent Events",
                generate_examples=False,
            )
        },
    )
    async def stream(
        self,
        events: ports.Events,
        compute: str | None = None,
        task: str | None = None,
        types: list[str] | None = None,
        last_event_id: str | None = Parameter(header="Last-Event-ID", default=None),
    ) -> ServerSentEvent:
        async def messages() -> AsyncGenerator[ServerSentEventMessage, None]:
            async for sequence, event_type, payload in events.stream(last_event_id, compute, task, tuple(types) if types else None):
                yield ServerSentEventMessage(id=str(sequence), event=event_type, data=payload)

        return ServerSentEvent(messages())
