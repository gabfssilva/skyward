from __future__ import annotations

from collections.abc import AsyncGenerator

from litestar import Controller, get
from litestar.params import Parameter
from litestar.response import ServerSentEvent, ServerSentEventMessage

from skyward.server.application import ports


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
            "source of truth."
        ),
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
