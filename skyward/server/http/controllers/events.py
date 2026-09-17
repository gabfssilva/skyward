from __future__ import annotations

from collections.abc import AsyncGenerator

from litestar import Controller, get
from litestar.di import Provide
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.response import Stream

from skyward.api import v1
from skyward.server.application import ports
from skyward.server.http.references import narrowed
from skyward.server.http.representation import recast

MESSAGE = b"id: %d\r\nevent: %s\r\ndata: %s\r\n\r\n"
"""One Server-Sent Event, byte for byte as litestar's ``ServerSentEvent`` frames it.

A payload is compact JSON, so it is always one ``data:`` line. The frames are written
here rather than by ``ServerSentEvent``, which sends every message on its own: the store
hands over runs of events, and a run goes out as one write of one message per event.
"""


class EventController(Controller):
    path = "/events"
    tags = ["events"]
    dependencies = {"compute_id": Provide(narrowed)}

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
            "`compute.lease.released`, `compute.abandoned`, `compute.deleting`, `compute.deletion_failed`, "
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
                v1.Event,
                media_type="text/event-stream",
                description="One `data:` payload per message, framed as Server-Sent Events",
                generate_examples=False,
            )
        },
    )
    async def stream(
        self,
        events: ports.Events,
        compute_id: str | None,
        task: str | None = None,
        types: list[str] | None = None,
        last_event_id: str | None = Parameter(header="Last-Event-ID", default=None),
    ) -> Stream:
        async def messages() -> AsyncGenerator[bytes, None]:
            async for run in events.stream(last_event_id, compute_id, task, tuple(types) if types else None):
                yield b"".join(MESSAGE % (sequence, event_type.encode(), payload) for sequence, event_type, payload in run)

        return Stream(
            messages(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    @get(
        "/log",
        summary="Read the event log",
        description=(
            "The recorded events, newest first, a page at a time: the last lines of every compute without replaying "
            "the log from its first. `types` filters on the frame name, as the stream does, and `cursor` is the "
            "`sequence` the previous page ended on.\n\n"
            "An entry's `sequence` is the id the stream gives the same event, so a page can be followed with the "
            "stream from its newest entry and nothing falls between the two. `compute.cost`, `node.metrics` and "
            "`node.progress` are published rather than recorded, and are never in the log.\n\n"
            "Every filter narrows the query rather than the page: `compute`, `node` and `task` scope it, and "
            "`contains` keeps the entries whose printed line holds any one of the strings, case-insensitively. A "
            "reader after one node's output, or after every line that said `Traceback`, asks for that."
        ),
    )
    async def log(
        self,
        events: ports.Events,
        compute_id: str | None,
        cursor: str | None = None,
        limit: int = Parameter(default=200, ge=1, le=1000),
        task: str | None = None,
        node: str | None = None,
        types: list[str] | None = None,
        contains: list[str] | None = Parameter(default=None, description="Keeps the entries whose printed line holds any one of these."),
    ) -> v1.Page[v1.LogEntryResource]:
        page = await events.log(
            cursor,
            limit,
            compute=compute_id,
            task=task,
            node=node,
            types=tuple(types) if types else None,
            contains=tuple(contains) if contains else None,
        )
        return recast(page, v1.Page[v1.LogEntryResource])
