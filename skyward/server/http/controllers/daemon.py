from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import msgspec
from litestar import Controller, get
from litestar.di import Provide
from litestar.exceptions import ValidationException
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.response import Stream

from skyward.api import v1
from skyward.server.application import ports
from skyward.server.http.controllers.events import MESSAGE, PING
from skyward.server.http.exceptions import failures
from skyward.server.http.references import narrowed
from skyward.server.http.representation import logged, summarized
from skyward.shared.observability import Query

WINDOW = 60 * 60 * 1000
"""The window a summary counts when it is not given one: the last hour."""

STEPS = 60
"""How many steps a summary's window is cut into when it is not told how long a step is."""

MOST_STEPS = 2000

ENCODER = msgspec.json.Encoder()


async def filtered(
    compute_id: str | None,
    level: v1.LogLevel = Parameter(default="DEBUG", description="The lowest level kept."),
    component: list[str] | None = Parameter(default=None, description="Keeps the lines of any one of these components."),
    node: str | None = Parameter(default=None, description="Keeps the lines about this node."),
    group: list[str] | None = Parameter(default=None, description="Keeps the lines of any one of these groups."),
    hide: list[str] | None = Parameter(default=None, description="Drops the lines of these groups."),
    contains: list[str] | None = Parameter(
        default=None, description="Keeps the lines that hold any one of these, ignoring case, in what they said or in the exception they carried."
    ),
) -> Query:
    """The filters every route over the daemon's log takes, and applies to the same effect."""
    return Query(
        level=level,
        components=frozenset(component or ()),
        compute=compute_id,
        node=node,
        groups=frozenset(group or ()),
        hidden=frozenset(hide or ()),
        contains=tuple(contains or ()),
    )


class DaemonLogController(Controller):
    path = "/daemon/log"
    tags = ["daemon"]
    dependencies = {"compute_id": Provide(narrowed), "wanted": Provide(filtered)}

    @get(
        summary="Read the daemon's log",
        description=(
            "What the daemon itself logged — reconciling, buying, connecting, dispatching, and every failure along the "
            "way — newest first, a page at a time. `cursor` is the `sequence` the previous page ended on.\n\n"
            "Every filter narrows the scan rather than the page, and they combine: `level` is the lowest level kept, "
            "a repeated `component`, `group` or `contains` keeps a line matching any one of them, and `hide` drops "
            "the groups it names. `since` and `until` bound it in time, `until` exclusive.\n\n"
            "Only a standalone daemon keeps a log; an application embedded in another process answers 404."
        ),
        responses=failures(404),
    )
    async def page(
        self,
        log: ports.DaemonLog,
        wanted: Query,
        since: int | None = Parameter(default=None, description="Milliseconds since the epoch the range starts at."),
        until: int | None = Parameter(default=None, description="Milliseconds since the epoch the range stops before."),
        cursor: str | None = None,
        limit: int = Parameter(default=200, ge=1, le=1000),
    ) -> v1.Page[v1.DaemonLogResource]:
        page = await log.page(replace(wanted, since=_moment(since), until=_moment(until)), cursor, limit)
        return v1.Page(items=tuple(logged(entry) for entry in page.items), next_cursor=page.next_cursor, total=None)

    @get(
        "/summary",
        summary="Summarize the daemon's log",
        description=(
            "A window of the log counted three ways: by level on each `step`, by component, and by group. A group is "
            "one logging call and the exception it was failing with, so a line repeated thirty thousand times is one "
            "group with a count, and a failure beside it is not lost among them. Groups come most severe first, then "
            "most frequent.\n\n"
            "The window is the last hour unless `since` and `until` say otherwise, cut into 60 steps unless `step` "
            f"does, and into at most {MOST_STEPS}. The component counts ignore the `component` filter, so they say "
            "what choosing another one would keep.\n\n"
            "`sequence` is the newest line counted: the stream opened with it as `Last-Event-ID` carries on from "
            "exactly there."
        ),
        responses=failures(404),
    )
    async def summary(
        self,
        log: ports.DaemonLog,
        wanted: Query,
        since: int | None = Parameter(default=None, description="Milliseconds since the epoch the window starts at."),
        until: int | None = Parameter(default=None, description="Milliseconds since the epoch the window stops before."),
        step: int | None = Parameter(default=None, ge=1, description="Milliseconds each count of the volume stands for."),
    ) -> v1.DaemonLogSummaryResource:
        end = until if until is not None else time.time_ns() // 1_000_000
        start = since if since is not None else end - WINDOW
        if start >= end:
            raise ValidationException("a window ends after it starts")
        width = step if step is not None else -(-(end - start) // STEPS)
        if end - start > width * MOST_STEPS:
            raise ValidationException(f"a window is counted in at most {MOST_STEPS} steps; ask for a longer `step`")
        return summarized(await log.summary(wanted, _at(start), _at(end), timedelta(milliseconds=width)))

    @get(
        "/stream",
        summary="Follow the daemon's log (SSE)",
        description=(
            "Every line the filters keep, as it is written. Each message's `id:` is the line's `sequence` and its "
            "`event:` is `log`. `Last-Event-ID` first replays what was written after it and then carries on, with "
            "nothing between the two.\n\n"
            "A reader that falls too far behind is hung up on and comes back from its last id. A stream with nothing "
            "to say for 15 seconds sends a comment line, `: ping`."
        ),
        responses={
            200: ResponseSpec(
                v1.DaemonLogResource,
                media_type="text/event-stream",
                description="One `data:` payload per line, framed as Server-Sent Events",
                generate_examples=False,
            ),
            **failures(404),
        },
    )
    async def stream(
        self,
        log: ports.DaemonLog,
        wanted: Query,
        last_event_id: str | None = Parameter(header="Last-Event-ID", default=None),
    ) -> Stream:
        async def messages() -> AsyncGenerator[bytes, None]:
            async for run in log.follow(wanted, int(last_event_id) if last_event_id else None):
                yield b"".join(MESSAGE % (entry.sequence, b"log", ENCODER.encode(logged(entry))) for entry in run) if run else PING

        return Stream(
            messages(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )


def _moment(ms: int | None) -> datetime | None:
    return None if ms is None else _at(ms)


def _at(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, UTC)
