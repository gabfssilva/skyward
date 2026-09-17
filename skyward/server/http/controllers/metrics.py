from __future__ import annotations

import time

from litestar import Controller, get
from litestar.exceptions import ValidationException
from litestar.params import Parameter

from skyward.api import v1
from skyward.server.application import ports
from skyward.server.http.exceptions import failures
from skyward.server.http.representation import recast

RAW_SPAN = 60 * 60 * 1000
"""The longest range answered sample by sample, in milliseconds. Past it, a range asks for a ``step``."""

BUCKETS = 2000
"""The most steps one range is answered in."""


class MetricController(Controller):
    path = "/computes/{compute:str}/metrics"
    tags = ["metrics"]

    @get(
        summary="Read a compute's metrics",
        description=(
            "Every metric of every node, as one series per node and name: `at` in milliseconds since the epoch on the "
            "node's clock, and the value measured then. Ask with exactly one of `since` and `after`.\n\n"
            "**A range** — `since`, and optionally `until` (open when left out, so a node whose clock runs ahead is not "
            "cut off). Without `step` the samples come as they were taken, over at most an hour; with `step`, as one "
            "value per step of that many milliseconds, starting on a multiple of it, `agg` saying which: `avg`, `min`, "
            "`max` or `last`, over at most 2000 steps.\n\n"
            "**A cursor** — `after`, the `cursor` a previous answer handed out. It answers with what was recorded since "
            "that answer, whenever it was measured: a node whose link dropped delivers its backlog late, and the backlog "
            "is still new. `reset: true` means some of it has already been compacted out of reach of a cursor — read the "
            "range again.\n\n"
            "Samples reach this a couple of seconds after the node takes them. `node` and `name` narrow it, each repeatable."
        ),
        responses=failures(404),
    )
    async def history(
        self,
        compute_id: str,
        metrics: ports.Metrics,
        since: int | None = Parameter(default=None, description="Milliseconds since the epoch the range starts at."),
        until: int | None = Parameter(default=None, description="Milliseconds since the epoch the range stops before."),
        after: str | None = Parameter(default=None, description="The `cursor` of a previous answer."),
        step: int | None = Parameter(default=None, ge=1, description="Milliseconds each value of a range stands for."),
        agg: v1.Aggregate = Parameter(default="avg", description="How the samples of one step become its value."),
        node: list[str] | None = Parameter(default=None, description="Keeps the nodes named."),
        name: list[str] | None = Parameter(default=None, description="Keeps the metrics named."),
    ) -> v1.MetricHistoryResource:
        match since, after:
            case None, str() as cursor if cursor.isdigit():
                return recast(await metrics.after(compute_id, cursor, node, name), v1.MetricHistoryResource)
            case None, str():
                raise ValidationException("`after` takes the `cursor` a previous answer handed out")
            case int() as start, None:
                span = (until if until is not None else time.time_ns() // 1_000_000) - start
                if step is None and span > RAW_SPAN:
                    raise ValidationException(f"a range of raw samples spans at most {RAW_SPAN} ms; ask for a `step`")
                if step is not None and span > step * BUCKETS:
                    raise ValidationException(f"a range is answered in at most {BUCKETS} steps; ask for a longer `step`")
                return recast(await metrics.series(compute_id, start, until, step, agg, node, name), v1.MetricHistoryResource)
            case _:
                raise ValidationException("ask with exactly one of `since` and `after`")

    @get(
        "/latest",
        summary="Read the newest value of each metric",
        description=(
            "One item per node and metric: the newest sample the daemon holds. A node that has been quiet for longer "
            "than a compaction window is answered from its compacted history, so a compute that is gone still says "
            "where its nodes were when they stopped. `node` and `name` narrow it, each repeatable."
        ),
        responses=failures(404),
    )
    async def latest(
        self,
        compute_id: str,
        metrics: ports.Metrics,
        node: list[str] | None = Parameter(default=None, description="Keeps the nodes named."),
        name: list[str] | None = Parameter(default=None, description="Keeps the metrics named."),
    ) -> v1.Page[v1.MetricSampleResource]:
        return v1.Page(items=recast(await metrics.latest(compute_id, node, name), tuple[v1.MetricSampleResource, ...]), next_cursor=None, total=None)
