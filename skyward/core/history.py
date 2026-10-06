"""What an open compute screen reads: the compute with its history, its tasks, its log.

:class:`~skyward.core.fleet.FleetObserver` follows the event stream, which is right
for what is happening now and says nothing of what happened before anybody was
looking. A screen on one compute wants more than that, and asks the API for it:
the last ten minutes of each node's utilisation, the tasks in the order they
matter, a page of the event log. This module is those reads and what is derived
from them, as values — no drawing, no widgets.

The metrics are read in fixed buckets of :data:`SPARK_STEP_MS` ending at the
bucket ``now`` falls in, so a sparkline is the same window on every repaint and
moves only when a bucket turns over. Idleness is read off the same window: the
fleet is idle since the bucket after the last one whose mean GPU utilisation
reached :data:`IDLE_GPU_PCT`, once that is :data:`IDLE_BUCKETS` buckets behind.

Every read lets errors propagate. What a failed read means is the caller's.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from skyward.api.v1 import (
    ComputeResource,
    LogEntryResource,
    MetricHistoryResource,
    NodeResource,
    NodeState,
    Page,
    TaskOrder,
    TaskResource,
    TaskState,
)
from skyward.core.client import Client

SPARK_STEP_MS = 30_000
SPARK_BUCKETS = 20
IDLE_GPU_PCT = 5.0
IDLE_BUCKETS = 4
RECENT = 5
TASKS_PER_PAGE = 8
LOG_LINES = 20

COUNTED: tuple[TaskState, ...] = ("running", "queued", "succeeded", "failed")

type NodeGroup = Literal["active", "bootstrapping", "stopping", "stopped", "failed"]

NODE_GROUPS: tuple[NodeGroup, ...] = ("active", "bootstrapping", "stopping", "stopped", "failed")

DRAINABLE: frozenset[NodeState] = frozenset({"requested", "provisioning", "connecting", "bootstrapping", "ready"})

_INCLUDE = "nodes.replaced,nodes.metrics,nodes.phases,nodes.running"


@dataclass(frozen=True, slots=True)
class Idle:
    since: datetime
    capped: bool
    """No bucket of the window was busy, so the idleness began before the window did."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    compute: ComputeResource
    sparks: Mapping[tuple[str, str], tuple[float | None, ...]]
    """Keyed by node id and metric name; each value has :data:`SPARK_BUCKETS` entries, oldest first."""
    averages: Mapping[str, float]
    idle: Idle | None
    at: datetime


@dataclass(frozen=True, slots=True)
class TaskPage:
    items: tuple[TaskResource, ...]
    total: int | None
    next: str | None
    counts: Mapping[TaskState, int]


@dataclass(frozen=True, slots=True)
class LogPage:
    entries: tuple[LogEntryResource, ...]
    next: str | None


async def snapshot(client: Client, compute_id: str, now: datetime) -> Snapshot:
    compute = await client.call("GET", f"/v1/computes/{compute_id}", ComputeResource, include=_INCLUDE)
    start = _window(now)
    history = (
        await client.call(
            "GET",
            f"/v1/computes/{compute_id}/metrics",
            MetricHistoryResource,
            name=("cpu", "gpu_util"),
            since=start,
            until=start + SPARK_BUCKETS * SPARK_STEP_MS,
            step=SPARK_STEP_MS,
            agg="avg",
        )
        if compute.nodes
        else None
    )
    sparks = _sparks(history, start)
    live = alive(compute)
    return Snapshot(compute, sparks, _averages(live), _idle(live, sparks, start), now)


async def recent(client: Client, limit: int = RECENT) -> tuple[ComputeResource, ...]:
    page = await client.call("GET", "/v1/computes", Page[ComputeResource], live=False, limit=limit)
    return page.items


async def tasks(
    client: Client,
    compute_id: str,
    *,
    order: TaskOrder = "state",
    function: str | None = None,
    cursor: str | None = None,
    seen: int = 0,
    limit: int = TASKS_PER_PAGE,
) -> TaskPage:
    """One page of a compute's tasks, and how many it holds in each of :data:`COUNTED`.

    ``seen`` is how many the pages before this one held. A last page exactly
    ``limit`` long still carries a cursor that leads to an empty page, so when the
    total is known the count is what ends the walk.
    """
    async with asyncio.TaskGroup() as requests:
        page = requests.create_task(
            client.call("GET", "/v1/tasks", Page[TaskResource], compute=compute_id, limit=limit, order=order, function=function, cursor=cursor)
        )
        counted = {
            state: requests.create_task(client.call("GET", "/v1/tasks", Page[TaskResource], compute=compute_id, function=function, state=state, limit=1))
            for state in COUNTED
        }

    read = page.result()
    last = read.total is not None and seen + len(read.items) >= read.total
    return TaskPage(
        items=read.items,
        total=read.total,
        next=None if last else read.next_cursor,
        counts={state: counted[state].result().total or 0 for state in COUNTED},
    )


async def log(
    client: Client,
    compute_id: str,
    *,
    node: str | None = None,
    task: str | None = None,
    term: str = "",
    cursor: str | None = None,
    limit: int = LOG_LINES,
) -> LogPage:
    page = await client.call(
        "GET",
        "/v1/events/log",
        Page[LogEntryResource],
        compute=compute_id,
        limit=limit,
        node=node,
        task=task,
        contains=term or None,
        cursor=cursor,
    )
    return LogPage(page.items, page.next_cursor)


def alive(compute: ComputeResource) -> tuple[NodeResource, ...]:
    return tuple(node for node in compute.nodes if node.terminated_at is None)


def group(node: NodeResource) -> NodeGroup:
    """Where a node stands in the fleet.

    A node given up on keeps the error that condemned it through ``deleting`` and
    ``deleted``, so on the way out the error, not the state, tells a failure from a
    node that was drained or scaled away.
    """
    match node.state:
        case "ready":
            return "active"
        case "requested" | "provisioning" | "connecting" | "bootstrapping":
            return "bootstrapping"
        case "lost" | "failed":
            return "failed"
        case _ if node.last_error is not None:
            return "failed"
        case "deleted":
            return "stopped"
        case _:
            return "stopping"


def deletable(compute: ComputeResource) -> bool:
    return compute.ended is None and compute.status.state not in ("deleting", "deleted")


def _window(now: datetime) -> int:
    return (int(now.timestamp() * 1000) // SPARK_STEP_MS - (SPARK_BUCKETS - 1)) * SPARK_STEP_MS


def _sparks(history: MetricHistoryResource | None, start: int) -> Mapping[tuple[str, str], tuple[float | None, ...]]:
    def buckets(times: tuple[int, ...], values: tuple[float, ...]) -> tuple[float | None, ...]:
        slots: list[float | None] = [None] * SPARK_BUCKETS
        for at, value in zip(times, values, strict=False):
            if 0 <= (index := (at - start) // SPARK_STEP_MS) < SPARK_BUCKETS:
                slots[index] = value
        return tuple(slots)

    if history is None:
        return {}
    return {(series.node, series.name): buckets(series.at, series.values) for series in history.series}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _averages(live: tuple[NodeResource, ...]) -> Mapping[str, float]:
    reported = [{name: gauge.value for name, gauge in node.metrics.items()} for node in live if isinstance(node.metrics, dict)]
    names = {name for gauges in reported for name in gauges}
    means = {name: _mean([gauges[name] for gauges in reported if name in gauges]) for name in names}
    return {name: mean for name, mean in means.items() if mean is not None}


def _idle(live: tuple[NodeResource, ...], sparks: Mapping[tuple[str, str], tuple[float | None, ...]], start: int) -> Idle | None:
    series = [sparks[key] for node in live if (key := (node.id, "gpu_util")) in sparks]
    window = [_mean([value for values in series if (value := values[index]) is not None]) for index in range(SPARK_BUCKETS)]
    busy = max((index for index, value in enumerate(window) if value is not None and value >= IDLE_GPU_PCT), default=-1)
    if all(value is None for value in window) or SPARK_BUCKETS - 1 - busy < IDLE_BUCKETS:
        return None
    return Idle(since=datetime.fromtimestamp((start + (busy + 1) * SPARK_STEP_MS) / 1000, UTC), capped=busy < 0)


__all__ = [
    "COUNTED",
    "DRAINABLE",
    "IDLE_BUCKETS",
    "IDLE_GPU_PCT",
    "LOG_LINES",
    "NODE_GROUPS",
    "RECENT",
    "SPARK_BUCKETS",
    "SPARK_STEP_MS",
    "TASKS_PER_PAGE",
    "Idle",
    "LogPage",
    "NodeGroup",
    "Snapshot",
    "TaskPage",
    "alive",
    "deletable",
    "group",
    "log",
    "recent",
    "snapshot",
    "tasks",
]
