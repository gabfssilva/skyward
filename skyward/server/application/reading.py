"""What a client reads about a compute, gathered in one pass.

A compute is read with the node holding each rank, a node with what it is holding,
and a task with the compute and the code it names — each answered by a different
store. This is where they meet, and where a block nobody asked for is never read.

What comes out are the daemon's own records, grouped: a machine's binding is still
in them, password and all. Deciding what of it a client sees is the edge's.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import fmean
from typing import Literal

import msgspec

from skyward.server.application import ports
from skyward.server.persistence.nodes import LIVE
from skyward.server.persistence.store import now
from skyward.shared.errors import NotFoundError
from skyward.shared.events import ConsoleEvent, PhaseEvent
from skyward.shared.provider import Machine
from skyward.shared.schemas import (
    Compute,
    ComputeState,
    DeletionCause,
    Function,
    MetricHistory,
    MetricSample,
    Node,
    Page,
    Task,
    TaskOrder,
    TaskState,
)

type Block = Literal["metrics", "phases", "running", "tail", "replaced", "latest", "pace", "utilization"]
"""What a read may be asked to carry beyond what it always does."""

TAIL = 20
"""How many of the lines a node printed last its tail carries."""

PHASES = 500
"""The most phase marks read for one node: a bootstrap turns over a few dozen."""

SPAN = 10 * 60 * 1000
"""How far back utilization reaches, in milliseconds."""

STEP = 10 * 1000
"""How long one value of utilization stands for, in milliseconds."""

PACED = timedelta(hours=1)
"""How far back the pace of a compute's tasks is counted."""

FAILED: tuple[TaskState, ...] = ("failed", "timed_out", "indeterminate")
"""The outcomes the last failure is picked from."""


@dataclass(frozen=True, slots=True)
class Code:
    """A function a task names, and the record of it — none for code nobody registered."""

    sha256: str
    function: Function | None


@dataclass(frozen=True, slots=True)
class Running:
    task: str
    ordinal: int
    code: Code
    started_at: datetime | None


@dataclass(frozen=True, slots=True)
class NodeReading:
    """A node, the machine behind it, and whichever blocks were asked for — ``None`` for the ones that were not."""

    node: Node
    machine: Machine | None
    busy: int
    metrics: tuple[MetricSample, ...] | None = None
    phases: tuple[PhaseEvent, ...] | None = None
    """The latest mark of each phase, in the order the phases began."""
    running: tuple[Running, ...] | None = None
    tail: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class Finished:
    task: Task
    code: Code


@dataclass(frozen=True, slots=True)
class Latest:
    succeeded: Finished | None
    failed: Finished | None


@dataclass(frozen=True, slots=True)
class Utilization:
    """The fleet's average of two readings, one value per ``step`` from ``since``; ``None`` where nobody reported."""

    since: int
    step: int
    gpu: tuple[float | None, ...]
    cpu: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class ComputeReading:
    compute: Compute
    nodes: tuple[NodeReading, ...]
    """The nodes holding a rank, and the replaced ones too when they were asked for."""
    rate: float
    latest: Latest | None = None
    pace: ports.Pace | None = None
    utilization: Utilization | None = None


@dataclass(frozen=True, slots=True)
class TaskReading:
    task: Task
    compute: str | None
    """The name of the compute it was given to."""
    code: Code


class Reader:
    def __init__(
        self,
        computes: ports.Computes,
        nodes: ports.Nodes,
        tasks: ports.Tasks,
        functions: ports.Functions,
        events: ports.Events,
        metrics: ports.Metrics,
    ) -> None:
        self._computes = computes
        self._nodes = nodes
        self._tasks = tasks
        self._functions = functions
        self._events = events
        self._metrics = metrics

    async def compute(self, ref: str, blocks: Collection[Block] = ()) -> ComputeReading:
        return await self._compute(await self._computes.get(ref), blocks)

    async def computes(
        self,
        cursor: str | None,
        limit: int,
        state: ComputeState | None,
        owned: bool | None,
        live: bool | None,
        cause: DeletionCause | None,
        blocks: Collection[Block] = (),
    ) -> Page[ComputeReading]:
        page = await self._computes.list(cursor, limit, state, owned, live, cause)
        return Page(items=tuple([await self._compute(compute, blocks) for compute in page.items]), next_cursor=page.next_cursor, total=page.total)

    async def nodes(self, compute: str, blocks: Collection[Block] = ()) -> tuple[NodeReading, ...]:
        return await self._read(compute, _shown(await self._nodes.of(compute), blocks), blocks)

    async def node(self, compute: str, ref: str, blocks: Collection[Block] = ()) -> NodeReading:
        """By id, or by rank: the node holding it, else the last one that did."""
        nodes = await self._nodes.of(compute)
        ranked = sorted((node for node in nodes if node.rank == int(ref)), key=lambda node: (node.state in LIVE, node.created_at)) if ref.isdigit() else ()
        match ranked or [node for node in nodes if node.id == ref]:
            case [*_, found]:
                (reading,) = await self._read(compute, (found,), blocks)
                return reading
            case _:
                raise NotFoundError(f"no such node: {ref}")

    async def task(self, task_id: str) -> TaskReading:
        (reading,) = await self._named((await self._tasks.get(task_id),))
        return reading

    async def tasks(
        self,
        cursor: str | None,
        limit: int,
        compute: str | None,
        states: tuple[TaskState, ...],
        correlation_id: str | None,
        function: str | None,
        order: TaskOrder,
    ) -> Page[TaskReading]:
        page = await self._tasks.list(cursor, limit, compute, states, correlation_id, function, order)
        return Page(items=await self._named(page.items), next_cursor=page.next_cursor, total=page.total)

    async def _compute(self, compute: Compute, blocks: Collection[Block]) -> ComputeReading:
        nodes = await self._nodes.of(compute.id)
        return ComputeReading(
            compute=compute,
            nodes=await self._read(compute.id, _shown(nodes, blocks), blocks),
            rate=round(sum(node.price_per_hour or 0 for node in nodes if node.launched_at is not None and node.terminated_at is None), 6),
            latest=await self._latest(compute.id) if "latest" in blocks else None,
            pace=await self._tasks.pace(compute.id, now() - PACED) if "pace" in blocks else None,
            utilization=await self._utilization(compute.id) if "utilization" in blocks else None,
        )

    async def _read(self, compute: str, nodes: Iterable[Node], blocks: Collection[Block]) -> tuple[NodeReading, ...]:
        held = await self._tasks.held(compute)
        busy = Counter(attempt.node for attempt in held)
        samples = await self._metrics.latest(compute) if "metrics" in blocks else ()
        codes = await self._codes({attempt.function for attempt in held}) if "running" in blocks else {}

        return tuple([
            NodeReading(
                node=node,
                machine=msgspec.convert(node.provider_binding, Machine) if node.provider_binding else None,
                busy=busy[node.id],
                metrics=tuple(sample for sample in samples if sample.node == node.id) if "metrics" in blocks else None,
                phases=await self._phases(node.id) if "phases" in blocks else None,
                running=(
                    tuple(Running(attempt.task, attempt.ordinal, codes[attempt.function], attempt.started_at) for attempt in held if attempt.node == node.id)
                    if "running" in blocks
                    else None
                ),
                tail=await self._tail(node.id) if "tail" in blocks else None,
            )
            for node in nodes
        ])

    async def _phases(self, node: str) -> tuple[PhaseEvent, ...]:
        page = await self._events.log(None, PHASES, node=node, types=("node.phase",))
        latest: dict[str, PhaseEvent] = {}
        for entry in reversed(page.items):
            match entry.data:
                case PhaseEvent() as mark:
                    latest[mark.phase] = mark
        return tuple(latest.values())

    async def _tail(self, node: str) -> tuple[str, ...]:
        page = await self._events.log(None, TAIL, node=node, types=("node.console",))
        lines: list[str] = []
        for entry in reversed(page.items):
            match entry.data:
                case ConsoleEvent(content=content):
                    lines.append(content)
        return tuple(lines)

    async def _latest(self, compute: str) -> Latest:
        succeeded = await self._tasks.list(None, 1, compute, ("succeeded",), order="finished")
        failed = await self._tasks.list(None, 1, compute, FAILED, order="finished")
        codes = await self._codes({task.function for task in (*succeeded.items, *failed.items)})
        return Latest(
            succeeded=next((Finished(task, codes[task.function]) for task in succeeded.items), None),
            failed=next((Finished(task, codes[task.function]) for task in failed.items), None),
        )

    async def _utilization(self, compute: str) -> Utilization:
        until = time.time_ns() // 1_000_000
        since = (until - SPAN) // STEP * STEP
        history = await self._metrics.series(compute, since, None, STEP, "avg", None, ("gpu_util", "cpu"))
        return Utilization(since=since, step=STEP, gpu=_averaged(history, "gpu_util", since, until), cpu=_averaged(history, "cpu", since, until))

    async def _named(self, tasks: tuple[Task, ...]) -> tuple[TaskReading, ...]:
        names = await self._computes.named({task.compute_id for task in tasks})
        codes = await self._codes({task.function for task in tasks})
        return tuple(TaskReading(task, names[task.compute_id], codes[task.function]) for task in tasks)

    async def _codes(self, digests: Collection[str]) -> dict[str, Code]:
        return {digest: Code(digest, await self._function(digest)) for digest in digests}

    async def _function(self, digest: str) -> Function | None:
        try:
            return await self._functions.get(digest)
        except NotFoundError:
            return None


def _shown(nodes: Iterable[Node], blocks: Collection[Block]) -> tuple[Node, ...]:
    """The nodes holding a rank, by rank — and the ones that held one before, when asked for."""
    return tuple(sorted((node for node in nodes if "replaced" in blocks or node.state in LIVE), key=lambda node: (node.rank, node.created_at)))


def _averaged(history: MetricHistory, name: str, since: int, until: int) -> tuple[float | None, ...]:
    """One reading across every node, one value per step: the mean of the nodes that reported in it."""
    steps: defaultdict[int, list[float]] = defaultdict(list)
    for series in history.series:
        if series.name == name:
            for at, value in zip(series.at, series.values, strict=True):
                steps[at].append(value)
    return tuple(fmean(values) if (values := steps.get(start)) else None for start in range(since, until, STEP))
