"""The compute, as one value: what every watcher of the event stream folds it into.

The daemon speaks in events — a node moved, a phase turned over, a gauge was
read — and anything watching a compute wants the accumulated answer, not the
stream. This module is that answer: an immutable :class:`ComputeView`, a pure
fold (:func:`observe`) that advances it one event at a time, and a hydration
step (:func:`refresh`) for the fields only the API carries, such as a node's
address or a task's timings.

It lives outside the UI because the UI is only one subscriber. The Rich panel,
the line console, and every callback a user registers with ``callbacks=`` are
handed the same view, folded once. The windows (``tail``, ``metrics``,
``errors``) are bounded so a compute that stays up for days never grows the
view with it — the full history is in the event log, not here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType

import msgspec

from skyward.shared import lifecycle
from skyward.shared.events import (
    ComputeDegraded,
    ComputeReleaseFailed,
    ConsoleEvent,
    CostEvent,
    Event,
    MetricEvent,
    NodeEvent,
    PhaseEvent,
    ProgressEvent,
    TaskEvent,
)
from skyward.shared.schemas import (
    Compute as ComputeResource,
)
from skyward.shared.schemas import (
    ComputeState,
    Node,
    NodeState,
    Page,
    Task,
    TaskEventState,
    TaskState,
)

HISTORY = 12
"""Samples kept per gauge, per node."""

TAIL = 40
"""Lines kept of what a node printed."""

ERRORS = 32
"""Messages kept of what has gone wrong, oldest dropped first."""


@dataclass(frozen=True, slots=True)
class PhaseView:
    """One bootstrap step on one node: named, and either underway, done, or broken."""

    name: str
    started: bool = False
    finished: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class NodeView:
    """One machine of the pool, as far as the stream and the API have said."""

    id: str
    rank: int = 0
    state: NodeState = "requested"
    machine: str | None = None
    address: str | None = None
    accelerator: str | None = None
    price_per_hour: float | None = None
    market: str | None = None
    error: str | None = None
    progress: str | None = None
    completion: float | None = None
    binding: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    phases: tuple[PhaseView, ...] = ()
    metrics: Mapping[str, tuple[float, ...]] = field(default_factory=lambda: MappingProxyType({}))
    tail: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskView:
    """One task, by id. ``submitted_at`` is ``None`` only until the API has been asked."""

    id: str
    state: TaskState = "queued"
    function: str = ""
    node: str | None = None
    submitted_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ComputeView:
    """The whole compute, end to end, at one moment.

    Frozen, so a callback that keeps a reference keeps that moment. The spec
    fields (``provider``, ``accelerator``, the bounds) arrive with the first
    :func:`refresh`; everything else moves with the stream.
    """

    id: str
    name: str | None = None
    state: ComputeState = "requested"
    provider: str = ""
    region: str | None = None
    accelerator: str | None = None
    accelerator_count: int = 1
    cpus: int = 0
    memory_gb: float = 0.0
    allocation: str = ""
    initial: int = 0
    minimum: int | None = None
    maximum: int | None = None
    created_at: datetime | None = None
    cost: float | None = None
    nodes_total: int = 0
    errors: tuple[str, ...] = ()
    nodes: tuple[NodeView, ...] = ()
    tasks: tuple[TaskView, ...] = ()

    @property
    def nodes_ready(self) -> int:
        return sum(node.state == "ready" for node in self.nodes)


type EventCallback = Callable[[Event, ComputeView], None]
"""What ``Compute(callbacks=...)`` takes: each event, and the view folded up to it."""


def observe(view: ComputeView, event: Event) -> ComputeView:
    """The fold: one event in, the view after it out.

    A compute event moves the state to wherever the lifecycle table says it leads —
    the same table the daemon moved the row by, read without the origin check,
    because a replay from the log's beginning is folded into a view the API may
    already have hydrated past it.
    """
    match event:
        case MetricEvent(node=node, name=name, value=value):
            return _sample(view, node, name, value)
        case CostEvent(cost=cost):
            return replace(view, cost=cost)
        case PhaseEvent(error=error):
            return _phased(_noted(view, error), event)
        case ProgressEvent(node=node, progress=progress, completion=completion):
            return _progressed(view, node, progress, completion)
        case ConsoleEvent(node=node, content=content):
            return _spoken(view, node, content)
        case NodeEvent(node=node, state=state, error=error):
            return _transition(_noted(view, error), node, state, error)
        case TaskEvent(task=task, state=state):
            return _tasked(view, task, state)
        case ComputeDegraded(error=error) | ComputeReleaseFailed(error=error):
            return replace(_noted(view, error), state=lifecycle.leads(event) or view.state)
        case _:
            return replace(view, state=lifecycle.leads(event) or view.state)


def refresh(view: ComputeView, compute: ComputeResource, nodes: Page[Node]) -> ComputeView:
    """Hydrate the fields only the API carries, keeping what only the stream saw.

    A node's address, price, and binding are never events; a node's tail,
    metrics, and phases are never resources. Each side keeps its half.
    """
    spec = compute.spec.specs[0] if compute.spec.specs else None
    previous = {node.id: node for node in view.nodes}
    rows = tuple(
        NodeView(
            id=node.id,
            state=node.state,
            rank=node.rank,
            machine=node.machine,
            address=node.address,
            accelerator=node.accelerator,
            price_per_hour=node.price_per_hour,
            market=node.market,
            error=node.last_error.message if node.last_error else None,
            binding=MappingProxyType(node.provider_binding),
            progress=previous[node.id].progress if node.id in previous else None,
            completion=previous[node.id].completion if node.id in previous else None,
            metrics=previous[node.id].metrics if node.id in previous else MappingProxyType({}),
            phases=previous[node.id].phases if node.id in previous else (),
            tail=previous[node.id].tail if node.id in previous else (),
        )
        for node in nodes.items
        if node.state != "deleted"
    )
    hydrated = replace(
        view,
        name=compute.name,
        state=compute.status.state,
        provider=spec.provider.kind if spec else "",
        region=spec.region if spec else None,
        accelerator=spec.accelerator if spec else None,
        accelerator_count=spec.accelerator_count if spec else 1,
        cpus=(spec.cpus or 0) if spec else 0,
        memory_gb=(spec.memory_gb or 0.0) if spec else 0.0,
        allocation=compute.spec.allocation,
        initial=compute.spec.nodes.initial,
        minimum=compute.spec.nodes.min,
        maximum=compute.spec.nodes.max,
        created_at=compute.created_at,
        nodes_total=compute.status.nodes_total,
        nodes=rows,
    )
    return _noted(hydrated, compute.status.last_error.message if compute.status.last_error else None)


def refresh_tasks(view: ComputeView, tasks: Page[Task], names: Mapping[str, str] = MappingProxyType({})) -> ComputeView:
    """The tasks as the API tells them, with the function's real name when known."""
    rows = tuple(
        TaskView(
            id=task.id,
            state=task.state,
            function=names.get(task.function) or task.function[:8],
            node=task.executions[-1].node_id if task.executions else None,
            submitted_at=task.submitted_at,
            started_at=task.executions[-1].started_at if task.executions else None,
            finished_at=task.finished_at,
        )
        for task in tasks.items
    )
    return replace(view, tasks=rows)


def decoded(payload: bytes) -> Event | None:
    """One event, or nothing at all if this build does not know it.

    A daemon may be newer than the client watching it, and an event it has
    learnt to send is not a reason to stop folding the ones it has always sent.
    """
    try:
        return msgspec.json.decode(payload, type=Event)
    except msgspec.ValidationError:
        return None


_METRICS = frozenset({"gpu_util", "gpu_mem_mb", "gpu_mem_total_mb", "cpu", "mem", "mem_used_mb", "mem_total_mb"})

_WAITING = frozenset({"requested", "provisioning"})
"""The states a machine is in while it is still on its way to an address."""


def _noted(view: ComputeView, error: str | None) -> ComputeView:
    if not error or (view.errors and view.errors[-1] == error):
        return view
    return replace(view, errors=(*view.errors, error)[-ERRORS:])


def _sample(view: ComputeView, node_id: str, name: str, value: float) -> ComputeView:
    if name not in _METRICS or not node_id:
        return view

    def add(node: NodeView) -> NodeView:
        history = (*node.metrics.get(name, ()), value)[-HISTORY:]
        return replace(node, metrics=MappingProxyType({**node.metrics, name: history}))

    return _amend(view, node_id, add)


def _phased(view: ComputeView, event: PhaseEvent) -> ComputeView:
    name = event.phase
    if not event.node or not name or name == "bootstrap":
        return view

    def mark(node: NodeView) -> NodeView:
        phases = {phase.name: phase for phase in node.phases}
        match event.event:
            case "started":
                phases[name] = PhaseView(name, started=True)
            case "completed":
                phases[name] = replace(phases.get(name, PhaseView(name)), finished=True)
            case "failed":
                phases[name] = replace(phases.get(name, PhaseView(name)), finished=True, error=event.error or "failed")
        return replace(node, phases=tuple(phases.values()))

    return _amend(view, event.node, mark)


def _spoken(view: ComputeView, node_id: str, content: str) -> ComputeView:
    if not node_id:
        return view
    return _amend(view, node_id, lambda node: replace(node, tail=(*node.tail, content)[-TAIL:]))


def _transition(view: ComputeView, node_id: str, state: NodeState, error: str | None) -> ComputeView:
    if not node_id:
        return view

    def move(node: NodeView) -> NodeView:
        moved = replace(node, state=state, error=error or node.error)
        return moved if state in _WAITING else replace(moved, progress=None, completion=None)

    return _amend(view, node_id, move)


def _progressed(view: ComputeView, node_id: str, progress: str, completion: float | None) -> ComputeView:
    """What the machine is doing while it is short of an address.

    Only a node the API has already named gets a line: a rankless placeholder
    row would put the words on the wrong slot of the footer.
    """
    if not any(node.id == node_id for node in view.nodes):
        return view
    return _amend(view, node_id, lambda node: replace(node, progress=progress, completion=completion))


def _tasked(view: ComputeView, task_id: str, state: TaskEventState) -> ComputeView:
    landed: TaskState
    match state:
        case "started":
            landed = "running"
        case "succeeded" | "failed" | "indeterminate":
            landed = state
    if not any(task.id == task_id for task in view.tasks):
        return replace(view, tasks=(*view.tasks, TaskView(id=task_id, state=landed)))
    return replace(view, tasks=tuple(replace(task, state=landed) if task.id == task_id else task for task in view.tasks))


def _amend(view: ComputeView, node_id: str, change: Callable[[NodeView], NodeView]) -> ComputeView:
    rows = view.nodes
    if not any(node.id == node_id for node in rows):
        rows = (*rows, NodeView(node_id))
    return replace(view, nodes=tuple(change(node) if node.id == node_id else node for node in rows))
