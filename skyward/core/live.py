"""The v1 Rich console with a v2 state adapter."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import aclosing, contextmanager, redirect_stdout, suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TextIO

import msgspec
from rich.console import Console, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text

from skyward.core._live_v1 import (
    _LOGO_LINES,
    DIM,
    WARNING_STYLE,
    _Accelerator,
    _badge_text,
    _BootstrapTimeline,
    _Cluster,
    _ClusterSpec,
    _emit,
    _emit_task,
    _Instance,
    _InstanceType,
    _LiveFooter,
    _make_badge,
    _node_label,
    _NodeStatus,
    _Offer,
    _Phase,
    _render_summary,
    _ssh_url,
    _State,
)
from skyward.core.client import Client
from skyward.shared.observability import NAME as LOGGER_NAME
from skyward.shared.schemas import Compute, Function, Node, Page, Task

POLL = 2.0
HISTORY = 12
TAIL = 40
_METRICS = frozenset({"gpu_util", "gpu_mem_mb", "gpu_mem_total_mb", "cpu", "mem", "mem_used_mb", "mem_total_mb"})
_NODE_STATES = frozenset({
    "requested",
    "provisioning",
    "connecting",
    "bootstrapping",
    "ready",
    "draining",
    "lost",
    "deleting",
    "deleted",
    "failed",
    "degraded",
})
_COMPUTE_STATES = frozenset({"requested", "provisioning", "ready", "degraded", "deleting", "deleted", "failed"})


class _Reading(msgspec.Struct, frozen=True):
    node: str
    name: str
    value: float


class _Accrued(msgspec.Struct, frozen=True):
    cost: float


@dataclass(frozen=True, slots=True)
class PhaseMark:
    name: str
    started: bool = False
    finished: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class NodeRow:
    id: str
    state: str = "requested"
    rank: int = 0
    machine: str | None = None
    address: str | None = None
    accelerator: str | None = None
    price: float | None = None
    market: str | None = None
    error: str | None = None
    provider_binding: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    metrics: Mapping[str, tuple[float, ...]] = field(default_factory=lambda: MappingProxyType({}))
    phases: tuple[PhaseMark, ...] = ()
    tail: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Pool:
    name: str = ""
    provider: str = ""
    region: str | None = None
    accelerator: str | None = None
    accelerator_count: int = 1
    cpus: int = 0
    memory_gb: float = 0.0
    allocation: str = ""
    state: str = "requested"
    total: int = 0
    desired: int = 0
    minimum: int | None = None
    maximum: int | None = None
    created_at: datetime | None = None
    cost: float | None = None


@dataclass(frozen=True, slots=True)
class TaskRow:
    state: str
    function: str
    node: str | None
    submitted_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


@dataclass(frozen=True, slots=True)
class View:
    pool: Pool = field(default_factory=Pool)
    nodes: tuple[NodeRow, ...] = ()
    tasks: tuple[TaskRow, ...] = ()
    progress: Mapping[int, str] = field(default_factory=lambda: MappingProxyType({}))


def observe(view: View, event: str, payload: bytes) -> View:
    if event == "node.metrics":
        reading = msgspec.json.decode(payload, type=_Reading)
        return _sample(view, reading.node, reading.name, reading.value)
    if event == "compute.cost":
        accrued = msgspec.json.decode(payload, type=_Accrued)
        return replace(view, pool=replace(view.pool, cost=accrued.cost))
    data = msgspec.json.decode(payload, type=dict[str, str])
    match event.split(".", 1):
        case ["node", "phase"]:
            return _phased(view, data)
        case ["node", "console"]:
            return _spoken(view, data)
        case ["node", state] if state in _NODE_STATES:
            return _transition(view, data.get("node", ""), state, data.get("error"))
        case ["compute", state] if state in _COMPUTE_STATES:
            return replace(view, pool=replace(view.pool, state=state))
        case _:
            return view


def refresh(view: View, compute: Compute, nodes: Page[Node]) -> View:
    spec = compute.spec.specs[0] if compute.spec.specs else None
    pool = Pool(
        name=compute.name or compute.id,
        provider=spec.provider.kind if spec else "",
        region=spec.region if spec else None,
        accelerator=spec.accelerator if spec else None,
        accelerator_count=spec.accelerator_count if spec else 1,
        cpus=(spec.cpus or 0) if spec else 0,
        memory_gb=(spec.memory_gb or 0.0) if spec else 0.0,
        allocation=compute.spec.allocation,
        state=compute.status.state,
        total=compute.status.nodes_total,
        desired=compute.spec.nodes.desired,
        minimum=compute.spec.nodes.min,
        maximum=compute.spec.nodes.max,
        created_at=compute.created_at,
        cost=view.pool.cost,
    )
    previous = {row.id: row for row in view.nodes}
    rows = tuple(
        NodeRow(
            id=node.id,
            state=node.state,
            rank=node.rank,
            machine=node.machine,
            address=node.address,
            accelerator=node.accelerator,
            price=node.price_per_hour,
            market=node.market,
            error=node.last_error.message if node.last_error else None,
            provider_binding=MappingProxyType(node.provider_binding),
            metrics=previous[node.id].metrics if node.id in previous else MappingProxyType({}),
            phases=previous[node.id].phases if node.id in previous else (),
            tail=previous[node.id].tail if node.id in previous else (),
        )
        for node in nodes.items
        if node.state != "deleted"
    )
    return replace(view, pool=pool, nodes=rows)


def refresh_tasks(
    view: View,
    tasks: Page[Task],
    names: Mapping[str, str] = MappingProxyType({}),
) -> View:
    rows = tuple(
        TaskRow(
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


def _sample(view: View, node_id: str, name: str, value: float) -> View:
    if name not in _METRICS or not node_id:
        return view

    def add(row: NodeRow) -> NodeRow:
        history = (*row.metrics.get(name, ()), value)[-HISTORY:]
        return replace(row, metrics=MappingProxyType({**row.metrics, name: history}))

    return _amend(view, node_id, add)


def _phased(view: View, data: dict[str, str]) -> View:
    node_id = data.get("node", "")
    name = data.get("phase", "")
    if not node_id or not name or name == "bootstrap":
        return view

    def mark(row: NodeRow) -> NodeRow:
        phases = {phase.name: phase for phase in row.phases}
        match data.get("event"):
            case "started":
                phases[name] = PhaseMark(name, started=True)
            case "completed":
                phases[name] = replace(phases.get(name, PhaseMark(name)), finished=True)
            case "failed":
                phases[name] = replace(phases.get(name, PhaseMark(name)), finished=True, error=data.get("error") or "failed")
        return replace(row, phases=tuple(phases.values()))

    return _amend(view, node_id, mark)


def _spoken(view: View, data: dict[str, str]) -> View:
    node_id = data.get("node", "")
    if not node_id:
        return view
    content = data.get("content", "")
    return _amend(view, node_id, lambda row: replace(row, tail=(*row.tail, content)[-TAIL:]))


def _transition(view: View, node_id: str, state: str, error: str | None) -> View:
    if not node_id:
        return view
    return _amend(view, node_id, lambda row: replace(row, state=state, error=error or row.error))


def _amend(view: View, node_id: str, change: Callable[[NodeRow], NodeRow]) -> View:
    rows = view.nodes
    if not any(row.id == node_id for row in rows):
        rows = (*rows, NodeRow(node_id))
    return replace(view, nodes=tuple(change(row) if row.id == node_id else row for row in rows))


def _state(view: View) -> _State:
    rows = sorted(view.nodes, key=lambda row: row.rank)
    statuses = MappingProxyType({row.rank: _node_status(row.state) for row in rows})
    instances = tuple(_instance(view.pool, row) for row in rows)
    metrics = MappingProxyType({
        row.rank: MappingProxyType(_metrics(row))
        for row in rows
        if row.metrics
    })
    spinners = MappingProxyType({
        row.rank: _BootstrapTimeline(
            phases=tuple(phase.name for phase in row.phases),
            completed=frozenset(phase.name for phase in row.phases if phase.finished and not phase.error),
            active=next((phase.name for phase in reversed(row.phases) if phase.started and not phase.finished), ""),
            output=row.tail[-1] if row.tail else "",
        )
        for row in rows
        if row.state in {"connecting", "bootstrapping"}
    })
    task_counts = Counter(task.state for task in view.tasks)
    latencies = tuple(
        (task.finished_at - task.started_at).total_seconds()
        for task in view.tasks
        if task.started_at and task.finished_at
    )
    per_function: dict[str, tuple[float, ...]] = {}
    failures: Counter[str] = Counter()
    per_node: Counter[int] = Counter()
    ranks = {row.id: row.rank for row in rows}
    for task in view.tasks:
        if task.started_at and task.finished_at:
            per_function[task.function] = (
                *per_function.get(task.function, ()),
                (task.finished_at - task.started_at).total_seconds(),
            )
        if task.state == "failed":
            failures[task.function] += 1
        if task.node in ranks:
            per_node[ranks[task.node]] += 1
    started = (
        time.monotonic() - max(0.0, (datetime.now(UTC) - view.pool.created_at).total_seconds())
        if view.pool.created_at
        else 0.0
    )
    first_task_at = (
        time.monotonic()
        - max(0.0, (datetime.now(UTC) - min(task.submitted_at for task in view.tasks)).total_seconds())
        if view.tasks
        else 0.0
    )
    active = sum(row.state not in {"deleted", "failed", "lost"} for row in rows)
    reconciler_state = (
        "draining"
        if any(row.state == "draining" for row in rows)
        else "scaling_up"
        if view.pool.desired > active
        else "watching"
    )
    return _State(
        total_nodes=view.pool.total,
        phase=_phase(view),
        nodes=statuses,
        tasks_queued=task_counts["queued"],
        tasks_running=task_counts["running"],
        tasks_done=task_counts["succeeded"],
        tasks_failed=task_counts["failed"],
        first_task_at=first_task_at,
        cluster=_Cluster(_ClusterSpec(view.pool.provider)),
        instances=instances,
        metrics=metrics,
        pool_started_at=started,
        task_latencies=latencies,
        task_fn_stats=MappingProxyType(per_function),
        task_fn_failed=MappingProxyType(dict(failures)),
        desired_nodes=view.pool.desired,
        pending_nodes=max(0, view.pool.desired - len(rows)),
        draining_nodes=sum(row.state == "draining" for row in rows),
        reconciler_state=reconciler_state,
        min_nodes=view.pool.minimum,
        max_nodes=view.pool.maximum,
        is_elastic=view.pool.minimum is not None or view.pool.maximum is not None,
        tasks_per_node=MappingProxyType(dict(per_node)),
        ssh_user=_binding_string(rows, "ssh_user"),
        ssh_key_path=_binding_string(rows, "ssh_key_path"),
        bootstrap_spinners=spinners,
        progress_lines=MappingProxyType(dict(view.progress)),
        node_instances=MappingProxyType({row.rank: instance for row, instance in zip(rows, instances, strict=True)}),
    )


def _phase(view: View) -> _Phase:
    if view.pool.state in {"deleted", "deleting"}:
        return _Phase.STOPPED
    if view.pool.state == "ready":
        return _Phase.READY
    states = {row.state for row in view.nodes}
    if "bootstrapping" in states:
        return _Phase.BOOTSTRAP
    if "connecting" in states:
        return _Phase.SSH
    return _Phase.PROVISIONING


def _node_status(state: str) -> _NodeStatus:
    if state == "ready":
        return _NodeStatus.READY
    if state == "bootstrapping":
        return _NodeStatus.SSH
    return _NodeStatus.WAITING


def _metrics(row: NodeRow) -> dict[str, float]:
    metrics = {name: history[-1] for name, history in row.metrics.items() if history}
    used = metrics.get("mem_used_mb")
    total = metrics.get("mem_total_mb")
    if "mem" not in metrics and used is not None and total:
        metrics["mem"] = used / total * 100
    return metrics


def _instance(pool: Pool, row: NodeRow) -> _Instance:
    accelerator_name = row.accelerator or pool.accelerator
    accelerator = _Accelerator(accelerator_name, pool.accelerator_count) if accelerator_name else None
    price = row.price
    spot = row.market == "spot" or (row.market is None and pool.allocation == "spot")
    return _Instance(
        id=row.machine or row.id,
        ip=row.address,
        ssh_port=_binding_int(row.provider_binding, "ssh_port", 22),
        region=_binding_string_value(row.provider_binding, "region") or pool.region or "",
        spot=spot,
        offer=_Offer(
            instance_type=_InstanceType(
                name=row.accelerator or pool.accelerator or "",
                vcpus=pool.cpus,
                memory_gb=pool.memory_gb,
                accelerator=accelerator,
            ),
            spot_price=price if spot else None,
            on_demand_price=price if not spot else None,
        ),
    )


def _binding_string(rows: list[NodeRow], key: str) -> str:
    return next((value for row in rows if (value := _binding_string_value(row.provider_binding, key))), "")


def _binding_string_value(binding: Mapping[str, object], key: str) -> str:
    value = binding.get(key)
    return value if isinstance(value, str) else ""


def _binding_int(binding: Mapping[str, object], key: str, default: int) -> int:
    value = binding.get(key)
    return value if isinstance(value, int) else default


def render(view: View) -> RenderableType:
    footer = _LiveFooter()
    footer.state = _state(view)
    return footer


@dataclass(frozen=True, slots=True)
class _Snapshot:
    compute: Compute
    nodes: Page[Node]
    tasks: Page[Task]
    names: Mapping[str, str]


class _LocalOutput:
    def __init__(self, console: Console, original: TextIO) -> None:
        self._console = console
        self._original = original

    def write(self, text: str) -> int:
        for line in text.splitlines():
            if stripped := line.rstrip():
                _emit(self._console, "local", stripped)
        return len(text)

    def flush(self) -> None:
        pass

    @property
    def encoding(self) -> str | None:
        return self._original.encoding

    @property
    def errors(self) -> str | None:
        return self._original.errors

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return False


class RichConsole:
    """Rich adaptive console: banner → live footer → event lines → summary."""

    def __init__(self, client: Client, compute: str, out: TextIO | None = None) -> None:
        self._client = client
        self._compute = compute
        self._console = Console(file=out or sys.stderr)
        self._view = View()
        self._footer = _LiveFooter()
        self._live: Live | None = None
        self._names: dict[str, str] = {}

    async def follow(self) -> None:
        with suppress(Exception):
            self._merge(await self._fetch())
        self._print_banner()
        self._footer.state = _state(self._view)
        with _quiet(self._console), redirect_stdout(_LocalOutput(self._console, sys.stdout)):
            live = Live(
                self._footer,
                console=self._console,
                refresh_per_second=8,
                screen=False,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live = live
            live.start()
            poll = asyncio.create_task(self._poll())
            try:
                async with aclosing(self._client.events(self._compute)) as stream:
                    async for event, payload in stream:
                        self._view = observe(self._view, event, payload)
                        self._footer.state = _state(self._view)
                        self._print_event(event, _safe(payload))
                        live.update(self._footer)
                        if event in {"compute.deleted", "compute.failed"}:
                            live.stop()
                            _emit(self._console, "skyward", "Shutting down...", WARNING_STYLE)
                            self._console.print(_render_summary(self._footer.state))
                            self._live = None
                            return
            finally:
                poll.cancel()
                if self._live is not None:
                    live.stop()
                    self._live = None

    def _print_banner(self) -> None:
        from skyward._version import __version__

        version = Text()
        version.append(f" v{__version__} ", style=_make_badge(140, 0.6))
        version.append("  Cloud accelerators with a single decorator", style=DIM)
        link = Text("https://gabfssilva.github.io/skyward/", style="underline dim")
        banner = Table.grid(padding=(0, 2))
        banner.add_column("logo")
        banner.add_column("info")
        for logo, info in zip(_LOGO_LINES, (Text(), version, link, Text()), strict=True):
            banner.add_row(logo, info)
        self._console.print()
        self._console.print(banner)
        self._console.print()

    def _print_event(self, event: str, data: dict[str, str]) -> None:
        state = _state(self._view)
        row = next((row for row in self._view.nodes if row.id == data.get("node")), None)
        node_id = row.rank if row else 0
        match event:
            case "node.console":
                if row is None or row.state not in {"connecting", "bootstrapping"}:
                    _emit(
                        self._console,
                        _node_label(state, node_id),
                        data.get("content", ""),
                        link=_ssh_url(state, node_id),
                    )
            case "node.ready":
                _emit(self._console, _node_label(state, node_id), "✓ Joined", "green bold", link=_ssh_url(state, node_id))
            case "node.failed" | "node.lost":
                _emit(self._console, "error", data.get("error") or event.removeprefix("node."), "red")
            case "task.queued":
                _emit_task(self._console, "skyward", "queued", data.get("task", ""))
            case "task.succeeded":
                _emit_task(self._console, _node_label(state, node_id), "done", "")
            case "task.failed":
                _emit_task(self._console, _node_label(state, node_id), "failed", "")
            case "compute.failed" | "compute.degraded":
                _emit(self._console, "error", data.get("error") or event.removeprefix("compute."), "red")
            case _:
                pass

    async def _poll(self) -> None:
        while True:
            await asyncio.sleep(POLL)
            try:
                snapshot = await self._fetch()
            except asyncio.CancelledError:
                raise
            except Exception:
                continue
            self._merge(snapshot)
            self._footer.state = _state(self._view)
            if self._live is not None:
                self._live.update(self._footer)

    async def _fetch(self) -> _Snapshot:
        compute = await self._client.call("GET", f"/v1/computes/{self._compute}", Compute)
        nodes = await self._client.call("GET", f"/v1/computes/{self._compute}/nodes", Page[Node])
        tasks: Page[Task] = Page(items=())
        with suppress(Exception):
            tasks = await self._client.call("GET", "/v1/tasks", Page[Task], compute=self._compute, limit=200)
        return _Snapshot(compute, nodes, tasks, await self._names_for(tasks))

    def _merge(self, snapshot: _Snapshot) -> None:
        self._view = refresh_tasks(
            refresh(self._view, snapshot.compute, snapshot.nodes),
            snapshot.tasks,
            snapshot.names,
        )

    async def _names_for(self, tasks: Page[Task]) -> Mapping[str, str]:
        for sha in {task.function for task in tasks.items} - self._names.keys():
            try:
                function = await self._client.call("GET", f"/v1/functions/{sha}", Function)
                self._names[sha] = function.name or sha[:8]
            except Exception:
                self._names[sha] = sha[:8]
        return self._names


def _event_line(view: View, event: str, data: dict[str, str]) -> Text:
    state = _state(view)
    row = next((row for row in view.nodes if row.id == data.get("node")), None)
    node_id = row.rank if row else 0
    if event == "node.ready":
        line = _badge_text(_node_label(state, node_id))
        line.append("  ✓ Joined")
        return line
    return Text(data.get("content") or event)


def _safe(payload: bytes) -> dict[str, str]:
    try:
        return msgspec.json.decode(payload, type=dict[str, str])
    except msgspec.ValidationError:
        return {}


class _LogSink(logging.Handler):
    def __init__(self, console: Console) -> None:
        super().__init__(logging.WARNING)
        self._console = console
        self.setFormatter(logging.Formatter("%(module)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._console.print(Text(self.format(record), style="yellow"))
        except Exception:
            self.handleError(record)


@contextmanager
def _quiet(console: Console) -> Iterator[None]:
    """Route warnings into the dashboard for the length of the block.

    Two loggers, because the daemon's records never reach the root one: they go to
    skyward's own non-propagating logger, and a record with nowhere to land would be
    printed to stderr by ``logging``'s last resort, through the live display.
    """
    sink = _LogSink(console)
    saved = [(target, target.handlers[:], target.level) for target in (logging.getLogger(), logging.getLogger(LOGGER_NAME))]
    for target, _, _ in saved:
        target.handlers = [sink]
        target.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for target, handlers, level in saved:
            target.handlers, target.level = handlers, level
