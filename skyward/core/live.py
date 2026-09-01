"""The live Rich console: the folded view in, widgets out.

The fold itself lives in :mod:`skyward.core.view`; this module only renders.
It is fed by the pool's :class:`~skyward.core.console.Observer` like any other
watcher — the panel is a subscriber of the same stream the user's callbacks
get, not a private pipeline of its own.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager, redirect_stdout
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TextIO

from rich.console import Console, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text

from skyward.core.view import ComputeView, NodeView
from skyward.core.widgets import (
    _LOGO_LINES,
    DIM,
    WARNING_STYLE,
    _Accelerator,
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
    _Progress,
    _render_summary,
    _ssh_url,
    _State,
)
from skyward.shared.observability import NAME as LOGGER_NAME
from skyward.shared.schemas import ComputeEvent, ConsoleEvent, Event, NodeEvent, TaskEvent


def _state(view: ComputeView) -> _State:
    rows = sorted(view.nodes, key=lambda row: row.rank)
    statuses = MappingProxyType({row.rank: _node_status(row.state) for row in rows})
    instances = tuple(_instance(view, row) for row in rows)
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
        time.monotonic() - max(0.0, (datetime.now(UTC) - view.created_at).total_seconds())
        if view.created_at
        else 0.0
    )
    submissions = [task.submitted_at for task in view.tasks if task.submitted_at]
    first_task_at = (
        time.monotonic() - max(0.0, (datetime.now(UTC) - min(submissions)).total_seconds())
        if submissions
        else 0.0
    )
    active = sum(row.state not in {"deleted", "failed", "lost"} for row in rows)
    pending = max(0, view.initial - len(rows))
    reconciler_state = (
        "draining"
        if any(row.state == "draining" for row in rows)
        else "scaling_up"
        if pending
        else "watching"
    )
    return _State(
        total_nodes=view.nodes_total,
        phase=_phase(view),
        nodes=statuses,
        tasks_queued=task_counts["queued"],
        tasks_running=task_counts["running"],
        tasks_done=task_counts["succeeded"],
        tasks_failed=task_counts["failed"],
        first_task_at=first_task_at,
        cluster=_Cluster(_ClusterSpec(view.provider)),
        instances=instances,
        metrics=metrics,
        pool_started_at=started,
        task_latencies=latencies,
        task_fn_stats=MappingProxyType(per_function),
        task_fn_failed=MappingProxyType(dict(failures)),
        target_nodes=active + pending,
        pending_nodes=pending,
        draining_nodes=sum(row.state == "draining" for row in rows),
        reconciler_state=reconciler_state,
        min_nodes=view.minimum,
        max_nodes=view.maximum,
        is_elastic=view.minimum is not None or view.maximum is not None,
        tasks_per_node=MappingProxyType(dict(per_node)),
        ssh_user=_binding_string(rows, "ssh_user"),
        ssh_key_path=_binding_string(rows, "ssh_key_path"),
        bootstrap_spinners=spinners,
        progress_lines=MappingProxyType({row.rank: _Progress(row.progress, row.completion) for row in rows if row.progress is not None}),
        node_instances=MappingProxyType({row.rank: instance for row, instance in zip(rows, instances, strict=True)}),
    )


def _phase(view: ComputeView) -> _Phase:
    if view.state in {"deleted", "deleting"}:
        return _Phase.STOPPED
    if view.state == "ready":
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


def _metrics(row: NodeView) -> dict[str, float]:
    metrics = {name: history[-1] for name, history in row.metrics.items() if history}
    used = metrics.get("mem_used_mb")
    total = metrics.get("mem_total_mb")
    if "mem" not in metrics and used is not None and total:
        metrics["mem"] = used / total * 100
    return metrics


def _instance(view: ComputeView, row: NodeView) -> _Instance:
    accelerator_name = row.accelerator or view.accelerator
    accelerator = _Accelerator(accelerator_name, view.accelerator_count) if accelerator_name else None
    price = row.price_per_hour
    spot = row.market == "spot" or (row.market is None and view.allocation == "spot")
    return _Instance(
        id=row.machine or row.id,
        ip=row.address,
        ssh_port=_binding_int(row.binding, "ssh_port", 22),
        region=_binding_string_value(row.binding, "region") or view.region or "",
        spot=spot,
        offer=_Offer(
            instance_type=_InstanceType(
                name=row.accelerator or view.accelerator or "",
                vcpus=view.cpus,
                memory_gb=view.memory_gb,
                accelerator=accelerator,
            ),
            spot_price=price if spot else None,
            on_demand_price=price if not spot else None,
        ),
    )


def _binding_string(rows: list[NodeView], key: str) -> str:
    return next((value for row in rows if (value := _binding_string_value(row.binding, key))), "")


def _binding_string_value(binding: Mapping[str, object], key: str) -> str:
    value = binding.get(key)
    return value if isinstance(value, str) else ""


def _binding_int(binding: Mapping[str, object], key: str, default: int) -> int:
    value = binding.get(key)
    return value if isinstance(value, int) else default


def render(view: ComputeView) -> RenderableType:
    footer = _LiveFooter()
    footer.state = _state(view)
    return footer


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
    """Rich adaptive console: banner → live footer → event lines → summary.

    A :class:`~skyward.core.console.Watcher`: the pool's observer owns the
    stream and the fold, and this class only draws what it is handed.
    """

    def __init__(self, out: TextIO | None = None) -> None:
        self._console = Console(file=out or sys.stderr)
        self._footer = _LiveFooter()
        self._live: Live | None = None
        self._stack = ExitStack()

    def opened(self, view: ComputeView) -> None:
        self._print_banner()
        self._footer.state = _state(view)
        self._stack.enter_context(_quiet(self._console))
        self._stack.enter_context(redirect_stdout(_LocalOutput(self._console, sys.stdout)))
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

    def event(self, event: Event, view: ComputeView) -> None:
        if self._live is None:
            return
        self._footer.state = _state(view)
        self._print_event(event, view)
        self._live.update(self._footer)
        if isinstance(event, ComputeEvent) and event.state in {"deleted", "failed"}:
            self._summarize()

    def refreshed(self, view: ComputeView) -> None:
        if self._live is None:
            return
        self._footer.state = _state(view)
        self._live.update(self._footer)

    def closed(self, view: ComputeView) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._stack.close()

    def _summarize(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        _emit(self._console, "skyward", "Shutting down...", WARNING_STYLE)
        self._console.print(_render_summary(self._footer.state))

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

    def _print_event(self, event: Event, view: ComputeView) -> None:
        state = _state(view)
        node = getattr(event, "node", None)
        row = next((row for row in view.nodes if row.id == node), None)
        node_id = row.rank if row else 0
        match event:
            case ConsoleEvent(content=content):
                if row is None or row.state not in {"connecting", "bootstrapping"}:
                    _emit(self._console, _node_label(state, node_id), content, link=_ssh_url(state, node_id))
            case NodeEvent(state="ready"):
                _emit(self._console, _node_label(state, node_id), "✓ Joined", "green bold", link=_ssh_url(state, node_id))
            case NodeEvent(state="failed" | "lost" as failure, error=error):
                _emit(self._console, "error", error or failure, "red")
            case TaskEvent(state="succeeded"):
                _emit_task(self._console, _node_label(state, node_id), "done", "")
            case TaskEvent(state="failed"):
                _emit_task(self._console, _node_label(state, node_id), "failed", "")
            case ComputeEvent(state="failed" | "degraded" as failure, error=error):
                _emit(self._console, "error", error or failure, "red")
            case _:
                pass

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
