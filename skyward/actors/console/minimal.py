"""Minimal console actor — single live status line over a cluster header.

Renders a compact, opinionated view of the session:

- A fixed cluster identity header grouped by ``(provider, region, instance_type)``.
- A single live status line that adapts to ``pool.phase``
  (``provisioning`` → ``ssh`` → ``bootstrap`` → ``ready``), showing
  aggregated metrics and task counters.
- A tail line per node during bootstrap with the current shell output.
- Above-live one-liners for salient events (lost/preempted nodes,
  bootstrap failures, provision failure, fatal errors, scaling changes).

The heavy lifting is delegated to Rich — ``Live`` owns the refresh loop,
``Group`` stacks the header + status + tails, and the ``Spinner`` animates
simply because ``__rich__`` is re-invoked on each tick.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from casty import ActorContext, Behavior, Behaviors
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from skyward.api.events import Error, Node, Pool
from skyward.api.projection import _throughput
from skyward.api.views import NodeStatus, NodeView, PoolPhase, PoolView, SessionView

from .messages import ConsoleInput, EventReceived, LocalOutput, LogReceived, ViewUpdated

__all__ = ["minimal_console_actor"]


# ── Helpers ──────────────────────────────────────────────────────


def _provider_name(pool: PoolView, inst: Any) -> str:
    if (
        pool.cluster is not None
        and getattr(pool.cluster, "spec", None) is not None
        and (name := pool.cluster.spec.provider)
    ):
        return name
    module = type(inst.offer.specific).__module__
    parts = module.split(".")
    if "providers" in parts:
        idx = parts.index("providers")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "?"


def _fmt_count(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else f"{n:g}"


def _hourly(instances: Iterable[Any]) -> float:
    return sum(
        (i.offer.spot_price if i.spot else i.offer.on_demand_price) or 0.0
        for i in instances
    )


def _elapsed_hours(pool: PoolView) -> float:
    if not pool.started_at:
        return 0.0
    return (time.monotonic() - pool.started_at) / 3600


def _group_instances(
    pool: PoolView,
) -> list[tuple[str, str, tuple[str, ...], int, Any, float]]:
    """Group by (provider, instance_type); regions collapse into a list.

    Each row carries the group's aggregated ``$/hr``.
    """
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    regions: dict[tuple[str, str], list[str]] = defaultdict(list)
    samples: dict[tuple[str, str], Any] = {}
    for inst in pool.instances:
        prov = _provider_name(pool, inst)
        itname = inst.offer.instance_type.name
        key = (prov, itname)
        groups[key].append(inst)
        region = inst.region or "?"
        if region not in regions[key]:
            regions[key].append(region)
        samples.setdefault(key, inst.offer.instance_type)
    return [
        (
            prov, itname, tuple(regions[(prov, itname)]),
            len(insts), samples[(prov, itname)], _hourly(insts),
        )
        for (prov, itname), insts in groups.items()
    ]


def _header(pool: PoolView) -> Text:
    if not pool.instances:
        return Text(f"{pool.name} · provisioning…", style="dim cyan")

    lines: list[Text] = []
    for prov, itname, regions, n, it, rate in _group_instances(pool):
        segs: list[tuple[str, str]] = [
            (prov, "cyan"),
            (", ".join(regions), "cyan"),
            (f"{n}× {itname}", "bold"),
        ]
        if it.accelerator is not None:
            segs.append(
                (f"{it.accelerator.name}×{_fmt_count(it.accelerator.count)}", "green"),
            )
        segs.extend([
            (f"{int(it.vcpus)} vCPU", "dim"),
            (f"{int(it.memory_gb)} GB", "dim"),
        ])
        if rate > 0:
            segs.append((f"${rate:.2f}/hr", "yellow"))
        lines.append(
            Text(" · ", style="dim").join(Text(s, style=st) for s, st in segs),
        )

    s = pool.scaling
    if s.is_elastic and s.min_nodes is not None and s.max_nodes is not None:
        lines[-1].append(f"  · autoscale {s.min_nodes}–{s.max_nodes}", style="dim yellow")
    return Text("\n").join(lines)


def _avg(pool: PoolView, metric: str) -> float | None:
    vals = [n.metrics[metric] for n in pool.nodes.values() if metric in n.metrics]
    return mean(vals) if vals else None


def _avg_prefix(pool: PoolView, prefix: str) -> float | None:
    """Average a metric across all nodes and all indexed variants.

    Handles both bare names (``"gpu_util"``) and indexed names emitted
    by multi-GPU collectors (``"gpu_util_0"``, ``"gpu_util_1"``).
    """
    vals: list[float] = []
    for n in pool.nodes.values():
        for name, value in n.metrics.items():
            if name == prefix or name.startswith(f"{prefix}_"):
                vals.append(value)
    return mean(vals) if vals else None


def _dominant_bootstrap_phase(pool: PoolView) -> str | None:
    counts: dict[str, int] = defaultdict(int)
    for n in pool.nodes.values():
        if n.bootstrap is not None and n.bootstrap.active:
            counts[n.bootstrap.active] += 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _phase_label(pool: PoolView) -> str:
    match pool.phase:
        case PoolPhase.PROVISIONING:
            return "provisioning"
        case PoolPhase.SSH:
            return "ssh"
        case PoolPhase.BOOTSTRAP | PoolPhase.WORKERS:
            if phase := _dominant_bootstrap_phase(pool):
                return f"bootstrap · {phase}"
            return "bootstrap"
        case PoolPhase.READY:
            return "ready"
        case PoolPhase.STOPPED:
            return "stopped"


def _status(pool: PoolView) -> Text:
    ready = sum(1 for n in pool.nodes.values() if n.status is NodeStatus.READY)
    s, t = pool.scaling, pool.tasks

    line = Text.assemble((_phase_label(pool), "bold cyan"))
    line.append(f"  {ready}/{pool.total_nodes} ready", style="bold")

    match s.reconciler_state:
        case "scaling_up" if s.pending:
            line.append(f" (+{s.pending} pending)", style="yellow")
        case "draining" if s.draining:
            line.append(f" (-{s.draining} draining)", style="yellow")

    if (v := _avg_prefix(pool, "gpu_util")) is not None:
        line.append(f" · gpu {v:.0f}%", style="magenta")
    if (v := _avg_prefix(pool, "gpu_mem_mb")) is not None:
        line.append(f" · vram {v / 1024:.1f}GB", style="magenta")
    if (v := _avg(pool, "cpu")) is not None:
        line.append(f" · cpu {v:.0f}%", style="magenta")
    if (v := _avg(pool, "mem")) is not None:
        line.append(f" · mem {v:.0f}%", style="magenta")

    if t.done or t.failed or t.running or t.queued:
        in_flight = t.queued + t.running
        line.append(f"  ·  tasks {in_flight}/{t.done} ✓", style="green")
        if t.failed:
            line.append(f"  {t.failed} ✗", style="red")
        if (tp := _throughput(t)) > 0:
            line.append(f" · {tp:.1f} t/min", style="dim")

    hourly = _hourly(pool.instances)
    if hourly > 0:
        total = hourly * _elapsed_hours(pool)
        line.append(f"  ·  Σ ${total:.2f}", style="yellow")
    return line


_INSTANCE_ID_WIDTH = 8


def _node_label(n: NodeView) -> str:
    if n.instance is not None and (iid := n.instance.id):
        return f"{iid[:_INSTANCE_ID_WIDTH]}/{n.node_id}"
    return f"node-{n.node_id}"


def _node_tails(pool: PoolView) -> list[Text]:
    if pool.phase not in (PoolPhase.SSH, PoolPhase.BOOTSTRAP, PoolPhase.WORKERS):
        return []
    tails: list[Text] = []
    for n in sorted(pool.nodes.values(), key=lambda n: n.node_id):
        if n.status is NodeStatus.READY:
            continue
        if n.bootstrap is None or not n.bootstrap.output:
            continue
        tails.append(
            Text.assemble(
                (f"  {_node_label(n)} │ ", "dim"),
                (n.bootstrap.output[:120], "cyan"),
            ),
        )
    return tails


def _first_pool(view: SessionView) -> PoolView | None:
    if view.pools:
        return next(iter(view.pools.values()))
    return None


def _summary(pool: PoolView, started_at: float) -> Text:
    elapsed = time.monotonic() - (pool.started_at or started_at)
    ready = sum(1 for n in pool.nodes.values() if n.status is NodeStatus.READY)
    line = Text.assemble(
        ("✓ done", "green bold"),
        ("  ·  ", "dim"),
        (f"{ready or pool.total_nodes} nodes", ""),
        ("  ·  ", "dim"),
        (f"{pool.tasks.done} tasks", "green"),
    )
    if pool.tasks.failed:
        line.append(f"  ·  {pool.tasks.failed} failed", style="red")
    line.append(f"  ·  elapsed {elapsed:.0f}s", style="dim")
    if (hourly := _hourly(pool.instances)) > 0:
        total = hourly * (elapsed / 3600)
        line.append(f"  ·  Σ ${total:.2f}", style="yellow")
    return line


# ── Renderable ───────────────────────────────────────────────────


@dataclass
class _View:
    view: SessionView = field(default_factory=SessionView)
    spinner: Spinner = field(default_factory=lambda: Spinner("dots", style="cyan"))

    def __rich__(self) -> Group:
        pool = _first_pool(self.view)
        if pool is None:
            return Group(Text("skyward · waiting…", style="dim"))
        row = Table.grid(padding=(0, 1))
        row.add_column(no_wrap=True)
        row.add_column()
        row.add_row(self.spinner, _status(pool))
        return Group(_header(pool), row, *_node_tails(pool))


# ── Actor ────────────────────────────────────────────────────────


def _print_event(live: Live, event: Any) -> None:
    match event:
        case Node.Lost(node_id=nid, reason=r):
            live.console.print(f"[red]✗[/] node-{nid} lost: {r}")
        case Node.Preempted(node_id=nid, reason=r):
            live.console.print(f"[yellow]⚠[/] node-{nid} preempted: {r}")
        case Node.ConnectionFailed(node_id=nid, error=e):
            live.console.print(f"[red]✗[/] node-{nid} ssh failed: {e}")
        case Node.WorkerFailed(node_id=nid, error=e):
            live.console.print(f"[red]✗[/] node-{nid} worker failed: {e}")
        case Node.Bootstrap.Failed(node_id=nid, phase=p, error=e):
            live.console.print(f"[red]✗[/] node-{nid} {p}: {e}")
        case Error.Occurred(message=m, fatal=True):
            live.console.print(f"[red bold]✗[/] {m}")
        case Error.Occurred(message=m):
            live.console.print(f"[red]✗[/] {m}")
        case Pool.ProvisionFailed(reason=r):
            live.console.print(f"[red bold]✗[/] provision failed: {r}")
        case _:
            pass


def minimal_console_actor() -> Behavior[ConsoleInput]:
    """Minimal console tells this story: idle → live → stopped."""

    console = Console(stderr=True)
    renderable = _View()
    live = Live(
        renderable, console=console, refresh_per_second=10,
        redirect_stdout=False, redirect_stderr=False,
    )
    started_at = time.monotonic()

    async def post_stop(_ctx: ActorContext[ConsoleInput]) -> None:
        if live.is_started:
            live.stop()

    async def setup(ctx: ActorContext[ConsoleInput]) -> Behavior[ConsoleInput]:
        live.start()
        return Behaviors.with_lifecycle(active(), post_stop=post_stop)

    def active() -> Behavior[ConsoleInput]:
        async def receive(
            ctx: ActorContext[ConsoleInput], msg: ConsoleInput,
        ) -> Behavior[ConsoleInput]:
            match msg:
                case ViewUpdated(view=view):
                    renderable.view = view

                case EventReceived(event=Pool.Stopped()):
                    pool = _first_pool(renderable.view)
                    if pool is not None:
                        live.update(Text())
                        live.stop()
                        console.print(_summary(pool, started_at))

                case EventReceived(event=ev):
                    if live.is_started:
                        _print_event(live, ev)

                case LogReceived(log=log):
                    if live.is_started:
                        live.console.print(log.message)

                case LocalOutput(line=line):
                    if live.is_started and (stripped := line.rstrip()):
                        live.console.print(stripped)

            return Behaviors.same()

        return Behaviors.receive(receive)

    return Behaviors.setup(setup)
