"""sky compute -- manage running compute pools via daemon."""

from __future__ import annotations

import asyncio
import json as _json
import time
from typing import Annotated, Any

from cyclopts import Parameter

from skyward.cli import compute_app
from skyward.cli._output import console, print_table


async def _daemon_request(request: Any) -> Any:
    """Send a request to the daemon and return the raw response."""
    from skyward.daemon.client import DaemonClient

    async with DaemonClient() as client:
        return await client.request(request)


def _run_async(coro: Any) -> Any:
    try:
        return asyncio.run(coro)
    except ConnectionError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from None


# -- commands ----------------------------------------------------------------


@compute_app.command(name="list")
def list_pools(
    *,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """List all running compute pools."""
    from skyward.daemon.protocol import GetPools, PoolList

    resp: PoolList = _run_async(_daemon_request(GetPools()))

    if not resp.pools:
        if json:
            print("[]")
        else:
            console.print("[dim]No running pools[/dim]")
        return

    if json:
        data = [
            {
                "name": p.name,
                "phase": p.phase,
                "nodes_ready": p.nodes_ready,
                "nodes_total": p.nodes_total,
                "tasks_done": p.tasks_done,
                "tasks_running": p.tasks_running,
                "started_at": p.started_at,
            }
            for p in resp.pools
        ]
        print(_json.dumps(data, default=str))
        return

    columns = ["Pool", "Phase", "Nodes", "Tasks", "Uptime"]
    rows = []
    for p in resp.pools:
        elapsed = time.time() - p.started_at if p.started_at else 0
        mins = int(elapsed / 60)
        nodes = f"{p.nodes_ready}/{p.nodes_total}"
        tasks = f"{p.tasks_done} done"
        if p.tasks_running:
            tasks += f", {p.tasks_running} running"
        rows.append((p.name, p.phase, nodes, tasks, f"{mins}m"))
    print_table(columns, rows)


@compute_app.default
def _default_list(
    *,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """List all running compute pools."""
    list_pools(json=json)


@compute_app.command(name="view")
def view_pool(
    name: Annotated[str, Parameter(help="Pool name")],
    *,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Show detailed view of a compute pool."""
    from skyward.daemon.protocol import DaemonError, GetPoolView, PoolViewResponse

    resp = _run_async(_daemon_request(GetPoolView(pool_name=name)))
    match resp:
        case DaemonError(error=err):
            console.print(f"[red]{err}[/red]")
        case PoolViewResponse(view=view):
            if json:
                print(_json.dumps(_pool_view_to_dict(view), default=str))
            else:
                _print_pool_view(view)


@compute_app.command(name="tasks")
def show_tasks(
    name: Annotated[str, Parameter(help="Pool name")],
    *,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Show task status for a compute pool."""
    from skyward.daemon.protocol import DaemonError, GetPoolView, PoolViewResponse

    resp = _run_async(_daemon_request(GetPoolView(pool_name=name)))
    match resp:
        case DaemonError(error=err):
            console.print(f"[red]{err}[/red]")
        case PoolViewResponse(view=view):
            if json:
                print(_json.dumps(_tasks_to_dict(view.tasks), default=str))
            else:
                _print_tasks(view)


@compute_app.command(name="stats")
def show_stats(
    name: Annotated[str, Parameter(help="Pool name")],
    *,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Show metrics for a compute pool."""
    from skyward.daemon.protocol import DaemonError, GetPoolView, PoolViewResponse

    resp = _run_async(_daemon_request(GetPoolView(pool_name=name)))
    match resp:
        case DaemonError(error=err):
            console.print(f"[red]{err}[/red]")
        case PoolViewResponse(view=view):
            if json:
                print(_json.dumps(_metrics_to_dict(view), default=str))
            else:
                _print_metrics(view)


@compute_app.command(name="start")
def start_daemon() -> None:
    """Start the Skyward daemon."""
    from skyward.daemon.spawn import ensure_daemon, is_daemon_running

    if is_daemon_running():
        console.print("[dim]Daemon already running[/dim]")
        return
    ensure_daemon()
    console.print("[green]Daemon started[/green]")


@compute_app.command(name="stop")
def stop_pool(
    name: Annotated[str, Parameter(help="Pool name")],
) -> None:
    """Stop a running compute pool."""
    from skyward.daemon.protocol import DaemonError, PoolShutdown, ShutdownPool

    resp = _run_async(_daemon_request(ShutdownPool(pool_name=name)))
    match resp:
        case PoolShutdown():
            console.print(f"[green]Pool '{name}' stopped[/green]")
        case DaemonError(error=err):
            console.print(f"[red]{err}[/red]")


@compute_app.command(name="console")
def console_pool(
    name: Annotated[str, Parameter(help="Pool name")],
) -> None:
    """Live console view of a compute pool."""
    _run_async(_console_stream(name))


async def _console_stream(name: str) -> None:
    from rich.console import Console as RichConsole
    from rich.live import Live
    from rich.text import Text

    from skyward.api.events import Log
    from skyward.api.views import NodeStatus, SessionView
    from skyward.daemon.client import DaemonClient

    rich = RichConsole(stderr=True)

    async with DaemonClient() as client:
        first = True
        with Live(Text("[dim]Connecting...[/dim]"), console=rich, refresh_per_second=4) as live:
            async for msg in client.subscribe(name):
                match msg:
                    case SessionView() as view:
                        pool = view.pools.get(name)
                        if pool is None:
                            continue
                        if first:
                            rich.print(f"[bold]{name}[/bold]  phase={pool.phase.name}")
                            first = False
                        ready = sum(
                            1 for n in pool.nodes.values()
                            if n.status.value >= NodeStatus.READY.value
                        )
                        total = pool.total_nodes
                        tasks_done = pool.tasks.done
                        tasks_running = pool.tasks.running
                        status = Text()
                        status.append(f" {name} ", style="bold")
                        status.append(f" {pool.phase.name} ", style="dim")
                        status.append(f" nodes {ready}/{total} ")
                        if tasks_done or tasks_running:
                            status.append(f" tasks: {tasks_done} done")
                            if tasks_running:
                                status.append(f", {tasks_running} running")
                        if pool.tasks.throughput > 0:
                            status.append(f" ({pool.tasks.throughput:.1f}/min)")
                        live.update(status)

                    case Log.Emitted(node_id=nid, message=message):
                        rich.print(f"  [dim]node-{nid}[/dim]  {message}")

    rich.print("[dim]Stream ended[/dim]")


# -- rendering helpers (private) --------------------------------------------


def _pool_view_to_dict(view: Any) -> dict[str, Any]:
    return {
        "name": view.name,
        "phase": view.phase.name,
        "total_nodes": view.total_nodes,
        "nodes": {
            str(nid): {
                "status": nv.status.name,
                "instance_id": nv.instance.id if nv.instance else None,
                "metrics": dict(nv.metrics),
            }
            for nid, nv in view.nodes.items()
        },
        "tasks": _tasks_to_dict(view.tasks),
        "scaling": {
            "desired": view.scaling.desired,
            "pending": view.scaling.pending,
            "draining": view.scaling.draining,
            "reconciler_state": view.scaling.reconciler_state,
            "is_elastic": view.scaling.is_elastic,
        },
    }


def _tasks_to_dict(tasks: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "queued": tasks.queued,
        "running": tasks.running,
        "done": tasks.done,
        "failed": tasks.failed,
        "throughput": tasks.throughput,
        "inflight": {
            tid: {
                "name": e.name,
                "kind": e.kind,
                "node_id": e.node_id,
                "started_at": e.started_at,
            }
            for tid, e in tasks.inflight.items()
        },
    }
    if tasks.fn_stats:
        result["fn_stats"] = {
            fn: {
                "count": len(lats),
                "avg": sum(lats) / len(lats),
                "min": min(lats),
                "max": max(lats),
            }
            for fn, lats in tasks.fn_stats.items()
        }
    else:
        result["fn_stats"] = {}
    return result


def _metrics_to_dict(view: Any) -> dict[str, Any]:
    return {str(nid): dict(nv.metrics) for nid, nv in view.nodes.items()}


def _print_pool_view(view: Any) -> None:
    console.print(f"\n[bold]{view.name}[/bold]  phase={view.phase.name}")
    console.print(f"  nodes: {view.total_nodes} total")
    for nid, nv in sorted(view.nodes.items()):
        iid = nv.instance.id[:12] if nv.instance else "\u2014"
        console.print(f"    node-{nid}  {nv.status.name}  {iid}")
    console.print(
        f"  scaling: desired={view.scaling.desired}"
        f" pending={view.scaling.pending}"
        f" draining={view.scaling.draining}",
    )
    console.print()


def _print_tasks(view: Any) -> None:
    tasks = view.tasks
    running = [(tid, e) for tid, e in tasks.inflight.items() if e.node_id >= 0]
    queued = [(tid, e) for tid, e in tasks.inflight.items() if e.node_id < 0]
    if running:
        console.print("\n[bold]RUNNING[/bold]")
        for tid, e in running:
            console.print(f"  {tid}  {e.name}  node-{e.node_id}")
    if queued:
        console.print("\n[bold]QUEUED[/bold]")
        for tid, e in queued:
            console.print(f"  {tid}  {e.name}")
    console.print("\n[bold]SUMMARY[/bold]")
    console.print(
        f"  done: {tasks.done}  failed: {tasks.failed}"
        f"  throughput: {tasks.throughput:.1f}/min",
    )
    if tasks.fn_stats:
        console.print()
        for fn, lats in tasks.fn_stats.items():
            avg = sum(lats) / len(lats)
            failed = tasks.fn_failed.get(fn, 0)
            suffix = f", {failed} failed" if failed else ""
            console.print(
                f"  {fn}  avg={avg:.1f}s  min={min(lats):.1f}s"
                f"  max={max(lats):.1f}s  ({len(lats)} calls{suffix})",
            )
    console.print()


def _print_metrics(view: Any) -> None:
    if not any(nv.metrics for nv in view.nodes.values()):
        console.print("[dim]No metrics available[/dim]")
        return
    console.print(f"\n[bold]{view.name}[/bold] metrics\n")
    metric_names: set[str] = set()
    for nv in view.nodes.values():
        metric_names.update(nv.metrics.keys())
    sorted_names = sorted(metric_names)
    columns = ["Node", *sorted_names]
    rows = []
    for nid, nv in sorted(view.nodes.items()):
        row: list[str] = [f"node-{nid}"]
        for mn in sorted_names:
            val = nv.metrics.get(mn)
            row.append(f"{val:.1f}" if val is not None else "\u2014")
        rows.append(tuple(row))
    print_table(columns, rows)
