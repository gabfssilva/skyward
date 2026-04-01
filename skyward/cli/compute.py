"""sky compute -- manage running compute pools via daemon."""

from __future__ import annotations

import asyncio
import json as _json
import time
from typing import Annotated, Any

from cyclopts import Parameter

from skyward.cli import compute_app
from skyward.cli._output import BRANCH, BRANCH_LAST, console, phase_label


async def _daemon_request(request: Any) -> Any:
    """Send a request to the daemon and return the raw response."""
    from skyward.daemon.client import DaemonClient

    async with DaemonClient() as client:
        return await client.request(request)


def _ensure_daemon_running() -> None:
    from skyward.daemon.spawn import ensure_daemon, is_daemon_running

    if not is_daemon_running():
        from rich.console import Console as RichConsole

        from skyward.cli._output import INACTIVE

        RichConsole(stderr=True).print(f"{INACTIVE} Daemon not running, starting...")
        ensure_daemon()


def _run_async(coro: Any) -> Any:
    _ensure_daemon_running()
    try:
        return asyncio.run(coro)
    except ConnectionError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from None


_PHASE_LABELS: dict[str, str] = {
    "PROVISIONING": "Provisioning instances",
    "SSH": "Connecting via SSH",
    "BOOTSTRAP": "Bootstrapping environment",
    "WORKERS": "Starting workers",
}


# -- commands ----------------------------------------------------------------


@compute_app.command(name="list")
def list_pools(
    *,
    all: Annotated[bool, Parameter(name="--all", help="Include stopped pools")] = False,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """List compute pools."""
    from skyward.daemon.protocol import GetPools, PoolList

    resp: PoolList = _run_async(_daemon_request(GetPools()))

    pools = resp.pools
    if not all:
        pools = tuple(p for p in pools if p.phase != "STOPPED")

    if not pools:
        if json:
            print("[]")
        else:
            console.print("[dim]  No pools running[/dim]")
        return

    if json:
        data = [
            {
                "name": p.name,
                "phase": p.phase,
                "provider": p.provider,
                "accelerator": p.accelerator,
                "nodes_ready": p.nodes_ready,
                "nodes_total": p.nodes_total,
                "tasks_done": p.tasks_done,
                "tasks_running": p.tasks_running,
                "started_at": p.started_at,
                "avg_cpu": p.avg_cpu,
                "avg_mem": p.avg_mem,
            }
            for p in pools
        ]
        print(_json.dumps(data, default=str))
        return

    for p in pools:
        elapsed = time.time() - p.started_at if p.started_at else 0
        mins = int(elapsed / 60)
        nodes = f"{p.nodes_ready}/{p.nodes_total} nodes"
        tasks = f"{p.tasks_done} done"
        if p.tasks_running:
            tasks += f", {p.tasks_running} running"

        spec = p.provider
        if p.accelerator:
            spec += f" · {p.accelerator}"

        phase = phase_label(p.phase)

        console.print(
            f"  [bold]{p.name}[/bold]    [dim]{spec}[/dim]    {phase}   "
            f"{nodes}   {tasks}   [dim]{mins}m[/dim]"
        )

        spec_parts = []
        if p.vcpus:
            spec_parts.append(f"{p.vcpus} vcpu")
        if p.memory:
            spec_parts.append(p.memory)
        if p.vram:
            spec_parts.append(f"{p.vram} vram")
        if p.disk:
            spec_parts.append(f"{p.disk} disk")
        if spec_parts:
            console.print(f"  [dim]           {' · '.join(spec_parts)}[/dim]")

        metrics_parts = []
        if p.avg_cpu is not None:
            metrics_parts.append(f"cpu {p.avg_cpu:.0f}%")
        if p.avg_mem is not None:
            metrics_parts.append(f"mem {p.avg_mem:.0f}%")
        if metrics_parts:
            console.print(f"  [dim]           {' · '.join(metrics_parts)}[/dim]")

        console.print()


@compute_app.default
def _default_list(
    *,
    all: Annotated[bool, Parameter(name="--all", help="Include stopped pools")] = False,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """List compute pools."""
    list_pools(all=all, json=json)


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


@compute_app.command(name="start")
def start_pool(
    name: Annotated[str, Parameter(help="Pool name from skyward.toml")],
    *,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Start a named compute pool via the daemon."""
    from pathlib import Path

    from skyward.daemon.protocol import PoolFailed, PoolLogLine, PoolProvisioning, PoolReady

    async def _start() -> None:
        from contextlib import nullcontext

        from rich.status import Status

        from skyward.cli._output import ERROR, PROGRESS
        from skyward.daemon.client import DaemonClient

        use_spinner = not json and console.is_terminal
        ctx = Status("", console=console, spinner="dots") if use_spinner else nullcontext()
        msg: object = None
        current_phase = ""

        with ctx as spinner:
            async with DaemonClient() as client:
                async for msg in client.ensure_pool_stream(
                    name, project_dir=str(Path.cwd()),
                ):
                    match msg:
                        case PoolProvisioning(phase=phase):
                            current_phase = _PHASE_LABELS.get(phase, phase)
                            if json:
                                print(_json.dumps({"pool": name, "phase": phase.lower()}))
                            elif spinner is not None:
                                spinner.update(f"{current_phase}...")
                            else:
                                console.print(f"{PROGRESS} {current_phase}...")
                        case PoolLogLine(message=log_msg):
                            if spinner is not None and current_phase:
                                short = log_msg[:60].strip()
                                spinner.update(f"{current_phase}: [dim]{short}[/dim]")
                        case PoolReady() | PoolFailed():
                            if spinner is not None:
                                spinner.stop()

        match msg:
            case PoolReady(pool_name=pn, node_count=n):
                if json:
                    print(_json.dumps({"pool": pn, "node_count": n, "status": "ready"}))
                else:
                    console.print(f"[green]✓[/green] Pool [bold]{pn}[/bold] ready ({n} nodes)")
            case PoolFailed(pool_name=pn, reason=reason):
                if json:
                    print(_json.dumps({"pool": pn, "status": "failed", "reason": reason}))
                else:
                    console.print(f"{ERROR} Pool [bold]{pn}[/bold] failed: {reason}")
                raise SystemExit(1)

    _run_async(_start())


@compute_app.command(name="logs")
def pool_logs(
    name: Annotated[str, Parameter(help="Pool name")],
    *,
    n: Annotated[int, Parameter(name=["-n", "--lines"], help="Number of lines")] = 50,
    follow: Annotated[bool, Parameter(name=["-f", "--follow"], help="Follow log output")] = False,
    all: Annotated[bool, Parameter(name="--all", help="All log files, not just latest")] = False,
    json: Annotated[bool, Parameter(name="--json", help="Raw JSONL output")] = False,
) -> None:
    """Show logs for a compute pool."""
    import subprocess

    from skyward.daemon.protocol import DaemonError, GetPoolLogs, PoolLogs

    resp = _run_async(_daemon_request(GetPoolLogs(pool_name=name, all=all)))
    match resp:
        case DaemonError(error=err):
            console.print(f"[red]{err}[/red]")
            raise SystemExit(1)
        case PoolLogs(paths=paths):
            pass
        case _:
            console.print(f"[red]Unexpected response: {resp}[/red]")
            raise SystemExit(1)

    import json as json_mod

    def _render_line(line: str) -> None:
        if json:
            print(line)
        else:
            try:
                entry = json_mod.loads(line)
                node = entry.get("node", "?")
                msg = entry.get("message", line)
                console.print(f"  [dim]node-{node}:[/dim] {msg}")
            except json_mod.JSONDecodeError:
                console.print(f"  {line}")

    if follow:
        import subprocess

        try:
            proc = subprocess.Popen(
                ["tail", "-f", paths[-1]],
                stdout=subprocess.PIPE,
                text=True,
            )
            assert proc.stdout is not None
            for raw in proc.stdout:
                _render_line(raw.rstrip())
        except KeyboardInterrupt:
            pass
        return

    from collections import deque
    from pathlib import Path

    lines: deque[str] = deque(maxlen=n)
    for path in paths:
        with Path(path).open() as fh:
            for raw in fh:
                lines.append(raw.rstrip())

    for line in lines:
        _render_line(line)


@compute_app.command(name="stop")
def stop_pool(
    name: Annotated[str, Parameter(help="Pool name")],
) -> None:
    """Stop a running compute pool."""
    from skyward.cli._output import ERROR, SUCCESS
    from skyward.daemon.protocol import DaemonError, PoolShutdown, ShutdownPool

    resp = _run_async(_daemon_request(ShutdownPool(pool_name=name)))
    match resp:
        case PoolShutdown():
            console.print(f"{SUCCESS} Pool [bold]{name}[/bold] stopped")
        case DaemonError(error=err):
            console.print(f"{ERROR} {err}")


@compute_app.command(name="console")
def console_pool(
    name: Annotated[str, Parameter(help="Pool name")],
) -> None:
    """Live console view of a compute pool."""
    _run_async(_console_stream(name))


async def _console_stream(name: str) -> None:
    from dataclasses import replace as _replace
    from types import MappingProxyType

    from rich.console import Console as RichConsole
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text

    from skyward.actors.console.actor import _print_event
    from skyward.actors.console.state import _State
    from skyward.actors.console.view import (
        _LOGO_LINES,
        DIM,
        WARNING_STYLE,
        _emit,
        _LiveFooter,
        _make_badge,
        _node_label,
        _render_summary,
        _state_from_pool_view,
    )
    from skyward.api.events import Log, Pool
    from skyward.api.views import SessionView
    from skyward.daemon.client import DaemonClient

    rich = RichConsole(stderr=True)
    footer = _LiveFooter()
    live: Live | None = None
    live_stopped = False
    latest_view = SessionView()
    progress: dict[int, str] = {}

    def _get_state() -> _State:
        pool = latest_view.pools.get(name)
        state = _state_from_pool_view(pool) if pool else _State(total_nodes=0)
        if progress:
            state = _replace(state, progress_lines=MappingProxyType(progress))
        return state

    def _update_footer(state: _State) -> None:
        nonlocal live
        if live_stopped:
            return
        footer.state = state
        if live is None:
            live = Live(
                footer, console=rich,
                refresh_per_second=8, screen=False,
                redirect_stdout=False, redirect_stderr=False,
            )
            live.start()

    def _stop_live(*, clear: bool = False) -> None:
        nonlocal live, live_stopped
        live_stopped = True
        if live is not None:
            if clear:
                live.update(Text())
            live.stop()
            live = None

    # -- banner --
    from skyward import __version__

    line1 = Text()
    line1.append(f" v{__version__} ", style=_make_badge(140, 0.6))
    line1.append("  Cloud accelerators with a single decorator", style=DIM)

    line2 = Text()
    line2.append("https://gabfssilva.github.io/skyward/", style="underline dim")

    right = [Text(), line1, line2, Text()]
    banner = Table.grid(padding=(0, 2))
    banner.add_column("logo")
    banner.add_column("info")
    for logo_line, info_line in zip(_LOGO_LINES, right, strict=True):
        banner.add_row(logo_line, info_line)
    rich.print()
    rich.print(banner)
    rich.print()

    # -- event loop --
    async with DaemonClient() as client:
        async for msg in client.subscribe(name):
            match msg:
                case SessionView() as view:
                    latest_view = view
                    state = _get_state()
                    _update_footer(state)

                case Log.Emitted(node_id=nid, message=message, overwrite=ow):
                    if ow:
                        progress[nid] = message
                        _update_footer(_get_state())
                    else:
                        if nid in progress:
                            state = _get_state()
                            _emit(rich, _node_label(state, nid), progress.pop(nid))
                        state = _get_state()
                        _emit(rich, _node_label(state, nid), message)

                case Pool.Stopped() | Pool.ProvisionFailed():
                    state = _get_state()
                    for nid, content in progress.items():
                        _emit(rich, _node_label(state, nid), content)
                    progress.clear()
                    _stop_live(clear=isinstance(msg, Pool.ProvisionFailed))
                    _print_event(rich, msg, state)
                    if isinstance(msg, Pool.Stopped):
                        _emit(rich, "skyward", "Shutting down...", WARNING_STYLE)
                    rich.print(_render_summary(state))

                case _:
                    state = _get_state()
                    _print_event(rich, msg, state)

    if not live_stopped:
        _stop_live()
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




def _print_pool_view(view: Any) -> None:
    console.print(f"  {phase_label(view.phase.name)} [bold]{view.name}[/bold] · {view.total_nodes} nodes")
    console.print()

    nodes = sorted(view.nodes.items())
    if nodes:
        console.print("  [bold]Nodes[/bold]")
        for i, (nid, nv) in enumerate(nodes):
            connector = BRANCH_LAST if i == len(nodes) - 1 else BRANCH
            iid = nv.instance.id[:12] if nv.instance else "\u2014"
            status = phase_label(nv.status.name)
            metrics_parts = []
            if "cpu" in nv.metrics:
                metrics_parts.append(f"cpu {nv.metrics['cpu']:.0f}%")
            if "mem" in nv.metrics:
                metrics_parts.append(f"mem {nv.metrics['mem']:.0f}%")
            metrics_str = f"   [dim]{' · '.join(metrics_parts)}[/dim]" if metrics_parts else ""
            console.print(f"  {connector} node-{nid}   {status}   [dim]{iid}[/dim]{metrics_str}")
        console.print()

    tasks = view.tasks
    console.print("  [bold]Tasks[/bold]")
    console.print(f"     {tasks.done} done · {tasks.failed} failed · {tasks.throughput:.1f}/min")

    if tasks.fn_stats:
        console.print()
        console.print("  [bold]Functions[/bold]")
        fns = list(tasks.fn_stats.items())
        for i, (fn, lats) in enumerate(fns):
            connector = BRANCH_LAST if i == len(fns) - 1 else BRANCH
            avg = sum(lats) / len(lats)
            n_calls = len(lats)
            call_word = "call" if n_calls == 1 else "calls"
            failed = tasks.fn_failed.get(fn, 0)
            suffix = f" · {failed} failed" if failed else ""
            console.print(
                f"  {connector} {fn}   {n_calls} {call_word}   "
                f"[dim]avg {avg:.1f}s · min {min(lats):.1f}s · max {max(lats):.1f}s{suffix}[/dim]"
            )

    running = [(tid, e) for tid, e in tasks.inflight.items() if e.node_id >= 0]
    queued = [(tid, e) for tid, e in tasks.inflight.items() if e.node_id < 0]
    if running:
        console.print()
        console.print("  [bold]Running[/bold]")
        for i, (tid, e) in enumerate(running):
            connector = BRANCH_LAST if i == len(running) - 1 else BRANCH
            elapsed = time.time() - e.started_at
            console.print(f"  {connector} {tid[:8]}   {e.name}   node-{e.node_id}   [dim]{elapsed:.0f}s[/dim]")
    if queued:
        console.print()
        console.print("  [bold]Queued[/bold]")
        for i, (tid, e) in enumerate(queued):
            connector = BRANCH_LAST if i == len(queued) - 1 else BRANCH
            console.print(f"  {connector} {tid[:8]}   {e.name}")

    console.print()
    scaling = view.scaling
    console.print("  [bold]Scaling[/bold]")
    console.print(f"     desired {scaling.desired} · pending {scaling.pending} · draining {scaling.draining}")
    console.print()


def _print_tasks(view: Any) -> None:
    tasks = view.tasks
    console.print(f"  [bold]{view.name}[/bold] · {tasks.done} done · {tasks.failed} failed · {tasks.throughput:.1f}/min")

    if tasks.fn_stats:
        console.print()
        console.print("  [bold]Functions[/bold]")
        fns = list(tasks.fn_stats.items())
        for i, (fn, lats) in enumerate(fns):
            connector = BRANCH_LAST if i == len(fns) - 1 else BRANCH
            avg = sum(lats) / len(lats)
            n_calls = len(lats)
            call_word = "call" if n_calls == 1 else "calls"
            failed = tasks.fn_failed.get(fn, 0)
            suffix = f" · {failed} failed" if failed else ""
            console.print(
                f"  {connector} {fn}   {n_calls} {call_word}   "
                f"[dim]avg {avg:.1f}s · min {min(lats):.1f}s · max {max(lats):.1f}s{suffix}[/dim]"
            )

    running = [(tid, e) for tid, e in tasks.inflight.items() if e.node_id >= 0]
    queued = [(tid, e) for tid, e in tasks.inflight.items() if e.node_id < 0]
    if running:
        console.print()
        console.print("  [bold]Running[/bold]")
        for i, (tid, e) in enumerate(running):
            connector = BRANCH_LAST if i == len(running) - 1 else BRANCH
            elapsed = time.time() - e.started_at
            console.print(f"  {connector} {tid[:8]}   {e.name}   node-{e.node_id}   [dim]{elapsed:.0f}s[/dim]")
    if queued:
        console.print()
        console.print("  [bold]Queued[/bold]")
        for i, (tid, e) in enumerate(queued):
            connector = BRANCH_LAST if i == len(queued) - 1 else BRANCH
            console.print(f"  {connector} {tid[:8]}   {e.name}")
    console.print()


