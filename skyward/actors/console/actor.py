from __future__ import annotations

import time
from dataclasses import replace
from types import MappingProxyType
from typing import Any

from casty import ActorContext, Behavior, Behaviors, SpyEvent, Terminated
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from skyward.actors.messages import (
    BootstrapCommand,
    BootstrapConsole,
    BootstrapDone,
    BootstrapPhase,
    ClusterReady,
    DesiredCountChanged,
    DrainComplete,
    DrainNode,
    Error,
    ExecuteOnNode,
    Log,
    Metric,
    NodeActivated,
    NodeBecameReady,
    NodeJoined,
    NodeLost,
    Preempted,
    Provision,
    ReconcilerNodeLost,
    ShutdownRequested,
    SpawnNodes,
    SubmitBroadcast,
    SubmitTask,
    TaskResult,
    TaskSubmitted,
)
from skyward.actors.node.messages import (
    _Connected,
    _ConnectionFailed,
    _LocalInstallDone,
    _PollResult,
    _PostBootstrapFailed,
    _UserCodeSyncDone,
    _WorkerFailed,
    _WorkerStarted,
)
from skyward.actors.pool.messages import (
    InstancesProvisioned,
    PoolStarted,
    PoolStopped,
    StartPool,
    StopPool,
    _ShutdownDone,
)

from .messages import ConsoleInput, LocalOutput
from .model import (
    _on_bootstrap_done,
    _on_broadcast_partial,
    _on_cluster_ready,
    _on_desired_changed,
    _on_drain_complete,
    _on_drain_node,
    _on_instances_provisioned,
    _on_metric,
    _on_node_joined,
    _on_reconciler_node_lost,
    _on_spawn_nodes,
    _on_spinner_remove,
    _on_ssh_connected,
    _on_start_pool,
    _on_task_assigned,
    _on_task_done,
    _on_task_failed,
    _on_task_submitted,
    _on_timeline_output,
    _on_timeline_phase_completed,
    _on_timeline_phase_started,
    _on_worker_started,
    _update_instance,
)
from .state import _Phase, _State
from .view import (
    DIM,
    WARNING_STYLE,
    _emit,
    _emit_task,
    _format_task,
    _LiveFooter,
    _make_badge,
    _node_id_from_path,
    _render_summary,
    _resolve_instance_id,
    _ssh_url,
)


def console_actor() -> Behavior[ConsoleInput]:
    """Console tells this story: idle -> observing -> stopped."""

    console = Console(stderr=True)
    _original_stdout: list[Any] = []

    def _install_stdout_redirect(ctx: ActorContext[ConsoleInput]) -> None:
        ref = ctx.self

        class _Writer:
            def __init__(self, original: Any) -> None:
                self._original = original

            def write(self, s: str) -> int:
                for line in s.splitlines(keepends=True):
                    if stripped := line.rstrip():
                        ref.tell(LocalOutput(line=stripped))
                return len(s)

            def flush(self) -> None:
                pass

            @property
            def encoding(self) -> str:
                return self._original.encoding

            @property
            def errors(self) -> str | None:
                return self._original.errors

            def fileno(self) -> int:
                return self._original.fileno()

            def isatty(self) -> bool:
                return False

        import sys

        _original_stdout.append(sys.stdout)
        sys.stdout = _Writer(sys.stdout)  # type: ignore[assignment]

    async def _restore_stdout(_ctx: ActorContext[ConsoleInput]) -> None:
        import sys

        if _original_stdout:
            sys.stdout = _original_stdout.pop()

    live: Live | None = None
    live_stopped = False
    footer = _LiveFooter()

    def _update_footer(state: _State) -> None:
        nonlocal live
        if live_stopped:
            return
        footer.state = state
        if live is None:
            live = Live(
                footer, console=console,
                refresh_per_second=8, screen=False,
                redirect_stdout=False, redirect_stderr=False,
            )
            live.start()

    def _stop_live() -> None:
        nonlocal live, live_stopped
        live_stopped = True
        if live is not None:
            live.stop()
            live = None

    def idle() -> Behavior[ConsoleInput]:
        async def setup(ctx: ActorContext[ConsoleInput]) -> Behavior[ConsoleInput]:
            _install_stdout_redirect(ctx)
            from skyward import __version__

            line1 = Text()
            line1.append(f" v{__version__} ", style=_make_badge(140, 0.6))
            line1.append("  Cloud accelerators with a single decorator", style=DIM)

            line2 = Text()
            line2.append("https://gabfssilva.github.io/skyward/", style="underline dim")

            from .view import _LOGO_LINES

            right = [Text(), line1, line2, Text()]

            banner = Table.grid(padding=(0, 2))
            banner.add_column("logo")
            banner.add_column("info")
            for logo_line, info_line in zip(_LOGO_LINES, right, strict=True):
                banner.add_row(logo_line, info_line)
            console.print()
            console.print(banner)
            console.print()
            state = _State(total_nodes=0)
            behavior = observing(state)
            return Behaviors.with_lifecycle(behavior, post_stop=_restore_stdout)

        return Behaviors.setup(setup)

    def observing(state: _State) -> Behavior[ConsoleInput]:
        async def receive(
            ctx: ActorContext[ConsoleInput], msg: ConsoleInput,
        ) -> Behavior[ConsoleInput]:
            match msg:
                case SpyEvent(event=Terminated()):
                    return Behaviors.same()

                case SpyEvent(event=StartPool(spec=pool_spec)):
                    accel_mem = pool_spec.accelerator.memory if pool_spec.accelerator else ""
                    new = replace(
                        _on_start_pool(state),
                        total_nodes=pool_spec.nodes.min,
                        desired_nodes=pool_spec.nodes.min,
                        min_nodes=pool_spec.nodes.min if pool_spec.nodes.auto_scaling else None,
                        max_nodes=pool_spec.nodes.max,
                        is_elastic=pool_spec.nodes.auto_scaling,
                        spec_accelerator_memory=accel_mem,
                    )
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=ClusterReady()):
                    new = _on_cluster_ready(state)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=InstancesProvisioned(cluster=cluster, instances=raw)):
                    new = _on_instances_provisioned(state, cluster, tuple(raw))
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=PoolStarted()):
                    new = replace(state, phase=_Phase.READY)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=StopPool()):
                    for iid, content in state.progress_lines.items():
                        _emit(console, iid, content, link=_ssh_url(state, iid))
                    _stop_live()

                    _emit(console, "skyward", "Shutting down...", WARNING_STYLE)
                    summary = _render_summary(state)
                    console.print(summary)
                    return observing(replace(state, phase=_Phase.STOPPING))

                case SpyEvent(event=PoolStopped() | _ShutdownDone()):
                    return Behaviors.same()

                case SpyEvent(event=Provision() | NodeBecameReady() | NodeActivated() | _PollResult()):
                    return Behaviors.same()

                case SpyEvent(event=NodeLost(node_id=nid, reason=reason)):
                    _emit(console, "error", f"Node {nid} lost: {reason}", "red")

                    return Behaviors.same()

                case SpyEvent(event=DesiredCountChanged(desired=desired)):
                    new = _on_desired_changed(state, desired)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=SpawnNodes(instances=insts, cluster=cl)):
                    new = _on_spawn_nodes(state, tuple(insts), cl)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=NodeJoined()):
                    new = _on_node_joined(state)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=DrainNode()):
                    new = _on_drain_node(state)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=DrainComplete(instance_id=iid)):
                    new = _on_drain_complete(state, iid)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=ReconcilerNodeLost()):
                    new = _on_reconciler_node_lost(state)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(actor_path=path, event=_Connected(instance=ni)):
                    nid = _node_id_from_path(path)
                    current = state
                    if ni is not None:
                        current = _update_instance(current, ni.instance)
                        if not current.ssh_user:
                            current = replace(
                                current,
                                ssh_user=ni.ssh_user,
                                ssh_key_path=ni.ssh_key_path,
                            )
                    iid = _resolve_instance_id(current, node_id=nid)
                    if iid:
                        new = _on_timeline_phase_started(
                            _on_ssh_connected(current, iid),
                            iid, "connecting",
                        )
                        _update_footer(new)
                        return observing(new)
                    return Behaviors.same()

                case SpyEvent(event=_ConnectionFailed(error=error)):
                    _emit(console, "error", f"SSH failed: {error}", "red")

                    return Behaviors.same()

                case SpyEvent(event=Preempted(reason=reason)):
                    _emit(console, "error", f"Preempted: {reason}", "red")

                    return Behaviors.same()

                case SpyEvent(event=BootstrapConsole() as ev):
                    content = ev.content.strip()
                    if not content or content.startswith("#"):
                        return Behaviors.same()
                    iid = ev.instance.instance.id
                    if iid in state.bootstrap_spinners:
                        new = _on_timeline_output(state, iid, content[:80])
                        _update_footer(new)
                        return observing(new)
                    if ev.overwrite:
                        progress = MappingProxyType({
                            **state.progress_lines, iid: content[:120],
                        })
                        new = replace(state, progress_lines=progress)
                        _update_footer(new)
                        return observing(new)
                    if iid in state.progress_lines:
                        _emit(console, iid, state.progress_lines[iid], link=_ssh_url(state, iid))
                        progress = MappingProxyType({
                            k: v for k, v in state.progress_lines.items() if k != iid
                        })
                        new = replace(state, progress_lines=progress)
                    else:
                        new = state
                    _emit(console, iid, content[:120], link=_ssh_url(state, iid))
                    return observing(new)

                case SpyEvent(event=BootstrapPhase() as ev):
                    iid = ev.instance.instance.id
                    match ev.event:
                        case "started" if ev.phase != "bootstrap":
                            new = _on_timeline_phase_started(state, iid, ev.phase)
                            _update_footer(new)
                            return observing(new)
                        case "completed" if ev.phase != "bootstrap":
                            if iid in state.bootstrap_spinners:
                                new = _on_timeline_phase_completed(state, iid, ev.phase)
                                _update_footer(new)
                                return observing(new)
                        case "failed":
                            new = _on_spinner_remove(state, iid)
                            link = _ssh_url(new, iid)
                            _emit(console, iid, f"\u2717 {ev.phase}: {ev.error}", "red", link=link)

                            return observing(new)
                    return Behaviors.same()

                case SpyEvent(event=BootstrapCommand() as ev):
                    iid = ev.instance.instance.id
                    if iid in state.bootstrap_spinners:
                        new = _on_timeline_output(state, iid, ev.command[:80])
                        _update_footer(new)
                        return observing(new)
                    return Behaviors.same()

                case SpyEvent(event=BootstrapDone(instance=inst, success=ok, error=err)):
                    iid = inst.instance.id
                    if ok:
                        new = _on_timeline_phase_started(
                            _on_bootstrap_done(state, iid),
                            iid, "worker",
                        )
                        _update_footer(new)
                        return observing(new)
                    new = _on_spinner_remove(state, iid)
                    link = _ssh_url(new, iid)
                    _emit(console, iid, f"\u2717 Bootstrap failed: {err}", "red", link=link)

                    return observing(new)

                case SpyEvent(event=_LocalInstallDone() | _UserCodeSyncDone()):
                    return Behaviors.same()

                case SpyEvent(event=_PostBootstrapFailed(error=err)):
                    _emit(console, "error", f"Post-bootstrap failed: {err}", "red")

                    return Behaviors.same()

                case SpyEvent(actor_path=path, event=_WorkerStarted()):
                    nid = _node_id_from_path(path)
                    iid = _resolve_instance_id(state, node_id=nid)
                    if iid:
                        started_at = state.bootstrap_started.get(iid)
                        elapsed = f" ({time.monotonic() - started_at:.1f}s)" if started_at else ""
                        new = _on_worker_started(_on_spinner_remove(state, iid), iid)
                        link = _ssh_url(new, iid)
                        _emit(console, iid, f"\u2713 Joined{elapsed}", "green bold", link=link)

                        _update_footer(new)
                        return observing(new)
                    return Behaviors.same()

                case SpyEvent(event=_WorkerFailed(error=error)):
                    _emit(console, "error", f"Worker failed: {error}", "red")

                    return Behaviors.same()

                case SpyEvent(event=SubmitTask(task_id=tid) as ev) if tid not in state.inflight:
                    name = _format_task(ev.fn, ev.args, ev.kwargs)
                    _emit_task(console, "skyward", "queued", name)

                    new = _on_task_submitted(state, tid, name, "single")
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(
                    event=SubmitBroadcast(task_id=tid) as ev,
                ) if tid not in state.inflight:
                    name = _format_task(ev.fn, ev.args, ev.kwargs)
                    n = len(state.instances)
                    _emit_task(console, "skyward", "queued", f"{name} \u2192 all {n} nodes")

                    new = _on_task_submitted(state, tid, name, "broadcast")
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=SubmitTask() | SubmitBroadcast()):
                    return Behaviors.same()

                case SpyEvent(event=TaskSubmitted(task_id=tid, node_id=nid)):
                    iid = _resolve_instance_id(state, node_id=nid) or ""
                    if iid:
                        entry = state.inflight.get(tid)
                        if entry:
                            _emit_task(console, iid, "running", entry.name, link=_ssh_url(state, iid))

                    new = _on_task_assigned(state, tid, iid)
                    return observing(new)

                case SpyEvent(event=TaskResult(task_id=tid, node_id=nid, error=is_err)):
                    entry = state.inflight.get(tid)
                    if entry is None:
                        return Behaviors.same()
                    iid = entry.instance_id or _resolve_instance_id(state, node_id=nid) or "skyward"
                    link = _ssh_url(state, iid)
                    match entry.kind:
                        case "broadcast":
                            new = _on_broadcast_partial(state, tid)
                            updated = new.inflight.get(tid)
                            if updated and updated.broadcast_done >= updated.broadcast_total:
                                elapsed = time.monotonic() - entry.started_at
                                _emit_task(console, iid, "done", f"{entry.name} in {elapsed:.1f}s", link=link)

                                new = _on_task_done(new, tid, elapsed)
                            _update_footer(new)
                            return observing(new)
                        case _:
                            elapsed = time.monotonic() - entry.started_at
                            if is_err:
                                _emit_task(console, iid, "failed", entry.name, link=link)
                                new = _on_task_failed(state, tid)
                            else:
                                _emit_task(console, iid, "done", f"{entry.name} in {elapsed:.1f}s", link=link)
                                new = _on_task_done(state, tid, elapsed)

                            _update_footer(new)
                            return observing(new)

                case SpyEvent(event=Log() as ev):
                    line = ev.line.strip()
                    if line:
                        iid = ev.instance.instance.id
                        if ev.overwrite:
                            progress = MappingProxyType({
                                **state.progress_lines, iid: line,
                            })
                            new = replace(state, progress_lines=progress)
                            _update_footer(new)
                            return observing(new)
                        if iid in state.progress_lines:
                            _emit(console, iid, state.progress_lines[iid], link=_ssh_url(state, iid))
                            progress = MappingProxyType({
                                k: v for k, v in state.progress_lines.items() if k != iid
                            })
                            new = replace(state, progress_lines=progress)
                        else:
                            new = state
                        _emit(console, iid, line, link=_ssh_url(state, iid))
                        return observing(new)
                    return Behaviors.same()

                case SpyEvent(event=Metric() as ev):
                    new = _on_metric(state, ev.instance.instance.id, ev.name, ev.value)
                    _update_footer(new)
                    return observing(new)

                case SpyEvent(event=ExecuteOnNode()):
                    return Behaviors.same()

                case SpyEvent(event=Error() as ev):
                    style = "red bold" if ev.fatal else "red"
                    _emit(console, "error", ev.message, style)

                    return Behaviors.same()

                case SpyEvent(event=ShutdownRequested()):
                    _stop_live()

                    _emit(console, "skyward", "Shutting down...", WARNING_STYLE)
                    summary = _render_summary(state)
                    console.print(summary)
                    return observing(replace(state, phase=_Phase.STOPPING))

                case LocalOutput(line=line):
                    if stripped := line.rstrip():
                        _emit(console, "local", stripped)

                    return Behaviors.same()

                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()
