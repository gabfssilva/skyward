from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from types import MappingProxyType
from typing import Any

from casty import ActorContext, ActorRef, Behavior, Behaviors, SpyEvent, Terminated
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from skyward.actors.messages import (
    BootstrapCommand,
    BootstrapConsole,
    BootstrapDone,
    BootstrapPhase,
    Error,
    ExecuteOnNode,
    Log,
    Metric,
    NodeActivated,
    NodeBecameReady,
    NodeConnected,
    NodeLost,
    Preempted,
    Provision,
    ShutdownRequested,
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
    PoolStopped,
    ProvisionFailed,
    StopPool,
    _ShutdownDone,
)
from skyward.actors.snapshot import PoolPhase, PoolSnapshot

from .messages import ConsoleInput, LocalOutput, _PollTick, _SetSession, _SnapshotReceived
from .model import (
    _advance,
    _on_bootstrap_done,
    _on_broadcast_partial,
    _on_metric,
    _on_node_lost,
    _on_spinner_remove,
    _on_ssh_connected,
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
    _node_label,
    _print_provisioning_error,
    _render_summary,
    _ssh_url,
    _task_cost_label,
)

_POLL_INTERVAL = 0.125

_PHASE_MAP: dict[PoolPhase, _Phase] = {
    PoolPhase.PROVISIONING: _Phase.PROVISIONING,
    PoolPhase.SSH: _Phase.SSH,
    PoolPhase.BOOTSTRAP: _Phase.BOOTSTRAP,
    PoolPhase.WORKERS: _Phase.WORKERS,
    PoolPhase.READY: _Phase.READY,
    PoolPhase.STOPPING: _Phase.STOPPING,
}


def _apply_snapshot(state: _State, snap: PoolSnapshot) -> _State:
    cluster = snap.cluster or state.cluster
    ssh_user = cluster.ssh_user if cluster else state.ssh_user
    ssh_key_path = cluster.ssh_key_path if cluster else state.ssh_key_path
    instances = snap.instances or state.instances
    accel_mem = state.spec_accelerator_memory
    if not accel_mem and instances:
        accel = instances[0].offer.instance_type.accelerator
        if accel and accel.memory:
            accel_mem = accel.memory
    return replace(
        state,
        phase=_advance(state.phase, _PHASE_MAP.get(snap.phase, state.phase)),
        total_nodes=len(snap.nodes) or len(instances) or state.total_nodes,
        desired_nodes=snap.scaling.desired_nodes,
        pending_nodes=snap.scaling.pending_nodes,
        draining_nodes=snap.scaling.draining_nodes,
        reconciler_state=snap.scaling.reconciler_state,
        is_elastic=snap.scaling.is_elastic,
        min_nodes=snap.scaling.min_nodes,
        max_nodes=snap.scaling.max_nodes,
        cluster=cluster,
        instances=instances,
        pool_started_at=snap.started_at or state.pool_started_at,
        ssh_user=ssh_user,
        ssh_key_path=ssh_key_path,
        spec_accelerator_memory=accel_mem,
    )


def console_actor() -> Behavior[ConsoleInput]:
    """Console tells this story: idle -> awaiting_session -> observing -> stopped."""

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

    def _stop_live(*, clear: bool = False) -> None:
        nonlocal live, live_stopped
        live_stopped = True
        if live is not None:
            if clear:
                live.update(Text())
            live.stop()
            live = None

    def _schedule_poll(ctx: ActorContext[ConsoleInput]) -> None:
        async def _tick() -> _PollTick:
            await asyncio.sleep(_POLL_INTERVAL)
            return _PollTick()

        ctx.pipe_to_self(
            _tick(),
            mapper=lambda r: r,
            on_failure=lambda _: _PollTick(),
        )

    def _fetch_snapshot(
        ctx: ActorContext[ConsoleInput],
        session_ref: ActorRef[Any],
    ) -> None:
        from skyward.actors.session.messages import GetSessionSnapshot, SessionSnapshot

        async def _ask() -> _SnapshotReceived:
            snap: SessionSnapshot = await ctx.system.ask(
                session_ref,
                lambda r: GetSessionSnapshot(reply_to=r),
                timeout=1.0,
            )
            return _SnapshotReceived(snapshots=snap.pools)

        ctx.pipe_to_self(
            _ask(),
            mapper=lambda r: r,
            on_failure=lambda _: _SnapshotReceived(snapshots=()),
        )

    def _handle_realtime(
        state: _State, msg: ConsoleInput,
    ) -> _State | None:
        match msg:
            case SpyEvent(event=Terminated()):
                return state

            case SpyEvent(event=PoolStopped() | _ShutdownDone()):
                return state

            case SpyEvent(event=ProvisionFailed(reason=reason)) if not state.provision_error_shown:
                _stop_live(clear=True)
                _print_provisioning_error(console, reason)
                return replace(state, provision_error_shown=True)

            case SpyEvent(event=Provision() | NodeBecameReady() | NodeActivated() | NodeConnected() | _PollResult()):
                return state

            case SpyEvent(event=NodeLost(node_id=nid, reason=reason)):
                label = _node_label(state, nid)
                _emit(console, "error", f"Node {nid} lost: {reason}", "red")
                new = _on_node_lost(state, nid)
                _update_footer(new)
                return new

            case SpyEvent(actor_path=path, event=_Connected(instance=ni)):
                nid = _node_id_from_path(path)
                if nid is None:
                    return state
                current = state
                if ni is not None:
                    current = _update_instance(current, ni.instance)
                    if not current.ssh_user:
                        current = replace(
                            current,
                            ssh_user=ni.ssh_user,
                            ssh_key_path=ni.ssh_key_path,
                        )
                new = _on_timeline_phase_started(
                    _on_ssh_connected(current, nid),
                    nid, "connecting",
                )
                _update_footer(new)
                return new

            case SpyEvent(event=_ConnectionFailed(error=error)):
                _emit(console, "error", f"SSH failed: {error}", "red")
                return state

            case SpyEvent(event=Preempted(reason=reason)):
                _emit(console, "error", f"Preempted: {reason}", "red")
                return state

            case SpyEvent(event=BootstrapConsole() as ev):
                content = ev.content.strip()
                if not content or content.startswith("#"):
                    return state
                nid = ev.instance.node
                label = _node_label(state, nid)
                if nid in state.bootstrap_spinners:
                    new = _on_timeline_output(state, nid, content)
                    _update_footer(new)
                    return new
                if ev.overwrite:
                    progress = MappingProxyType({
                        **state.progress_lines, nid: content,
                    })
                    new = replace(state, progress_lines=progress)
                    _update_footer(new)
                    return new
                if nid in state.progress_lines:
                    _emit(console, label, state.progress_lines[nid], link=_ssh_url(state, nid))
                    progress = MappingProxyType({
                        k: v for k, v in state.progress_lines.items() if k != nid
                    })
                    new = replace(state, progress_lines=progress)
                else:
                    new = state
                _emit(console, label, content, link=_ssh_url(state, nid))
                return new

            case SpyEvent(event=BootstrapPhase() as ev):
                nid = ev.instance.node
                label = _node_label(state, nid)
                match ev.event:
                    case "started" if ev.phase != "bootstrap":
                        new = _on_timeline_phase_started(state, nid, ev.phase)
                        _update_footer(new)
                        return new
                    case "completed" if ev.phase != "bootstrap":
                        if nid in state.bootstrap_spinners:
                            new = _on_timeline_phase_completed(state, nid, ev.phase)
                            _update_footer(new)
                            return new
                    case "failed":
                        new = _on_spinner_remove(state, nid)
                        link = _ssh_url(new, nid)
                        _emit(console, label, f"\u2717 {ev.phase}: {ev.error}", "red", link=link)
                        return new
                return state

            case SpyEvent(event=BootstrapCommand() as ev):
                nid = ev.instance.node
                if nid in state.bootstrap_spinners:
                    new = _on_timeline_output(state, nid, ev.command[:80])
                    _update_footer(new)
                    return new
                return state

            case SpyEvent(event=BootstrapDone(instance=inst, success=ok, error=err)):
                nid = inst.node
                label = _node_label(state, nid)
                if ok:
                    new = _on_timeline_phase_started(
                        _on_bootstrap_done(state, nid),
                        nid, "worker",
                    )
                    _update_footer(new)
                    return new
                new = _on_spinner_remove(state, nid)
                link = _ssh_url(new, nid)
                _emit(console, label, f"\u2717 Bootstrap failed: {err}", "red", link=link)
                return new

            case SpyEvent(event=_LocalInstallDone() | _UserCodeSyncDone()):
                return state

            case SpyEvent(event=_PostBootstrapFailed(error=err)):
                _emit(console, "error", f"Post-bootstrap failed: {err}", "red")
                return state

            case SpyEvent(actor_path=path, event=_WorkerStarted()):
                nid = _node_id_from_path(path)
                if nid is not None:
                    label = _node_label(state, nid)
                    started_at = state.bootstrap_started.get(nid)
                    elapsed = f" ({time.monotonic() - started_at:.1f}s)" if started_at else ""
                    new = _on_worker_started(_on_spinner_remove(state, nid), nid)
                    link = _ssh_url(new, nid)
                    _emit(console, label, f"\u2713 Joined{elapsed}", "green bold", link=link)
                    _update_footer(new)
                    return new
                return state

            case SpyEvent(event=_WorkerFailed(error=error)):
                _emit(console, "error", f"Worker failed: {error}", "red")
                return state

            case SpyEvent(event=SubmitTask(task_id=tid) as ev) if tid not in state.inflight:
                name = _format_task(ev.fn, ev.args, ev.kwargs)
                _emit_task(console, "skyward", "queued", name)
                new = _on_task_submitted(state, tid, name, "single")
                _update_footer(new)
                return new

            case SpyEvent(
                event=SubmitBroadcast(task_id=tid) as ev,
            ) if tid not in state.inflight:
                name = _format_task(ev.fn, ev.args, ev.kwargs)
                n = len(state.instances)
                _emit_task(console, "skyward", "queued", f"{name} \u2192 all {n} nodes")
                new = _on_task_submitted(state, tid, name, "broadcast")
                _update_footer(new)
                return new

            case SpyEvent(event=SubmitTask() | SubmitBroadcast()):
                return state

            case SpyEvent(event=TaskSubmitted(task_id=tid, node_id=nid)):
                label = _node_label(state, nid)
                entry = state.inflight.get(tid)
                if entry:
                    _emit_task(console, label, "running", entry.name, link=_ssh_url(state, nid))
                new = _on_task_assigned(state, tid, nid)
                return new

            case SpyEvent(event=TaskResult(task_id=tid, node_id=nid, error=is_err)):
                entry = state.inflight.get(tid)
                if entry is None:
                    return state
                resolved_nid = entry.node_id if entry.node_id >= 0 else nid
                label = _node_label(state, resolved_nid)
                link = _ssh_url(state, resolved_nid)
                match entry.kind:
                    case "broadcast":
                        new = _on_broadcast_partial(state, tid)
                        updated = new.inflight.get(tid)
                        if updated and updated.broadcast_done >= updated.broadcast_total:
                            elapsed = time.monotonic() - entry.started_at
                            cost = _task_cost_label(state, nid, elapsed, broadcast=True)
                            _emit_task(console, label, "done", f"{entry.name} in {elapsed:.1f}s", link=link, cost=cost)
                            new = _on_task_done(new, tid, elapsed)
                        _update_footer(new)
                        return new
                    case _:
                        elapsed = time.monotonic() - entry.started_at
                        if is_err:
                            _emit_task(console, label, "failed", entry.name, link=link)
                            new = _on_task_failed(state, tid)
                        else:
                            cost = _task_cost_label(state, resolved_nid, elapsed)
                            _emit_task(console, label, "done", f"{entry.name} in {elapsed:.1f}s", link=link, cost=cost)
                            new = _on_task_done(state, tid, elapsed)
                        _update_footer(new)
                        return new

            case SpyEvent(event=Log() as ev):
                line = ev.line.strip()
                if line:
                    nid = ev.instance.node
                    label = _node_label(state, nid)
                    if ev.overwrite:
                        progress = MappingProxyType({
                            **state.progress_lines, nid: line,
                        })
                        new = replace(state, progress_lines=progress)
                        _update_footer(new)
                        return new
                    if nid in state.progress_lines:
                        _emit(console, label, state.progress_lines[nid], link=_ssh_url(state, nid))
                        progress = MappingProxyType({
                            k: v for k, v in state.progress_lines.items() if k != nid
                        })
                        new = replace(state, progress_lines=progress)
                    else:
                        new = state
                    _emit(console, label, line, link=_ssh_url(state, nid))
                    return new
                return state

            case SpyEvent(event=Metric() as ev):
                new = _on_metric(state, ev.instance.node, ev.name, ev.value)
                _update_footer(new)
                return new

            case SpyEvent(event=ExecuteOnNode()):
                return state

            case SpyEvent(event=Error() as ev):
                style = "red bold" if ev.fatal else "red"
                _emit(console, "error", ev.message, style)
                return state

            case SpyEvent(event=ShutdownRequested()):
                _stop_live()
                _emit(console, "skyward", "Shutting down...", WARNING_STYLE)
                summary = _render_summary(state)
                console.print(summary)
                return replace(state, phase=_Phase.STOPPING)

            case SpyEvent(event=StopPool()):
                for nid, content in state.progress_lines.items():
                    label = _node_label(state, nid)
                    _emit(console, label, content, link=_ssh_url(state, nid))
                _stop_live()
                _emit(console, "skyward", "Shutting down...", WARNING_STYLE)
                summary = _render_summary(state)
                console.print(summary)
                return replace(state, phase=_Phase.STOPPING)

            case LocalOutput(line=line):
                if stripped := line.rstrip():
                    _emit(console, "local", stripped)
                return state

        return None

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
            return awaiting_session(state)

        return Behaviors.setup(setup)

    def awaiting_session(state: _State) -> Behavior[ConsoleInput]:
        async def receive(
            ctx: ActorContext[ConsoleInput], msg: ConsoleInput,
        ) -> Behavior[ConsoleInput]:
            match msg:
                case _SetSession(ref=session_ref):
                    _schedule_poll(ctx)
                    return Behaviors.with_lifecycle(
                        observing(state, session_ref), post_stop=_restore_stdout,
                    )
                case _:
                    new_state = _handle_realtime(state, msg)
                    if new_state is not None and new_state is not state:
                        return awaiting_session(new_state)
                    return Behaviors.same()

        return Behaviors.receive(receive)

    def observing(state: _State, session_ref: ActorRef[Any]) -> Behavior[ConsoleInput]:
        async def receive(
            ctx: ActorContext[ConsoleInput], msg: ConsoleInput,
        ) -> Behavior[ConsoleInput]:
            match msg:
                case _PollTick():
                    _fetch_snapshot(ctx, session_ref)
                    _schedule_poll(ctx)
                    return Behaviors.same()

                case _SnapshotReceived(snapshots=snaps) if snaps:
                    snap = snaps[0]
                    new = _apply_snapshot(state, snap)
                    _update_footer(new)
                    return observing(new, session_ref)

                case _SnapshotReceived():
                    return Behaviors.same()

                case _SetSession():
                    return Behaviors.same()

                case _:
                    new_state = _handle_realtime(state, msg)
                    if new_state is not None and new_state is not state:
                        return observing(new_state, session_ref)
                    return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()
