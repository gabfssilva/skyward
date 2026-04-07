from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Any

from casty import ActorContext, Behavior, Behaviors
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from skyward.api.events import Error, Node, Pool, Task
from skyward.api.views import PoolView, SessionView

from .messages import ConsoleInput, EventReceived, LocalOutput, LogReceived, ViewUpdated
from .state import _State
from .view import (
    DIM,
    WARNING_STYLE,
    _emit,
    _emit_task,
    _LiveFooter,
    _make_badge,
    _node_label,
    _print_provisioning_error,
    _render_summary,
    _ssh_url,
    _state_from_pool_view,
)


def _first_pool(view: SessionView) -> PoolView | None:
    if view.pools:
        return next(iter(view.pools.values()))
    return None


def _print_event(console: Console, event: object, state: _State) -> None:
    match event:
        case Pool.ProvisionFailed(reason=reason):
            _print_provisioning_error(console, reason)
        case Node.Ready(node_id=nid):
            label = _node_label(state, nid)
            _emit(console, label, "\u2713 Joined", "green bold", link=_ssh_url(state, nid))
        case Node.Lost(node_id=nid, reason=reason):
            _emit(console, "error", f"Node {nid} lost: {reason}", "red")
        case Node.ConnectionFailed(error=error):
            _emit(console, "error", f"SSH failed: {error}", "red")
        case Node.Preempted(reason=reason):
            _emit(console, "error", f"Preempted: {reason}", "red")
        case Node.WorkerFailed(error=error):
            _emit(console, "error", f"Worker failed: {error}", "red")
        case Node.Bootstrap.Failed(node_id=nid, phase=phase, error=err):
            label = _node_label(state, nid)
            _emit(console, label, f"\u2717 {phase}: {err}", "red", link=_ssh_url(state, nid))
        case Task.Queued(name=name, kind="broadcast"):
            n = len(state.instances)
            _emit_task(console, "skyward", "queued", f"{name} \u2192 all {n} nodes")
        case Task.Queued(name=name):
            _emit_task(console, "skyward", "queued", name)
        case Task.Completed(node_id=nid, elapsed=elapsed):
            label = _node_label(state, nid)
            _emit_task(console, label, "done", f"in {elapsed:.1f}s", link=_ssh_url(state, nid))
        case Task.Failed(node_id=nid):
            label = _node_label(state, nid)
            _emit_task(console, label, "failed", "", link=_ssh_url(state, nid))
        case Error.Occurred(message=message, fatal=fatal):
            style = "red bold" if fatal else "red"
            _emit(console, "error", message, style)
        case Pool.Stopped():
            pass
        case _:
            pass


def console_actor() -> Behavior[ConsoleInput]:
    """Console tells this story: idle -> rendering."""

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
            return Behaviors.with_lifecycle(
                rendering(SessionView()), post_stop=_restore_stdout,
            )

        return Behaviors.setup(setup)

    def _get_state(
        view: SessionView,
        progress: MappingProxyType[int, str] = MappingProxyType({}),
    ) -> _State:
        pool = _first_pool(view)
        state = _state_from_pool_view(pool) if pool else _State(total_nodes=0)
        if progress:
            state = replace(state, progress_lines=progress)
        return state

    def rendering(
        view: SessionView,
        progress: MappingProxyType[int, str] = MappingProxyType({}),
    ) -> Behavior[ConsoleInput]:
        async def receive(
            ctx: ActorContext[ConsoleInput], msg: ConsoleInput,
        ) -> Behavior[ConsoleInput]:
            match msg:
                case ViewUpdated(view=new_view):
                    state = _get_state(new_view, progress)
                    _update_footer(state)
                    return rendering(new_view, progress)

                case EventReceived(event=event):
                    state = _get_state(view, progress)
                    match event:
                        case Pool.Stopped() | Pool.ProvisionFailed():
                            for nid, content in progress.items():
                                _emit(console, _node_label(state, nid), content)
                            _stop_live(clear=isinstance(event, Pool.ProvisionFailed))
                            _print_event(console, event, state)
                            if isinstance(event, Pool.Stopped):
                                _emit(console, "skyward", "Shutting down...", WARNING_STYLE)
                            summary = _render_summary(state)
                            console.print(summary)
                            return rendering(view, MappingProxyType({}))
                        case Node.Lost(node_id=nid) if nid in progress:
                            _emit(console, _node_label(state, nid), progress[nid])
                            _print_event(console, event, state)
                            new_progress = MappingProxyType({
                                k: v for k, v in progress.items() if k != nid
                            })
                            return rendering(view, new_progress)
                        case _:
                            _print_event(console, event, state)
                    return Behaviors.same()

                case LogReceived(log=log):
                    nid = log.node_id
                    if log.overwrite:
                        new_progress = MappingProxyType({**progress, nid: log.message})
                        state = _get_state(view, new_progress)
                        _update_footer(state)
                        return rendering(view, new_progress)
                    if nid in progress:
                        state = _get_state(view, progress)
                        _emit(console, _node_label(state, nid), progress[nid])
                        new_progress = MappingProxyType({
                            k: v for k, v in progress.items() if k != nid
                        })
                    else:
                        new_progress = progress
                    state = _get_state(view, new_progress)
                    _emit(console, _node_label(state, nid), log.message)
                    return rendering(view, new_progress)

                case LocalOutput(line=line):
                    if stripped := line.rstrip():
                        _emit(console, "local", stripped)
                    return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()
