from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

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
    _print_no_offers,
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
        case Pool.NoOffers(specs=specs):
            _print_no_offers(console, specs)
        case Node.Ready(node_id=nid):
            label = _node_label(state, nid)
            _emit(console, label, "✓ Joined", "green bold", link=_ssh_url(state, nid))
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
            _emit(console, label, f"✗ {phase}: {err}", "red", link=_ssh_url(state, nid))
        case Task.Queued(name=name, kind="broadcast"):
            n = len(state.instances)
            _emit_task(console, "skyward", "queued", f"{name} → all {n} nodes")
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


class _Writer:
    def __init__(self, original: Any, post: Callable[[ConsoleInput], None]) -> None:
        self._original = original
        self._post = post

    def write(self, s: str) -> int:
        for line in s.splitlines(keepends=True):
            if stripped := line.rstrip():
                self._post(LocalOutput(line=stripped))
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


class RichConsole:
    """Rich adaptive console: banner → live footer → event lines → summary."""

    tick_interval: float | None = None

    def __init__(self) -> None:
        self._console = Console(stderr=True)
        self._original_stdout: Any = None
        self._live: Live | None = None
        self._live_stopped = False
        self._footer = _LiveFooter()
        self._view = SessionView()
        self._progress: dict[int, str] = {}

    def start(self, post: Callable[[ConsoleInput], None]) -> None:
        self._original_stdout = sys.stdout
        sys.stdout = _Writer(sys.stdout, post)  # type: ignore[assignment]
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
        self._console.print()
        self._console.print(banner)
        self._console.print()

    def stop(self) -> None:
        self._stop_live()
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
            self._original_stdout = None

    def tick(self) -> None:
        pass

    def _update_footer(self, state: _State) -> None:
        if self._live_stopped:
            return
        self._footer.state = state
        if self._live is None:
            self._live = Live(
                self._footer, console=self._console,
                refresh_per_second=8, screen=False,
                redirect_stdout=False, redirect_stderr=False,
            )
            self._live.start()

    def _stop_live(self, *, clear: bool = False) -> None:
        self._live_stopped = True
        if self._live is not None:
            if clear:
                self._live.update(Text())
            self._live.stop()
            self._live = None

    def _get_state(self) -> _State:
        pool = _first_pool(self._view)
        state = _state_from_pool_view(pool) if pool else _State(total_nodes=0)
        if self._progress:
            from dataclasses import replace
            from types import MappingProxyType

            state = replace(state, progress_lines=MappingProxyType(dict(self._progress)))
        return state

    def handle(self, msg: ConsoleInput) -> None:
        match msg:
            case ViewUpdated(view=new_view):
                self._view = new_view
                self._update_footer(self._get_state())

            case EventReceived(event=event):
                state = self._get_state()
                match event:
                    case Pool.Stopped() | Pool.ProvisionFailed() | Pool.NoOffers():
                        for nid, content in self._progress.items():
                            _emit(self._console, _node_label(state, nid), content)
                        self._stop_live(
                            clear=isinstance(event, Pool.ProvisionFailed | Pool.NoOffers),
                        )
                        _print_event(self._console, event, state)
                        if isinstance(event, Pool.Stopped):
                            _emit(self._console, "skyward", "Shutting down...", WARNING_STYLE)
                        self._console.print(_render_summary(state))
                        self._progress = {}
                    case Node.Lost(node_id=nid) if nid in self._progress:
                        _emit(self._console, _node_label(state, nid), self._progress[nid])
                        _print_event(self._console, event, state)
                        self._progress.pop(nid, None)
                    case _:
                        _print_event(self._console, event, state)

            case LogReceived(log=log):
                nid = log.node_id
                if log.overwrite:
                    self._progress[nid] = log.message
                    self._update_footer(self._get_state())
                    return
                if nid in self._progress:
                    state = self._get_state()
                    _emit(self._console, _node_label(state, nid), self._progress[nid])
                    self._progress.pop(nid, None)
                state = self._get_state()
                _emit(self._console, _node_label(state, nid), log.message)

            case LocalOutput(line=line):
                if stripped := line.rstrip():
                    _emit(self._console, "local", stripped)
