"""The log section of a compute screen: a page of the compute's recorded events, newest first.

The log is append-only, so a page behind a cursor never changes and is read once;
only the newest page is read again on every :data:`~skyward.core.tui.cells.POLL`.
Paging goes toward older entries only, and the stack of cursors that led to the
current page is what "newer" pops. A filter — a node, a task, a search term —
changes which entries there are, so it empties the stack and reads at once. A
failed read leaves what is drawn and says so on the title line. The poll never
replaces a read still in flight: a daemon slower than the poll is waited for.

The title line is links: a filter in force is dropped by clicking it, and the
paging and the search are a click as well as a key.
"""

from __future__ import annotations

from collections.abc import Mapping

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static
from textual.worker import Worker

from skyward.api.v1 import (
    ComputeDegradedEvent,
    ComputeDeletionFailedEvent,
    LogEntryResource,
    NodeConsoleEvent,
    NodeMetricsEvent,
    NodePhaseEvent,
    NodeProgressEvent,
    NodeStateEvent,
)
from skyward.core import history
from skyward.core.client import FAILURES, Client
from skyward.core.tui.cells import ERROR_STYLE, POLL, Links, clock, link, rank_badge
from skyward.core.tui.dialogs import Ask
from skyward.core.widgets import DIM


class Log(Vertical):
    """A page of a compute's event log, with the filters in force on its title line."""

    DEFAULT_CSS = """
    Log { height: 1fr; }
    Log > #title { height: auto; }
    Log > #body { height: 1fr; }
    Log > #body > Static { height: auto; }
    """

    BINDINGS = [
        Binding("/", "search", "search"),
        Binding("[", "older", "older"),
        Binding("]", "newer", "newer"),
        Binding("0", "latest", "latest"),
        Binding("x", "clear", "clear filters"),
    ]

    def __init__(self, client: Client, compute_id: str) -> None:
        super().__init__()
        self._client = client
        self._compute = compute_id
        self._ranks: Mapping[str, int] = {}
        self._node_id: str | None = None
        self._task_id: str | None = None
        self._term = ""
        self._cursors: tuple[str, ...] = ()
        self._page: history.LogPage | None = None
        self._current = False
        self._failure: str | None = None
        self._reader: Worker[None] | None = None

    @property
    def node_id(self) -> str | None:
        return self._node_id

    @property
    def task_id(self) -> str | None:
        return self._task_id

    def compose(self) -> ComposeResult:
        yield Links(id="title")
        with VerticalScroll(id="body"):
            yield Static(id="lines")

    def on_mount(self) -> None:
        self.set_interval(POLL, self._poll)
        self._reload()

    def set_ranks(self, ranks: Mapping[str, int]) -> None:
        self._ranks = ranks
        self._draw()

    def show_node(self, node_id: str | None) -> None:
        self._node_id = node_id
        self._restart()

    def show_task(self, task_id: str | None) -> None:
        self._task_id = task_id
        self._restart()

    def action_search(self) -> None:
        self.app.push_screen(Ask("filter", self._term), self._searched)

    def action_older(self) -> None:
        if self._page is not None and self._page.next is not None:
            self._cursors = (*self._cursors, self._page.next)
            self._reload()

    def action_newer(self) -> None:
        if self._cursors:
            self._cursors = self._cursors[:-1]
            self._reload()

    def action_latest(self) -> None:
        self._restart()

    def action_drop(self, which: str) -> None:
        match which:
            case "node":
                self._node_id = None
            case "task":
                self._task_id = None
            case _:
                self._term = ""
        self._restart()

    async def on_links_pressed(self, message: Links.Pressed) -> None:
        message.stop()
        await self.run_action(message.action)

    def action_clear(self) -> None:
        self._node_id = None
        self._task_id = None
        self._term = ""
        self._restart()

    def _searched(self, text: str | None) -> None:
        if text is not None:
            self._term = text
            self._restart()

    def _restart(self) -> None:
        self._cursors = ()
        self._reload()

    def _poll(self) -> None:
        idle = self._reader is None or self._reader.is_finished
        if idle and (not self._cursors or not self._current):
            self._reload()

    def _reload(self) -> None:
        self._current = False
        self._reader = self.run_worker(self._read(), exclusive=True, group="log")

    async def _read(self) -> None:
        try:
            page = await history.log(
                self._client,
                self._compute,
                node=self._node_id,
                task=self._task_id,
                term=self._term,
                cursor=self._cursors[-1] if self._cursors else None,
            )
        except FAILURES as failure:
            self._failure = str(failure) or type(failure).__name__
        else:
            self._page = page
            self._current = True
            self._failure = None
        if self.display:
            self._draw()

    def _draw(self) -> None:
        self.query_one("#title", Static).update(self._title())
        self.query_one("#lines", Static).update(self._lines())

    def _title(self) -> Text:
        newest = not self._cursors
        older = self._page is not None and self._page.next is not None
        parts = [
            Text("Log", style="bold"),
            *((link(f"node #{self._ranks.get(self._node_id, '?')} ✕", "drop('node')"),) if self._node_id is not None else ()),
            *((link(f"task {self._task_id} ✕", "drop('task')"),) if self._task_id is not None else ()),
            *((link(f'"{self._term}" ✕', "drop('term')"),) if self._term else ()),
            Text("live" if newest else f"page {len(self._cursors) + 1}", style=DIM),
            link("‹ older", "older") if older else Text("‹ older", style=DIM),
            Text("newer ›", style=DIM) if newest else link("newer ›", "newer"),
            Text("latest »", style=DIM) if newest else link("latest »", "latest"),
            link("search", "search"),
        ]
        text = Text("  ").join(parts)
        if self._failure is not None:
            text.append(f" · daemon request failed: {self._failure}", style=ERROR_STYLE)
        return text

    def _lines(self) -> Text:
        if self._page is None:
            return Text()
        if not self._page.entries:
            filtered = self._node_id is not None or self._task_id is not None or self._term != ""
            return Text("no matching events" if filtered else "no events", style=DIM)
        text = Text()
        for index, entry in enumerate(self._page.entries):
            if index:
                text.append("\n")
            self._append(text, entry)
        return text

    def _append(self, text: Text, entry: LogEntryResource) -> None:
        node, body, failed = _line(entry)
        text.append(clock(entry.at), style=DIM)
        text.append(" ")
        text.append_text(rank_badge(self._ranks.get(node)) if node is not None else Text("    "))
        text.append(" ")
        text.append(body, style=ERROR_STYLE if failed else "" if entry.type == "node.console" else DIM)


def _line(entry: LogEntryResource) -> tuple[str | None, str, bool]:
    """The node an entry names, what it says, and whether it carries an error."""
    match entry.data:
        case NodeConsoleEvent(node=node, content=content):
            return node, content.rstrip("\n"), False
        case NodePhaseEvent(node=node, phase=phase, event=mark, error=error):
            return node, _said(f"{phase} {mark}", error), error is not None
        case NodeStateEvent(node=node, error=error):
            return node, _said(entry.type, error), error is not None
        case NodeProgressEvent(node=node) | NodeMetricsEvent(node=node):
            return node, entry.type, False
        case ComputeDegradedEvent(error=error) | ComputeDeletionFailedEvent(error=error):
            return None, _said(entry.type, error), True
        case _:
            return None, entry.type, False


def _said(text: str, error: str | None) -> str:
    return text if error is None else f"{text}: {error}"


__all__ = ["Log"]
