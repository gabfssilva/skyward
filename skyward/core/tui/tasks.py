"""The tasks section of a compute screen: a page of the compute's tasks, one line each.

The section owns its reads. The fleet observer folds an event stream, which knows
nothing of tasks finished before anybody looked; a page of them is a question for
the API, so this widget asks :func:`skyward.core.history.tasks` on mount and every
:data:`~skyward.core.tui.cells.POLL` seconds, and at once when the sort, the
filter or the page changes. Paging is by cursor and only forward: the stack of
cursors that led to the current page is what "previous" pops, and its depth times
the page size is how many tasks the pages behind held. A failed read leaves what
is drawn and says so on the title line.

A change of sort, filter or page replaces the read in flight, since its answer
is no longer the one wanted. The poll does not: a tick that finds a read still
out waits for it, or a daemon slower than the poll would never be heard.

Each task is drawn as the Claude Code panel draws it: a glyph for its state, the
node it ran on, its function, how it went and, at the end of the line, ``logs``;
its error, whole, on the line under it. The function's name is a link that
narrows the list to that function, and ``logs`` opens the log on that task. The
arrow keys walk the lines for a keyboard, and enter and ``l`` act on the one
under the cursor — a band drawn only while the lines have the focus, so that it
says where the keys go. The status line is links as well: sorting, filtering and
paging are a click as well as a key.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Static
from textual.worker import Worker

from skyward.api.v1 import ExecutionResource, TaskOrder, TaskResource
from skyward.core import history
from skyward.core.client import FAILURES, Client
from skyward.core.tui.cells import ERROR_STYLE, POLL, TASK_GLYPHS, TASK_STYLES, Links, aware, dots, link, rank_badge
from skyward.core.tui.dialogs import Ask
from skyward.core.widgets import DIM, _format_duration

ORDERS: tuple[TaskOrder, ...] = ("state", "submitted", "finished")


class Tasks(Vertical):
    """A page of a compute's tasks, each on a line of its own with its error under it."""

    DEFAULT_CSS = """
    Tasks { height: auto; }
    Tasks > #title, Tasks > #status, Tasks > #rows { height: auto; }
    """

    BINDINGS = [
        Binding("up", "move(-1)", "up", show=False),
        Binding("down", "move(1)", "down", show=False),
        Binding("o", "order", "sort"),
        Binding("f", "filter", "function"),
        Binding("enter", "function", "this function"),
        Binding("n", "next", "next page"),
        Binding("p", "previous", "previous page"),
        Binding("l", "logs", "logs"),
    ]

    @dataclass
    class Logs(Message):
        task_id: str

    def __init__(self, client: Client, compute_id: str, logging: Callable[[], str | None]) -> None:
        super().__init__()
        self._client = client
        self._compute = compute_id
        self._logging = logging
        self._ranks: Mapping[str, int] = {}
        self._order: TaskOrder = ORDERS[0]
        self._function: str | None = None
        self._cursors: tuple[str, ...] = ()
        self._page: history.TaskPage | None = None
        self._failure: str | None = None
        self._selected: str | None = None
        self._reader: Worker[None] | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="title")
        yield Links(id="status")
        rows = Links(id="rows")
        rows.can_focus = True
        yield rows

    def on_mount(self) -> None:
        self.set_interval(POLL, self._poll)
        self._reload()

    def set_ranks(self, ranks: Mapping[str, int]) -> None:
        self._ranks = ranks
        self._draw()

    def redraw(self) -> None:
        """Draw the page again: what the screen asks once the log has moved to another task."""
        self._draw()

    def on_descendant_focus(self) -> None:
        self._draw()

    def on_descendant_blur(self) -> None:
        self._draw()

    async def on_links_pressed(self, message: Links.Pressed) -> None:
        message.stop()
        await self.run_action(message.action)

    def action_move(self, step: int) -> None:
        if self._page is None or not self._page.items:
            return
        keys = [task.id for task in self._page.items]
        at = keys.index(self._selected) if self._selected in keys else 0
        self._selected = keys[max(0, min(len(keys) - 1, at + step))]
        self._draw()

    def action_order(self) -> None:
        self._order = ORDERS[(ORDERS.index(self._order) + 1) % len(ORDERS)]
        self._restart()

    def action_filter(self) -> None:
        self.app.push_screen(Ask("function", self._function or ""), self._filtered)

    def action_function(self, task_id: str | None = None) -> None:
        task = self._named(task_id)
        if task is not None and (name := task.function.name) is not None:
            self._function = None if self._function == name else name
            self._restart()

    def action_next(self) -> None:
        if self._page is not None and self._page.next is not None:
            self._cursors = (*self._cursors, self._page.next)
            self._reload()

    def action_previous(self) -> None:
        if self._cursors:
            self._cursors = self._cursors[:-1]
            self._reload()

    def action_logs(self, task_id: str | None = None) -> None:
        task = self._named(task_id)
        if task is not None and task.state != "queued":
            self._selected = task.id
            self.post_message(self.Logs(task.id))

    def _filtered(self, text: str | None) -> None:
        if text is not None:
            self._function = text or None
            self._restart()

    def _restart(self) -> None:
        self._cursors = ()
        self._reload()

    def _poll(self) -> None:
        if self._reader is None or self._reader.is_finished:
            self._reload()

    def _reload(self) -> None:
        self._reader = self.run_worker(self._read(), exclusive=True, group="tasks")

    async def _read(self) -> None:
        try:
            page = await history.tasks(
                self._client,
                self._compute,
                order=self._order,
                function=self._function,
                cursor=self._cursors[-1] if self._cursors else None,
                seen=len(self._cursors) * history.TASKS_PER_PAGE,
            )
        except* FAILURES as failures:
            error = failures.exceptions[0]
            self._failure = str(error) or type(error).__name__
        else:
            self._page = page
            self._failure = None
        if self.display:
            self._draw()

    def _named(self, task_id: str | None) -> TaskResource | None:
        wanted = self._selected if task_id is None else task_id
        if self._page is None or wanted is None:
            return None
        return next((task for task in self._page.items if task.id == wanted), None)

    def _draw(self) -> None:
        self.query_one("#title", Static).update(self._title())
        self.query_one("#status", Static).update(self._status())
        if self._page is not None:
            self.query_one("#rows", Static).update(self._rows(self._page.items))

    def _title(self) -> Text:
        text = Text("Tasks", style="bold")
        if self._page is not None:
            page = self._page
            counted = {state: page.counts.get(state, 0) for state in history.COUNTED}
            other = (page.total or 0) - sum(counted.values())
            text = Text(f"Tasks ({'?' if page.total is None else page.total})", style="bold")
            text.append("  ")
            text.append(dots(*(f"{count} {state}" for state, count in counted.items()), *((f"{other} other",) if other > 0 else ())), style=DIM)
        if self._failure is not None:
            text.append(f" · daemon request failed: {self._failure}", style=ERROR_STYLE)
        return text

    def _status(self) -> Text:
        if self._page is None:
            return Text()
        items = self._page.items
        first = len(self._cursors) * history.TASKS_PER_PAGE
        total = "?" if self._page.total is None else self._page.total
        span = f"{first + 1}–{first + len(items)} of {total}" if items else "0 of 0"
        parts = (
            link(f"sort: {self._order} ›", "order"),
            link(f"function: {self._function or 'all'}", "filter"),
            link("‹ prev", "previous") if self._cursors else Text("‹ prev", style=DIM),
            Text(span, style=DIM),
            link("next ›", "next") if self._page.next is not None else Text("next ›", style=DIM),
        )
        return Text("  ").join(parts)

    def _rows(self, items: tuple[TaskResource, ...]) -> Text:
        if not items:
            return Text("no tasks" if self._function is None else f"no {self._function} tasks", style=DIM)
        if self._selected not in {task.id for task in items}:
            self._selected = items[0].id
        now = datetime.now(UTC)
        band = Style(bgcolor=self.app.theme_variables.get("panel"))
        cursor = self._selected if self.query_one("#rows").has_focus else None
        lines: list[Text] = []
        for task in items:
            head = self._head(task, now)
            if task.id == cursor:
                head.stylize(band)
            lines.append(head)
            latest = _latest(task)
            if latest is not None and latest.error is not None:
                lines.append(Text(f"{_INDENT}{latest.error.message}", style=ERROR_STYLE))
        return Text("\n").join(lines)

    def _head(self, task: TaskResource, now: datetime) -> Text:
        latest = _latest(task)
        name = task.function.name
        text = Text(TASK_GLYPHS.get(task.state, "·"), style=TASK_STYLES.get(task.state, DIM))
        text.append(" ")
        text.append_text(self._rank(latest))
        text.append(" ")
        text.append_text(
            Text(task.function.sha256[:12], style=DIM) if name is None else link(name, f"function('{task.id}')", "bold" if self._function == name else "")
        )
        text.append(f"  {_progress(task, now)}", style=DIM)
        if len(task.executions) > 1:
            text.append(f" · {len(task.executions)} attempts", style=DIM)
        if task.state != "queued":
            text.append("  ")
            text.append_text(link("logs ✕" if self._logging() == task.id else "logs", f"logs('{task.id}')"))
        return text

    def _rank(self, latest: ExecutionResource | None) -> Text:
        if latest is None or latest.node_id is None:
            return Text(" " * 4)
        return rank_badge(self._ranks.get(latest.node_id))


_INDENT = " " * 7
"""What the error under a task is set in by: the glyph, the badge and the spaces between them."""


def _latest(task: TaskResource) -> ExecutionResource | None:
    return max(task.executions, key=lambda execution: execution.ordinal, default=None)


def _progress(task: TaskResource, now: datetime) -> str:
    state = task.state.replace("_", " ")
    if task.state == "queued":
        return f"queued {_age(now, task.submitted_at)}"
    started = min((execution.started_at for execution in task.executions if execution.started_at is not None), default=None)
    took = None if started is None else _format_duration((aware(task.finished_at or now) - aware(started)).total_seconds())
    if task.finished_at is None:
        return state if took is None else f"{state} {took}"
    ago = f"{_age(now, task.finished_at)} ago"
    if took is None:
        return dots(state, ago)
    return dots(f"{state} {'in' if task.state == 'succeeded' else 'after'} {took}", ago)


def _age(now: datetime, moment: datetime) -> str:
    return _format_duration((now - aware(moment)).total_seconds())


__all__ = ["Tasks"]
