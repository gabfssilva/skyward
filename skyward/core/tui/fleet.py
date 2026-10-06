"""The fleet screen: the live computes as a list, and the ones that ended under them.

It is the list the Claude Code panel opens on, drawn the same way: one line per
compute — its name, a glyph for its state and how many of its machines are up,
the accelerator, the provider, what it costs an hour and what it has cost, its
age — a total under the live ones, and the last few that ended below, dim. A
compute with an error says it on the line after its own.

The observer folds the event stream; this screen only draws, on a timer rather
than per event — a compute streaming gauges from eight machines would otherwise
repaint the list faster than a terminal can show it. What has ended is not in the
stream, so the screen asks the API for the last few every :data:`RECENT_POLL`
seconds. A read that fails is the daemon being unreachable, and the list gives way
to saying so, as the panel does.

Every name is a link to its compute, and the arrow keys walk the rows for a
keyboard: enter opens the one under the cursor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Static
from textual.worker import Worker

from skyward.api.v1 import ComputeResource
from skyward.core import history
from skyward.core.client import FAILURES, Client, address
from skyward.core.fleet import Fleet, FleetObserver
from skyward.core.tui.cells import BILLING, ERROR_STYLE, REFRESH, Links, aware, dots, link, money
from skyward.core.tui.compute import ComputeScreen
from skyward.core.view import ComputeView
from skyward.core.widgets import DIM, _accelerator_label

RECENT_POLL = 5.0
"""Seconds between reads of the computes that have ended."""

GAP = 2
"""Columns between two cells."""


@dataclass(frozen=True, slots=True)
class Row:
    """One compute as the list says it, live or ended."""

    id: str
    name: str
    state: str
    ready: int
    total: int
    accelerator: str | None
    accelerator_count: int
    provider: str
    rate: float | None
    cost: float | None
    age: float
    error: str | None
    ended: bool


@dataclass(frozen=True, slots=True)
class Column:
    title: str
    width: int
    right: bool
    cell: Callable[[Row], Text]


class FleetScreen(Screen[None]):
    """The list of computes: the live ones, a total, and the ones that ended."""

    BINDINGS = [
        Binding("enter", "open", "open compute", priority=True),
        Binding("up", "move(-1)", "up", show=False),
        Binding("down", "move(1)", "down", show=False),
    ]

    def __init__(self, fleet: FleetObserver, url: str, client: Client) -> None:
        super().__init__()
        self._fleet = fleet
        self._url = url
        self._client = client
        self._selected: str | None = None
        self._tick = 0
        self._recent: tuple[ComputeResource, ...] = ()
        self._loaded = False
        self._failure: str | None = None
        self._at = datetime.now(UTC)
        self._reading: Worker[None] | None = None
        self._shown: tuple[Fleet, tuple[ComputeResource, ...], str | None, str | None] | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="page"):
            yield Static(id="summary")
            with VerticalScroll(can_focus=False):
                yield Links(id="computes")
        yield Footer()

    def on_mount(self) -> None:
        self._timer = self.set_interval(REFRESH, self._repaint)
        self._reader = self.set_interval(RECENT_POLL, self._fetch)
        self._fetch()
        self._repaint()

    def on_screen_suspend(self) -> None:
        self._timer.pause()
        self._reader.pause()

    def on_screen_resume(self) -> None:
        self._timer.resume()
        self._reader.resume()
        self._fetch()
        self._repaint()

    async def on_links_pressed(self, message: Links.Pressed) -> None:
        await self.run_action(message.action)

    def action_open(self, compute_id: str | None = None) -> None:
        if (opened := compute_id or self._selected) is not None:
            self.app.push_screen(ComputeScreen(self._client, opened))

    def action_move(self, step: int) -> None:
        keys = [row.id for row in self._rows()]
        if not keys:
            return
        at = keys.index(self._selected) if self._selected in keys else 0
        self._selected = keys[max(0, min(len(keys) - 1, at + step))]
        self._repaint()

    def _fetch(self) -> None:
        """Read the ended computes again, unless the last read is still out: a slow daemon is waited for, not asked twice."""
        if self._reading is None or self._reading.is_finished:
            self._reading = self.run_worker(self._read(), group="recent")

    async def _read(self) -> None:
        try:
            self._recent = await history.recent(self._client)
        except FAILURES as error:
            self._failure = str(error) or type(error).__name__
        else:
            self._failure = None
        self._loaded = True

    def _rows(self) -> list[Row]:
        now = datetime.now(UTC)
        fleet = self._fleet.views
        live = sorted(fleet.values(), key=lambda view: aware(view.created_at) if view.created_at else now, reverse=True)
        ended = (compute for compute in self._recent if compute.id not in fleet)
        return [*(_live(view, now) for view in live), *(_ended(compute, now) for compute in ended)]

    def _repaint(self) -> None:
        self._tick += 1
        fleet = self._fleet.views
        drawn = (fleet, self._recent, self._selected, self._failure)
        if self._shown is not None and all(now is before for now, before in zip(drawn, self._shown, strict=True)) and self._tick % 4:
            return
        if self._shown is None or self._shown[0] is not fleet or self._shown[1] is not self._recent:
            self._at = datetime.now(UTC)
        self._shown = drawn
        self.query_one("#summary", Static).update(Text(dots(where(self._url), f"updated {self._at.astimezone():%H:%M:%S}"), style=DIM))
        rows = self._rows()
        if self._selected not in {row.id for row in rows}:
            self._selected = rows[0].id if rows else None
        self.query_one("#computes", Static).update(self._page(rows))

    def _page(self, rows: list[Row]) -> Text:
        if self._failure is not None:
            return Text("\n").join((Text(f"daemon request failed at {self._url}: {self._failure}", style=ERROR_STYLE), Text("sky server start", style=DIM)))
        if not self._loaded and not rows:
            return Text("loading…", style=DIM)
        band = Style(bgcolor=self.app.theme_variables.get("panel"))
        columns = _columns(rows)
        width = sum(column.width for column in columns) + GAP * (len(columns) - 1)
        rule = Text("─" * width, style=DIM)
        live = [row for row in rows if not row.ended]
        ended = [row for row in rows if row.ended]
        lines = [_line(columns, tuple(Text(column.title, style=DIM) for column in columns)), rule]
        if not live:
            lines.append(Text("no live computes", style=DIM))
        for row in live:
            lines.extend(self._lines(columns, row, band))
        if live:
            lines.extend((rule, _line(columns, _totals(columns, live))))
        if ended:
            lines.extend((Text(""), Text("recent", style=DIM)))
            for row in ended:
                lines.extend(self._lines(columns, row, band))
        return Text("\n").join(lines)

    def _lines(self, columns: tuple[Column, ...], row: Row, band: Style) -> list[Text]:
        line = _line(columns, tuple(column.cell(row) for column in columns))
        if row.ended:
            line.stylize(DIM)
        if row.id == self._selected:
            line.stylize(band)
        lines = [line]
        if row.error is not None and not row.ended:
            lines.append(Text(f"  {row.error}", style=ERROR_STYLE))
        return lines


def where(url: str) -> str:
    """The daemon, as a person would say it: ``@ :17590`` on this machine, the url anywhere else."""
    host, port = address(url)
    return f"@ :{port}" if host in _LOCAL else url


def rate(view: ComputeView) -> float:
    """What the compute costs per hour right now: the price of every machine still held."""
    return sum(node.price_per_hour or 0.0 for node in view.nodes if node.state in BILLING)


def age(seconds: float) -> str:
    """A duration as the panel says it: the one unit that fits."""
    if seconds < 60:
        return f"{max(0, int(seconds))}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 48 * 3600:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


_LOCAL = frozenset({"127.0.0.1", "localhost", "::1"})

_STATE_GLYPHS = {"requested": "◌", "provisioning": "◐", "ready": "●", "degraded": "●", "deleting": "◑", "deleted": "○"}
_STATE_STYLES = {
    "requested": "yellow",
    "provisioning": "yellow",
    "connecting": "yellow",
    "bootstrapping": "yellow",
    "ready": "green",
    "degraded": "red",
    "draining": "magenta",
    "deleting": "magenta",
}


def _live(view: ComputeView, now: datetime) -> Row:
    return Row(
        id=view.id,
        name=view.name or view.id,
        state=view.state,
        ready=view.nodes_ready,
        total=view.nodes_total or len(view.nodes),
        accelerator=view.accelerator,
        accelerator_count=view.accelerator_count,
        provider=view.provider or "?",
        rate=rate(view),
        cost=view.cost,
        age=(now - aware(view.created_at)).total_seconds() if view.created_at else 0.0,
        error=view.errors[-1] if view.errors else None,
        ended=False,
    )


def _ended(compute: ComputeResource, now: datetime) -> Row:
    spec = compute.spec.specs[0] if compute.spec.specs else None
    accelerator, count = _shape(compute)
    return Row(
        id=compute.id,
        name=compute.name or compute.id,
        state=compute.status.state,
        ready=0,
        total=0,
        accelerator=accelerator,
        accelerator_count=count,
        provider=compute.provider.name if compute.provider else (spec.provider.kind if spec else "?"),
        rate=None,
        cost=compute.ended.cost if compute.ended else compute.cost,
        age=(now - aware(compute.created_at)).total_seconds(),
        error=compute.status.last_error.message if compute.status.last_error else None,
        ended=True,
    )


def _shape(compute: ComputeResource) -> tuple[str | None, int]:
    """The accelerator bought, or the one asked for while nothing was."""
    if compute.offer:
        return compute.offer.accelerator, compute.offer.accelerator_count
    if compute.spec.specs:
        return compute.spec.specs[0].accelerator, compute.spec.specs[0].accelerator_count
    return None, 1


def _state(row: Row) -> Text:
    if row.ended:
        return Text(_STATE_GLYPHS["deleted"])
    return Text(f"{_STATE_GLYPHS.get(row.state, '●')} {row.ready}/{row.total}", style=_STATE_STYLES.get(row.state, ""))


def _gpu(row: Row) -> Text:
    if row.accelerator is None:
        return Text("—")
    label = _accelerator_label(row.accelerator)
    return Text(f"{row.accelerator_count}×{label}" if row.accelerator_count > 1 else label)


def _columns(rows: list[Row]) -> tuple[Column, ...]:
    """The panel's columns; the name takes the room the longest one needs, up to a point."""
    longest = max((len(row.name) for row in rows), default=0)
    return (
        Column("NAME", min(32, max(16, longest)), False, lambda row: link(row.name, f"open('{row.id}')", "" if row.ended else "bold")),
        Column("STATE", 9, False, _state),
        Column("GPU", 10, False, _gpu),
        Column("PROVIDER", 9, False, lambda row: Text(row.provider)),
        Column("$/H", 7, True, lambda row: Text(money(row.rate))),
        Column("COST", 7, True, lambda row: Text(money(row.cost))),
        Column("AGE", 4, True, lambda row: Text(age(row.age))),
    )


def _totals(columns: tuple[Column, ...], live: list[Row]) -> tuple[Text, ...]:
    def summed(values: list[float | None]) -> str:
        return money(None if any(value is None for value in values) else sum(value or 0.0 for value in values))

    sums = {"NAME": "total", "$/H": summed([row.rate for row in live]), "COST": summed([row.cost for row in live])}
    return tuple(Text(sums.get(column.title, ""), style="bold") for column in columns)


def _line(columns: tuple[Column, ...], cells: tuple[Text, ...]) -> Text:
    line = Text()
    for index, (column, cell) in enumerate(zip(columns, cells, strict=True)):
        if index:
            line.append(" " * GAP)
        cell.truncate(column.width, overflow="ellipsis")
        if column.right:
            line.append(" " * (column.width - cell.cell_len))
            line.append_text(cell)
        else:
            line.append_text(cell)
            line.append(" " * (column.width - cell.cell_len))
    return line


__all__ = ["GAP", "RECENT_POLL", "Column", "FleetScreen", "Row", "age", "rate", "where"]
