"""The notifications of a screen: what failed lately and is not yet seen, one entry per kind of failure.

The section owns its reads, as the tasks and the log do: every
:data:`~skyward.core.tui.cells.POLL` seconds it asks for the failures of the last
day — of every compute on the fleet screen, of one on a compute screen — and
draws the ones the reader's :class:`~skyward.core.notifications.Seen` does not
hold. Each entry is a line saying what failed, how many times and when, the
reason under it whole, and ``seen`` at the end of the line: a link that
dismisses it. ``mark all seen`` on the title, and ``a`` on either screen,
dismiss every entry drawn. A dismissal is drawn at once, not at the next read.

With nothing left to say, the section is not drawn at all. On the fleet screen
the compute's name is a link to it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Static
from textual.worker import Worker

from skyward.api.v1 import LogEntryResource
from skyward.core import notifications
from skyward.core.client import FAILURES, Client
from skyward.core.notifications import Notification, Seen
from skyward.core.tui.cells import ERROR_STYLE, POLL, Links, aware, link
from skyward.core.widgets import DIM, _format_duration


class Notifications(Vertical):
    """The failures not yet seen, of every compute or of one."""

    DEFAULT_CSS = """
    Notifications { height: auto; }
    Notifications > #title, Notifications > #entries { height: auto; }
    """

    @dataclass
    class Open(Message):
        compute_id: str

    def __init__(self, client: Client, seen: Seen, compute: str | None = None, names: Callable[[], Mapping[str, str]] = lambda: {}) -> None:
        super().__init__()
        self._client = client
        self._seen = seen
        self._compute = compute
        self._names = names
        self._window: tuple[LogEntryResource, ...] = ()
        self._dismissed: frozenset[int] = frozenset()
        self._reader: Worker[None] | None = None
        self.display = False

    def compose(self) -> ComposeResult:
        yield Links(id="title")
        yield Links(id="entries")

    def on_mount(self) -> None:
        self.set_interval(POLL, self._poll)
        self._poll()

    @property
    def shown(self) -> tuple[Notification, ...]:
        return notifications.unseen(self._window, self._dismissed)

    async def on_links_pressed(self, message: Links.Pressed) -> None:
        message.stop()
        await self.run_action(message.action)

    def action_seen(self, newest: int) -> None:
        self._dismiss(next((notification.sequences for notification in self.shown if notification.sequences[0] == newest), ()))

    def action_seen_all(self) -> None:
        self._dismiss(tuple(sequence for notification in self.shown for sequence in notification.sequences))

    def action_open(self, compute_id: str) -> None:
        self.post_message(self.Open(compute_id))

    def _dismiss(self, sequences: tuple[int, ...]) -> None:
        if not sequences:
            return
        self._seen.mark(sequences, datetime.now(UTC))
        self._dismissed = self._dismissed | frozenset(sequences)
        self._draw()

    def _poll(self) -> None:
        if self._reader is None or self._reader.is_finished:
            self._reader = self.run_worker(self._read(), group="notifications")

    async def _read(self) -> None:
        try:
            self._window = await notifications.window(self._client, datetime.now(UTC), self._compute)
        except FAILURES:
            return
        self._dismissed = self._seen.read()
        self._draw()

    def _draw(self) -> None:
        shown = self.shown
        self.display = bool(shown)
        if not shown:
            return
        title = Text(f"Notifications ({len(shown)})", style="bold")
        title.append("  ")
        title.append_text(link("mark all seen", "seen_all"))
        self.query_one("#title", Static).update(title)
        now = datetime.now(UTC)
        self.query_one("#entries", Static).update(Text("\n").join(line for notification in shown for line in self._entry(notification, now)))

    def _entry(self, notification: Notification, now: datetime) -> tuple[Text, ...]:
        count = len(notification.sequences)
        head = Text("✗ ", style=ERROR_STYLE)
        if self._compute is None:
            head.append_text(link(self._names().get(notification.compute, notification.compute), f"open('{notification.compute}')", "bold"))
            head.append("  ")
        head.append(notifications.describe(notification.kind), style=ERROR_STYLE)
        if count > 1:
            head.append(f" ×{count}", style=ERROR_STYLE)
        head.append(f"  {_format_duration((now - aware(notification.at)).total_seconds())} ago  ", style=DIM)
        head.append_text(link("seen", f"seen({notification.sequences[0]})"))
        if notification.reason is None:
            return (head,)
        return head, Text(f"{_INDENT}{notification.reason}")


_INDENT = "  "


__all__ = ["Notifications"]
