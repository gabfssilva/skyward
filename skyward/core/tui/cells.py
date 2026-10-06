"""The words and colours ``sky app`` draws with, shared by every screen.

The fleet screen, the compute screen and its sections show the same states, the
same tasks and the same numbers, and a state must not be green on one screen and
yellow on the next. This module holds that vocabulary — the state sets that move
a spinner, the styles of nodes and tasks, the rank badges, the sparkline and the
small formatters — so each screen composes cells and none of them defines a
colour. The tables are the console's own (:mod:`skyward.core.widgets`); what is
added here is what only a screen that stays open needs.

The row under a cursor is marked by a band behind it and keeps its colours
(:func:`table`): a state that is green is still green when it is the one
selected, which is when it is being read.

A screen that stays open is also clicked on. :func:`link` is a word that can be,
and :class:`Links` is the line that holds such words: whatever a key does, the
link beside it does too, by running the same action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from rich.style import Style
from rich.text import Text
from textual.message import Message
from textual.widgets import DataTable, Static

from skyward.core.widgets import DIM, _badge_style

REFRESH = 0.25
"""Seconds between repaints — the spinner's pace."""

POLL = 2.0
"""Seconds between API reads of an open compute screen."""

NAMED_TASKS = 3
"""Up to this many tasks on one node are named in its row; past it, the row says how many."""

ERROR_STYLE = "red"

MOVING = frozenset({"requested", "provisioning", "deleting"})
NODE_MOVING = frozenset({"requested", "provisioning", "connecting", "bootstrapping", "draining", "deleting"})
BILLING = frozenset({"provisioning", "connecting", "bootstrapping", "ready", "draining", "deleting"})
BADGE_KEYS: Mapping[str, str] = {"degraded": "failed", "deleting": "shutting down", "requested": "queued"}
NODE_STYLES: Mapping[str, str] = {
    "requested": "bright_black",
    "provisioning": "bright_black",
    "connecting": "yellow",
    "bootstrapping": "yellow",
    "ready": "green",
    "draining": "yellow",
    "lost": "red",
    "failed": "red",
    "deleting": "color(238)",
    "deleted": "color(238)",
}
TASK_GLYPHS: Mapping[str, str] = {
    "running": "●",
    "queued": "◌",
    "succeeded": "✓",
    "failed": "✗",
    "timed_out": "✗",
    "cancelled": "○",
    "indeterminate": "?",
}
TASK_STYLES: Mapping[str, str] = {
    "running": "green",
    "failed": "red",
    "timed_out": "red",
    "indeterminate": "magenta",
}
RANK_COLORS = ("#3b6fd8", "#8e5bd6", "#1f8f8f", "#c7702a", "#c2427a", "#6b8e23", "#546e7a", "#8d6e63")
SPARK_GLYPHS = "▁▂▃▄▅▆▇█"


class Links(Static):
    """A line of text with links in it, each asking the widget around the line to act.

    A clicked link runs its action on the widget it was drawn in, and a line of
    text has no actions of its own. So the line only says which one was asked
    for, and the widget that owns the line runs it: the action a key is bound to.

    A link is drawn as it was written, not underlined and recoloured the way
    Textual marks one: the words that can be clicked are the ones that name an
    action, and a state that is green stays green when it is also a link. The
    one under the mouse is still lit, as Textual lights it.
    """

    @dataclass
    class Pressed(Message):
        action: str

    @property
    def link_style(self) -> Style:
        return Style()

    def action_press(self, action: str) -> None:
        self.post_message(self.Pressed(action))


def table(name: str) -> DataTable[Text]:
    """A table with a row cursor that leaves each cell the colour it was drawn in."""
    return DataTable[Text](id=name, cursor_type="row", zebra_stripes=False, cursor_foreground_priority="renderable")


def link(label: str, action: str, style: str = "") -> Text:
    """``label`` as a link of a :class:`Links` line, running ``action`` on the widget that owns the line."""
    return Text(label, style=Style.parse(style) + Style(meta={"@click": f"press({action!r})"}))


def state_badge(state: str, frame: str) -> Text:
    label = f"{frame} {state}" if state in MOVING else state
    return Text(f" {label} ", style=_badge_style(BADGE_KEYS.get(state, state)))


def rank_badge(rank: int | None, muted: bool = False) -> Text:
    if rank is None:
        return Text(" #? ", style=DIM)
    if muted:
        return Text(f" #{rank} ", style=DIM)
    return Text(f" #{rank} ", style=f"bold white on {RANK_COLORS[rank % len(RANK_COLORS)]}")


def spark(values: Sequence[float | None]) -> str:
    return "".join(" " if value is None else SPARK_GLYPHS[max(0, min(7, int(value / 100 * 8)))] for value in values)


def dots(*parts: str) -> str:
    return " · ".join(parts)


def short(machine: str | None) -> str:
    if machine is None:
        return ""
    return machine if len(machine) <= 12 else f"{machine[:5]}…{machine[-4:]}"


def aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def clock(moment: datetime) -> str:
    return aware(moment).astimezone().strftime("%H:%M:%S")


def money(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


__all__ = [
    "BADGE_KEYS",
    "BILLING",
    "ERROR_STYLE",
    "Links",
    "MOVING",
    "NAMED_TASKS",
    "NODE_MOVING",
    "NODE_STYLES",
    "POLL",
    "REFRESH",
    "TASK_GLYPHS",
    "TASK_STYLES",
    "aware",
    "clock",
    "dots",
    "link",
    "money",
    "rank_badge",
    "short",
    "spark",
    "state_badge",
    "table",
]
