"""``sky app``: the fleet on one screen, and one compute under the cursor.

A Textual application over :class:`~skyward.core.fleet.FleetObserver`. The
observer folds; the screens draw. The application itself holds the observer, the
daemon's address and the theme: a white page on a light terminal, Textual's own
dark theme on a dark one. Every screen draws in a column of :data:`WIDTH`
characters at most, centred in the terminal: a list of a dozen computes read on
a wide monitor is a column, not a line stretched across it. The fleet's list is
narrower still and takes only the width it needs, so it is the list that sits in
the middle and not a column with the list at its left edge.
"""

from __future__ import annotations

from textual.app import App
from textual.binding import Binding
from textual.theme import Theme

from skyward.core.client import Client
from skyward.core.fleet import FleetObserver
from skyward.core.tui.fleet import FleetScreen
from skyward.core.widgets import _DARK

WIDTH = 120
"""The most columns a screen uses; a narrower terminal is used whole."""

LIGHT = Theme(
    name="skyward-light",
    primary="#1f6b3a",
    secondary="#2f5a8a",
    warning="#b45309",
    error="#b91c1c",
    success="#1f6b3a",
    accent="#2f5a8a",
    foreground="#111111",
    background="#ffffff",
    surface="#ffffff",
    panel="#f2f2f2",
    dark=False,
    variables={"footer-background": "#f2f2f2", "footer-key-foreground": "#1f6b3a"},
)
"""A white page: the terminal's own light background is the theme, not a tint of it."""


class Dashboard(App[None]):
    """The screen ``sky app`` opens: every live compute, and the one under the cursor."""

    TITLE = "skyward"
    CSS = f"""
    Screen {{ layout: vertical; align-horizontal: center; }}
    #page {{ width: {WIDTH}; max-width: 100%; height: 1fr; }}
    FleetScreen #page, FleetScreen #page > VerticalScroll, #summary, #computes {{ width: auto; }}
    #summary {{ height: 1; margin: 1 2 1 2; }}
    #computes {{ height: auto; margin: 0 2; }}
    DataTable > .datatable--header {{ color: $text-muted; background: transparent; text-style: none; }}
    DataTable > .datatable--cursor {{ background: $panel; }}
    """
    BINDINGS = [Binding("q", "quit", "quit")]
    ENABLE_COMMAND_PALETTE = False

    def __init__(self, client: Client, url: str) -> None:
        super().__init__()
        self.register_theme(LIGHT)
        self.theme = "textual-dark" if _DARK else LIGHT.name
        self._client = client
        self._fleet = FleetObserver(client)
        self._url = url

    def on_mount(self) -> None:
        self.run_worker(self._fleet.follow(), exclusive=True)
        self.push_screen(FleetScreen(self._fleet, self._url, self._client))


__all__ = ["LIGHT", "WIDTH", "Dashboard"]
