"""The Textual application: themes, timers, and screen wiring."""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from textual.app import App
from textual.binding import Binding
from textual.css.query import NoMatches

from skyward.tui.screens import DashboardScreen, NodeScreen
from skyward.tui.sources import DataSource, SimulatedSource
from skyward.tui.theme import SKYWARD_DARK, SKYWARD_LIGHT

__all__ = ["SkywardTUI", "run"]


class SkywardTUI(App[None]):
    """Skyward cluster monitor.

    Drives a :class:`~skyward.tui.sources.DataSource` on a one-second tick
    and a faster pulse timer that animates the in-progress status dots.

    Parameters
    ----------
    source : DataSource
        Provider of cluster snapshots.
    start_dark : bool
        Start in the dark theme (default matches the mockup: light).
    tick_interval : float
        Seconds between data ticks.
    on_ready : Callable[[], None] | None
        Called once the app is mounted and its message loop is live — used
        by :class:`~skyward.core.application.Application` to release the
        worker thread only after the dashboard can accept ``call_from_thread``.
    """

    CSS_PATH = "app.tcss"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("t", "toggle_theme", "theme"),
        Binding("q", "quit", "quit"),
    ]

    def __init__(
        self,
        source: DataSource,
        *,
        start_dark: bool = False,
        tick_interval: float = 1.0,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._source = source
        self._start_dark = start_dark
        self._tick_interval = tick_interval
        self._on_ready = on_ready
        self.pulse_on = True

    @property
    def source(self) -> DataSource:
        """The active data source."""
        return self._source

    def get_theme_variable_defaults(self) -> dict[str, str]:
        """Provide fallbacks so ``app.tcss`` parses before our theme is active.

        The stylesheet is parsed under the default Textual theme, which does
        not define ``$edge``/``$dim``/etc.  These defaults keep parsing valid;
        the Skyward themes override them per-theme once applied.
        """
        return {
            "edge": "#222a35", "dim": "#6e7787", "faint": "#2c3440",
            "tint": "#15202f", "hover": "#161c26", "bright": "#f0f3f6",
            "cyan": "#39c5cf",
        }

    def on_mount(self) -> None:
        """Register themes, show the dashboard, and start the timers."""
        self.register_theme(SKYWARD_DARK)
        self.register_theme(SKYWARD_LIGHT)
        self.theme = "skyward-dark" if self._start_dark else "skyward-light"
        self.push_screen(DashboardScreen())
        self.set_interval(self._tick_interval, self._tick)
        self.set_interval(0.5, self._pulse)
        if self._on_ready is not None:
            self._on_ready()

    def _tick(self) -> None:
        self._source.tick()
        self._refresh()

    def _refresh(self) -> None:
        # NoMatches: the screen is mid-mount during a push; the next tick repaints.
        if isinstance(self.screen, (DashboardScreen, NodeScreen)):
            with contextlib.suppress(NoMatches):
                self.screen.refresh_data(self._source.snapshot())

    def _pulse(self) -> None:
        self.pulse_on = not self.pulse_on
        if isinstance(self.screen, (DashboardScreen, NodeScreen)):
            with contextlib.suppress(NoMatches):
                self.screen.apply_pulse()

    def action_toggle_theme(self) -> None:
        """Swap between the light and dark Skyward themes."""
        self.theme = "skyward-light" if self.theme == "skyward-dark" else "skyward-dark"
        self._refresh()


def run(*, dark: bool = False, tick_interval: float = 1.0, seed: int = 7) -> None:
    """Launch the TUI with the simulated data source.

    Parameters
    ----------
    dark : bool
        Start in the dark theme (default is light, matching the mockup).
    tick_interval : float
        Seconds between simulated ticks.
    seed : int
        Seed for the simulation's jitter.
    """
    SkywardTUI(
        SimulatedSource(seed=seed), start_dark=dark, tick_interval=tick_interval,
    ).run()
