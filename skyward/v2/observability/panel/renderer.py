"""Panel renderer.

Manages Rich Live display with auto-refresh.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.style import Style
from rich.text import Text

from .components import PanelLayout

if TYPE_CHECKING:
    from .state import PanelState

# Style constants for final output
STYLE_SUCCESS = Style(color="green")
STYLE_DIM = Style(color="bright_black")
STYLE_SPOT = Style(color="green")


def _format_duration(seconds: float) -> str:
    """Format seconds as Xm XXs."""
    mins, secs = divmod(int(seconds), 60)
    if mins > 0:
        return f"{mins}m{secs:02d}s"
    return f"{secs}s"


class PanelRenderable:
    """Rich renderable that builds panel from state on each render.

    Rich Live calls __rich_console__ at refresh_per_second rate.
    We build a fresh ViewModel from state each time.
    """

    def __init__(self, state: PanelState) -> None:
        self._state = state

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = options.max_width
        height = console.size.height
        blink_on = int(time.monotonic() * 2) % 2 == 0
        vm = self._state.to_view_model(width, height, blink_on)
        yield PanelLayout(vm).render()


class PanelRenderer:
    """Manages Rich Live display - simplified, no background thread.

    Rich Live handles auto-refresh at 4fps. We just pass a PanelRenderable
    that reads from state on each render.
    """

    def __init__(self) -> None:
        self._live: Live | None = None
        self._state: PanelState | None = None

    @property
    def console(self) -> Console:
        """Access to the console for final output."""
        if self._live:
            return self._live.console
        return Console()

    def start(self, state: PanelState) -> None:
        """Initialize and start Live display.

        Args:
            state: Mutable state that PanelRenderable will read on each refresh.
        """
        self._state = state
        self._live = Live(
            PanelRenderable(state),
            refresh_per_second=4,
            transient=False,
        )
        self._live.console.clear()
        self._live.start()

    def stop(self, grace_period: float = 2.5) -> None:
        """Stop Live display with grace period for final logs.

        Args:
            grace_period: Seconds to wait before stopping, allowing final events to render.
        """
        def stop_later():
            if grace_period > 0:
                time.sleep(grace_period)

            if self._live:
                self._live.stop()
                self._live = None

        threading.Thread(target=stop_later, daemon=True).start()

    def print_final_status(
        self,
        has_error: bool,
        total_cost: float,
        elapsed: float,
        savings: float,
    ) -> None:
        """Print final status after pool stops."""
        console = self.console
        if has_error:
            console.print("[bold red]x[/bold red] Pool stopped with errors")
        else:
            final = Text()
            final.append("v ", style=STYLE_SUCCESS)
            final.append("Complete", style="bold green")
            if total_cost > 0:
                final.append(f" | Cost: ${total_cost:.4f}", style=STYLE_DIM)
            if elapsed > 0:
                final.append(f" | Duration: {_format_duration(elapsed)}", style=STYLE_DIM)
            if savings > 0:
                final.append(f" | Saved: ${savings:.4f}", style=STYLE_SPOT)
            console.print(final)
