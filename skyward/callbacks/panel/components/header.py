"""Header component for the panel.

Renders: ● s k y w a r d  ·  $0.42  ·  5:23
"""

from __future__ import annotations

from rich.text import Text

from ..viewmodel import HeaderVM


def _format_duration(seconds: float) -> str:
    """Format seconds as M:SS."""
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}:{secs:02d}"


class Header:
    """Renders the skyward header with status indicator, cost, and elapsed time.

    Example: ● s k y w a r d  ·  $0.42  ·  5:23

    The marker blinks when there's an active phase:
    - ● (cyan, blinking) - work in progress
    - ○ (green) - all phases completed
    - ● (dim) - idle
    """

    def __init__(self, vm: HeaderVM) -> None:
        self._vm = vm

    def render(self, width: int) -> Text:
        """Render header to Rich Text."""
        vm = self._vm
        text = Text()

        # Determine marker state
        has_active = any(s == "in_progress" for s in vm.phases.values())
        all_done = all(s == "completed" for s in vm.phases.values())

        if all_done:
            marker, marker_style = "○", "green bold"
        elif has_active:
            marker = "●" if vm.blink_on else "○"
            marker_style = "cyan bold"
        else:
            marker, marker_style = "●", "dim"

        # Compose header
        text.append(f"{marker} ", style=marker_style)
        text.append("s k y w a r d", style="bold")
        text.append(
            f"  ·  ${vm.cost:.2f}  ·  {_format_duration(vm.elapsed_seconds)}", style="dim"
        )

        return text
