"""Metric visualization components: Sparkline and MetricBadge."""

from __future__ import annotations

from rich.text import Text

from ..viewmodel import MetricVM

# Sparkline characters (8 levels from empty to full)
_SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


class Sparkline:
    """Renders metric history as unicode sparkline.

    Converts a sequence of percentage values (0-100) into a visual
    sparkline using unicode block characters.

    Example output: ▁▂▃▄▅▆▇█
    """

    def __init__(self, history: tuple[float, ...], width: int = 8, style: str = "green") -> None:
        self._history = history
        self._width = width
        self._style = style

    def render(self) -> Text:
        """Render sparkline to Rich Text."""
        if not self._history:
            return Text(_SPARKLINE_CHARS[0] * self._width, style=self._style)

        # Take last `width` values
        values = list(self._history)[-self._width :]

        # Pad with first char if not enough values
        while len(values) < self._width:
            values.insert(0, 0.0)

        # Convert each value to a sparkline character
        chars: list[str] = []
        for v in values:
            # Clamp to 0-100
            v = max(0.0, min(100.0, v))
            # Map to 0-7 index
            idx = int((v / 100.0) * 7)
            idx = min(idx, 7)  # Ensure we don't exceed index
            chars.append(_SPARKLINE_CHARS[idx])

        return Text("".join(chars), style=self._style)


class MetricBadge:
    """Renders a labeled metric with sparkline and current value.

    Example output: cpu ▁▂▃▄▅▆▇█ 78%
    """

    def __init__(self, vm: MetricVM) -> None:
        self._vm = vm

    def render(self) -> Text:
        """Render metric badge to Rich Text."""
        vm = self._vm
        result = Text()

        # Label (expected to be 4 chars, e.g. "cpu ", "mem ", "gpu ", "temp")
        result.append(vm.label, style="dim")
        result.append(" ")  # separator

        # Sparkline
        sparkline = Sparkline(vm.history, width=vm.spark_width, style=vm.style)
        result.append_text(sparkline.render())

        # Current value
        result.append(f" {vm.value:>2.0f}{vm.unit}", style=f"{vm.style} bold")

        return result


def temp_color(temp: float) -> str:
    """Return color based on temperature: green < 60, yellow 60-75, red > 75."""
    if temp < 60:
        return "green"
    if temp < 75:
        return "yellow"
    return "red"
