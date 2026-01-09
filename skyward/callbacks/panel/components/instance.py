"""Instance row component.

Renders instance with configurable number of log lines.
"""

from __future__ import annotations

from rich.text import Text

from ..viewmodel import InstanceVM
from .metrics import MetricBadge


def _short_id(instance_id: str, length: int = 10) -> str:
    """Shorten instance ID for display."""
    return instance_id[:length] if len(instance_id) > length else instance_id


class InstanceRow:
    """Renders instance with header and optional log lines.

    log_lines=0: Just header
        i-abc12345 spot   cpu ▁▂▃▄ 78%  mem ▁▁▂▃ 45%  gpu ▅▆▇█ 92%  temp 67°

    log_lines=2: Header + 2 log lines (secondary nodes)
        i-abc12345 spot   cpu ▁▂▃▄ 78%  mem ▁▁▂▃ 45%  gpu ▅▆▇█ 92%  temp 67°
        │ log line 1
        └ log line 2

    log_lines=N: Header + N log lines (head node fills screen)
        i-def67890 spot   cpu ▁▁▂▃ 72%  mem ▁▁▂▃ 45%  gpu ▄▅▆▇ 89%  temp 65°
        │ log line 1
        │ log line 2
        │ ...
        └ log line N
    """

    def __init__(self, vm: InstanceVM, log_lines: int = 0) -> None:
        self._vm = vm
        self._log_lines = log_lines

    def render(self, width: int) -> list[Text]:
        """Render instance row(s)."""
        lines: list[Text] = []

        # Header line with all metrics
        header = self._render_header()
        lines.append(header)

        # Log lines with │ and └ borders - only render actual logs
        num_logs = len(self._vm.logs)
        lines_to_render = min(self._log_lines, num_logs) if self._log_lines > 0 else 0

        for i in range(lines_to_render):
            log_line = Text()
            border = "└ " if i == lines_to_render - 1 else "│ "
            log_line.append(border, style="dim")

            # Parse ANSI codes (Keras progress bars use them)
            log_text = self._vm.logs[i]
            max_log_len = width - 6
            if len(log_text) > max_log_len:
                log_text = log_text[: max_log_len - 3] + "..."
            # Use from_ansi to render ANSI escape codes
            log_line.append_text(Text.from_ansi(log_text))

            lines.append(log_line)

        return lines

    def _render_header(self) -> Text:
        """Render the header line with instance ID and metrics."""
        vm = self._vm
        line = Text()

        # Instance ID and market
        id_style = "cyan bold" if vm.market == "spot" else "yellow bold"
        market_label = "spot" if vm.market == "spot" else "od"

        line.append(_short_id(vm.instance_id), style=id_style)
        line.append(f" {market_label}", style="dim")
        line.append("   ")

        # All metrics with sparklines
        line.append_text(MetricBadge(vm.cpu).render())
        line.append("  ")
        line.append_text(MetricBadge(vm.mem).render())
        line.append("  ")
        line.append_text(MetricBadge(vm.gpu).render())
        line.append("  ")
        line.append_text(MetricBadge(vm.temp).render())

        return line
