"""Panel layout component.

Assembles all components into the final panel Table.
"""

from __future__ import annotations

from rich.align import Align
from rich.table import Table
from rich.text import Text

from ..viewmodel import PanelViewModel
from .infra import create_footer, create_header
from .instance import InstanceRow

HEADER_LINES = 1
SEPARATOR_LINES = 2
FOOTER_LINES = 1
SECONDARY_LOG_LINES = 2


def _build_separator(width: int) -> Text:
    return Text("─" * width, style="dim")


class PanelLayout:
    """Assembles all components into the final panel.

    Layout structure:
    1. Header (centered) - 1 line
    2. Separator - 1 line
    3. Secondary instances (header + 2 log lines each)
    4. Head instance (fills remaining screen)
    5. Padding (pushes footer to bottom)
    6. Separator - 1 line
    7. Footer (centered) - 1 line
    """

    def __init__(self, vm: PanelViewModel) -> None:
        self._vm = vm

    def render(self) -> Table:
        vm = self._vm
        width = vm.terminal_width
        height = vm.terminal_height

        secondary_nodes = [inst for inst in vm.instances if not inst.is_active]
        head_node = next((inst for inst in vm.instances if inst.is_active), None)

        lines_used = HEADER_LINES + SEPARATOR_LINES + FOOTER_LINES

        secondary_total = len(secondary_nodes) * (1 + SECONDARY_LOG_LINES)
        lines_used += secondary_total

        head_log_lines = max(5, height - lines_used - (1 if head_node else 0))

        table = Table(box=None, show_header=False, padding=0, expand=False)
        table.add_column(width=width)

        # 1. Header
        header = create_header(vm.header, vm.infra, width)
        table.add_row(header)

        # 2. Separator
        table.add_row(Align.center(_build_separator(width)))

        # 3. Secondary instances
        content_lines = 0
        for inst in secondary_nodes:
            for line in InstanceRow(inst, log_lines=SECONDARY_LOG_LINES).render(width):
                table.add_row(line)
                content_lines += 1

        # 4. Head instance
        if head_node:
            for line in InstanceRow(head_node, log_lines=head_log_lines).render(width):
                table.add_row(line)
                content_lines += 1

        # 5. Pad to push footer to bottom
        expected_content = secondary_total + (1 + head_log_lines if head_node else head_log_lines)
        padding = max(0, expected_content - content_lines)
        for _ in range(padding):
            table.add_row(Text(""))

        # 6. Separator
        table.add_row(Align.center(_build_separator(width)))

        # 7. Footer
        footer = create_footer(vm.cluster, vm.header)
        table.add_row(Align.center(footer))

        return table
