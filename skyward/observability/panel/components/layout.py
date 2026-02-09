"""Panel layout component.

Assembles all components into the final panel Table.
"""

from __future__ import annotations

from rich.align import Align
from rich.table import Table
from rich.text import Text

from ..viewmodel import PanelViewModel
from .header import Header
from .infra import create_cluster_section
from .instance import InstanceRow

# Fixed layout constants
HEADER_LINES = 1
INFRA_LINES = 3
SECONDARY_LOG_LINES = 2  # Each secondary node gets 2 log lines


def _build_separator(width: int) -> Text:
    sep_width = min(80, width)
    return Text("─" * sep_width, style="dim")


class PanelLayout:
    """Assembles all components into the final panel.

    Layout structure:
    1. Header (centered) - 1 line
    2. Infra + Cluster avg (centered) - 3 lines
    3. Secondary instances (header + 2 log lines each)
    4. Head instance (header + remaining lines fill screen)
    """

    def __init__(self, vm: PanelViewModel) -> None:
        self._vm = vm

    def render(self) -> Table:
        """Render complete panel as a Rich Table."""
        vm = self._vm
        width = vm.terminal_width
        height = vm.terminal_height

        # Count secondary nodes
        secondary_nodes = [inst for inst in vm.instances if not inst.is_active]
        head_node = next((inst for inst in vm.instances if inst.is_active), None)

        # Calculate available lines for head
        # Layout: header (1) + infra (3) + separator (1) + secondary (3 each) + head header (1) + head logs
        separator_line = 1
        head_header = 1 if head_node else 0
        fixed_overhead = HEADER_LINES + INFRA_LINES + separator_line + head_header
        secondary_total = len(secondary_nodes) * (1 + SECONDARY_LOG_LINES)

        # Head gets remaining space (minimum 5 log lines)
        head_log_lines = max(5, height - fixed_overhead - secondary_total)

        # Main container table
        table = Table(box=None, show_header=False, padding=0, expand=False)
        table.add_column(width=width)

        # 1. Header (centered)
        header = Header(vm.header).render(width)
        table.add_row(Align.center(header))

        # 2. Infra + Cluster section (centered)
        cluster_section = create_cluster_section(vm.infra, vm.cluster)
        table.add_row(Align.center(cluster_section))

        # 3. Separator line
        separator = _build_separator(width)
        table.add_row(Align.center(separator))

        # 4. Secondary instances (header + 2 log lines each)
        for inst in secondary_nodes:
            for line in InstanceRow(inst, log_lines=SECONDARY_LOG_LINES).render(width):
                table.add_row(line)

        # 5. Head instance (fills remaining screen)
        if head_node:
            for line in InstanceRow(head_node, log_lines=head_log_lines).render(width):
                table.add_row(line)

        return table
