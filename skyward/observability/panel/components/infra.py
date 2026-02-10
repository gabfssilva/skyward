"""Infrastructure and cluster panels.

Components for the header/footer of the panel.
"""

from __future__ import annotations

import time

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text

from ..viewmodel import ClusterVM, HeaderVM, InfraVM

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def _spinner_frame() -> str:
    idx = int(time.monotonic() * 10) % len(SPINNER_FRAMES)
    return SPINNER_FRAMES[idx]


def _format_duration(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}:{secs:02d}"


class HeaderBar:
    """Single-line header: branding + infra info.

    ● s k y w a r d   aws > us-east-1 > g5g.xlarge · 4 vCPU · 8 GB · 2 spot @ $0.27/hr
    """

    def __init__(self, header: HeaderVM, infra: InfraVM | None) -> None:
        self._header = header
        self._infra = infra

    def render(self, width: int) -> RenderableType:
        vm = self._header
        has_active = any(s == "in_progress" for s in vm.phases.values())
        all_done = all(s == "completed" for s in vm.phases.values())

        match (all_done, has_active):
            case (True, _):
                marker, marker_style = "", "green bold"
            case (_, True):
                marker = "" if vm.blink_on else ""
                marker_style = "cyan bold"
            case _:
                marker, marker_style = "", "dim"

        # Left: branding
        left = Text()
        left.append(f"{marker} ", style=marker_style)
        left.append("s k y w a r d", style="bold")

        infra = self._infra

        # Center: provider > region > instance · specs
        center = Text()
        if infra is not None:
            spinner = _spinner_frame()
            center.append(infra.provider, style="yellow")
            center.append(" > ", style="dim")
            center.append(infra.region)
            center.append(" > ", style="dim")
            center.append(infra.instance_type or spinner, style="bold" if infra.instance_type else "dim")

            if infra.vcpus > 0:
                center.append(" · ", style="dim")
                if infra.gpu_info:
                    center.append(infra.gpu_info, style="magenta")
                    center.append(" · ", style="dim")
                center.append(f"{infra.vcpus} vCPU", style="white")
                center.append(" · ", style="dim")
                center.append(f"{infra.memory_gb} GB", style="white")
            elif infra.gpu_info:
                center.append(" · ", style="dim")
                center.append(infra.gpu_info, style="magenta")

        # Right: allocation @ rate
        right = Text()
        if infra is not None:
            spinner = _spinner_frame()
            right.append(infra.allocation, style="cyan")
            right.append(" @ ", style="dim")
            if infra.hourly_rate == "$0.00/hr":
                right.append(spinner, style="dim")
            else:
                right.append(infra.hourly_rate, style="green bold")

        table = Table(box=None, show_header=False, padding=0, expand=True)
        table.add_column(justify="left")
        table.add_column(justify="center", ratio=1)
        table.add_column(justify="right")
        table.add_row(left, center, right)

        return table


class FooterBar:
    """Single-line footer: cluster metrics (left) + cost/elapsed (right).

    cpu ▂▂ 21%  gpu ▁▁ 0%  mem ▁▁ 10%  vram ▁▁ 0%      estimated total: ~$0.42  elapsed: 5:23
    """

    def __init__(self, cluster: ClusterVM, header: HeaderVM) -> None:
        self._cluster = cluster
        self._header = header

    def render(self) -> Text:
        vm = self._header
        cluster = self._cluster
        text = Text()

        text.append("estimated total: ", style="dim")
        text.append(f"~${vm.cost:.2f}", style="green bold")
        text.append(" · ", style="dim")
        text.append("elapsed: ", style="dim")
        text.append(_format_duration(vm.elapsed_seconds), style="white")
        text.append(" · ", style="dim")
        text.append("averages: ", style="dim")

        for i, metric in enumerate((cluster.cpu, cluster.gpu, cluster.mem, cluster.gpu_mem)):
            if i > 0:
                text.append(", ", style="dim")
            text.append(f"{metric.label} ", style="dim")
            text.append(f"{metric.value:.0f}{metric.unit}", style=f"{metric.style} bold")

        return text


def create_header(header: HeaderVM, infra: InfraVM | None, width: int) -> RenderableType:
    return HeaderBar(header, infra).render(width)


def create_footer(cluster: ClusterVM, header: HeaderVM) -> Text:
    return FooterBar(cluster, header).render()
