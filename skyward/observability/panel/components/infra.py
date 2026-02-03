"""Infrastructure and cluster panels.

Components for the infra/cluster section of the panel.
"""

from __future__ import annotations

import time

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text

from ..viewmodel import ClusterVM, InfraVM
from .metrics import MetricBadge

# Spinner frames for pending values (dots style)
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def _spinner_frame() -> str:
    """Get current spinner frame based on time (animates at ~10fps)."""
    idx = int(time.monotonic() * 10) % len(SPINNER_FRAMES)
    return SPINNER_FRAMES[idx]


class InitializingPanel:
    """Shown while waiting for cluster data.

    Just displays "Initializing..." centered.
    """

    def render(self) -> RenderableType:
        return Text("Initializing...", style="dim")


class InfraPanel:
    """Left side: infrastructure info.

    Layout:
        aws > us-east-1 > p4d.24xlarge
        96 vCPU  1152 GB  8x A100-40GB (320 GB)
        12 nodes (10 spot + 2 od) @ $393/hr
    """

    def __init__(self, vm: InfraVM) -> None:
        self._vm = vm

    def render(self) -> RenderableType:
        vm = self._vm
        spinner = _spinner_frame()

        table = Table.grid()
        table.add_column(justify="center")

        # Line 1: provider > region > instance_type
        line1 = Text()
        line1.append(vm.provider, style="yellow")
        line1.append(" > ", style="dim")
        line1.append(vm.region)
        line1.append(" > ", style="dim")
        if vm.instance_type:
            line1.append(vm.instance_type, style="bold")
        else:
            line1.append(spinner, style="dim")
        table.add_row(line1)

        # Line 2: GPU · vCPU · memory (show spinner if no instance data yet)
        line2 = Text()
        if vm.vcpus > 0:
            if vm.gpu_info:
                line2.append(vm.gpu_info, style="magenta")
                line2.append(" · ", style="dim")
            line2.append(f"{vm.vcpus} vCPU", style="white")
            line2.append(" · ", style="dim")
            line2.append(f"{vm.memory_gb} GB", style="white")
        elif vm.gpu_info:
            # We have GPU info from spec but not instance details yet
            line2.append(vm.gpu_info, style="magenta")
            line2.append(f" {spinner}", style="dim")
        else:
            line2.append(f"{spinner} loading instance details...", style="dim")
        table.add_row(line2)

        # Line 3: allocation @ rate (show spinner if rate is zero)
        line3 = Text()
        line3.append(vm.allocation, style="cyan")
        line3.append(" @ ", style="dim")
        if vm.hourly_rate == "$0.00/hr":
            line3.append(f"{spinner}", style="dim")
        else:
            line3.append(vm.hourly_rate, style="green bold")
        table.add_row(line3)

        return table


class MetricsPanel:
    """Right side: cluster-wide average metrics.

    Layout:
        cluster avg
        cpu ▁▂▃▄▅▆▇█ 76%   gpu ▅▆▇█ 91%
        mem ▁▁▂▃▃▃▃▃ 43%   temp ▂▂▃▃ 67
    """

    def __init__(self, vm: ClusterVM) -> None:
        self._vm = vm

    def render(self) -> RenderableType:
        vm = self._vm

        metrics_panel = Table.grid(padding=(0, 2))
        metrics_panel.add_column(justify="center")

        metrics = Table.grid(padding=(0, 2))

        metrics.add_column(justify="center")
        metrics.add_column(justify="center")
        #
        # # Header row
        # table.add_row(Text("cluster avg", style="dim bold"))

        # Metrics rows
        metrics.add_row(
            MetricBadge(vm.cpu).render(),
            MetricBadge(vm.gpu).render(),
        )
        metrics.add_row(
            MetricBadge(vm.mem).render(),
            MetricBadge(vm.gpu_mem).render(),
        )

        metrics_panel.add_row(
            Text("cluster avg", style="dim bold")
        )

        metrics_panel.add_row(
            metrics
        )

        return metrics_panel


class ClusterPanel:
    """Full cluster section with infra (left) and metrics (right).

    Only rendered when we have actual cluster data.
    Uses Rich Table.grid for side-by-side layout.
    """

    def __init__(self, infra: InfraVM, cluster: ClusterVM) -> None:
        self._infra = infra
        self._cluster = cluster

    def render(self) -> RenderableType:
        table = Table.grid(padding=(0, 3))
        table.add_column(justify="center")
        table.add_column(justify="center")

        left = InfraPanel(self._infra).render()
        right = MetricsPanel(self._cluster).render()

        table.add_row(left, right)

        return table


def create_cluster_section(infra: InfraVM | None, cluster: ClusterVM) -> RenderableType:
    """Factory that returns the appropriate panel based on state.

    Returns:
        InitializingPanel if no infra data, ClusterPanel otherwise.
    """
    if infra is None:
        return InitializingPanel().render()
    return ClusterPanel(infra, cluster).render()
