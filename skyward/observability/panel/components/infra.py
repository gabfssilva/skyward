"""Infrastructure and cluster panels.

Components for the infra/cluster section of the panel.
"""

from __future__ import annotations

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text

from ..viewmodel import ClusterVM, InfraVM
from .metrics import MetricBadge


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

        table = Table.grid()
        table.add_column(justify="center")

        # Line 1: provider > region > instance_type
        line1 = Text()
        line1.append(vm.provider, style="yellow")
        line1.append(" > ", style="dim")
        line1.append(vm.region)
        line1.append(" > ", style="dim")
        line1.append(vm.instance_type, style="bold")
        table.add_row(line1)

        # Line 2: vCPU · memory · GPU
        line2 = Text()
        line2.append(f"{vm.vcpus} vCPU", style="white")
        line2.append(" · ", style="dim")
        line2.append(f"{vm.memory_gb} GB", style="white")
        if vm.gpu_info:
            line2.append(" · ", style="dim")
            line2.append(vm.gpu_info, style="magenta")
        table.add_row(line2)

        # Line 3: allocation @ rate
        line3 = Text()
        line3.append(vm.allocation, style="cyan")
        line3.append(" @ ", style="dim")
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
