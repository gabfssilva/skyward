"""Immutable ViewModels for panel rendering.

These frozen dataclasses represent snapshots of state ready for rendering.
They are produced by PanelState.to_view_model() and consumed by Components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# =============================================================================
# Primitive ViewModels
# =============================================================================


@dataclass(frozen=True, slots=True)
class MetricVM:
    """ViewModel for a single metric with sparkline history."""

    label: str  # "cpu", "mem", "gpu", "temp"
    value: float  # Current value (0-100 for %, degrees for temp)
    history: tuple[float, ...]  # History for sparkline (0-100)
    unit: str = "%"  # "%" or "°"
    style: str = "green"  # Rich style name
    spark_width: int = 8  # Sparkline width in chars


@dataclass(frozen=True, slots=True)
class HeaderVM:
    """ViewModel for the header line."""

    cost: float  # Total accumulated cost
    elapsed_seconds: float  # Time since pool started
    phases: dict[str, Literal["pending", "in_progress", "completed"]]
    blink_on: bool = True  # For blinking indicator


@dataclass(frozen=True, slots=True)
class InfraVM:
    """ViewModel for infrastructure info."""

    provider: str  # "aws", "vastai", etc.
    region: str  # "us-east-1"
    instance_type: str  # "p4d.24xlarge"
    vcpus: int  # 96
    memory_gb: int  # 1152
    gpu_info: str | None  # "8× A100-40GB (320 GB)" or None
    allocation: str  # "12 nodes (10 spot + 2 od)"
    hourly_rate: str  # "$393/hr"


@dataclass(frozen=True, slots=True)
class ClusterVM:
    """ViewModel for cluster-wide average metrics."""

    cpu: MetricVM
    mem: MetricVM
    gpu: MetricVM
    gpu_mem: MetricVM  # GPU memory utilization
    temp: MetricVM


@dataclass(frozen=True, slots=True)
class InstanceVM:
    """ViewModel for a single instance."""

    instance_id: str  # Full instance ID
    short_id: str  # Truncated for display
    market: Literal["spot", "on-demand"]
    status: Literal["bootstrapping", "ready", "done"]  # Instance lifecycle status
    cpu: MetricVM
    mem: MetricVM
    gpu: MetricVM
    gpu_mem: MetricVM  # GPU memory utilization
    temp: MetricVM
    logs: tuple[str, ...]  # Last 5 log lines
    is_active: bool = False  # Whether this is the expanded instance
    blink_on: bool = True  # For blinking status indicator


# =============================================================================
# Root ViewModel
# =============================================================================


@dataclass(frozen=True, slots=True)
class PanelViewModel:
    """Complete snapshot for rendering the entire panel.

    This is the root ViewModel that contains all data needed to render
    the panel. It is produced by PanelState.to_view_model() and consumed
    by PanelLayout to orchestrate all component rendering.
    """

    header: HeaderVM
    infra: InfraVM | None
    cluster: ClusterVM
    instances: tuple[InstanceVM, ...]
    active_instance_id: str | None
    terminal_width: int
    terminal_height: int
