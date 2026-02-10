"""Panel UI module for v2 event system.

Provides a Rich terminal dashboard for cluster monitoring.
The panel subscribes to v2 events and displays real-time
metrics, logs, and cost tracking.

Usage:
    # Add PanelModule to your injector modules
    from skyward.observability.panel import PanelModule

    modules = [
        SkywardModule(),
        PoolConfigModule(spec=spec, provider_config=provider),
        PanelModule(),  # Enables the Rich panel UI
    ]
"""

from .renderer import PanelRenderer
from .state import InfraState, InstanceState, MetricsState, PanelState

__all__ = [
    "PanelRenderer",
    "PanelState",
    "InstanceState",
    "MetricsState",
    "InfraState",
]
