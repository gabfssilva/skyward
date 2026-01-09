"""Rich-based panel callback with MVC architecture.

This module provides a beautiful terminal UI for monitoring Skyward pools.
The architecture follows MVVM + Component System pattern:

- **State**: Mutable state updated by events (PanelState)
- **ViewModel**: Immutable snapshots for rendering (PanelViewModel, etc.)
- **Components**: Reusable Rich renderers (Header, InfraSection, InstanceRow)
- **Controller**: Event dispatcher (PanelController)
- **Renderer**: Rich Live display manager (PanelRenderer)

Example:
    >>> from skyward.callbacks import panel
    >>> from skyward.core.callback import use_callback
    >>>
    >>> with use_callback(panel()):
    ...     # Events will be rendered to the terminal
    ...     pass
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .controller import PanelController
from .renderer import PanelRenderer

if TYPE_CHECKING:
    from skyward.core.callback import Callback

__all__ = ["panel"]


def panel() -> Callback:
    """Create a Rich panel callback for beautiful terminal UI.

    The panel shows a hybrid layout with:
    - Header: blinking indicator + cost + elapsed time
    - Infrastructure info (left) + cluster avg metrics (right)
    - Expanded instance with 5 log lines (most recently active)
    - Compact instances with 1 log line each

    Cost tracking is built-in using prices from InstanceSpec events.
    No need to compose with cost_tracker().

    Returns:
        A callback that renders an animated Rich panel.

    Example:
        >>> with use_callback(panel()):
        ...     emit(PoolStarted())
        ...     # ... panel updates ...
        ...     emit(PoolStopping())
    """
    renderer = PanelRenderer()
    controller = PanelController(renderer)
    return controller.handle
