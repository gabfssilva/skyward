"""Built-in callbacks for Skyward events.

This module provides standard callbacks for cost tracking
and terminal UI display.

Example:
    from skyward.callbacks import cost_tracker, panel
    from skyward.callback import compose, use_callback

    callback = compose(cost_tracker(), panel())

    with use_callback(callback):
        # events are tracked with Rich panel UI
        ...
"""

from skyward.callbacks.cost import cost_tracker
from skyward.callbacks.panel import panel

__all__ = [
    "cost_tracker",
    "panel",
]
