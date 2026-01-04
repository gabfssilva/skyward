"""Built-in callbacks for Skyward events.

This module provides standard callbacks for cost tracking
and terminal UI display.

Example:
    from skyward.callbacks import cost_tracker, spinner
    from skyward.callback import compose, use_callback

    callback = compose(cost_tracker(), spinner())

    with use_callback(callback):
        # events are tracked with spinner UI
        ...
"""

from skyward.callbacks.cost import cost_tracker
from skyward.callbacks.spinner import spinner

__all__ = [
    "cost_tracker",
    "spinner",
]
