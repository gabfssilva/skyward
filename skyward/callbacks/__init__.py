"""Built-in callbacks for Skyward events.

This module provides standard callbacks for logging, cost tracking,
and terminal UI display.

Example:
    from skyward.callbacks import log, cost_tracker, spinner
    from skyward.callback import compose, use_callback

    callback = compose(cost_tracker(), log)

    with use_callback(callback):
        # events are logged and costs tracked
        ...
"""

from skyward.callbacks.cost import cost_tracker
from skyward.callbacks.log import log
from skyward.callbacks.spinner import spinner

__all__ = [
    "log",
    "cost_tracker",
    "spinner",
]
