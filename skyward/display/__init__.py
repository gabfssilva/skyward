"""Display module for skyward terminal UI."""

from skyward.display.consumers.log import LogConsumer
from skyward.display.consumers.spinner import SpinnerConsumer

__all__ = [
    "LogConsumer",
    "SpinnerConsumer",
]
