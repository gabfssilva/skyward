"""What skyward says to the person at the terminal, in the shape of a log line.

The SDK and the CLI say a handful of things on stderr: a daemon started, a daemon
on another version, a compute left up. The terminal they say it on is the one the
user's own code prints to, so each goes out with a time and a level, laid out as
the daemon's log lays out its records, and reads as skyward's rather than as the
output of whatever skyward is running. On a terminal the level is coloured and the
rest of the frame is dimmed; written anywhere else the line is plain text.

Not a sink of :data:`~skyward.shared.observability.logger`: that one carries
skyward's own diagnostics, which a user's process has no sink for and does not
want on its terminal. These are few, and meant to be read.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from datetime import datetime

from skyward.shared.observability.logfile import Severity

COLOURS: Mapping[Severity, str] = {"DEBUG": "2", "INFO": "36", "WARNING": "33", "ERROR": "1;31"}
"""The SGR code each level is painted in."""


def notice(severity: Severity, message: str) -> None:
    """Write ``message`` to stderr as one line: when, how serious, and what.

    ``sys.stderr`` is read at every call rather than held, because a notebook
    rebinds it and a line written to the stream it replaced reaches nobody.
    """
    out = sys.stderr
    now = datetime.now()
    when = f"{now:%Y-%m-%d %H:%M:%S}.{now.microsecond // 1000:03d}"
    level = f"{severity:<8}"
    if out.isatty():
        print(f"\033[2m{when} |\033[0m \033[{COLOURS[severity]}m{level}\033[0m \033[2m|\033[0m {message}", file=out, flush=True)
    else:
        print(f"{when} | {level} | {message}", file=out, flush=True)


__all__ = ["notice"]
