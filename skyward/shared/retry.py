"""Whether an attempt that did not answer should be made again.

The answer is a function of the user's, ``(reason, attempt) -> bool``, and it is
asked on whichever side holds the reason. A function that raised is a live
exception on the worker, in the venv that raised it, and that is where the
function is asked. An attempt that was lost — the process died under it, the
worker restarted, the machine went away — is a :class:`Lost` on the daemon, which
never unpickles a user's exception and never needs to.

Either way the daemon is the one that tries again: it creates the next execution
of the same task and places it, preferring a node other than the one that failed.
The function decides; it never runs anything.

The default is to try once more after a loss and never after an exception: a
function that raised is a fact about the function, and running it again on the
strength of nothing is the user's call to make.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from skyward.shared.observability import logger

log = logger.bind(component="retry")

type Cause = Literal["never_started", "process_died", "worker_restarted", "node_gone", "unknown"]
"""What the daemon can actually tell about a lost attempt.

``never_started`` — it was still on its way to the worker; nothing of the user's
ran. ``process_died`` — the worker saw the process running the function die.
``worker_restarted`` — the worker came back and had never heard of it. ``node_gone``
— the machine holding it is gone. ``unknown`` — the call failed in a way the daemon
cannot place. Nothing here says *why* a machine went away: a preemption and a
crash look the same from here, and the name does not pretend otherwise.
"""


@dataclass(frozen=True, slots=True)
class Lost:
    """An attempt that ended without the function answering.

    The function may have run, in part or in full — this is what a retry decision
    weighs against duplicate side effects.
    """

    cause: Cause
    node_id: str | None = None


type Reason = Exception | Lost
type Attempt = int
"""The attempt that just failed, counted from one."""
type Retry = Callable[[Reason, Attempt], bool]

CEILING = 10
"""Attempts, all told, past which nothing is asked. A decision that always says yes
is a task that never ends."""


def default(reason: Reason, attempt: Attempt) -> bool:
    """Once more after a loss; never after an exception."""
    return isinstance(reason, Lost) and attempt <= 1


def decide(retry: Retry | None, reason: Reason, attempt: Attempt) -> bool:
    """Ask, and take a decision that could not be made as a no.

    A decision function that raises has decided nothing, and the task is not
    retried on the strength of a bug in the thing that was meant to say whether
    to — it is logged, and the attempt stands as it ended.
    """
    if retry is None or attempt >= CEILING:
        return False
    try:
        return bool(retry(reason, attempt))
    except Exception:
        log.exception("the retry decision raised; the attempt stands as it ended")
        return False
