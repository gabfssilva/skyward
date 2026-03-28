"""Daemon mode -- persistent background pools with TTL and crash recovery.

Not exported at ``skyward`` top level. Used internally by
``ComputePool.Named()`` when config has ``daemon = true``.
"""

from .pool import DaemonPool
from .spawn import ensure_daemon, is_daemon_running

__all__ = [
    "DaemonPool",
    "ensure_daemon",
    "is_daemon_running",
]
