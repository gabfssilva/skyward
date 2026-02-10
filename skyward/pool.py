"""Pool state definitions.

The pool lifecycle is now managed by PoolActor (skyward.actors.pool).
This module retains PoolState for backwards compatibility.
"""


class PoolState:
    """Pool lifecycle states."""

    INIT = "init"
    REQUESTING = "requesting"
    PROVISIONING = "provisioning"
    READY = "ready"
    SHUTTING_DOWN = "shutting_down"
    DESTROYED = "destroyed"


__all__ = [
    "PoolState",
]
