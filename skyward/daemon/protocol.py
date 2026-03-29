"""Client-daemon protocol messages.

All messages are frozen dataclasses transported via cloudpickle
over a Unix domain socket. The type unions define the contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.api.views import PoolView

# -- Requests (client -> daemon) -------------------------------------------

@dataclass(frozen=True, slots=True)
class EnsurePool:
    """Provision pool if not running, or return existing handle."""
    name: str
    project_dir: str | None = None


@dataclass(frozen=True, slots=True)
class SubmitTask:
    """Execute a function on one node (round-robin)."""
    pool_name: str
    payload: bytes  # cloudpickle(PendingFunction)
    timeout: float = 300.0
    client_id: str = ""


@dataclass(frozen=True, slots=True)
class SubmitBroadcast:
    """Execute a function on ALL nodes."""
    pool_name: str
    payload: bytes
    timeout: float = 300.0
    client_id: str = ""


@dataclass(frozen=True, slots=True)
class GetNodeCount:
    """Query ready node count."""
    pool_name: str


@dataclass(frozen=True, slots=True)
class Disconnect:
    """Client disconnecting from a pool (don't kill it)."""
    pool_name: str
    client_id: str = ""


@dataclass(frozen=True, slots=True)
class ShutdownPool:
    """Explicitly tear down a pool."""
    pool_name: str


@dataclass(frozen=True, slots=True)
class Ping:
    """Health check."""


@dataclass(frozen=True, slots=True)
class GetPools:
    """List all running pools."""


@dataclass(frozen=True, slots=True)
class GetPoolView:
    """Get full view of a specific pool."""
    pool_name: str


@dataclass(frozen=True, slots=True)
class SubscribeEvents:
    """Subscribe to live event stream for a pool."""
    pool_name: str


# -- Responses (daemon -> client) ------------------------------------------

@dataclass(frozen=True, slots=True)
class PoolReady:
    """Pool is provisioned and ready for tasks."""
    pool_name: str
    node_count: int


@dataclass(frozen=True, slots=True)
class PoolFailed:
    """Pool provisioning failed."""
    pool_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class TaskSucceeded:
    """Task completed successfully."""
    payload: bytes  # cloudpickle(result)


@dataclass(frozen=True, slots=True)
class TaskFailed:
    """Task raised an exception."""
    error: str
    traceback: str


@dataclass(frozen=True, slots=True)
class BroadcastSucceeded:
    """Broadcast completed successfully."""
    payload: bytes  # cloudpickle(list[result])


@dataclass(frozen=True, slots=True)
class NodeCount:
    """Current ready node count."""
    ready: int


@dataclass(frozen=True, slots=True)
class Disconnected:
    """Acknowledged client disconnect."""


@dataclass(frozen=True, slots=True)
class PoolShutdown:
    """Pool has been torn down."""


@dataclass(frozen=True, slots=True)
class Pong:
    """Health check response."""


@dataclass(frozen=True, slots=True)
class DaemonError:
    """Generic error from daemon."""
    error: str
    traceback: str | None = None


@dataclass(frozen=True, slots=True)
class PoolSummary:
    """Lightweight pool info for list view."""
    name: str
    phase: str
    nodes_ready: int
    nodes_total: int
    tasks_done: int
    tasks_running: int
    started_at: float


@dataclass(frozen=True, slots=True)
class PoolList:
    """Response to GetPools."""
    pools: tuple[PoolSummary, ...]


@dataclass(frozen=True, slots=True)
class PoolViewResponse:
    """Response to GetPoolView."""
    view: PoolView


# -- Type unions -----------------------------------------------------------

type DaemonRequest = (
    EnsurePool | SubmitTask | SubmitBroadcast | GetNodeCount
    | Disconnect | ShutdownPool | Ping
    | GetPools | GetPoolView | SubscribeEvents
)

type DaemonResponse = (
    PoolReady | PoolFailed | TaskSucceeded | TaskFailed
    | BroadcastSucceeded | NodeCount | Disconnected | PoolShutdown
    | Pong | DaemonError
    | PoolList | PoolViewResponse
)
