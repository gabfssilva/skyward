from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol, runtime_checkable

from skyward.protocol.schemas import (
    Compute,
    ComputeCreate,
    ComputeSpecPatch,
    ComputeState,
    Execution,
    ExecutionCreate,
    Function,
    Generation,
    GenerationCreate,
    Lease,
    LeaseClaim,
    Node,
    NodeState,
    Offer,
    Page,
    Provider,
    ProviderCreate,
    Task,
    TaskCreate,
    TaskState,
)

type Route = Literal["round_robin"]
"""How a forwarded connection picks among the ready nodes."""


@runtime_checkable
class Computes(Protocol):
    async def create(self, body: ComputeCreate, idempotency_key: str) -> tuple[Compute, bool]:
        """Returns the compute and whether it was newly created (False = idempotent replay)."""
        ...

    async def get(self, ref: str) -> Compute: ...

    async def list(self, cursor: str | None, limit: int, state: ComputeState | None, owned: bool | None) -> Page[Compute]: ...

    async def patch(self, ref: str, body: ComputeSpecPatch, expected_revision: int) -> Compute: ...

    async def delete(self, ref: str, expected_revision: int, idempotency_key: str) -> Compute: ...

    async def claim_lease(self, ref: str, claim: LeaseClaim) -> Lease: ...

    async def release_lease(self, ref: str) -> None: ...


@runtime_checkable
class Generations(Protocol):
    async def list(self, compute: str) -> Page[Generation]: ...

    async def get(self, compute: str, number: int) -> Generation: ...

    async def create(self, compute: str, body: GenerationCreate, expected_revision: int, idempotency_key: str) -> Generation:
        """Applies pending drift, or rolls back to `body.source`. Quiesce, drain, replace."""
        ...


@runtime_checkable
class Nodes(Protocol):
    async def list(self, compute: str, include_terminal: bool, generation: int | None) -> Page[Node]: ...

    async def get(self, compute: str, node_id: str) -> Node: ...

    async def drain(self, compute: str, node_id: str, idempotency_key: str) -> Node: ...


@runtime_checkable
class Functions(Protocol):
    async def exists(self, sha256: str) -> bool: ...

    async def get(self, sha256: str) -> Function: ...

    async def list(self, cursor: str | None, limit: int) -> Page[Function]: ...

    async def register(self, sha256: str, blob: bytes, name: str | None) -> tuple[Function, bool]:
        """Returns the function and whether it was newly registered."""
        ...


@runtime_checkable
class Blobs(Protocol):
    async def exists(self, sha256: str) -> bool: ...

    async def put(self, sha256: str, blob: bytes) -> bool:
        """Returns whether the blob was newly written."""
        ...

    async def get(self, sha256: str) -> bytes: ...


@runtime_checkable
class Tasks(Protocol):
    async def submit(self, body: TaskCreate, idempotency_key: str) -> tuple[Task, bool]: ...

    async def get(self, task_id: str) -> Task: ...

    async def list(self, cursor: str | None, limit: int, compute: str | None, state: TaskState | None, correlation_id: str | None) -> Page[Task]: ...

    async def cancel(self, task_id: str, idempotency_key: str) -> Task: ...

    async def result(self, task_id: str, wait_seconds: int) -> bytes | None:
        """None means no terminal outcome yet. Raises on non-success terminal outcomes."""
        ...

    async def expire(self) -> tuple[str, ...]:
        """Time out the tasks past their deadline. Returns the ones it timed out."""
        ...


@runtime_checkable
class Executions(Protocol):
    async def list(self, task_id: str) -> Page[Execution]: ...

    async def get(self, task_id: str, ordinal: int) -> Execution: ...

    async def create(self, task_id: str, body: ExecutionCreate, idempotency_key: str) -> Task:
        """Retry: a new physical attempt of the same task. Never a new task."""
        ...


@runtime_checkable
class Events(Protocol):
    def stream(
        self,
        last_event_id: str | None,
        compute: str | None,
        task: str | None,
        types: tuple[str, ...] | None,
    ) -> AsyncIterator[tuple[int, str, bytes]]:
        """Yields (sequence, event_type, payload) starting after last_event_id."""
        ...


@runtime_checkable
class Providers(Protocol):
    async def create(self, body: ProviderCreate) -> Provider: ...

    async def get(self, ref: str) -> Provider: ...

    async def list(self) -> Page[Provider]: ...

    async def delete(self, ref: str) -> None: ...


@runtime_checkable
class Offers(Protocol):
    async def list(
        self,
        provider: str | None,
        kind: str | None,
        accelerator: str | None,
        min_count: int | None,
        min_vram: float | None,
        max_price: float | None,
        refresh: bool,
    ) -> Page[Offer]:
        """Serve from cache, refreshing whatever the provider's TTL says is stale.

        A refresh that fails leaves the stale rows in place and records the error
        on the provider: a provider that is down should degrade the answer, not
        erase the catalog.
        """
        ...


@runtime_checkable
class Reconciler(Protocol):
    """Decides how many machines a compute should have, and writes that down.

    Called with a key, never with a payload: that is what lets the emitter coalesce
    N wakeups for the same compute into one pass. It reads the current state itself,
    so a lost event costs latency, not correctness — the tick finds the same work.

    It writes rows and nothing else. Buying the machines, logging into them and
    placing work on them are three other things, and they react to what it wrote.
    """

    async def compute(self, compute_id: str) -> None: ...

    async def observed(self, compute_id: str, node_id: str, state: NodeState, error: str) -> None:
        """What a node's own lifecycle reported about it.

        The one thing the reconciler is told rather than reads, because it is the
        one thing no query can answer: whether the SSH connection this process is
        holding got as far as a running worker.
        """
        ...

    async def unsettled(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Computes and tasks whose intent has not been realized yet."""
        ...


@runtime_checkable
class Dispatcher(Protocol):
    """Puts one written-down attempt on one machine, and brings its answer back."""

    async def task(self, task_id: str) -> None: ...

    async def resume(self, compute_id: str) -> None:
        """Offer the queue a slot that just came free, or a machine that just arrived."""
        ...

    def stream(self, task_id: str) -> AsyncIterator[bytes]:
        """Dispatch a streaming task and forward its frames to whoever is reading.

        The one dispatch nothing does on its own: a stream has a far end, and only
        the caller consuming it can hold that.
        """
        ...


@runtime_checkable
class Forwarder(Protocol):
    """Bridges one local TCP connection to a node port, as two half-duplex streams.

    A forwarded connection is two requests — the caller's bytes going up, the
    node's coming back down — because nothing in HTTP/1.1 carries both on one. The
    id the caller mints ties them together: :meth:`up` opens the channel to a ready
    node and :meth:`down` rides the same one back. Both take that id and nothing
    about a node, because which node answered is the forwarder's to decide.
    """

    async def up(self, compute_id: str, cid: str, remote_port: int, route: Route, chunks: AsyncIterator[bytes]) -> None:
        """Open a channel to a ready node and pump the caller's bytes into it."""
        ...

    def down(self, cid: str) -> AsyncIterator[bytes]:
        """The node's bytes back to the caller, for the ``up`` that opened this id."""
        ...


@runtime_checkable
class Health(Protocol):
    async def live(self) -> bool: ...

    async def ready(self) -> bool: ...

    async def dependencies(self) -> dict[str, str]: ...
