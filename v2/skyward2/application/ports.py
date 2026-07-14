from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from skyward2.protocol.schemas import (
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
    """Closes the gap between intent and observation.

    Called with a key, never with a payload: that is what lets the emitter
    coalesce N wakeups for the same compute into one run. The reconciler reads
    the current state itself, so a lost event costs latency, not correctness —
    the periodic sweep finds the same work.
    """

    async def compute(self, compute_id: str) -> None: ...

    async def task(self, task_id: str) -> None: ...

    def stream(self, task_id: str) -> AsyncIterator[bytes]:
        """Dispatch a streaming task and forward its frames to whoever is reading.

        The one dispatch the reconciler does not do on its own: a stream has a far
        end, and only the caller consuming it can hold that.
        """
        ...

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
class Health(Protocol):
    async def live(self) -> bool: ...

    async def ready(self) -> bool: ...

    async def dependencies(self) -> dict[str, str]: ...
