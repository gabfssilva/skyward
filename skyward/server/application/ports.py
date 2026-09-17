from __future__ import annotations

from collections.abc import AsyncIterator, Collection, Iterable, Sequence
from datetime import datetime
from typing import Literal, NamedTuple, Protocol, runtime_checkable

from skyward.server.application.ssh import Pty, Result
from skyward.shared.events import LogEntry
from skyward.shared.schemas import (
    Aggregate,
    Compute,
    ComputeCreate,
    ComputeSpecPatch,
    ComputeState,
    DeletionCause,
    DependencyState,
    Execution,
    ExecutionCreate,
    Function,
    Generation,
    GenerationCreate,
    Lease,
    LeaseClaim,
    MetricHistory,
    MetricSample,
    Node,
    NodeState,
    Offer,
    OfferSort,
    Page,
    Provider,
    ProviderCreate,
    Task,
    TaskCreate,
    TaskOrder,
    TaskState,
)

type Route = Literal["round_robin"]
"""How a forwarded connection picks among the ready nodes."""

type Target = Literal["all"] | int
"""Which nodes an operation reaches: every ready one, or the one holding this rank.

There is no head to name. A rank is only an index into a symmetric world, so rank
zero is where a caller with no opinion lands and means nothing more than that.
"""


class Held(NamedTuple):
    """One attempt a machine is holding: placed and not yet answered for, or answered for and not yet let go of."""

    node: str
    task: str
    ordinal: int
    function: str
    started_at: datetime | None


class Pace(NamedTuple):
    """How many tasks finished over a while, and how long they took on average from their first attempt starting."""

    finished: int
    mean_seconds: float | None


@runtime_checkable
class Computes(Protocol):
    async def create(self, body: ComputeCreate, idempotency_key: str) -> tuple[Compute, bool]:
        """Returns the compute and whether it was newly created (False = idempotent replay)."""
        ...

    async def get(self, ref: str) -> Compute: ...

    async def identify(self, ref: str) -> str:
        """The id behind a name or an id, which is what every other store is keyed by."""
        ...

    async def named(self, ids: Collection[str]) -> dict[str, str | None]:
        """The name each of these computes goes by, or None for one that has none."""
        ...

    async def list(
        self,
        cursor: str | None,
        limit: int,
        state: ComputeState | None,
        owned: bool | None,
        live: bool | None,
        cause: DeletionCause | None = None,
    ) -> Page[Compute]: ...

    async def patch(self, ref: str, body: ComputeSpecPatch, expected_revision: int) -> Compute: ...

    async def delete(self, ref: str, expected_revision: int, idempotency_key: str) -> Compute: ...

    async def claim_lease(self, ref: str, claim: LeaseClaim) -> Lease: ...

    async def release_lease(self, ref: str) -> None: ...


@runtime_checkable
class Generations(Protocol):
    async def list(self, compute: str) -> Page[Generation]: ...

    async def get(self, compute: str, number: int) -> Generation: ...

    async def create(self, compute: str, body: GenerationCreate, expected_revision: int, idempotency_key: str) -> Generation:
        """Makes generation `body.source` current again, as a new generation."""
        ...


@runtime_checkable
class Nodes(Protocol):
    async def of(self, compute: str) -> tuple[Node, ...]:
        """Every node the compute ever had, replaced ones included."""
        ...

    async def drain(self, compute: str, node_id: str, idempotency_key: str) -> Node: ...


@runtime_checkable
class Functions(Protocol):
    async def exists(self, sha256: str) -> bool: ...

    async def get(self, sha256: str) -> Function: ...

    async def list(self, cursor: str | None, limit: int, latest: bool = False, lineage: str | None = None) -> Page[Function]: ...

    async def register(self, sha256: str, blob: bytes, name: str | None, source: str | None = None) -> tuple[Function, bool]:
        """Returns the function and whether it was newly registered.

        ``source`` is the text it was written as, for one that was written rather
        than pickled from a callable that was already running somewhere.
        """
        ...

    async def excerpt(self, sha256: str, text: str) -> Function:
        """Keep the text the SDK read for a function it already uploaded."""
        ...


@runtime_checkable
class Blobs(Protocol):
    async def exists(self, sha256: str) -> bool: ...

    async def put(self, sha256: str, blob: bytes) -> bool:
        """Returns whether the blob was newly written."""
        ...

    async def get(self, sha256: str) -> bytes: ...

    async def rechunk(self) -> None:
        """Convert content stored before chunking; returns when there is none left."""
        ...


@runtime_checkable
class Tasks(Protocol):
    async def submit(self, body: TaskCreate, idempotency_key: str) -> tuple[Task, bool]: ...

    async def get(self, task_id: str) -> Task: ...

    async def list(
        self,
        cursor: str | None,
        limit: int,
        compute: str | None = None,
        states: Sequence[TaskState] = (),
        correlation_id: str | None = None,
        function: str | None = None,
        order: TaskOrder = "submitted",
    ) -> Page[Task]: ...

    async def cancel(self, task_id: str, idempotency_key: str) -> Task: ...

    async def result(self, task_id: str, wait_seconds: int) -> bytes | None:
        """None means no terminal outcome yet. Raises on non-success terminal outcomes."""
        ...

    def close(self) -> None:
        """End every wait in :meth:`result` now: the daemon is going away."""
        ...

    async def held(self, compute: str) -> tuple[Held, ...]:
        """The attempts this compute's machines are holding, the earliest started first."""
        ...

    async def pace(self, compute: str, since: datetime) -> Pace:
        """How many of this compute's tasks finished since ``since``, and how long they took."""
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
    ) -> AsyncIterator[tuple[tuple[int, str, bytes], ...]]:
        """Yields (sequence, event_type, payload) after last_event_id, in runs of what was ready together."""
        ...

    async def log(
        self,
        cursor: str | None,
        limit: int,
        *,
        compute: str | None = None,
        task: str | None = None,
        node: str | None = None,
        types: tuple[str, ...] | None = None,
        contains: tuple[str, ...] | None = None,
    ) -> Page[LogEntry]:
        """The recorded events, newest first; ``cursor`` is the sequence the previous page ended on.

        ``contains`` matches the line a node printed, any one of the strings, so a
        search is a query over the whole log rather than a filter over the page in hand.
        """
        ...


@runtime_checkable
class Metrics(Protocol):
    def add(self, compute: str, samples: Iterable[MetricSample]) -> None:
        """Hold samples for the next :meth:`flush`."""
        ...

    async def flush(self) -> None:
        """Write every sample held; one already recorded is kept once."""
        ...

    async def compact(self, now: int | None = None) -> None:
        """Fold every window closed for longer than the grace into its node's chunk."""
        ...

    async def series(
        self,
        compute: str,
        since: int,
        until: int | None = None,
        step: int | None = None,
        aggregate: Aggregate = "avg",
        nodes: Sequence[str] | None = None,
        names: Sequence[str] | None = None,
    ) -> MetricHistory:
        """Samples measured from ``since`` up to ``until`` (open when not given), or one value per ``step`` milliseconds."""
        ...

    async def after(self, compute: str, cursor: str, nodes: Sequence[str] | None = None, names: Sequence[str] | None = None) -> MetricHistory:
        """What was recorded after ``cursor``, whenever it was measured; ``reset`` when some of it was compacted first."""
        ...

    async def latest(self, compute: str, nodes: Sequence[str] | None = None, names: Sequence[str] | None = None) -> tuple[MetricSample, ...]:
        """The newest sample of each node and name."""
        ...


@runtime_checkable
class Providers(Protocol):
    async def create(self, body: ProviderCreate) -> Provider: ...

    async def update(self, ref: str, body: ProviderCreate) -> Provider: ...

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
        *,
        spot: bool | None = None,
        sort: OfferSort = "price",
        limit: int | None = None,
    ) -> Page[Offer]:
        """Serve from cache, refreshing whatever the provider's TTL says is stale.

        A refresh that fails leaves the stale rows in place and records the error
        on the provider: a provider that is down should degrade the answer, not
        erase the catalog.

        A reader after the cheapest few asks for a ``limit`` — nobody reads the four
        thousandth cheapest machine — and ``total`` says how many the filters matched.
        Unset is the whole catalog, which is what the planner and the CLI want.
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

    async def observed(self, compute_id: str, node_id: str, state: NodeState, error: str | None) -> None:
        """What a node's own lifecycle reported about it.

        The one thing the reconciler is told rather than reads, because it is the
        one thing no query can answer: whether the SSH connection this process is
        holding got as far as a running worker.
        """
        ...

    async def unsettled(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Computes and tasks whose intent has not been realized yet.

        A deleted compute is among the computes while a task of it has an attempt without
        a verdict, and that task is not among the tasks: it is answered for when the
        compute is, not offered to a dispatcher that has nothing left to place it on.
        """
        ...


@runtime_checkable
class Dispatcher(Protocol):
    """Puts one written-down attempt on one machine, and brings its answer back."""

    async def task(self, task_id: str) -> None: ...

    async def resume(self, compute_id: str) -> None:
        """Offer the queue a slot that just came free, or a machine that just arrived."""
        ...

    async def deleted(self, compute_id: str) -> None:
        """Answer for every attempt a deleted compute still owed, with the verdict that is left."""
        ...

    async def expire(self) -> None:
        """Time out the attempts past their deadline, and ask the machines still running one to stop."""
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

    async def down(self, cid: str) -> AsyncIterator[bytes]:
        """The node's bytes back to the caller, for the ``up`` that opened this id.

        Waits for the channel and then hands back the stream, so a channel that
        cannot be opened is raised here rather than part-way through a body.
        """
        ...


@runtime_checkable
class Files(Protocol):
    """One compute's filesystem and shell, over the links the daemon already holds.

    Deliberately not the byte proxy a :class:`Forwarder` is. That endpoint carries
    raw bytes to a port on the node, so running SFTP across it would be SSH inside
    SSH, and the far end would need the compute's private key — which is minted per
    compute precisely so it never leaves the daemon.

    Every operation takes a :data:`Target` rather than a node, because a file
    written to one machine of four is not written to the compute. Reading is the
    exception: it has a single stream, so it takes a single rank.
    """

    async def ls(self, compute_id: str, target: Target, path: str) -> tuple[tuple[str, Result], ...]: ...

    async def rm(self, compute_id: str, target: Target, path: str) -> tuple[tuple[str, Result], ...]: ...

    async def put(self, compute_id: str, target: Target, path: str, content: bytes) -> tuple[tuple[str, str | None], ...]:
        """Per node, what stopped the write, or None where nothing did."""
        ...

    def get(self, compute_id: str, rank: int, path: str) -> AsyncIterator[bytes]:
        """One node's copy of a file, as it is read off the machine."""
        ...

    async def run(self, compute_id: str, target: Target, command: str) -> tuple[tuple[str, Result], ...]:
        """One shell command on each targeted node, and what each one said."""
        ...


@runtime_checkable
class Shell(Protocol):
    """Bridges one interactive session to a node's terminal, as two half-duplex streams.

    The transport a :class:`Forwarder` uses, carrying a pseudo-terminal instead of a
    socket: keystrokes up, everything the terminal paints down, tied by the id the
    caller mints. Unlike a forward it takes a rank, because a shell is somebody
    sitting at one machine — picking a different one per connection would be the
    wrong answer to every question they ask it.

    The machine does not have to be ready. A terminal is how a bootstrap is watched
    while it happens, so it is offered on every machine the daemon holds a link to,
    which is every machine that has answered SSH.
    """

    async def up(
        self,
        compute_id: str,
        cid: str,
        rank: int | None,
        command: str | None,
        term: str,
        size: tuple[int, int],
        chunks: AsyncIterator[bytes],
    ) -> None:
        """Open a terminal on the machine at ``rank`` and pump the caller's keystrokes into it."""
        ...

    async def down(self, cid: str) -> AsyncIterator[bytes]:
        """What the terminal paints, for the ``up`` that opened this id.

        Waits for the terminal and then hands back the stream, so a refusal is an
        answer with a status on it rather than a body that stops mid-chunk.
        """
        ...

    async def open(self, compute_id: str, rank: int | None, command: str | None, term: str, size: tuple[int, int]) -> Pty:
        """The terminal itself, for a transport that carries both directions at once.

        The pair above is what HTTP/1.1 leaves room for. A caller holding one socket
        needs neither half nor the id that ties them, and gets the one thing the
        pair has nowhere to put: the screen's new shape, sent mid-session.
        """
        ...


@runtime_checkable
class Health(Protocol):
    async def live(self) -> bool: ...

    async def ready(self) -> bool: ...

    async def dependencies(self) -> dict[str, DependencyState]: ...
