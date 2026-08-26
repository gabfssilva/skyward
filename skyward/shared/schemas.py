import hashlib
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal

import msgspec
from msgspec import UNSET, Struct, UnsetType, field
from msgspec.structs import force_setattr

from skyward.shared.accelerators import resolve
from skyward.shared.architectures import Architecture, architecture

type ComputeState = Literal[
    "requested",
    "provisioning",
    "recovering",
    "ready",
    "degraded",
    "deleting",
    "deleted",
    "failed",
]

type NodeState = Literal[
    "requested",
    "provisioning",
    "connecting",
    "bootstrapping",
    "ready",
    "draining",
    "lost",
    "deleting",
    "deleted",
    "failed",
]

type TaskState = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "timed_out",
    "indeterminate",
]

type ExecutionState = Literal[
    "created",
    "assigned",
    "dispatching",
    "accepted",
    "started",
    "cancel_requested",
    "succeeded",
    "failed",
    "cancelled",
    "timed_out",
    "indeterminate",
]

type Dispatch = Literal["one", "all", "stream"]
"""One node, every node, or one node that answers in pieces.

A stream is a task like any other — it is admitted, written down and given an
execution — and it is dispatched by the request that reads it rather than by the
reconciler. Nobody can hold the far end of a stream but the caller who is consuming
it, so nobody else can start one on their behalf.
"""
type Desired = Literal["running", "deleted"]
type NodeDesired = Literal["present", "deleted"]
type Allocation = Literal["spot", "on_demand", "spot_if_available", "cheapest"]
type Market = Literal["spot", "on_demand"]
"""Which price an offer was taken at — what ``Allocation`` resolves to once an offer is in hand.

An allocation is a preference and may not be satisfiable; a market is a decision,
and it is what the machines are actually billed under.
"""
type BillingUnit = Literal["second", "minute", "hour"]
"""The granularity a provider bills at — prices are always per hour, this is the rounding.

A machine kept for 61 seconds costs 2 minutes on a per-minute provider and a full
hour on a per-hour one. Stamped by each adapter on its offers, because it is a fact
about the provider's billing, not a preference.
"""
type Selection = Literal["cheapest", "first"]
type Executor = Literal["thread", "process", "loky"]
type SkywardSource = Literal["auto", "local", "github", "pypi"]

type ErrorCode = Literal[
    "not_found",
    "revision_conflict",
    "idempotency_conflict",
    "lease_held",
    "name_taken",
    "compute_not_connected",
    "compute_not_accepting",
    "compute_not_resizable",
    "unsupported_provider",
    "unsupported_plugin",
    "secret_in_definition",
    "hash_mismatch",
    "task_failed",
    "task_indeterminate",
    "duplication_not_acknowledged",
    "capability_mismatch",
    "release_pending",
]


class Error(Struct, frozen=True):
    """Every failure, in one shape, whatever produced it.

    ``code`` is the closed set a client matches on, so nobody parses prose.
    ``retryable`` is a property of the failure rather than a guess left to the
    caller: a revision conflict is worth re-reading and re-sending, and a spec no
    provider can satisfy is not.
    """

    code: ErrorCode
    message: str
    retryable: bool
    request_id: str | None = None
    details: dict[str, Any] | None = None


class ProviderRef(Struct, frozen=True):
    """Which account a spec wants to buy from.

    A kind and nothing else: how the account is configured belongs to the provider
    row, which is what the daemon builds its adapter from. A copy of the settings
    riding on the spec would be a second answer to the same question, and the one
    nobody reads — so a spec names the kind, and ``sky providers set`` or the SDK's
    own account object is what says what that kind means here.
    """

    kind: str


class PluginRef(Struct, frozen=True):
    """A plugin as it travels: a name and its parameters, never an object.

    A plugin is rebuilt on the node from exactly this, which is why it cannot hold
    a closure or a live handle. Its behaviour is its class's methods; its identity
    is ``kind``.
    """

    kind: str
    params: dict[str, Any] = field(default_factory=dict)


class PipIndex(Struct, frozen=True):
    """A package index the resolver should reach for, and what it may serve.

    ``packages`` is the scope: only those names resolve from ``url`` (uv's
    ``explicit = true``), so a private index cannot silently answer for a public
    package. An empty scope makes it an ordinary extra index.
    """

    url: str
    packages: Sequence[str] = ()

    def __post_init__(self) -> None:
        force_setattr(self, "packages", (self.packages,) if isinstance(self.packages, str) else tuple(self.packages))


class MetricSpec(Struct, frozen=True):
    """One reading a node takes about itself: a shell command, sampled on a period.

    The command must print a bare number; anything else is dropped rather than
    reported. ``interval`` is in seconds.
    """

    name: str
    command: str
    interval: float


class Image(Struct, frozen=True):
    """The environment a node builds before it runs anything.

    The base, the interpreter, the packages and where they resolve from. What the
    user shipped from their own machine is not here — ``includes`` is packed into
    a blob client-side and only its hash travels, because a spec is written to the
    compute row and served back by the API.
    """

    base: str | None = None
    python: str | None = None
    pip: Sequence[str] = ()
    apt: Sequence[str] = ()
    pip_indexes: Sequence[PipIndex] = ()
    env: dict[str, str] = field(default_factory=dict)
    shell_vars: dict[str, str] = field(default_factory=dict)
    includes: Sequence[str] = ()
    excludes: Sequence[str] = ()
    includes_sha256: str | None = None
    """The user-code tarball, once the client has built it and put it in the blob
    store. ``includes``/``excludes`` are the client's inputs; this is what the node
    reads."""
    metrics: Sequence[MetricSpec] | None = None
    """``None`` leaves the built-in collectors in place; a list replaces them."""
    bootstrap_timeout: int = 900
    skyward: SkywardSource = "auto"
    warm: bool = False
    """Whether a machine that finished bootstrapping is kept as a boot image.

    Off because what it creates is never removed: an AMI holds a snapshot that bills
    for its storage until it is deregistered, and nothing here deregisters it. Turning
    it on is taking that on. What is created carries :meth:`content_hash` as a tag, on
    the image and on the snapshot behind it, so it can be found again and removed.
    Only providers that can snapshot a running machine honor it.
    """

    def __post_init__(self) -> None:
        for name in ("pip", "apt", "pip_indexes", "includes", "excludes"):
            value = getattr(self, name)
            force_setattr(self, name, (value,) if isinstance(value, str) else tuple(value))
        if self.metrics is not None:
            force_setattr(self, "metrics", tuple(self.metrics))

    def content_hash(self, source: str) -> str:
        """Name the environment a bootstrapped machine ends up in.

        Covers what the bootstrap installs — the base, the interpreter, the packages
        and the indexes they are resolved from — together with ``source``, which is
        what stands in for a skyward version now that a node installs whatever the
        daemon is running.

        Left out is everything the bootstrap re-applies on every boot: the exports,
        the shell vars, the metric commands, and the user code, which is synced per
        run. Folding those in would split the images over changes that cost nothing
        to redo.

        Parameters
        ----------
        source : str
            :attr:`skyward.server.application.source.Source.argument` — what follows
            ``uv pip install``. Never a locally built wheel: its bytes change with
            every edit, so a name derived from it would outlive what it named.

        Returns
        -------
        str
            Twelve hex characters — long enough to name an image, short enough to read
            in one.
        """
        identity = (self.base, self.python, self.pip, self.apt, self.pip_indexes, source)
        return hashlib.sha256(msgspec.json.encode(identity)).hexdigest()[:12]


class Volume(Struct, frozen=True):
    """A bucket the nodes read and write as a directory.

    ``bucket`` names an object-storage bucket, except on providers that attach a
    volume of their own rather than mounting one — RunPod reads it as the id or
    name of a network volume.

    Where the credentials come from is what ``storage_sha256`` decides, and it is
    the only reason the field exists. ``None`` means the daemon resolves them from
    the provider record it already holds, and nothing about the bucket's access
    ever reaches this struct. A digest means the client brought its own, put them
    in the blob store, and left only the hash here — because a spec is written to
    the compute row and served back by the compute API, and a secret written there
    is a secret published.
    """

    bucket: str
    mount: str
    prefix: str = ""
    read_only: bool = True
    storage_sha256: str | None = None


class Endpoint(Struct, frozen=True):
    """Where a bucket is reached and what signs for it.

    A blob, never a spec field: it carries the secret that :class:`Volume` refuses
    to. ``access_key`` unset means the machine signs with its own instance
    identity, which is how a bucket in the account that bought the machine is
    reached with no credential in flight at all.
    """

    url: str
    access_key: str | None = None
    secret_key: str | None = None
    path_style: bool = False


class Worker(Struct, frozen=True):
    """How much work a node takes at once, and what runs it.

    ``concurrency`` unset lets the node decide from what it has. The executor is
    the backend the tasks run on: threads by default, processes or loky when the
    work holds the GIL.
    """

    concurrency: int | None = None
    executor: Executor = "thread"
    reuse: bool = True
    buffer: int = 0
    """How many tasks the worker accepts beyond ``concurrency``.

    ``concurrency`` is the executor's width — how many tasks run at once. ``buffer``
    is the slack casty admits on top of it: the extra tasks arrive, are unpickled and
    wait at the executor's door, so a slot that frees has the next task in hand rather
    than a round trip away. It is also the depth the daemon reads as backpressure —
    the mailbox only fills once the buffer is full, which is the point at which
    another node would actually help.
    """


class Spec(Struct, frozen=True):
    """One shape of machine that would do, from one account.

    A compute carries several of these as a preference list and buys exactly one:
    everything here is what the market filters offers on, and everything that must
    be true of the whole fleet — volumes, plugins, node counts — is on the compute
    instead.
    """

    provider: ProviderRef
    accelerator: str | None = None
    accelerator_count: int = 1
    cpus: int | None = None
    memory_gb: int | None = None
    region: str | None = None
    disk_gb: int | None = None
    """The least disk a machine must carry to satisfy this spec."""
    architecture: Architecture | None = None
    """The instruction set the machine must run, when the payload only has wheels for one.

    An offer that does not report its architecture does not satisfy this, because
    it cannot be shown to: shipping x86 wheels to an arm machine is not a slow
    node, it is a node that never runs a task.
    """
    max_hourly_cost: float | None = None
    """The most an hour of one machine may cost, at the price it is actually bought on."""


class NodeBounds(Struct, frozen=True):
    """How many machines, and how much of that is negotiable.

    ``desired`` is the target. ``min`` is the count at which work may start, which
    is what lets a job of eight begin on four. ``max`` is the ceiling autoscaling
    may reach. Both unset means the target is also the floor and the ceiling.
    """

    desired: int
    min: int | None = None
    max: int | None = None


class RetryPolicy(Struct, frozen=True):
    """How many times to try again, split by whether trying again is safe.

    A task that never started can be re-run freely. A task whose node went silent
    after it may have run is a different question, and it defaults to zero because
    the system does not know whether it had side effects and will not pretend to.
    """

    safe_retries: int = 3
    ambiguous_retries: int = 0


class Options(Struct, frozen=True):
    """Operational knobs the daemon reads off the spec.

    Each defaults to the value the runtime hard-coded before the knob existed, so a
    spec built without options behaves exactly as one built with ``Options()``. The
    client-side timeouts are not here: they govern how long the owning process waits
    for its own pool, never leave it, and so ride the SDK's ``Options`` rather than
    the wire.
    """

    ssh_connect_timeout: float = 240.0
    ssh_reconnect_attempts: int = 30
    ssh_retry_delay: float = 2.0
    worker_timeout: float = 180.0
    """Seconds the worker has to answer once it has been started.

    Measured on RunPod: the two nodes that *formed* a cluster answered ~25s after
    bootstrap; a third node *joining* the formed cluster took over 68s, every time.
    Founding is cheap and joining is not, so this covers the slower of the two.
    """
    provision_timeout: float = 300.0
    """Seconds a bought machine has to publish an address before it is given up on.

    The SSH timeouts start when there is somewhere to dial; this is the window
    before that, and nothing else measures it. A machine the provider reports as
    running and never gives an address to is a machine that bills for as long as
    the compute lives — so it becomes ``lost``, is terminated, and the deficit is
    closed with another one. ``0`` waits forever.
    """
    autoscale_idle_timeout: float = 30.0
    autoscale_cooldown: float = 0.0
    """Seconds between autoscaling decisions. ``0`` is no cooldown — today's behavior."""
    default_compute_timeout: float = 0.0
    """Seconds a task may run when it names no deadline of its own. ``0`` is unbounded."""
    health_command: str | None = None
    """A shell command run on each node to ask whether the machine is still usable.

    A nonzero exit is one failure, and ``health_failures`` consecutive ones make the
    node ``lost`` — the same word a machine that went away gets, so the deficit it
    leaves is closed the way any other is. ``None`` probes nothing, which is what the
    runtime did before the knob existed.
    """
    health_interval: float = 30.0
    health_failures: int = 3
    health_function: bytes | None = None
    health_timeout: float = 15.0
    health_initial_delay: float = 0.0
    cluster: bool | None = None


class ComputeSpec(Struct, frozen=True):
    """Everything a compute was asked to be. Intent, never observation.

    Only a client writes it, and only through ``PATCH``. Of its fields exactly one
    is mutable in place — ``nodes``, which resizes — and changing any other is
    drift: it is recorded, the applied definition is kept, and replacing the
    machines takes a new generation.
    """

    specs: tuple[Spec, ...]
    nodes: NodeBounds
    selection: Selection = "cheapest"
    allocation: Allocation = "spot_if_available"
    image: Image = field(default_factory=Image)
    worker: Worker = Worker()
    options: Options = Options()
    plugins: tuple[PluginRef, ...] = ()
    retry: RetryPolicy = RetryPolicy()
    delete_on_exit: bool = False
    desired: Desired = "running"
    ttl: int = 600
    """Seconds a machine may go with no control-plane connection before it self-terminates.

    A safety net against a leaked machine that bills forever: the daemon holds an SSH
    connection to every node for the pool's whole life, so a node that has had none for
    ``ttl`` seconds is one no daemon is coming back for. ``0`` disables it. Only providers
    that boot the machine from a container entrypoint (RunPod) honor it today."""
    volumes: tuple[Volume, ...] = ()
    """Buckets every node of the compute mounts.

    On the compute and not on a :class:`Spec`, because the specs are a preference
    list and only one of them is ever bought: a volume named against the spec that
    lost would be a mount the fleet does not have.
    """


class ComputeSpecPatch(Struct, frozen=True):
    """The one field of a spec that can change without replacing machines."""

    nodes: NodeBounds


class ComputeCreate(Struct, frozen=True):
    """What it takes to ask for a compute: a definition, and optionally a name to find it by."""

    spec: ComputeSpec
    name: str | None = None


class ComputeStatus(Struct, frozen=True):
    """What the reconciler has observed. Never written by a client.

    ``observed_generation`` against the compute's ``generation`` is the progress:
    the gap between them *is* the pending work, which is why there is no operation
    resource to poll.
    """

    state: ComputeState
    observed_generation: int
    nodes_ready: int
    nodes_total: int
    drift: tuple[str, ...] = ()
    last_error: Error | None = None


class Lease(Struct, frozen=True):
    """Who currently owns the compute, and until when.

    A liveness signal the owning process renews, not a lock on the resource. Zero
    owners is legitimate and temporary — a daemon restarting, a script killed —
    and only the holder opens SSH connections to the machines. Losing it destroys
    nothing by itself.
    """

    owner: str | None = None
    expires_at: datetime | None = None


class Compute(Struct, frozen=True):
    """A set of machines held under one intention, as the API serves it.

    ``spec`` is what was asked for and ``status`` is what was observed, written by
    different actors and kept in one resource so that reading both is one call.
    ``revision`` is the concurrency token behind ``ETag`` and ``If-Match``;
    ``generation`` counts definitions, not writes.
    """

    id: str
    name: str | None
    revision: int
    generation: int
    spec: ComputeSpec
    status: ComputeStatus
    lease: Lease
    created_at: datetime


class LeaseClaim(Struct, frozen=True):
    """A bid for ownership: who is claiming, and for how long before it lapses."""

    owner: str
    ttl_seconds: int


class Node(Struct, frozen=True):
    """One machine, ranked, as the control plane knows it.

    The row exists before the provider is asked for a machine, which is what makes
    provisioning idempotent — ``machine`` is null until there is one to name. A
    node that died keeps its row and its ``provider_binding`` until the provider
    confirms the instance is gone, so nothing goes missing unnoticed.
    """

    id: str
    compute_id: str
    generation: int
    rank: int
    revision: int
    desired: NodeDesired
    state: NodeState
    provider_binding: dict[str, Any]
    created_at: datetime
    machine: str | None = None
    """The provider's name for the machine, once it has one. A ``requested`` node has none."""
    address: str | None = None
    accelerator: str | None = None
    price_per_hour: float | None = None
    market: Market | None = None
    billing_unit: BillingUnit | None = None
    launched_at: datetime | None = None
    last_error: Error | None = None
    terminated_at: datetime | None = None


class Function(Struct, frozen=True):
    """A registered piece of code, named by the hash of its serialized bytes.

    The metadata only. The code is a blob, fetched as one — which is what lets a
    task name its function without carrying it.
    """

    sha256: str
    size_bytes: int
    codec: str
    created_at: datetime
    name: str | None = None


class Execution(Struct, frozen=True):
    """One physical attempt at a task.

    ``ordinal`` counts the attempts and ``rank`` says which node took it. A retry
    is another execution pointing ``retry_of`` at the last one — never another
    task, because the task id is the handle the caller is holding.
    """

    id: str
    rank: int
    ordinal: int
    state: ExecutionState
    node_id: str | None = None
    retry_of: str | None = None
    result_sha256: str | None = None
    error: Error | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TaskCreate(Struct, frozen=True):
    """One call to place: the code, its arguments, and how widely to run it.

    Arguments travel inline below a size threshold and as a blob above it, which
    is why there are two fields for them and exactly one is set.
    """

    compute: str
    function: str
    dispatch: Dispatch
    args_inline: bytes | None = None
    args_sha256: str | None = None
    rank: int | None = None
    timeout_seconds: int | None = None
    retry: RetryPolicy | UnsetType = UNSET
    correlation_id: str | None = None


class Task(Struct, frozen=True):
    """One call — function plus arguments — and its one terminal outcome.

    Append-only. ``state`` is derived from the executions rather than written
    beside them, and is stored only so that listing by state is a query instead of
    a scan. ``correlation_id`` is how the tasks of one ``&``, ``gather`` or ``map``
    are found together: a field on each of them, not a resource of their own.
    """

    id: str
    compute_id: str
    generation: int
    function: str
    args_sha256: str
    dispatch: Dispatch
    state: TaskState
    retry: RetryPolicy
    executions: tuple[Execution, ...]
    submitted_at: datetime
    correlation_id: str | None = None
    deadline_at: datetime | None = None
    result_sha256: str | None = None
    finished_at: datetime | None = None


class ExecutionCreate(Struct, frozen=True):
    """A retry, and the admission that it may be a second run.

    ``acknowledge_duplication`` is required to retry an indeterminate outcome: the
    system does not know whether the previous attempt had side effects, and the
    caller is the only one in a position to say that running twice is acceptable.
    """

    acknowledge_duplication: bool = False
    ranks: tuple[int, ...] | None = None


class Generation(Struct, frozen=True):
    """One frozen definition of a compute, and whether the machines match it yet.

    History is kept rather than overwritten, because a rollback is a generation
    too — it is a new number carrying an old spec, not an erasure of what happened
    in between.
    """

    number: int
    spec: ComputeSpec
    hash: str
    created_at: datetime
    applied: bool


class GenerationCreate(Struct, frozen=True):
    """Replace the machines: with the pending drift, or with an older definition.

    Without ``source`` this applies whatever drift the status is carrying. With
    one, it rolls back to that generation. ``force`` marks unresolved tasks
    indeterminate rather than refusing while executions are still live.
    """

    source: int | None = None
    force: bool = False


class Page[T](Struct, frozen=True):
    """A slice of a listing, and the cursor that continues it.

    ``next_cursor`` is null on the last page. Cursors are opaque and are not
    offsets — a row inserted mid-walk does not shift what a held cursor returns.
    """

    items: tuple[T, ...]
    next_cursor: str | None = None


class ProviderKind(Struct, frozen=True):
    """A kind of cloud this build can talk to, and what registering one needs.

    Capability negotiation, before anything is created: a kind absent from this
    list cannot be registered, usually because its SDK is not installed.
    """

    kind: str
    credential_fields: tuple[str, ...]
    offers_ttl_seconds: int


class ProviderCreate(Struct, frozen=True):
    """An account to register: what to call it, which cloud it is, and what opens it.

    ``credentials`` are validated against the kind's declared fields before the
    row is written, and no read path returns them afterwards.
    """

    name: str
    kind: str
    credentials: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


class Provider(Struct, frozen=True):
    """A registered account, as the API serves it — which is to say, without its secrets.

    ``offers_fetched_at`` and ``offers_count`` describe the cached catalog behind
    it, and ``last_error`` is why the last refresh failed. A failed refresh leaves
    the stale offers in place: stale offers beat no offers.
    """

    id: str
    name: str
    kind: str
    config: dict[str, Any]
    offers_ttl_seconds: int
    created_at: datetime
    offers_fetched_at: datetime | None = None
    offers_count: int = 0
    last_error: Error | None = None


class Offer(Struct, frozen=True):
    """One buyable machine shape from one account, normalized.

    The accelerator name and its VRAM are parsed into the shared vocabulary here
    rather than in each adapter, which is what stops the same GPU from being two
    different accelerators depending on who is selling it. ``price`` is the
    cheapest the offer can be had for, and it is what ordering and budget filters
    compare on.
    """

    id: str
    provider_id: str
    provider_name: str
    kind: str
    instance_type: str
    accelerator_count: int
    cpus: int
    memory_gb: float
    fetched_at: datetime
    expires_at: datetime
    accelerator: str | None = None
    region: str | None = None
    disk_gb: float | None = None
    architecture: str | None = None
    spot_price: float | None = None
    on_demand_price: float | None = None
    billing_unit: BillingUnit = "hour"
    available: int | None = None
    vram: float | None = None
    price: float | None = None
    specific: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize what an adapter handed over into what the market compares on.

        ``accelerator`` may be handed in exactly as the provider spells it
        ("NVIDIA H100 80GB SXM5", "H100-80G-PCIe"); it is normalized here into
        the shared vocabulary, and ``vram`` falls out of the same parse. Doing
        it in the struct rather than in each adapter is the point: eleven
        adapters normalizing on their own is how `h100` and `h100-sxm` ended up
        being different accelerators.

        ``architecture`` is normalized here for the same reason, with the opposite
        treatment of the unknown: a spelling the vocabulary does not have becomes
        ``None``. An adapter passes whatever its API said and cannot invent a
        third architecture, and an offer that reports nothing stays unsellable to
        a spec that asked for one.

        ``price`` is the cheapest the offer can be had for. Ordering and budget
        filters run on it, not on ``on_demand_price`` — several providers
        publish spot-only flavors, and comparing those on a price they do not
        have would hide them from every question about cost.
        """
        accelerator, vram = resolve(self.accelerator, self.vram)
        force_setattr(self, "accelerator", accelerator)
        force_setattr(self, "vram", vram)
        force_setattr(self, "architecture", architecture(self.architecture))

        prices = [price for price in (self.spot_price, self.on_demand_price) if price is not None]
        force_setattr(self, "price", min(prices) if prices else None)


type DependencyState = Literal["ok", "unreachable"]


class Liveness(Struct, frozen=True):
    """Whether the process answers. Says nothing about the store or the providers."""

    live: bool


class Readiness(Struct, frozen=True):
    """Whether the daemon can serve: schema in place, recovery done."""

    ready: bool


type PhaseMark = Literal["started", "completed", "failed"]
"""Whether a bootstrap phase opened, closed, or broke."""

type TaskEventState = Literal["started", "succeeded", "failed", "indeterminate"]
"""What the stream says about a task: that it began, or how it ended.

Narrower than :data:`TaskState`, which is the task resource's own vocabulary. A
task that was never placed has a state and no event, and ``started`` is a moment
rather than a state — the two are not the same alphabet and are not merged.
"""


class ComputeEvent(Struct, frozen=True, tag_field="type", tag="compute.state"):
    """A compute reached a state worth saying out loud.

    Carried by ``compute.ready``, ``compute.degraded`` and ``compute.deleted``.
    Not every state gets one: the stream reports the transitions a watcher acts
    on, not every pass the reconciler makes.
    """

    compute: str
    state: ComputeState
    error: str | None = None


class ComputeAbandoned(Struct, frozen=True, tag_field="type", tag="compute.abandoned"):
    """Nothing renewed the lease and ``delete_on_exit`` was set, so it is going away.

    Its own fact rather than a compute state, because a lapsed lease is not a
    failure — it is what a client exiting looks like from in here.
    """

    compute: str


class CostEvent(Struct, frozen=True, tag_field="type", tag="compute.cost"):
    """What the compute has accrued so far, and over how many live machines.

    Published rather than recorded: a gauge sampled every few seconds has no
    replay value, and the event log has no GC to save it from one.
    """

    compute: str
    cost: float
    nodes: int
    at: datetime


class NodeEvent(Struct, frozen=True, tag_field="type", tag="node.state"):
    """One machine's lifecycle moved, carried by every ``node.{state}``.

    ``state`` repeats the event name because a payload that has been written
    down, exported, or replayed out of the stream has to say what it is without
    the frame that carried it.
    """

    compute: str
    node: str
    state: NodeState
    error: str | None = None


class ConsoleEvent(Struct, frozen=True, tag_field="type", tag="node.console"):
    """A line a node printed, and the work it belongs to when it belongs to some.

    Recorded, because output that only existed live would be output a client that
    reconnected could never see.

    ``task`` carries the *execution* — the attempt on this node — because that is
    what the machine that wrote the line was handed. A reader after a whole task's
    output wants every execution of it, not a string equal to the task's id.
    """

    compute: str
    node: str
    content: str
    task: str | None = None


class PhaseEvent(Struct, frozen=True, tag_field="type", tag="node.phase"):
    """A bootstrap phase turning over, so a late subscriber replays the checklist.

    ``phase`` names the step; ``event`` says whether it opened, closed, or broke.
    """

    compute: str
    node: str
    event: PhaseMark
    phase: str
    at: datetime
    error: str | None = None


class MetricEvent(Struct, frozen=True, tag_field="type", tag="node.metrics"):
    """One gauge reading off one node. Published rather than recorded, like ``compute.cost``."""

    compute: str
    node: str
    name: str
    value: float


class TaskEvent(Struct, frozen=True, tag_field="type", tag="task.state"):
    """A task began, or reached the one terminal outcome it is allowed."""

    compute: str
    task: str
    state: TaskEventState


type Event = (
    ComputeEvent | ComputeAbandoned | CostEvent | NodeEvent | ConsoleEvent | PhaseEvent | MetricEvent | TaskEvent
)
"""Everything the SSE stream carries, as one tagged union.

The SSE frame's ``event:`` field is the name a subscriber filters on, and it is
finer than this: ten node states and four task outcomes share a struct each. The
``type`` tag inside the payload is the coarser of the two, and it is what makes
the payload decodable on its own — by a client that matches on the struct instead
of on a string, and by a reader of the OpenAPI document, which has no way to
express a discriminator that lives outside the body.
"""
