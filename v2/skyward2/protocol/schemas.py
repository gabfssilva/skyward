from datetime import datetime
from typing import Any, Literal

from msgspec import UNSET, Struct, UnsetType, field
from msgspec.structs import force_setattr

from skyward2.protocol.accelerators import resolve

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
type Selection = Literal["cheapest", "first"]
type Executor = Literal["thread", "process"]
type SkywardSource = Literal["auto", "local", "github", "pypi"]

type ErrorCode = Literal[
    "not_found",
    "revision_conflict",
    "idempotency_conflict",
    "lease_held",
    "compute_not_accepting",
    "unsupported_provider",
    "unsupported_plugin",
    "secret_in_definition",
    "hash_mismatch",
    "task_failed",
    "task_indeterminate",
    "duplication_not_acknowledged",
    "capability_mismatch",
]


class Error(Struct, frozen=True):
    code: ErrorCode
    message: str
    retryable: bool
    request_id: str | None = None
    details: dict[str, Any] | None = None


class ProviderRef(Struct, frozen=True):
    kind: str
    config: dict[str, Any] = field(default_factory=dict)


class PluginRef(Struct, frozen=True):
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


class Image(Struct, frozen=True):
    base: str | None = None
    python: str | None = None
    packages: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    skyward: SkywardSource = "auto"


class Worker(Struct, frozen=True):
    concurrency: int | None = None
    executor: Executor | None = None


class Spec(Struct, frozen=True):
    provider: ProviderRef
    accelerator: str | None = None
    accelerator_count: int = 1
    cpus: int | None = None
    memory_gb: int | None = None
    region: str | None = None


class NodeBounds(Struct, frozen=True):
    desired: int
    min: int | None = None
    max: int | None = None


class RetryPolicy(Struct, frozen=True):
    safe_retries: int = 3
    ambiguous_retries: int = 0


class ComputeSpec(Struct, frozen=True):
    specs: tuple[Spec, ...]
    nodes: NodeBounds
    selection: Selection = "cheapest"
    allocation: Allocation = "spot_if_available"
    image: Image = field(default_factory=Image)
    worker: Worker = Worker()
    plugins: tuple[PluginRef, ...] = ()
    retry: RetryPolicy = RetryPolicy()
    delete_on_exit: bool = False
    desired: Desired = "running"


class ComputeSpecPatch(Struct, frozen=True):
    nodes: NodeBounds


class ComputeCreate(Struct, frozen=True):
    spec: ComputeSpec
    name: str | None = None


class ComputeStatus(Struct, frozen=True):
    state: ComputeState
    observed_generation: int
    nodes_ready: int
    nodes_total: int
    drift: tuple[str, ...] = ()
    last_error: Error | None = None


class Lease(Struct, frozen=True):
    owner: str | None = None
    expires_at: datetime | None = None


class Compute(Struct, frozen=True):
    id: str
    name: str | None
    revision: int
    generation: int
    spec: ComputeSpec
    status: ComputeStatus
    lease: Lease
    created_at: datetime


class LeaseClaim(Struct, frozen=True):
    owner: str
    ttl_seconds: int


class Node(Struct, frozen=True):
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
    last_error: Error | None = None
    terminated_at: datetime | None = None


class Function(Struct, frozen=True):
    sha256: str
    size_bytes: int
    codec: str
    created_at: datetime
    name: str | None = None


class Execution(Struct, frozen=True):
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
    acknowledge_duplication: bool = False
    ranks: tuple[int, ...] | None = None


class Generation(Struct, frozen=True):
    number: int
    spec: ComputeSpec
    hash: str
    created_at: datetime
    applied: bool


class GenerationCreate(Struct, frozen=True):
    source: int | None = None
    force: bool = False


class Page[T](Struct, frozen=True):
    items: tuple[T, ...]
    next_cursor: str | None = None


class ProviderKind(Struct, frozen=True):
    kind: str
    credential_fields: tuple[str, ...]
    offers_ttl_seconds: int


class ProviderCreate(Struct, frozen=True):
    name: str
    kind: str
    credentials: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


class Provider(Struct, frozen=True):
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
    spot_price: float | None = None
    on_demand_price: float | None = None
    available: int | None = None
    vram: float | None = None
    price: float | None = None
    specific: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Derive the two fields an offer is compared on: its accelerator and its price.

        ``accelerator`` may be handed in exactly as the provider spells it
        ("NVIDIA H100 80GB SXM5", "H100-80G-PCIe"); it is normalized here into
        the shared vocabulary, and ``vram`` falls out of the same parse. Doing
        it in the struct rather than in each adapter is the point: eleven
        adapters normalizing on their own is how `h100` and `h100-sxm` ended up
        being different accelerators.

        ``price`` is the cheapest the offer can be had for. Ordering and budget
        filters run on it, not on ``on_demand_price`` — several providers
        publish spot-only flavors, and comparing those on a price they do not
        have would hide them from every question about cost.
        """
        accelerator, vram = resolve(self.accelerator, self.vram)
        force_setattr(self, "accelerator", accelerator)
        force_setattr(self, "vram", vram)

        prices = [price for price in (self.spot_price, self.on_demand_price) if price is not None]
        force_setattr(self, "price", min(prices) if prices else None)
