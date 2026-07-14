from datetime import datetime
from typing import Any, Literal

from msgspec import UNSET, Struct, UnsetType, field

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

type Dispatch = Literal["one", "all"]
type Desired = Literal["running", "deleted"]
type NodeDesired = Literal["present", "deleted"]
type Allocation = Literal["spot", "on_demand", "spot_if_available", "cheapest"]
type Selection = Literal["cheapest", "first"]
type Executor = Literal["thread", "process"]

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


class Provider(Struct, frozen=True):
    kind: str
    credentials_resolvable: bool
    daemon_supported: bool
    capabilities: tuple[str, ...]


class Offer(Struct, frozen=True):
    provider: str
    instance_type: str
    accelerator_count: int
    cpus: int
    memory_gb: float
    region: str
    price_per_hour: float
    spot: bool
    accelerator: str | None = None
    available: int | None = None
