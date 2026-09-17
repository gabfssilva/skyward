"""Everything ``/v1`` takes and answers with.

A ``*Resource`` is a document an endpoint answers with or takes; one it takes is named
for what it does, as ``CreateComputeResource``. ``*Event`` is what the event stream and
the event log carry, named after the ``type`` tag it goes out under, and ``*Frame`` is
one message of a stream — a streaming task's body, or a terminal's socket. ``Page`` and
``Error`` wrap the rest, and everything else in the module is a part of one of them.

A default is what the daemon takes for a field a request left out. A response carries
every field, except one typed ``| UnsetType``: a block the caller asked for with
``include``, absent from the document when it was not asked for, which is how "not
asked" stays distinct from "asked, and there is none".

An event's fields are the ones the daemon records, defaults included: a log entry
written before a field existed still decodes.
"""

from datetime import datetime
from typing import Literal

from msgspec import UNSET, Struct, UnsetType, field

type ComputeState = Literal["requested", "provisioning", "ready", "degraded", "deleting", "deleted"]
type NodeState = Literal["requested", "provisioning", "connecting", "bootstrapping", "ready", "draining", "lost", "deleting", "deleted", "failed"]
type NodeDesired = Literal["present", "deleted"]
type TaskState = Literal["queued", "running", "succeeded", "failed", "cancelled", "timed_out", "indeterminate"]
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
type Desired = Literal["running", "deleted"]
type Allocation = Literal["spot", "on_demand", "spot_if_available", "cheapest"]
type Selection = Literal["cheapest", "first"]
type Market = Literal["spot", "on_demand"]
type BillingUnit = Literal["second", "minute", "hour"]
type DeletionCause = Literal["requested", "abandoned"]
type Executor = Literal["thread", "process", "loky"]
type SkywardSource = Literal["auto", "local", "github", "pypi"]
type Architecture = Literal["x86_64", "arm64"]
type PhaseState = Literal["started", "completed", "failed"]
type TaskEventState = Literal["started", "retrying", "succeeded", "failed", "timed_out", "indeterminate"]
type DependencyState = Literal["ok", "unreachable"]
type OfferSort = Literal["price", "vram", "available"]
type Aggregate = Literal["avg", "min", "max", "last"]
type TaskOrder = Literal["submitted", "state", "finished"]
type Reading = Literal[
    "cpu",
    "mem_used_mb",
    "mem_total_mb",
    "gpu_util",
    "gpu_mem_mb",
    "gpu_mem_total_mb",
    "gpu_temp_c",
    "gpu_power_w",
    "net_rx_kbps",
    "net_tx_kbps",
    "disk_used_pct",
]
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
    "hash_mismatch",
    "task_failed",
    "task_indeterminate",
    "duplication_not_acknowledged",
    "capability_mismatch",
    "release_pending",
    "illegal_transition",
    "reconcile_failed",
    "source_rejected",
]

type ComputeInclude = Literal[
    "nodes.metrics",
    "nodes.phases",
    "nodes.running",
    "nodes.tail",
    "nodes.replaced",
    "tasks.latest",
    "tasks.pace",
    "utilization",
]
"""What ``include`` may ask a compute for, beyond what it always carries."""

type NodeInclude = Literal["metrics", "phases", "running", "tail", "replaced"]
"""The same blocks, asked of a node directly — where there is no ``nodes.`` to prefix them with."""


class Error(Struct, frozen=True, kw_only=True):
    code: ErrorCode
    message: str
    retryable: bool
    request_id: str | None
    details: dict[str, object] | None


class ProviderRef(Struct, frozen=True, kw_only=True):
    kind: str
    name: str | None = None
    """The account's name. Left out, any account of the kind will do."""


class PipIndex(Struct, frozen=True, kw_only=True):
    url: str
    packages: tuple[str, ...] = ()
    """The only names that resolve from ``url``. Empty makes it an ordinary extra index."""


class MetricSpec(Struct, frozen=True, kw_only=True):
    name: str
    command: str
    interval: float


class PluginRef(Struct, frozen=True, kw_only=True):
    kind: str
    params: dict[str, object] = field(default_factory=dict)


class Volume(Struct, frozen=True, kw_only=True):
    bucket: str
    mount: str
    prefix: str = ""
    read_only: bool = True
    storage_sha256: str | None = None
    """The blob holding the bucket's credentials. Null takes them from the provider account."""


class Worker(Struct, frozen=True, kw_only=True):
    concurrency: int | None = None
    executor: Executor = "thread"
    reuse: bool = True
    buffer: int = 0


class NodeBounds(Struct, frozen=True, kw_only=True):
    initial: int
    min: int | None = None
    max: int | None = None


class Options(Struct, frozen=True, kw_only=True):
    ssh_connect_timeout: float = 240.0
    ssh_reconnect_attempts: int = 30
    ssh_retry_delay: float = 2.0
    worker_timeout: float = 180.0
    provision_timeout: float = 300.0
    autoscale_idle_timeout: float = 120.0
    autoscale_cooldown: float = 0.0
    task_queue_timeout: float = 0.0
    task_run_timeout: float = 0.0
    health_command: str | None = None
    health_interval: float = 30.0
    health_failures: int = 3
    health_function: bytes | None = None
    health_timeout: float = 15.0
    health_initial_delay: float = 0.0
    cluster: bool | None = None


class Spec(Struct, frozen=True, kw_only=True):
    provider: ProviderRef
    accelerator: str | None = None
    accelerator_count: int = 1
    cpus: int | None = None
    memory_gb: int | None = None
    disk_gb: int | None = None
    region: str | None = None
    architecture: Architecture | None = None
    max_hourly_cost: float | None = None


class Image(Struct, frozen=True, kw_only=True):
    base: str | None = None
    python: str | None = None
    pip: tuple[str, ...] = ()
    apt: tuple[str, ...] = ()
    pip_indexes: tuple[PipIndex, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    shell_vars: dict[str, str] = field(default_factory=dict)
    includes: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    includes_sha256: str | None = None
    metrics: tuple[Reading | MetricSpec, ...] | None = None
    """Null is every reading."""
    bootstrap_timeout: int = 900
    skyward: SkywardSource = "auto"
    warm: bool = False


class ComputeSpec(Struct, frozen=True, kw_only=True):
    specs: tuple[Spec, ...]
    """The machine shapes that would do, in order of preference. One of them is bought."""
    nodes: NodeBounds
    selection: Selection = "cheapest"
    allocation: Allocation = "spot_if_available"
    image: Image = field(default_factory=Image)
    worker: Worker = Worker()
    options: Options = Options()
    plugins: tuple[PluginRef, ...] = ()
    retry: str | None = None
    """The blob holding the retry decision tasks take when they name none. Null is no retry."""
    delete_on_exit: bool = False
    desired: Desired = "running"
    ttl: int = 600
    """Seconds a machine may go with no daemon connected before it terminates itself. Zero is never."""
    volumes: tuple[Volume, ...] = ()


class Call(Struct, frozen=True, kw_only=True):
    """A call's arguments as JSON, for a caller that cannot pickle them."""

    args: tuple[object, ...] | None = None
    kwargs: dict[str, object] | None = None


class ComputeStatus(Struct, frozen=True, kw_only=True):
    state: ComputeState
    observed_generation: int
    """The generation the machines were last reconciled to. Behind ``generation`` is work still pending."""
    last_error: Error | None


class ProviderSummary(Struct, frozen=True, kw_only=True):
    id: str
    name: str
    kind: str


class ComputeSummary(Struct, frozen=True, kw_only=True):
    id: str
    name: str | None


class FunctionSummary(Struct, frozen=True, kw_only=True):
    """The code a task names. ``name`` and ``version`` are null for code the daemon holds no record of."""

    sha256: str
    name: str | None
    version: int | None


class LeaseResource(Struct, frozen=True, kw_only=True):
    """Who owns the compute, and until when. Zero owners is legitimate and temporary."""

    owner: str | None
    expires_at: datetime | None


class Refusal(Struct, frozen=True, kw_only=True):
    reason: str
    retry_at: datetime


class Ending(Struct, frozen=True, kw_only=True):
    at: datetime
    cause: DeletionCause
    cost: float


class TaskSummary(Struct, frozen=True, kw_only=True):
    id: str
    function: FunctionSummary
    state: TaskState
    finished_at: datetime | None
    error: Error | None


class LatestTasks(Struct, frozen=True, kw_only=True):
    succeeded: TaskSummary | None
    failed: TaskSummary | None
    """The latest to fail, time out or end indeterminate."""


class Pace(Struct, frozen=True, kw_only=True):
    finished_last_hour: int
    mean_seconds: float | None
    """How long the tasks that finished in the last hour ran, on average."""


class TaskCounts(Struct, frozen=True, kw_only=True):
    queued: int
    running: int
    succeeded: int
    failed: int
    cancelled: int
    timed_out: int
    indeterminate: int
    latest: LatestTasks | UnsetType = UNSET
    pace: Pace | UnsetType = UNSET


class Utilization(Struct, frozen=True, kw_only=True):
    """The average across the compute's nodes, one value per ``step`` milliseconds since ``since``. Null is a step nobody reported."""

    since: int
    step: int
    gpu: tuple[float | None, ...]
    cpu: tuple[float | None, ...]


class Ssh(Struct, frozen=True, kw_only=True):
    host: str
    port: int
    user: str


class Progress(Struct, frozen=True, kw_only=True):
    step: str
    completion: float | None
    """Between zero and one, for providers that count it."""


class Gauge(Struct, frozen=True, kw_only=True):
    at: int
    value: float


class Phase(Struct, frozen=True, kw_only=True):
    name: str
    state: PhaseState
    at: datetime
    error: str | None


class Running(Struct, frozen=True, kw_only=True):
    task: str
    ordinal: int
    function: FunctionSummary
    started_at: datetime | None


class MetricSeries(Struct, frozen=True, kw_only=True):
    node: str
    name: str
    at: tuple[int, ...]
    values: tuple[float, ...]


class Page[T](Struct, frozen=True, kw_only=True):
    """A slice of a listing. ``next_cursor`` is null on the last page; ``total`` is null where nothing counted."""

    items: tuple[T, ...]
    next_cursor: str | None
    total: int | None


class OfferResource(Struct, frozen=True, kw_only=True):
    """One machine shape one account sells. ``price`` is the cheapest it can be had for."""

    id: str
    provider_id: str
    provider_name: str
    kind: str
    instance_type: str
    accelerator: str | None
    accelerator_count: int
    vram: float | None
    cpus: int
    memory_gb: float
    disk_gb: float | None
    architecture: Architecture | None
    region: str | None
    spot_price: float | None
    on_demand_price: float | None
    price: float | None
    billing_unit: BillingUnit
    available: int | None
    fetched_at: datetime
    expires_at: datetime
    specific: dict[str, object]


class NodeResource(Struct, frozen=True, kw_only=True):
    """One machine holding one rank of a compute."""

    id: str
    rank: int
    generation: int
    created_at: datetime
    state: NodeState
    desired: NodeDesired
    machine: str | None
    """The provider's name for the machine. Null until one was bought."""
    address: str | None
    """Where the other nodes reach it."""
    ssh: Ssh | None
    accelerator: str | None
    market: Market | None
    price_per_hour: float | None
    billing_unit: BillingUnit | None
    launched_at: datetime | None
    terminated_at: datetime | None
    last_error: Error | None
    progress: Progress | None
    """What a machine without an address yet is doing, as the provider tells it."""
    busy: int
    """How many attempts are running on it."""
    metrics: dict[str, Gauge] | UnsetType = UNSET
    phases: tuple[Phase, ...] | UnsetType = UNSET
    running: tuple[Running, ...] | UnsetType = UNSET
    tail: tuple[str, ...] | UnsetType = UNSET
    """The last lines the node printed."""


class ComputeResource(Struct, frozen=True, kw_only=True):
    """A set of machines held under one intention, and the machines holding each rank of it.

    ``spec`` is what was asked for and ``status`` what was observed. ``nodes`` is the
    node currently holding each rank, and ``include=nodes.replaced`` adds the ones that
    held a rank before.
    """

    id: str
    name: str | None
    revision: int
    generation: int
    created_at: datetime
    status: ComputeStatus
    spec: ComputeSpec
    provider: ProviderSummary | None
    """The account the compute was bound to. Null until it is bound, since a spec may name several."""
    offer: OfferResource | None
    lease: LeaseResource
    cost: float
    """What its machines have cost so far, in dollars."""
    rate: float
    """What its machines cost right now, in dollars per hour."""
    tasks: TaskCounts
    placement: Refusal | None
    """Why the last machine could not be bought, while the compute is waiting to try again."""
    ended: Ending | None
    nodes: tuple[NodeResource, ...]
    utilization: Utilization | UnsetType = UNSET


class ExecutionResource(Struct, frozen=True, kw_only=True):
    """One physical attempt at a task. A retry is another execution, never another task."""

    id: str
    rank: int
    ordinal: int
    state: ExecutionState
    node_id: str | None
    retry_of: str | None
    result_sha256: str | None
    error: Error | None
    started_at: datetime | None
    finished_at: datetime | None
    deadline_at: datetime | None
    stopping: bool


class TaskResource(Struct, frozen=True, kw_only=True):
    """One call — a function and its arguments — and its one terminal outcome."""

    id: str
    compute: ComputeSummary
    generation: int
    function: FunctionSummary
    args_sha256: str
    dispatch: Dispatch
    state: TaskState
    retry: str | None
    executions: tuple[ExecutionResource, ...]
    submitted_at: datetime
    finished_at: datetime | None
    rank: int | None
    """The node the task named. Null is any node with a free slot."""
    correlation_id: str | None
    queue_timeout_seconds: float | None
    run_timeout_seconds: float | None
    result_sha256: str | None


class GenerationResource(Struct, frozen=True, kw_only=True):
    """One frozen definition of a compute, and whether the machines were built to it."""

    number: int
    spec: ComputeSpec
    hash: str
    created_at: datetime
    applied: bool


class FunctionResource(Struct, frozen=True, kw_only=True):
    """A registered piece of code, named by the hash of its serialized bytes.

    ``lineage`` is one function — the same name in the same file — and ``version``
    counts the changes to its code along it.
    """

    sha256: str
    size_bytes: int
    codec: str
    created_at: datetime
    lineage: str
    version: int
    name: str | None
    qualname: str | None
    origin: str | None
    source: str | None
    excerpt: str | None


class ProviderResource(Struct, frozen=True, kw_only=True):
    """A registered account, without its credentials."""

    id: str
    name: str
    kind: str
    config: dict[str, object]
    offers_ttl_seconds: int
    created_at: datetime
    offers_fetched_at: datetime | None
    offers_count: int
    last_error: Error | None


class ProviderKindResource(Struct, frozen=True, kw_only=True):
    """A kind of cloud this daemon can talk to, and what registering one takes."""

    kind: str
    credential_fields: tuple[str, ...]
    offers_ttl_seconds: int


class AcceleratorResource(Struct, frozen=True, kw_only=True):
    """One accelerator the catalog knows, and the card behind the name."""

    name: str
    vram: float
    manufacturer: str
    architecture: str
    cuda_min: str
    cuda_max: str


class MetricHistoryResource(Struct, frozen=True, kw_only=True):
    """A compute's metrics over a range or since a cursor. ``reset`` says the cursor fell behind compaction."""

    series: tuple[MetricSeries, ...]
    cursor: str
    reset: bool


class MetricSampleResource(Struct, frozen=True, kw_only=True):
    """One reading off one node. ``at`` is milliseconds since the epoch, on the node's clock."""

    node: str
    name: str
    at: int
    value: float


class LivenessResource(Struct, frozen=True, kw_only=True):
    live: bool
    version: str


class ReadinessResource(Struct, frozen=True, kw_only=True):
    ready: bool


class CommandResultResource(Struct, frozen=True, kw_only=True):
    """What one command said on one machine."""

    exit_code: int
    stdout: str
    stderr: str


class CreateComputeResource(Struct, frozen=True, kw_only=True):
    spec: ComputeSpec
    name: str | None = None


class UpdateComputeResource(Struct, frozen=True, kw_only=True):
    """A resize: ``nodes`` is the one part of a spec that changes without replacing the machines."""

    nodes: NodeBounds


class CreateGenerationResource(Struct, frozen=True, kw_only=True):
    """An earlier generation's definition made current again, as a new generation."""

    source: int


class ClaimLeaseResource(Struct, frozen=True, kw_only=True):
    owner: str
    ttl_seconds: int


class CreateTaskResource(Struct, frozen=True, kw_only=True):
    """One call to place. The arguments are exactly one of ``args_inline``, ``args_sha256`` and ``call``."""

    compute: str
    function: str
    dispatch: Dispatch
    args_inline: bytes | None = None
    args_sha256: str | None = None
    call: Call | None = None
    rank: int | None = None
    """The node to run on, for ``one`` and ``stream``. Given, the task waits for that one."""
    queue_timeout_seconds: float | None = None
    """How long each attempt may wait to start. Null takes the compute's; zero is no limit."""
    run_timeout_seconds: float | None = None
    """How long each attempt may run. Null takes the compute's; zero is no limit."""
    retry: str | None | UnsetType = UNSET
    """The blob holding this task's retry decision. Left out takes the compute's; null is no retry."""
    correlation_id: str | None = None


class CreateExecutionResource(Struct, frozen=True, kw_only=True):
    """A retry. An indeterminate outcome is retried only with ``acknowledge_duplication``, since it may run twice."""

    acknowledge_duplication: bool = False
    ranks: tuple[int, ...] | None = None


class WriteFunctionResource(Struct, frozen=True, kw_only=True):
    """A function as text: a module, and the name in it to call."""

    name: str
    source: str


class AttachExcerptResource(Struct, frozen=True, kw_only=True):
    """The text of a function already registered as a pickle, as the console shows it."""

    text: str


class CreateProviderResource(Struct, frozen=True, kw_only=True):
    """An account, and what opens it. Also what replaces one wholesale; ``credentials`` are never read back."""

    name: str
    kind: str
    credentials: dict[str, str] = field(default_factory=dict)
    config: dict[str, object] = field(default_factory=dict)


class ComputeCreatedEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.created"):
    """The definition was accepted. The compute exists and owns nothing yet."""

    compute: str


class ComputeBoundEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.bound"):
    """The compute was given an offer, a region and the markets to buy on. ``previous`` is the offer it left."""

    compute: str
    offer: str
    instance_type: str
    region: str | None
    markets: tuple[Market, ...]
    previous: str | None = None


class ComputeAdoptedEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.adopted"):
    """Another daemon bound the compute first, and this one carries on under its binding."""

    compute: str


class ComputeProvisioningEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.provisioning"):
    """Fewer machines answer than the floor asks for."""

    compute: str
    nodes_ready: int
    nodes_total: int
    generation: int


class ComputeReadyEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.ready"):
    """Enough machines answer to satisfy the floor."""

    compute: str
    nodes_ready: int
    nodes_total: int
    generation: int


class ComputeDegradedEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.degraded"):
    """A reconcile pass broke on the compute. The next pass tries again."""

    compute: str
    error: str
    code: ErrorCode = "reconcile_failed"


class ComputeGenerationCreatedEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.generation.created"):
    """A new definition was frozen: a resize, or an earlier generation brought back."""

    compute: str
    number: int


class ComputeGenerationAppliedEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.generation.applied"):
    """The machines now reflect this definition."""

    compute: str
    number: int


class ComputeLeaseClaimedEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.lease.claimed"):
    compute: str
    owner: str


class ComputeLeaseReleasedEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.lease.released"):
    compute: str


class ComputeAbandonedEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.abandoned"):
    """Nothing renewed the lease and ``delete_on_exit`` was set, so the compute is going away."""

    compute: str


class ComputeDeletingEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.deleting"):
    compute: str
    nodes_ready: int
    nodes_total: int


class ComputeDeletionFailedEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.deletion_failed"):
    """A teardown pass broke. The next one carries on giving the machines back."""

    compute: str
    error: str
    code: ErrorCode = "reconcile_failed"


class ComputeStraysTerminatedEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.strays_terminated"):
    """Machines the provider held under the compute, and that no node owned, were terminated."""

    compute: str
    machines: tuple[str, ...]


class ComputeDeletedEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.deleted"):
    """Every machine is gone and the binding is released. Nothing bills any more."""

    compute: str
    nodes_ready: int = 0
    nodes_total: int = 0


class ComputeCostEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="compute.cost"):
    """What the compute has cost so far, over how many live machines. Streamed, never logged."""

    compute: str
    cost: float
    nodes: int
    at: datetime


class NodeStateEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="node.state"):
    """A node's state moved. Its stream frame is named ``node.{state}``."""

    compute: str
    node: str
    state: NodeState
    error: str | None = None


class NodeProgressEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="node.progress"):
    """What a machine without an address yet is doing. Streamed, never logged."""

    compute: str
    node: str
    progress: str
    completion: float | None = None


class NodeConsoleEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="node.console"):
    """A line a node printed, and the task and attempt it belongs to when it belongs to one."""

    compute: str
    node: str
    content: str
    task: str | None = None
    execution: str | None = None


class NodePhaseEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="node.phase"):
    """A bootstrap phase opened, closed or broke."""

    compute: str
    node: str
    event: PhaseState
    phase: str
    at: datetime
    error: str | None = None


class NodeMetricsEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="node.metrics"):
    """One reading off one node. Streamed, never logged."""

    compute: str
    node: str
    name: str
    value: float


class TaskStateEvent(Struct, frozen=True, kw_only=True, tag_field="type", tag="task.state"):
    """A task started, is being retried, or ended. Its stream frame is named ``task.{state}``."""

    compute: str
    task: str
    state: TaskEventState
    attempt: int = 1


type Event = (
    ComputeCreatedEvent
    | ComputeBoundEvent
    | ComputeAdoptedEvent
    | ComputeProvisioningEvent
    | ComputeReadyEvent
    | ComputeDegradedEvent
    | ComputeGenerationCreatedEvent
    | ComputeGenerationAppliedEvent
    | ComputeLeaseClaimedEvent
    | ComputeLeaseReleasedEvent
    | ComputeAbandonedEvent
    | ComputeDeletingEvent
    | ComputeDeletionFailedEvent
    | ComputeStraysTerminatedEvent
    | ComputeDeletedEvent
    | ComputeCostEvent
    | NodeStateEvent
    | NodeProgressEvent
    | NodeConsoleEvent
    | NodePhaseEvent
    | NodeMetricsEvent
    | TaskStateEvent
)
"""One message of the event stream's ``data:``, and one entry of the log, told apart by ``type``."""


class LogEntryResource(Struct, frozen=True, kw_only=True):
    """One recorded event. ``sequence`` is the stream's ``id:`` for the same event; ``at`` is when the daemon recorded it."""

    sequence: int
    type: str
    """The stream frame's name, which for a node or task state is finer than ``data.type``."""
    at: datetime
    data: Event


class ChunkFrame(Struct, frozen=True, kw_only=True, tag_field="status", tag="chunk"):
    """One item the generator yielded, pickled."""

    value: bytes


class FailedFrame(Struct, frozen=True, kw_only=True, tag_field="status", tag="failed"):
    """The generator raised. Always the last frame."""

    error: str
    traceback: str


type Frame = ChunkFrame | FailedFrame
"""One message of ``GET /tasks/{id}/stream``: msgpack, after four big-endian bytes of length."""


class ResizeFrame(Struct, frozen=True, kw_only=True):
    """A terminal's new shape, as a text message up ``/computes/{compute}/shell/attach``."""

    columns: int
    rows: int


__all__ = [
    "AcceleratorResource",
    "Aggregate",
    "Allocation",
    "Architecture",
    "AttachExcerptResource",
    "BillingUnit",
    "Call",
    "ChunkFrame",
    "ClaimLeaseResource",
    "CommandResultResource",
    "ComputeAbandonedEvent",
    "ComputeAdoptedEvent",
    "ComputeBoundEvent",
    "ComputeCostEvent",
    "ComputeCreatedEvent",
    "ComputeDegradedEvent",
    "ComputeDeletedEvent",
    "ComputeDeletingEvent",
    "ComputeDeletionFailedEvent",
    "ComputeGenerationAppliedEvent",
    "ComputeGenerationCreatedEvent",
    "ComputeInclude",
    "ComputeLeaseClaimedEvent",
    "ComputeLeaseReleasedEvent",
    "ComputeProvisioningEvent",
    "ComputeReadyEvent",
    "ComputeResource",
    "ComputeSpec",
    "ComputeState",
    "ComputeStatus",
    "ComputeStraysTerminatedEvent",
    "ComputeSummary",
    "CreateComputeResource",
    "CreateExecutionResource",
    "CreateGenerationResource",
    "CreateProviderResource",
    "CreateTaskResource",
    "DeletionCause",
    "DependencyState",
    "Desired",
    "Dispatch",
    "Ending",
    "Error",
    "ErrorCode",
    "Event",
    "ExecutionResource",
    "ExecutionState",
    "Executor",
    "FailedFrame",
    "Frame",
    "FunctionResource",
    "FunctionSummary",
    "Gauge",
    "GenerationResource",
    "Image",
    "LatestTasks",
    "LeaseResource",
    "LivenessResource",
    "LogEntryResource",
    "Market",
    "MetricHistoryResource",
    "MetricSampleResource",
    "MetricSeries",
    "MetricSpec",
    "NodeBounds",
    "NodeConsoleEvent",
    "NodeDesired",
    "NodeInclude",
    "NodeMetricsEvent",
    "NodePhaseEvent",
    "NodeProgressEvent",
    "NodeResource",
    "NodeState",
    "NodeStateEvent",
    "OfferResource",
    "OfferSort",
    "Options",
    "Pace",
    "Page",
    "Phase",
    "PhaseState",
    "PipIndex",
    "PluginRef",
    "Progress",
    "ProviderKindResource",
    "ProviderRef",
    "ProviderResource",
    "ProviderSummary",
    "ReadinessResource",
    "Reading",
    "Refusal",
    "ResizeFrame",
    "Running",
    "Selection",
    "SkywardSource",
    "Spec",
    "Ssh",
    "TaskCounts",
    "TaskEventState",
    "TaskOrder",
    "TaskResource",
    "TaskState",
    "TaskStateEvent",
    "TaskSummary",
    "UpdateComputeResource",
    "Utilization",
    "Volume",
    "Worker",
    "WriteFunctionResource",
]
