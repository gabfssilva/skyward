from piccolo.columns import JSONB, Boolean, Bytea, Float, Integer, Serial, Text, Timestamptz, Varchar
from piccolo.table import Table


class ProviderRow(Table, tablename="providers"):
    """A registered provider account.

    ``name`` is the alias, ``kind`` picks the adapter. Two rows may share a kind:
    two AWS accounts, two Vast keys, two regions. That is why the compute refers
    to a provider id, not to a kind.

    ``credentials`` holds the secret in the clear for now. It never leaves this
    table: no read path selects it, and the API never returns it.

    ``offers_fetched_at`` is when the catalog was last fetched successfully;
    ``offers_attempted_at`` is when it was last asked for, successfully or not.
    """

    id = Varchar(primary_key=True)
    name = Varchar(unique=True, index=True)
    kind = Varchar(index=True)
    credentials = JSONB(default={})
    config = JSONB(default={})
    created_at = Timestamptz()
    offers_fetched_at = Timestamptz(null=True, default=None)
    offers_attempted_at = Timestamptz(null=True, default=None)
    last_error = Varchar(null=True, default=None)


class OfferRow(Table, tablename="offers"):
    """A cached offer.

    A cache, not a ledger: a refresh replaces a provider's rows wholesale,
    because an offer that vanished from the catalog must vanish here — keeping
    it would let a compute be planned against hardware that no longer exists.

    ``expires_at`` comes from the provider's own TTL. A marketplace expires in
    minutes; a fixed fleet can hold for hours.
    """

    id = Varchar(primary_key=True)
    offer_id = Varchar(index=True)
    provider_id = Varchar(index=True)
    provider_name = Varchar()
    kind = Varchar(index=True)
    instance_type = Varchar()
    accelerator = Varchar(null=True, default=None, index=True)
    accelerator_count = Integer(default=0, index=True)
    vram = Float(null=True, default=None, index=True)
    cpus = Integer(default=0)
    memory_gb = Float(default=0.0)
    disk_gb = Float(null=True, default=None)
    architecture = Varchar(null=True, default=None)
    region = Varchar(null=True, default=None)
    spot_price = Float(null=True, default=None)
    on_demand_price = Float(null=True, default=None)
    price = Float(null=True, default=None, index=True)
    billing_unit = Varchar(default="hour")
    available = Integer(null=True, default=None)
    specific = JSONB(default={})
    fetched_at = Timestamptz()
    expires_at = Timestamptz(index=True)


class ComputeRow(Table, tablename="computes"):
    """What was asked for, and what has been observed of it.

    ``spec`` is intent and only the user writes it; the ``status_*`` columns are
    observation and only ``ComputeStore.apply`` writes them, as the projection of
    the event that observed it. Keeping them in one row is what makes a reconcile
    one read.

    ``binding`` is the provider's per-compute state — the network it created, the
    availability zone it pinned. It is not in the API's ``Compute``: it is
    infrastructure bookkeeping, and it is here rather than in memory because the
    compute outlives by days the process that started it. The same goes for
    ``private_key``: the daemon that reconnects to these machines after a restart
    is not the daemon that provisioned them, and a key held in memory would strand
    every machine it paid for. ``authority`` is the certificate authority of the
    compute's own cluster, and is here for that reason twice over: the workers only
    admit what it signed, so a daemon that lost it can log into the machines and
    still not be allowed to speak to them.

    ``revision`` is the optimistic-concurrency token behind ``If-Match``. Every
    write bumps it; a write that expected an older one is refused.

    ``placement_*`` is why the last launch was refused and when buying is tried
    again — the one thing about a compute stuck in ``provisioning`` that no other
    column says.

    ``deletion_cause`` and ``deleted_at`` are how the compute ended. The cause is
    written with the intent to delete, by whoever had it — a client, or the
    reconciler letting go of a lease nobody renews — and the moment by the move
    into ``deleted``, once, the way a node's ``terminated_at`` is.
    """

    id = Varchar(primary_key=True)
    name = Varchar(null=True, default=None, index=True)
    revision = Integer(default=1)
    generation = Integer(default=1)
    spec = JSONB(default="{}")

    provider_id = Varchar(null=True, default=None)
    offer_id = Varchar(null=True, default=None)
    offer = JSONB(null=True, default=None)
    binding = JSONB(default="{}")
    private_key = Text(null=True, default=None)
    authority = JSONB(null=True, default=None)
    markets = JSONB(default="[]")
    volumes = JSONB(default="[]")

    status_state = Varchar(index=True)
    status_observed_generation = Integer(default=0)
    status_error = JSONB(null=True, default=None)
    placement_reason = Varchar(null=True, default=None)
    placement_retry_at = Timestamptz(null=True, default=None)
    deletion_cause = Varchar(null=True, default=None)
    deleted_at = Timestamptz(null=True, default=None)

    lease_owner = Varchar(null=True, default=None)
    lease_expires_at = Timestamptz(null=True, default=None)

    created_at = Timestamptz()


class GenerationRow(Table, tablename="generations"):
    """One definition of a compute, frozen.

    A new generation is how infrastructure gets replaced: same compute id, new
    definition, the old one destroyed. The history is kept because a rollback is
    a generation too — it names the one it goes back to.
    """

    id = Varchar(primary_key=True)
    compute_id = Varchar(index=True)
    number = Integer()
    spec = JSONB(default="{}")
    hash = Varchar()
    applied = Boolean(default=False)
    created_at = Timestamptz()


class NodeRow(Table, tablename="nodes"):
    """One machine, as the control plane knows it.

    The row exists before the machine does. It is written in ``requested`` with no
    ``machine_id`` at all, and that is what makes the loop idempotent: a node being
    provisioned right now is a row that already counts, so the next pass does not
    buy a second one. The provider is asked to create a machine *for this row*, and
    the id it answers with is written back here.

    Which leaves exactly one gap, and it is the one every payment gateway has: a
    crash between the provider creating the machine and us recording its id. The
    row is still ``requested``, the machine is real, and nothing points at it. That
    machine is found by listing the binding and matched against the rows that claim
    no machine — which is what ``machine_id`` being indexed and nullable is for.
    """

    id = Varchar(primary_key=True)
    compute_id = Varchar(index=True)
    machine_id = Varchar(index=True, null=True, default=None)
    generation = Integer()
    rank = Integer()
    revision = Integer(default=1)
    desired = Varchar(default="present")
    state = Varchar(index=True)
    provider_binding = JSONB(default="{}")
    address = Varchar(null=True, default=None)
    accelerator = Varchar(null=True, default=None)
    price_per_hour = Float(null=True, default=None)
    market = Varchar(null=True, default=None)
    billing_unit = Varchar(null=True, default=None)
    last_error = JSONB(null=True, default=None)
    created_at = Timestamptz()
    launched_at = Timestamptz(null=True, default=None)
    terminated_at = Timestamptz(null=True, default=None)


class BlobRow(Table, tablename="blobs"):
    """Content, addressed by its hash.

    Functions, arguments and results all live here. The same argument sent to a
    hundred nodes is stored once, and a result read twice is not consumed the
    first time.

    The content itself is kept as chunks, so what blobs share is stored once: a row
    is the ordered 32-byte digests of its chunks. Its name is still the hash of the
    whole, the bytes as they were uploaded.
    """

    sha256 = Varchar(primary_key=True)
    size_bytes = Integer()
    chunks = Bytea()
    created_at = Timestamptz()


class ChunkRow(Table, tablename="chunks"):
    """A piece of blob content, named by the sha256 of its bytes and stored zlib-compressed."""

    sha256 = Varchar(primary_key=True)
    data = Bytea()


class FunctionRow(Table, tablename="functions"):
    """What a blob of code is, so a task can name it without carrying it.

    ``source`` is the text a function written in the console was built from;
    ``excerpt`` is the text the SDK read off the file of one it pickled. The blob is
    still what runs either way.

    ``lineage`` and ``shape`` are read off the payload when it arrives, without
    unpickling it: one function across all its uploads, and its code apart from
    where it lives and what it captured. The version is not stored — it is counted
    from the shapes, in order, each time it is read.
    """

    sha256 = Varchar(primary_key=True)
    size_bytes = Integer()
    codec = Varchar()
    name = Varchar(null=True, default=None)
    source = Text(null=True, default=None)
    excerpt = Text(null=True, default=None)
    created_at = Timestamptz()
    lineage = Varchar(index=True)
    qualname = Varchar(null=True, default=None)
    origin = Varchar(null=True, default=None)
    shape = Varchar(null=True, default=None)


class TaskRow(Table, tablename="tasks"):
    """One call: function plus arguments, one terminal outcome.

    ``state`` is derived from the executions and recomputed whenever one of them
    changes — never written beside them. It is stored rather than computed on read
    only so that listing by state is a query and not a scan.
    """

    id = Varchar(primary_key=True)
    compute_id = Varchar(index=True)
    generation = Integer()
    function = Varchar(index=True)
    args_sha256 = Varchar()
    dispatch = Varchar()
    state = Varchar(index=True)
    rank = Integer(null=True, default=None)
    """The node the caller named, or null for any of them. Not the execution's rank:
    that one says which node of a broadcast an attempt is, and every task has it."""
    decision = Varchar(null=True, default=None)
    """The digest of the task's retry decision. Not ``retry``: that column exists in
    older files as a ``NOT NULL`` object nobody read, and a file cannot be told to
    let it go — so it is left where it is, filled by its own default, and this one
    is the column that is read."""
    correlation_id = Varchar(null=True, default=None, index=True)
    submitted_at = Timestamptz()
    queue_timeout = Float(null=True, default=None)
    """Seconds each attempt may wait to start, settled at admission; null is no limit."""
    run_timeout = Float(null=True, default=None)
    """Seconds each attempt may run once started, settled at admission; null is no limit."""
    result_sha256 = Varchar(null=True, default=None)
    finished_at = Timestamptz(null=True, default=None)


class ExecutionRow(Table, tablename="executions"):
    """One physical attempt at a task.

    A retry is another of these, never another task — which is what lets the
    caller's handle survive it. ``ordinal`` counts the attempts; ``rank`` says
    which node of a broadcast this one is.
    """

    id = Varchar(primary_key=True)
    task_id = Varchar(index=True)
    rank = Integer(default=0)
    ordinal = Integer()
    state = Varchar(index=True)
    node_id = Varchar(null=True, default=None)
    retry_of = Varchar(null=True, default=None)
    result_sha256 = Varchar(null=True, default=None)
    error = JSONB(null=True, default=None)
    started_at = Timestamptz(null=True, default=None)
    finished_at = Timestamptz(null=True, default=None)
    deadline_at = Timestamptz(null=True, default=None)
    """When the phase the attempt is in runs out: its wait until it starts, its run after."""
    stopping = Boolean(default=False)
    """Answered for while a machine still runs it: the slot is the worker's until it lets go."""


class EventRow(Table, tablename="events"):
    """The log the SSE stream replays from.

    ``sequence`` is the cursor a client resumes on, and it is the primary key
    because it must be monotonic and gapless in the order things were committed.
    Nothing here is garbage-collected: a cursor that was valid stays valid.

    ``compute_id``, ``node_id`` and ``task_id`` are what a reader narrows by. The
    table holds every line every node ever printed, so one node's output has to be
    a filter on an index rather than a walk through everybody else's.
    """

    sequence = Serial(primary_key=True)
    type = Varchar(index=True)
    compute_id = Varchar(null=True, default=None, index=True)
    node_id = Varchar(null=True, default=None, index=True)
    task_id = Varchar(null=True, default=None, index=True)
    payload = Text()
    created_at = Timestamptz()


class MetricSampleRow(Table, tablename="metric_samples"):
    """A node's readings not yet folded into a chunk, one row each.

    Only the recent past lives here: compaction moves every window that has closed
    into :class:`MetricChunkRow` and deletes its rows, so the table holds the open
    window, the grace behind it, and whatever arrived too late for either.

    ``id`` is the order samples were recorded in, which is not the order they were
    measured in — a node whose link dropped delivers its backlog late. That is what
    a reader following along keeps as its cursor. ``at`` is milliseconds since the
    epoch on the node's clock. A sample is unique by where it came from and when,
    so reading a node's log again does not record it twice.
    """

    id = Serial(primary_key=True)
    compute_id = Varchar()
    node_id = Varchar()
    name = Varchar()
    at = Integer()
    value = Float()


class MetricChunkRow(Table, tablename="metric_chunks"):
    """One node's samples over one closed window, compressed into a single blob.

    ``since`` and ``until`` bound the window in milliseconds. ``last_id`` is the
    highest :class:`MetricSampleRow` id folded in, which is how a reader holding a
    cursor learns that rows it had not read yet were compacted away from under it.
    """

    id = Serial(primary_key=True)
    compute_id = Varchar()
    node_id = Varchar()
    since = Integer()
    until = Integer()
    last_id = Integer()
    data = Bytea()


class IdempotencyRow(Table, tablename="idempotency"):
    """What a key has already been used to do.

    The fingerprint is what makes a replay distinguishable from a collision: the
    same key with the same request is the caller retrying and gets the original
    resource back; the same key with a different request is a bug, and gets a
    409 rather than a second resource.
    """

    key = Varchar(primary_key=True)
    fingerprint = Varchar()
    resource_id = Varchar()
    created_at = Timestamptz()


TABLES = (
    ProviderRow,
    OfferRow,
    ComputeRow,
    GenerationRow,
    NodeRow,
    BlobRow,
    ChunkRow,
    FunctionRow,
    TaskRow,
    ExecutionRow,
    EventRow,
    MetricSampleRow,
    MetricChunkRow,
    IdempotencyRow,
)
