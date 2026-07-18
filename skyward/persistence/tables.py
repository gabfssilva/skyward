from piccolo.columns import JSONB, Boolean, Bytea, Float, Integer, Serial, Text, Timestamptz, Varchar
from piccolo.table import Table


class ProviderRow(Table, tablename="providers"):
    """A registered provider account.

    ``name`` is the alias, ``kind`` picks the adapter. Two rows may share a kind:
    two AWS accounts, two Vast keys, two regions. That is why the compute refers
    to a provider id, not to a kind.

    ``credentials`` holds the secret in the clear for now. It never leaves this
    table: no read path selects it, and the API never returns it.
    """

    id = Varchar(primary_key=True)
    name = Varchar(unique=True, index=True)
    kind = Varchar(index=True)
    credentials = JSONB(default={})
    config = JSONB(default={})
    created_at = Timestamptz()
    offers_fetched_at = Timestamptz(null=True, default=None)
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
    observation and only the reconciler writes them. Keeping them in one row is
    what makes a reconcile one read.

    ``binding`` is the provider's per-compute state — the network it created, the
    availability zone it pinned. It is not in the API's ``Compute``: it is
    infrastructure bookkeeping, and it is here rather than in memory because the
    compute outlives by days the process that started it. The same goes for
    ``private_key``: the daemon that reconnects to these machines after a restart
    is not the daemon that provisioned them, and a key held in memory would strand
    every machine it paid for.

    ``revision`` is the optimistic-concurrency token behind ``If-Match``. Every
    write bumps it; a write that expected an older one is refused.
    """

    id = Varchar(primary_key=True)
    name = Varchar(unique=True, null=True, default=None, index=True)
    revision = Integer(default=1)
    generation = Integer(default=1)
    spec = JSONB(default="{}")

    provider_id = Varchar(null=True, default=None)
    offer_id = Varchar(null=True, default=None)
    offer = JSONB(null=True, default=None)
    binding = JSONB(default="{}")
    private_key = Text(null=True, default=None)
    markets = JSONB(default="[]")
    volumes = JSONB(default="[]")

    status_state = Varchar(index=True)
    status_observed_generation = Integer(default=0)
    status_nodes_ready = Integer(default=0)
    status_nodes_total = Integer(default=0)
    status_drift = JSONB(default="[]")
    status_error = JSONB(null=True, default=None)

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
    """

    sha256 = Varchar(primary_key=True)
    data = Bytea()
    created_at = Timestamptz()


class FunctionRow(Table, tablename="functions"):
    """What a blob of code is, so a task can name it without carrying it."""

    sha256 = Varchar(primary_key=True)
    size_bytes = Integer()
    codec = Varchar()
    name = Varchar(null=True, default=None)
    created_at = Timestamptz()


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
    retry = JSONB(default="{}")
    correlation_id = Varchar(null=True, default=None, index=True)
    submitted_at = Timestamptz()
    deadline_at = Timestamptz(null=True, default=None)
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


class EventRow(Table, tablename="events"):
    """The log the SSE stream replays from.

    ``sequence`` is the cursor a client resumes on, and it is the primary key
    because it must be monotonic and gapless in the order things were committed.
    Nothing here is garbage-collected: a cursor that was valid stays valid.
    """

    sequence = Serial(primary_key=True)
    type = Varchar(index=True)
    compute_id = Varchar(null=True, default=None, index=True)
    task_id = Varchar(null=True, default=None, index=True)
    payload = Text()
    created_at = Timestamptz()


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
    FunctionRow,
    TaskRow,
    ExecutionRow,
    EventRow,
    IdempotencyRow,
)
