# Persistence

A compute outlives the process that asked for it. That single requirement is what decides everything on this page: if the daemon can be restarted, killed, or replaced while your machines keep running and billing, then nothing it needs to reconnect to them may live in memory.

The store is one SQLite file. Eleven tables, no migrations framework, no second datastore.

## Where it lives

```
~/.skyward/skyward.sqlite
```

The directory is created with mode `0700`. Override it with `database=` on `Compute`, `--database` on the CLI, or `sky server start --database`. `sky config show` prints what a call actually resolved.

The file is opened in **WAL** mode, which is what lets a `sky.Compute` in one script and a daemon in another write to the same database without an exclusive lock. WAL is a journal mode, not a change feed — nothing reads the WAL itself. Statements wait up to 30 seconds for a write lock before reporting `database is locked`; that ceiling is the longest write the store ever holds, an offers refresh rewriting a provider's whole catalog.

Tables are created on connect if absent, and a table that has gained a column since the file was written gains it on open: each model is compared against `PRAGMA table_info` and what is missing is added. There is no migration step and no schema version, which holds because a column added to a table that already has rows is always nullable — a row written before the column existed has nothing to say about it.

## Intent and observation

The most important convention in the schema is that **desired state and observed state are different columns, written by different actors**.

On `computes`, `spec` is intent, and only a client writes it through `PATCH`. The `status_*` columns are observation, and only the reconciler writes them. They live in one row rather than two tables because that makes a reconcile pass a single read.

| Column | Written by | Meaning |
|---|---|---|
| `spec` | client | the definition you asked for |
| `generation` | client | which revision of that definition is current |
| `status_observed_generation` | reconciler | which one has actually been applied |
| `status_state` | `ComputeStore.apply` | `requested`, `provisioning`, `ready`, `degraded`, `deleting`, `deleted` — moved only by an event, recorded in the same transaction |
| `status_nodes_ready` / `status_nodes_total` | reconciler | how many machines answer, of how many that exist |
| `status_drift` | reconciler | what differs between intent and reality |
| `revision` | either | the optimistic-concurrency token behind `If-Match` |

`generation` against `status_observed_generation` is the progress bar. It is why there is no operation resource to poll — the gap between the two *is* the pending work.

## What only exists because the daemon can die

Four columns are the whole reason this is a database and not a cache.

`computes.binding` is the provider's per-compute state: the network it created, the availability zone it pinned, the security group it owns. It is not part of the API's `Compute` — it is infrastructure bookkeeping — and it is on disk because the compute outlives by days the process that started it.

`computes.private_key` is the SSH key for the machines. The daemon that reconnects after a restart is not the daemon that provisioned them; a key held in memory would strand every machine it paid for.

`computes.authority` is the certificate authority of the compute's own cluster. Its workers admit what it signed and refuse everything else, so a daemon that lost it could still log into the machines over SSH and not be allowed to speak to them. It is null on a compute bound before certificates existed, and those computes keep speaking plaintext: the material a running worker was handed cannot be changed underneath it.

`nodes.machine_id` is nullable and indexed, and that is a deliberate pair. A node row is written in `requested` **before** the provider is asked for a machine, which is what makes the provisioning loop idempotent: a machine being bought right now is already a row that counts, so the next pass does not buy a second one.

That leaves exactly one gap, and it is the one every payment gateway has — a crash between the provider creating the machine and the daemon recording its id. The row still says `requested`, the machine is real and billing, and nothing points at it. Those machines are found by listing the provider binding and matching against rows that claim no machine. This is also why every provider adapter method must be idempotent: the daemon can die between an API call and the commit that records it.

## The tables

| Table | Holds | Notes |
|---|---|---|
| `providers` | one registered account | `credentials` is stored here and nowhere else; no read path selects it |
| `offers` | the cached hardware catalog | a cache, not a ledger |
| `computes` | intent, observation, binding, lease | one row per compute; also the SSH key and the cluster's authority |
| `generations` | one frozen definition each | history kept, because a rollback is a generation too |
| `nodes` | one machine as the control plane knows it | `requested` → `provisioning` → `connecting` → `bootstrapping` → `ready` |
| `blobs` | content addressed by SHA-256 | code, arguments, results |
| `functions` | what a blob of code is | so a task can name it without carrying it |
| `tasks` | one call, one terminal outcome | `state` derived from executions |
| `executions` | one physical attempt | `ordinal` counts attempts, `rank` says which node |
| `events` | the log the SSE stream replays | `sequence` is monotonic and gapless |
| `idempotency` | what a key has already been used to do | key plus request fingerprint |

A few of these carry decisions worth stating outright.

**`offers` is replaced wholesale.** A refresh deletes a provider's rows and writes the new catalog, because an offer that vanished upstream must vanish here — keeping it would let a compute be planned against hardware that no longer exists. `expires_at` comes from the provider's own TTL, so a marketplace expires in minutes and a fixed fleet holds for hours. A *failed* refresh is the exception: it leaves the stale rows in place and records the error on the provider row, because stale offers beat no offers.

**`tasks.state` is derived, never written beside the executions.** It is recomputed whenever an execution changes. It is stored rather than computed on read only so that listing by state is a query instead of a scan.

**`blobs` deduplicates by construction.** The same argument broadcast to a hundred nodes is stored once, and a result read twice is not consumed the first time.

**`events` is never garbage-collected.** `sequence` is the primary key because it must be monotonic and gapless in commit order, and nothing prunes it. A cursor that was once valid stays valid — which is what lets a client reconnect and print a bootstrap it was not around to watch.

**`idempotency` stores a fingerprint, not just a key.** That is what distinguishes a replay from a collision: the same key with the same request is the caller retrying, and gets the original resource; the same key with a different request is a bug, and gets a `409`.

## What is deliberately not written down

An SSH connection is not a row. Neither is a forwarded port, nor the casty client that dials the cluster.

Everything live is held in `Runtimes` — one per compute per process — and it is **rebuilt** from the store when a daemon comes up, never recovered from it. The store says which machines exist and at which addresses; the daemon redials them.

This is what the compute's **lease** protects. A lease is a liveness signal the owning process renews in the background, stored as `lease_owner` and `lease_expires_at`. Two daemons holding SSH connections to the same machines would both believe they were the one bootstrapping it, so only the lease holder opens a runtime. Releasing a lease is not deletion — with `delete_on_exit=False`, the compute stays up and the next process attaches to it. When renewals simply stop, the reconciler may reclaim it.

## Reading the store directly

It's a SQLite file, so it opens with anything:

```console
$ sqlite3 ~/.skyward/skyward.sqlite '.tables'
$ sqlite3 ~/.skyward/skyward.sqlite \
    'select id, name, status_state, status_nodes_ready from computes'
```

Read freely. Writing behind the daemon's back is how you end up with a compute whose `revision` no client expects and machines nobody reclaims — use the API or the CLI, which is what they are for.

To back it up, stop the daemon and copy the file together with its `-wal` and `-shm` siblings, or use `sqlite3 ... '.backup'` while it runs.

## Next steps

- **[HTTP API](http-api.md)** — the interface these rows are served through
- **[Architecture](architecture.md)** — the components that read and write them
- **[Reconciliation and provisioning](provision-controllers.md)** — how intent becomes machines
- **[Events](reference/events.md)** — the stream the event log feeds
