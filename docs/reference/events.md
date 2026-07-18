# Events

Everything observable about a compute is an event in one log. Node stdout, bootstrap phases, node state changes, task outcomes — all of it goes to the same place, and there is no second source of truth to reconcile against. `sky log`, the SDK's live console, and any HTTP client read the same stream.

## The stream

```
GET /v1/events
```

Server-Sent Events. Each message carries the event name as its `event:` field, a JSON object as `data:`, and the log's **global sequence** as `id:`.

Query parameters, all optional and all AND-ed:

| Parameter | Effect |
|-----------|--------|
| `compute` | Only events belonging to this compute id |
| `task` | Only events belonging to this task id |
| `types` | Only these event names — repeatable |

The `Last-Event-ID` header replays from that sequence onward.

### Replay

A subscriber replays from the table and is then attached to the live feed, and it is subscribed *before* the replay reads — so an event committed in between is buffered rather than lost, and dropped on the way out if the replay already carried it. The snapshot and the cursor are captured in the same logical order, so there is no gap between reading state and subscribing.

There is no automatic event GC, so any valid cursor stays resumable. That is what lets a client print a bootstrap it was not around to watch.

### Slow consumers

A slow consumer never blocks a commit. Its queue fills, the daemon closes the connection, and the client reconnects from its last id.

## Recorded vs published

Two kinds of event ride the same feed.

**Recorded** events are written to the event table and get a sequence. They replay. Everything below is recorded except where noted.

**Published** events are said once to whoever is listening and kept nowhere. A gauge sampled every couple of seconds has no replay value, and the event table has no GC to save it from one; a late subscriber simply misses the samples it was not there for. Published events carry the last sequence seen rather than one of their own.

## The events

Every payload is a flat JSON object. All of them carry `compute`; the ones about a node carry `node`, and the ones about a task carry `task`.

### Compute

| Event | When |
|-------|------|
| `compute.ready` | Enough nodes are ready to satisfy the lower bound, and it was not ready before |
| `compute.degraded` | A capability mismatch was hit; carries `error` |
| `compute.deleted` | The compute's machines are gone |
| `compute.abandoned` | Nothing has held the compute and `delete_on_exit` was set, so it is being deleted |
| `compute.cost` | *Published.* Accrued cost so far: `cost`, `nodes`, `at` |

### Node

`node.{state}` is recorded whenever a node's lifecycle reports a new state, where `state` is one of the node states: `requested`, `provisioning`, `connecting`, `bootstrapping`, `ready`, `draining`, `lost`, `deleting`, `deleted`, `failed`. The payload carries `node`, and `error` when the transition came with one.

Three more come from the node itself rather than from the reconciler:

| Event | Payload | Notes |
|-------|---------|-------|
| `node.console` | `node`, `content`, and `task` when the line belongs to one | A node's stdout/stderr. Recorded |
| `node.phase` | `node`, `event`, `phase`, `at`, `error` | A bootstrap phase turning over, so a late subscriber replays the checklist. Recorded |
| `node.metrics` | `node`, `name`, `value` | *Published.* One gauge reading |

Console output goes straight to the log rather than through the daemon's internal wakeup bus: that bus coalesces identical payloads, and two identical lines of a user's `print` are two lines, not a duplicate.

### Task

| Event | When |
|-------|------|
| `task.started` | The task was placed on a node and began |
| `task.succeeded` | It returned |
| `task.failed` | It raised, timed out, or its node went away |
| `task.indeterminate` | Its outcome cannot be established — the node is gone and the result never arrived |

Each carries `compute` and `task`.

## Reading it

From the CLI:

```bash
sky log research -f -o json
```

From the SDK's client, which yields `(event_name, payload_bytes)`:

```python
async for name, payload in client.events(compute_id):
    ...
```

Or directly, with any SSE client:

```bash
http --stream GET localhost:7590/v1/events compute==<id> types==task.failed
```

## What is not here

There is no `logs` resource. Task output and node bootstrap output are events in this stream, and nowhere else.

Names that are not in the table above and that you may see in the source — `compute.dispatch`, `compute.changed`, `task.changed`, `node.observed`, `node.connect`, `node.requested` — are the daemon's internal wakeups, not stream events. They exist to make a reconcile pass happen and are never recorded or published.
