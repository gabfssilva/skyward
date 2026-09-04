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

Every payload is a flat JSON object carrying a `type` tag. All of them carry `compute`; the ones about a node carry `node`, and the ones about a task carry `task`.

For a compute the `event:` name and the `type` tag are the same word, one per fact. For a node or a task the tag is coarser than the name — ten node states share one payload shape, and so do four task outcomes:

| `event:` | `type` |
|---|---|
| `compute.{fact}` — see below | the same name |
| `node.{state}` | `node.state` |
| `node.console` | `node.console` |
| `node.phase` | `node.phase` |
| `node.metrics` | `node.metrics` |
| `task.started`, `task.succeeded`, `task.failed`, `task.indeterminate` | `task.state` |

Filter on the name; decode on the tag. The name is what `types` matches, and the tag is what makes a payload readable once it is out of the frame that carried it — written to a file, replayed by `sky log export`, or handed to a client that never saw the SSE envelope.

### Compute

Every change of a compute's `status.state` is one of the events below — the daemon has no way to move the state without recording the event that moved it, in the same transaction. Five of them move the state; the rest are facts that move nothing. An event that leads to the state the compute is already in is not recorded again, and neither is a fact whose every field the compute's status already shows: the reconciler says `ready` to a ready compute on every pass, and the provider refuses to release with the same words tick after tick, and neither is news.

| Event | Moves to | When |
|-------|----------|------|
| `compute.created` | — | The definition was accepted; the compute exists in `requested` |
| `compute.bound` | — | An offer was chosen and the binding landed: `offer`, `instance_type`, `region`, `markets`; `previous` when a region refused and the compute followed the next offer |
| `compute.adopted` | — | Another daemon bound the compute first and this one is carrying on under its binding |
| `compute.provisioning` | `provisioning` | Fewer nodes answer than the floor asks for: `nodes_ready`, `nodes_total`, `generation`. Said again on the way back down |
| `compute.ready` | `ready` | Enough nodes are ready to satisfy the floor: `nodes_ready`, `nodes_total`, `generation` |
| `compute.degraded` | `degraded` | A reconcile pass broke on a compute on its way up; carries `error` and `code`. Once per transition, not once per failing tick |
| `compute.generation.created` | — | A new definition was frozen: `number` |
| `compute.generation.applied` | — | The machines reflect that definition: `number` |
| `compute.lease.claimed` | — | A process took ownership: `owner`. Renewals are not said |
| `compute.lease.released` | — | The owner let go without asking for anything to be destroyed |
| `compute.abandoned` | — | Nothing has held the compute and `delete_on_exit` was set, so it is being deleted |
| `compute.deleting` | `deleting` | Destruction was asked for: `nodes_ready`, `nodes_total` |
| `compute.deletion_failed` | — | A pass broke on a compute on its way out; carries `error` and `code` (`release_pending` when the provider would not release the binding). Once per distinct failure, not once per failing tick |
| `compute.strays_terminated` | — | Machines under the compute that no row owns were terminated: `machines` |
| `compute.deleted` | `deleted` | The machines are gone and the binding is released: `nodes_ready`, `nodes_total`, both zero |
| `compute.cost` | — | *Published.* Accrued cost so far: `cost`, `nodes`, `at` |

The state machine is `requested → provisioning ⇄ ready`, either of which may fall to `degraded` and come back; anything before deletion may go to `deleting`, and only `deleting` goes to `deleted`. The table is `skyward.shared.lifecycle.COMPUTE`, and it is the same table the client folds the stream by.

### Node

`node.{state}` is recorded whenever a node's lifecycle reports a new state, where `state` is one of the node states: `requested`, `provisioning`, `connecting`, `bootstrapping`, `ready`, `draining`, `lost`, `deleting`, `deleted`, `failed`. The payload carries `node` and `state`, and `error` when the transition came with one. `state` repeats what the event name already said, which is the point: a payload that has been written down or exported has to say what it is without the frame that carried it.

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

Each carries `compute`, `task`, and the `state` its name says.

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
http --stream GET localhost:17590/v1/events compute==<id> types==task.failed
```

## What is not here

There is no `logs` resource. Task output and node bootstrap output are events in this stream, and nowhere else.

Names that are not in the table above and that you may see in the source — `compute.dispatch`, `compute.changed`, `task.changed`, `node.observed`, `node.connect`, `node.requested` — are the daemon's internal wakeups, not stream events. They exist to make a reconcile pass happen and are never recorded or published.
