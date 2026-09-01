# Watching a compute

Everything observable about a compute — machines moving through their lifecycle, bootstrap phases turning over, lines the nodes print, gauges, cost, task outcomes — is an event in one log. The pool's live console is a reader of that log, and nothing more. `callbacks=` gives your code the same seat: every event, as it happens, together with the whole compute folded into one value.

## Callbacks

A callback is a plain callable, `Callable[[Event, ComputeView], None]`. You hand any number of them to the pool, and each one sees every event:

```python
import skyward as sky

def observer(event: sky.Event, compute: sky.ComputeView) -> None:
    match event:
        case sky.NodeEvent(node=node, state="ready"):
            print(f"{node} up — {compute.nodes_ready}/{compute.nodes_total}")
        case sky.PhaseEvent(phase=phase, event="completed"):
            print(f"bootstrap: {phase} done")
        case sky.ConsoleEvent(node=node, content=line):
            forward_to_logging(node, line)
        case sky.CostEvent():
            budget.observe(compute.cost)
        case _:
            pass

with sky.Compute(provider=sky.AWS(), nodes=4, callbacks=(observer,)) as pool:
    train(data) >> pool
```

`Event` is a tagged union — `ComputeEvent`, `NodeEvent`, `PhaseEvent`, `ConsoleEvent`, `ProgressEvent`, `MetricEvent`, `CostEvent`, `TaskEvent`, `ComputeAbandoned` — so `match` is the whole dispatch mechanism. There is no registration API to learn: one callable, one `match`, and the cases you did not write fall through.

Three properties make callbacks safe to lean on:

- **The stream replays.** The event log starts at the compute's creation and the subscription reads it from the beginning, so a callback registered at construction still sees the provisioning and bootstrap it could not possibly have been early enough for. This is why `callbacks=` is a constructor argument: the pool provisions inside `__enter__`, and by the time your code runs inside the `with` block, the interesting part already happened.
- **Callbacks run off the event loop.** They are dispatched from the pool's background thread, in registration order, and never sit between your script and its tasks. A slow callback delays the next event, not the training run.
- **A callback that raises is reported and skipped.** The exception is printed to stderr and the stream carries on. A broken observer must not take the run with it.

## The view

The second argument is the fold: every event that has passed, accumulated into one immutable `ComputeView`. It answers the question an event alone cannot — a `ConsoleEvent` does not know whether the compute is `ready`; the view does.

```python
compute.state            # "requested" → "provisioning" → "ready" → ... one ComputeState word
compute.cost             # accrued so far, from the cost gauge
compute.errors           # tuple[str, ...] — what has gone wrong, most recent last
compute.nodes            # tuple[NodeView, ...]
compute.tasks            # tuple[TaskView, ...]
compute.nodes_ready      # counted from the rows
```

Each `NodeView` carries the machine's lifecycle `state`, its `address`, `price_per_hour` and `market` once the API has said them, the bootstrap checklist as `phases` (each `PhaseView` named and either underway, done, or broken — the pip installs your plugins asked for show up here), a short `metrics` history per gauge, the `tail` of what the node printed, and the `progress` line a machine reports while it is still short of an address. Each `TaskView` carries the task's `state`, the function's real name, and its timings.

The view is a `frozen` dataclass: a callback that keeps a reference keeps that moment, and two callbacks handed the same event are handed the same value. Its windows are bounded — the last 40 tail lines, 12 samples per gauge, 32 error messages — so a compute that stays up for days never grows the view with it. The full history is the event log itself, which `sky log export` hands you whole.

The view is fed from both directions. Events move what events can say; the fields only the API carries — a node's address, a task's `submitted_at`, the spec the compute was created from — are hydrated by reads the pool makes when a node or task transition lands. You never see the seam: by the time your callback runs, the view is the best of both.

## One stream, every watcher

The pool holds **one** SSE connection per compute, whoever is watching. The live console, the Rich dashboard, and all of your callbacks hang off the same consumer: the stream is read once, folded once, and each subscriber is handed the same view. Registering five callbacks costs five function calls per event, not five connections.

The connection is nursed. Every frame carries the log's global sequence, so when the transport drops — a daemon bounce, a flaky network — the client reconnects with `Last-Event-ID` and resumes exactly after the last event it delivered, never repeating one. What a gap can cost you are the published gauges (cost, metrics, progress) that fell inside it, which do not replay by design. A daemon that stays silent past the client's retry window is a different situation, and then the error is the answer.

## The raw stream

Callbacks are sugar over an iterator. When you would rather own the loop — a dashboard's render cycle, an asyncio app bridging events into its own queue — `pool.events()` hands you the decoded stream directly:

```python
with sky.Compute.attached("training") as pool:
    for event in pool.events():
        match event:
            case sky.TaskEvent(task=task, state="failed"):
                alert(task)
            case sky.ComputeEvent(state="deleted"):
                break
```

The iterator replays the log and then follows it, with the same reconnect behaviour, for as long as you keep reading — the stream has no end while the compute exists, so the consumer decides when to stop, and a `break` closes the connection. Each `events()` call is its own subscription with its own connection; the shared one belongs to the callbacks and the console.

## Building on it

`Compute.attached` takes `callbacks=` too, which is what turns this into an integration surface rather than a logging convenience: a process that did not create the compute can join it and watch it.

```python
with sky.Compute.attached("training", callbacks=(post_to_slack,)) as pool:
    ...
```

A monitoring sidecar, a Slack bot announcing failed tasks, a cost widget on an internal dashboard — each is an attach, a callback, and a `match` over the cases it cares about. The daemon owns the compute either way; watchers come and go.

## Next steps

- **[Events](reference/events.md)** — the wire-level reference: every event's payload, recorded vs published, replay and cursors
- **[Compute and task dispatch](reference/pool.md)** — the pool's full constructor surface
- **[Core Concepts](concepts.md)** — the programming model the events narrate
- **[CLI](cli.md)** — `sky monitor` and `sky log export`, the same stream from a shell
