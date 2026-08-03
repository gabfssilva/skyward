# Distributed collections

Distributed collections let functions running in the same compute share state. They are backed by Casty's replicated collections and acknowledge writes according to the selected consistency level. Values are serialized by Skyward; keys must be hashable.

The constructors are available inside a running `@sky.function` call:

```python
import skyward as sky


@sky.function
def process(items: list[str]) -> int:
    seen = sky.set("seen")
    for item in sky.shard(items):
        seen.add(item)
    return len(seen)
```

Multi-node replication requires cluster networking. A standalone compute can run independent tasks, but it cannot provide shared state between workers. See [Standalone Workers](guides/standalone-workers.md).

Collections are named. Calling the same constructor with the same name returns a proxy to the same collection for the lifetime of the compute. Replication uses up to three replicas, limited by the number of nodes.

## Dict

`sky.dict(name)` provides shared key-value operations:

```python
@sky.function
def process_with_cache(items: list[str]) -> int:
    cache = sky.dict("embeddings")
    for item in sky.shard(items):
        value = cache.get(item)
        if value is None:
            cache[item] = compute_embedding(item)
    return len(cache)
```

Supported operations are `cache[key]`, assignment, `key in cache`, `get`, `pop`, `items`, `len`, and `clear`. `items()` returns a list of key-value pairs.

## Counter

`sky.counter(name)` is an atomic integer with `add`, `get`, and `reset`:

```python
@sky.function
def train_step() -> int:
    steps = sky.counter("global_steps")
    steps.add()
    return steps.get()
```

`add(delta=1)` changes the value, `get()` reads it, and `reset()` returns it to zero.

## Set

`sky.set(name)` stores unique values:

```python
@sky.function
def deduplicate(batch_id: int) -> str:
    seen = sky.set("processed_batches")
    key = f"batch:{batch_id}"
    if key in seen:
        return "skipped"
    seen.add(key)
    return "processed"
```

Supported operations are membership, `add`, `remove`, `items`, `len`, and `clear`.

## Queue

`sky.queue(name)` is a FIFO queue for dynamic work distribution. `poll()` is non-blocking and returns `None` when the queue is empty:

```python
@sky.function
def producer(tasks: list[int]) -> None:
    work = sky.queue("work")
    if sky.instance_info().is_head:
        for task in tasks:
            work.offer(task)


@sky.function
def worker() -> list[int]:
    work = sky.queue("work")
    results = []
    while (task := work.poll()) is not None:
        results.append(task * 2)
    return results
```

Supported operations are `offer`, `poll`, `len`, and `clear`.

## Barrier

`sky.barrier(name, parties)` creates a synchronization point. Each participant calls `wait()`:

```python
@sky.function
def synchronized_epoch(epoch: int) -> dict:
    info = sky.instance_info()
    sync = sky.barrier("epoch_sync", parties=info.nodes)

    loss = train_one_epoch(epoch)
    sync.wait()
    return {"rank": info.rank, "loss": loss}
```

The barrier releases all participants once `parties` calls have arrived. `wait(timeout=...)` can bound the wait.

## Lock

`sky.lock(name)` provides a lease-backed critical section:

```python
@sky.function
def safe_checkpoint(step: int) -> bool:
    state = sky.dict("checkpoint")
    with sky.lock("checkpoint_lock"):
        current_best = state.get("best_loss", float("inf"))
        my_loss = evaluate(step)
        if my_loss < current_best:
            state["best_loss"] = my_loss
            state["best_step"] = step
            return True
    return False
```

The lock has a configurable `ttl` and optional acquisition `timeout`. If the holder disappears, the lease expires.

## Registry

`sky.registry(name)` stores named values with explicit registry operations:

```python
@sky.function
def publish(step: int, model: object) -> None:
    models = sky.registry("checkpoints")
    models.register(step, model)


@sky.function
def load(step: int) -> object | None:
    return sky.registry("checkpoints").lookup(step)
```

Use `register`, `lookup`, `unregister`, and `list`. Registry keys follow the same hashability rule as dict keys.

## Consistency

Collections use strong consistency by default. Pass `consistency="eventual"` when a lower acknowledgement cost is acceptable:

```python
strong = sky.dict("source_of_truth")
eventual = sky.dict("cache", consistency="eventual")
```

Strong writes use the replicated collection's default quorum. Eventual consistency acknowledges one replica and may return a stale read while replication catches up.

## Next steps

- [Distributed Training](distributed-training.md) — multi-node training and topology
- [Clustering](architecture.md) — how the cluster powers shared state
- [API Reference](reference/distributed.md) — the public collection constructors
