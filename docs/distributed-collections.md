# Distributed Collections

When compute functions run across multiple nodes, they sometimes need to share state. A training loop might track global progress. A batch processor might deduplicate work across workers. A distributed pipeline might need a synchronization point where all nodes wait before proceeding. A checkpoint routine might need mutual exclusion so two nodes don't write conflicting state simultaneously.

Skyward provides distributed data structures for these cases. They look like their Python counterparts — `dict`, `set`, `counter`, `queue`, `barrier`, `lock` — but they're backed by Casty's distributed actor system and replicated across the [cluster](architecture.md). Any node can read and write to them, and the cluster handles synchronization automatically.

Collections are created lazily by name. Calling `sky.dict("cache")` on any node returns a proxy to the same shared dict — if the collection doesn't exist yet, the cluster creates it; if it already exists, you get a reference to the existing one. Inside `@sky.compute` functions, use the module-level shortcuts (`sky.dict`, `sky.counter`, `sky.set`, etc.) which resolve the active pool automatically through a `ContextVar`. You can also access them via the pool directly: `pool.dict("cache")`, `pool.counter("steps")`, etc.

## Dict

The distributed dict is a shared key-value store across all workers. The most common use case is a distributed cache: if one node computes an expensive result, other nodes can read it instead of recomputing.

```python
@sky.compute
def process_with_cache(items: list[str]) -> dict:
    cache = sky.dict("embeddings")

    for item in sky.shard(items):
        if item in cache:
            result = cache[item]
        else:
            result = compute_embedding(item)
            cache[item] = result

    return {"processed": len(items)}
```

The dict supports standard Python syntax: `cache[key]`, `cache[key] = value`, `del cache[key]`, `key in cache`, `cache.get(key, default)`, `cache.update(...)`, and `cache.pop(key, default)`.

Internally, each key is managed by a separate actor (entity-per-key), which means reads and writes to different keys don't contend with each other. This design makes the dict highly concurrent — hundreds of nodes can read and write to different keys simultaneously without coordination overhead. The trade-off is that enumeration methods — `keys()`, `values()`, `items()`, `len()` — are not available, because there's no single actor that knows about all keys. Think of it as a distributed cache with get/put/delete/contains semantics, not a full `dict` replacement.

## Counter

The distributed counter is an atomic integer shared across all workers. Every node can increment and decrement it, and all operations are serialized through the counter's backing actor — no lost updates, no race conditions.

```python
@sky.compute
def train_step(batch) -> dict:
    progress = sky.counter("global_steps")
    progress.increment()
    return {"step": int(progress)}
```

The counter supports `progress.value` (current value), `progress.increment(n=1)`, `progress.decrement(n=1)`, `progress.reset(value=0)`, and `int(progress)` for casting. It's useful for tracking global progress across workers, counting processed items, or implementing simple distributed coordination where all you need is a shared integer.

## Set

The distributed set tracks unique values across workers. The typical use case is deduplication — ensuring that a batch isn't processed twice even when multiple nodes receive overlapping work.

```python
@sky.compute
def deduplicate(batch_id: int) -> str:
    seen = sky.set("processed_batches")
    key = f"batch:{batch_id}"

    if key in seen:
        return "skipped"

    seen.add(key)
    return "processed"
```

The set supports `value in s`, `len(s)`, `s.add(value)`, and `s.discard(value)`. Unlike the dict, the set does support `len()` because it's backed by a single replicated actor rather than entity-per-key.

## Queue

The distributed queue is a FIFO work queue for dynamic task distribution. Unlike the static partitioning that `shard()` provides — where each node gets a predetermined slice of the data — a queue lets workers pull tasks at their own pace. Fast workers process more items, slow workers process fewer, and the overall throughput adapts to heterogeneous performance.

```python
@sky.compute
def producer(tasks: list[int]):
    queue = sky.queue("work")
    info = sky.instance_info()
    if info.is_head:
        for task in tasks:
            queue.put(task)

@sky.compute
def worker() -> list:
    queue = sky.queue("work")
    results = []
    while True:
        task = queue.get(timeout=0.5)
        if task is None:
            break
        results.append(task * 2)
    return results
```

The queue supports `queue.put(value)`, `queue.get(timeout=None)` (returns `None` on timeout), `queue.empty()`, and `len(queue)`. The producer-consumer pattern shown above is a common way to implement dynamic load balancing: the head node fills the queue, all workers drain it at their own pace, and the work naturally distributes based on each worker's processing speed.

## Barrier

The distributed barrier is a synchronization point where all workers must arrive before any can proceed. This is useful when your distributed computation has phases that must complete globally before the next phase begins — for example, ensuring all nodes have finished an epoch before aggregating results, or waiting for all nodes to load a model before starting inference.

```python
@sky.compute
def synchronized_epoch(epoch: int) -> dict:
    info = sky.instance_info()
    sync = sky.barrier("epoch_sync", n=info.total_nodes)

    loss = train_one_epoch(epoch)
    sync.wait()  # blocks until all n workers arrive

    return {"node": info.node, "loss": loss}
```

The barrier is created with a participant count `n`. When `n` workers have called `wait()`, all of them are released simultaneously. If any worker fails to arrive — because of an error, a timeout, or a preempted spot instance — the others will block until their task times out. Barriers work best when all nodes are expected to reach the synchronization point reliably.

## Lock

The distributed lock provides mutual exclusion across the cluster. When multiple nodes need to update shared state atomically — like writing the best checkpoint so far, or coordinating access to an external resource — a lock ensures only one node enters the critical section at a time.

```python
@sky.compute
def safe_checkpoint(step: int) -> bool:
    lock = sky.lock("checkpoint_lock")
    state = sky.dict("checkpoint")

    with lock:
        current_best = state.get("best_loss", float("inf"))
        my_loss = evaluate(step)
        if my_loss < current_best:
            state["best_loss"] = my_loss
            state["best_step"] = step
            return True
    return False
```

The lock supports `lock.acquire()`, `lock.release()`, and the context manager protocol (`with lock:`). Casty's distributed locking ensures that only one holder exists across the cluster at any time — if node 0 holds the lock, node 1's `acquire()` blocks until node 0 releases it, regardless of which physical machine each node runs on.

## Consistency

Collections support two consistency levels, configured at creation time:

```python
# Default: eventual consistency (faster)
cache = sky.dict("cache")

# Strong consistency (slower, always up-to-date)
cache = sky.dict("cache", consistency="strong")
```

With **eventual consistency** (the default), reads are fast but may see slightly stale values. A write on node 0 might not be visible on node 1 for a brief window while replication propagates. This is sufficient for most use cases — caches, progress counters, deduplication sets — where reading a slightly outdated value doesn't affect correctness.

With **strong consistency**, every read returns the latest written value. This is slower because it requires coordination with the actor managing the data, but it's necessary when correctness depends on seeing the most recent state — for example, when using a lock and a dict together to coordinate checkpoint writes, you want the dict reads inside the critical section to be strongly consistent.

## Async Interface

All collections expose async methods for use in async contexts. The naming convention adds `_async` to each operation:

```python
await cache.get_async(key)
await cache.set_async(key, value)
await counter.increment_async()
await queue.put_async(value)
await lock.acquire_async()
```

The sync interface (the default, used in most `@sky.compute` functions) blocks the calling thread while waiting for the actor response. The async interface returns awaitables, which is useful if you're writing custom async logic inside a worker or integrating with an existing async codebase.

## Next Steps

- [Distributed Training](distributed-training.md) — Multi-node training with shared state across workers
- [Clustering](architecture.md) — How the Casty actor cluster powers distributed collections
- [API Reference](reference/distributed.md) — Full API documentation
