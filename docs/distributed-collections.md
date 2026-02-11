# Distributed Collections

Skyward provides distributed data structures that are shared across all nodes in a pool. They work like their Python counterparts but are automatically replicated and synchronized across the cluster.

## Available Collections

| Collection | Create | Python Equivalent |
|------------|--------|-------------------|
| `pool.dict(name)` | Key-value store | `dict` |
| `pool.set(name)` | Unique value store | `set` |
| `pool.counter(name)` | Atomic counter | `int` |
| `pool.queue(name)` | FIFO queue | `queue.Queue` |
| `pool.barrier(name, n)` | Synchronization point | `threading.Barrier` |
| `pool.lock(name)` | Mutual exclusion | `threading.Lock` |

Collections are created lazily by name — calling `pool.dict("cache")` on any node returns the same shared dict. Inside `@sky.compute` functions, use the module-level shortcuts (`sky.dict`, `sky.counter`, etc.) which reference the active pool automatically.

## Dict

Shared key-value store across all workers. Supports standard Python dict syntax.

```python
import skyward as sky

@sky.compute
def process_with_cache(items: list[str]) -> dict:
    cache = sky.dict("embeddings")
    info = sky.instance_info()

    for item in sky.shard(items):
        if item in cache:
            result = cache[item]
        else:
            result = compute_embedding(item)
            cache[item] = result

    return {"node": info.node, "processed": len(items)}

with sky.ComputePool(provider=sky.AWS(), nodes=4) as pool:
    results = process_with_cache(items) @ pool
```

**API:**

- `cache[key]` / `cache[key] = value` / `del cache[key]`
- `key in cache`
- `cache.get(key, default=None)`
- `cache.update({...})`
- `cache.pop(key, default=None)`

!!! note
    Dict uses entity-per-key internally, so enumeration methods (`keys()`, `values()`, `items()`, `len()`) are not available. Use it as a distributed cache with get/put/delete/contains semantics.

## Counter

Atomic distributed counter with increment/decrement.

```python
@sky.compute
def train_step(batch) -> dict:
    progress = sky.counter("global_steps")
    progress.increment()
    return {"step": int(progress)}
```

**API:**

- `progress.value` — current value
- `progress.increment(n=1)` / `progress.decrement(n=1)`
- `progress.reset(value=0)`
- `int(progress)` — cast to int

## Set

Distributed set for tracking unique values across workers.

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

**API:**

- `value in s`
- `len(s)`
- `s.add(value)`
- `s.discard(value)`

## Queue

FIFO work queue for dynamic task distribution.

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

with sky.ComputePool(provider=sky.AWS(), nodes=4) as pool:
    producer(list(range(20))) @ pool
    results = worker() @ pool
```

**API:**

- `queue.put(value)`
- `queue.get(timeout=None)` — returns `None` on timeout
- `queue.empty()`
- `len(queue)`

## Barrier

Synchronization point — all workers must arrive before any can proceed.

```python
@sky.compute
def synchronized_epoch(epoch: int) -> dict:
    info = sky.instance_info()
    sync = sky.barrier("epoch_sync", n=info.total_nodes)

    loss = train_one_epoch(epoch)
    sync.wait()  # all nodes reach here before continuing

    return {"node": info.node, "loss": loss}
```

**API:**

- `barrier.wait()` — blocks until `n` workers arrive

## Lock

Mutual exclusion for critical sections. Supports context manager protocol.

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

**API:**

- `lock.acquire()` / `lock.release()`
- `with lock:` — context manager

## Consistency

Collections support two consistency levels:

| Level | Behavior |
|-------|----------|
| `"eventual"` | Faster reads, may see slightly stale values (default) |
| `"strong"` | Linearizable reads, always see latest value |

```python
# Default: eventual consistency (faster)
cache = sky.dict("cache")

# Strong consistency (slower, always up-to-date)
cache = sky.dict("cache", consistency="strong")
```

Eventual consistency is sufficient for most use cases (caches, progress counters). Use strong consistency when correctness depends on reading the latest value (e.g., checkpoint coordination).

## Async Interface

All collections also expose async methods for use in async contexts:

```python
await cache.get_async(key)
await cache.set_async(key, value)
await counter.increment_async()
await queue.put_async(value)
await lock.acquire_async()
```

---

## Related Topics

- [Distributed Training](distributed-training.md) — Multi-node training guides
- [API Reference](reference/distributed.md) — Full API documentation
