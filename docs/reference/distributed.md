# Distributed collections

Distributed collections are available inside functions running on a compute. They are replicated across the compute's nodes and accessed synchronously from the task's worker.

Values are serialized by Skyward. Map keys must be hashable because they are used to route operations. Collections replicate to `min(3, nodes)` members.

## Consistency

`strong` is the default. Writes require a majority acknowledgement. `eventual` acknowledges on one replica and reads the nearest copy; use it for state where lower coordination cost is acceptable.

```python
counts = sky.counter("processed", consistency="eventual")
```

## Collections

| Factory | Operations |
|---|---|
| `sky.dict(name)` | Mapping operations, `get`, `pop`, `items`, `clear` |
| `sky.set(name)` | `add`, `remove`, membership, `items`, `clear` |
| `sky.counter(name)` | `add`, `get`, `reset` |
| `sky.queue(name)` | Non-blocking `offer`, `poll`, `len`, `clear` |
| `sky.registry(name)` | `register`, `lookup`, `unregister`, `list` |
| `sky.barrier(name, parties)` | Wait for the configured number of participants |
| `sky.lock(name, ttl=30, timeout=None)` | Context-manager lease across the compute |

```python
@sky.function
def save_checkpoint(step, model):
    checkpoints = sky.registry("checkpoints")
    checkpoints.register(step, model)
    return checkpoints.list()

@sky.function
def worker_step(batch):
    processed = sky.counter("processed")
    processed.add(len(batch))
    with sky.lock("checkpoint"):
        write_checkpoint()
```

`queue.poll()` does not block: an empty queue returns `None`. A lock is released when its context exits; its lease also expires if the holder dies.

::: skyward.Consistency

::: skyward.DistributedRegistry

::: skyward.dict

::: skyward.set

::: skyward.counter

::: skyward.registry

::: skyward.queue

::: skyward.barrier

::: skyward.lock
