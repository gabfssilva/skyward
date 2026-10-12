# Runtime

The runtime API is available inside a function running on a compute node. Calling `instance_info()` on the dispatching process raises an error.

## Node information

`sky.instance_info()` returns an `Info` value. It contains the node id, compute id, rank, peer addresses, local worker slot, and worker concurrency.

```python
import skyward as sky

@sky.function
def report():
    info = sky.instance_info()
    return {
        "rank": info.rank,
        "nodes": info.nodes,
        "worker": info.worker,
        "total_workers": info.total_workers,
        "head": info.head_addr,
    }
```

The derived properties are:

| Property | Meaning |
|---|---|
| `nodes` | Number of peer nodes |
| `total_workers` | `nodes * workers_per_node` |
| `global_worker_index` | Position of this worker across all node slots |
| `host` | Address of this node in `peers` |
| `head` / `head_addr` | Address of rank zero |
| `head_port` | Worker rendezvous port on rank zero |
| `job_id` | Compute id |
| `is_head` | Whether `rank == 0` |

`sky.is_head()` is the shorthand for `sky.instance_info().is_head`.

## Sharding

`sky.shard()` gives the current node its contiguous slice of one or more sequences. All nodes receive the full input and calculate the same rank split locally. With `shuffle=True`, the same deterministic permutation is applied to each input; `seed` controls the permutation. `drop_last=True` truncates the input so every node receives the same number of elements.

```python
@sky.function
def train_epoch(features, labels):
    features, labels = sky.shard(features, labels, shuffle=True, seed=7, drop_last=True)
    return len(features)
```

Pass `node` and `total_nodes` explicitly when testing a shard outside a node. Otherwise they are read from `Info`.

## Events

`sky.emit(name, payload)` sends a JSON value to the daemon, which records it in the compute's event log as a `node.emitted` event. It can be called anywhere on a node — a task, a thread it started, a plugin's `setup`, an executor subprocess — and names the task whose code is running, when there is one. The output policy does not apply to it.

```python
@sky.function
def train(epochs):
    for epoch in range(epochs):
        loss = step()
        sky.emit("epoch", {"epoch": epoch, "loss": loss})
```

The events are read like any other: `sky log`, `GET /v1/events/log?types=node.emitted`, or the stream. An event is limited to 64 KiB; larger values belong in the task's result or in storage.

::: skyward.emit

## Output and callbacks

`stdout`, `stderr`, and `silent` control which task output is sent back to the client. `redirect_output(callback)` temporarily sends both streams to a callback and yields callback writers.

::: skyward.Info

::: skyward.instance_info

::: skyward.shard

::: skyward.is_head

::: skyward.stdout

::: skyward.stderr

::: skyward.silent

::: skyward.CallbackWriter

::: skyward.redirect_output
