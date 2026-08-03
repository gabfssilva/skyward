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
