# Broadcast

Some operations need to run on every node in the pool — not just one. Distributed training setups, data-parallel processing, cache warming, model loading. The `@` operator handles this: it sends the same computation to all nodes and returns a list with one result per node. Combined with `shard()`, each node can operate on its own partition of the data while receiving the same function and arguments.

## Processing Data on Every Node

Define a compute function that uses `shard()` to get its portion of the data:

```python
--8<-- "examples/guides/03_broadcast.py:6:17"
```

The function receives the *full* dataset as its argument, but `shard()` returns only the portion assigned to the current node. This means the serialization cost is paid once (the full dataset is sent to every node), but each node processes a different slice. The sharding is automatic — `shard()` reads the node's position from `instance_info()` and uses modulo striding (`indices[node::total_nodes]`) to divide the data evenly.

`instance_info()` returns an `InstanceInfo` with the node's index, the total cluster size, whether it's the head node, and the addresses of all peers. This is how the function knows where it sits in the cluster without any explicit configuration.

## Broadcasting with `@`

Use `@` instead of `>>` to run the function on every node:

```python
--8<-- "examples/guides/03_broadcast.py:21:26"
```

Where `>>` sends work to a single node (round-robin), `@` sends it to *all* nodes. The pool serializes the function and arguments once, dispatches a copy to each worker, waits for every node to complete, and collects the results. The return type is `list[T]` — one entry per node, in node order.

This is the foundation for distributed patterns in Skyward. When every node runs the same function but `shard()` gives each one different data, you get data parallelism without any explicit coordination. The function body is identical across nodes — the differentiation happens at runtime based on each node's position in the cluster.

## Aggregating Results

Each node returns a partial result. Since broadcast returns a list, you combine them locally on the client side:

```python
--8<-- "examples/guides/03_broadcast.py:28:33"
```

This map-reduce pattern — broadcast a function, shard the data inside, aggregate the results — is the simplest form of distributed computation in Skyward. More complex patterns (distributed training with gradient synchronization, for example) build on the same foundation but use framework integrations to handle the inter-node communication.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/03_broadcast.py
```

---

**What you learned:**

- **`@` operator** broadcasts a function to every node in the pool, returning `list[T]`.
- **`shard()`** divides data for the current node — each node processes its own slice of the full dataset.
- **`instance_info()`** provides the node's identity: index, total count, head status, peer addresses.
- **Map-reduce pattern** — broadcast + shard inside + aggregate locally — is the foundation for distributed computation.
