# Broadcast

In this guide you'll run a function on **every node** in your pool. You'll learn the `@` operator for broadcasting and `shard()` for automatic data partitioning.

## Processing Data on Every Node

Define a compute function that uses `shard()` to get its portion of the data:

```python
--8<-- "examples/guides/03_broadcast.py:6:17"
```

`shard()` automatically divides the input for the current node — node 0 gets the first quarter, node 1 the second, and so on. `instance_info()` returns metadata about the current node.

## Broadcasting with @

Use `@` instead of `>>` to run the function on **all nodes**:

```python
--8<-- "examples/guides/03_broadcast.py:21:26"
```

Where `>>` sends work to a single node, `@` broadcasts to every node in the pool. The result is a list — one entry per node.

## Aggregating Results

Each node returns its partial result. Combine them locally:

```python
--8<-- "examples/guides/03_broadcast.py:28:33"
```

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/03_broadcast.py
```

---

**What you learned:**

- **`@` operator** broadcasts a function to every node in the pool.
- **`shard()`** automatically partitions data for the current node.
- **`instance_info()`** returns the node's identity and metadata.
- Results come back as a **list**, one entry per node.
