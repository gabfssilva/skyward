# Data Sharding

In this guide you'll learn how to **distribute data across nodes** using `shard()`. Each node automatically gets its portion of the data — no manual splitting needed.

## Automatic Sharding

Pass full datasets to the compute function. Inside, call `shard()` to get this node's portion:

```python
--8<-- "examples/guides/05_data_sharding.py:8:22"
```

`shard()` divides the data evenly across all nodes. With 4 nodes and 1000 samples, each node gets ~250. The `shuffle` and `seed` parameters ensure deterministic, shuffled splits.

## Sharding Multiple Arrays

Pass multiple arrays to `shard()` to split them in sync:

```python
--8<-- "examples/guides/05_data_sharding.py:13:13"
```

Both arrays are split at the same indices — row 0 of `x` always pairs with row 0 of `y`.

## Type Preservation

`shard()` preserves the input type — lists return lists, tuples return tuples, numpy arrays return arrays:

```python
--8<-- "examples/guides/05_data_sharding.py:25:42"
```

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/05_data_sharding.py
```

---

**What you learned:**

- **`shard()`** automatically partitions data for the current node.
- **Multiple arrays** can be sharded in sync with a single call.
- **Shuffle and seed** ensure deterministic, reproducible splits.
- **Type preservation** — lists stay lists, arrays stay arrays.
