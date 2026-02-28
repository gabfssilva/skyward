# Data sharding

The most common pattern in distributed computing is sending the same function to every node but having each node operate on a different slice of the data. `sky.shard()` automates this: it reads the current node's position from `instance_info()` and returns only the portion of the data that belongs to this node. No manual index math, no configuration — just pass the full dataset and get back your shard.

## Automatic sharding

Pass the full dataset to the compute function. Inside, call `shard()` to get this node's portion:

```python
--8<-- "examples/guides/05_data_sharding.py:8:22"
```

The function receives the *full* dataset as arguments — `full_x` and `full_y` are the complete arrays. `shard()` divides them using modulo striding: with 4 nodes and 1000 samples, node 0 gets indices `[0, 4, 8, ...]`, node 1 gets `[1, 5, 9, ...]`, and so on. Each node ends up with ~250 samples, evenly distributed regardless of whether the total is divisible by the node count.

The `shuffle=True` parameter randomizes the order before sharding, with a fixed `seed` ensuring all nodes agree on the same permutation. This is important for training: without shuffling, each node would get a contiguous block of the original data order, which can introduce bias if the data is sorted.

## Sharding multiple arrays

When you pass multiple arrays to `shard()`, the same indices are selected from each one — so paired data stays consistent:

```python
--8<-- "examples/guides/05_data_sharding.py:13:13"
```

This is critical for supervised learning: features and labels, inputs and targets, questions and answers. After sharding, `x[i]` still corresponds to `y[i]` because the same positions were selected from both arrays. You can pass any number of arrays to a single `shard()` call, and they'll all be split at the same indices.

## Type preservation

`shard()` returns the same type it receives. Lists produce lists, tuples produce tuples, NumPy arrays produce arrays, PyTorch tensors produce tensors:

```python
--8<-- "examples/guides/05_data_sharding.py:25:42"
```

This means you can shard a tensor and immediately pass it to a model without type conversions or wrapping. The sharding operation is transparent to downstream code — it doesn't know (or care) that it's working with a subset.

## Equal-size shards with `drop_last`

By default, striding can produce shards of slightly different sizes when the total isn't evenly divisible. If your training loop requires fixed batch dimensions (common with compiled models or certain padding strategies), use `drop_last=True`:

```python
x, y = sky.shard(x_full, y_full, drop_last=True)
```

This switches from striding to contiguous blocks and discards leftover elements, guaranteeing every node gets exactly the same number of samples.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/05_data_sharding.py
```

---

**What you learned:**

- **`shard()`** automatically partitions data for the current node using modulo striding.
- **Multiple arrays** sharded in a single call stay aligned — same indices selected from each.
- **`shuffle=True` + `seed`** randomize the split deterministically, avoiding bias from data ordering.
- **Type preservation** — lists, tuples, arrays, and tensors all stay their original type after sharding.
- **`drop_last=True`** guarantees equal-size shards by discarding leftover elements.
