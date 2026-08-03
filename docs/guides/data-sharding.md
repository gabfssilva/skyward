# Data sharding

The most common pattern in distributed computing is sending the same function to every node but having each node operate on a different slice of the data. `sky.shard()` automates this: it reads the current node's position from `instance_info()` and returns only the portion of the data that belongs to this node. No manual index math, no configuration — just pass the full dataset and get back your shard.

## Automatic sharding

Pass the full dataset to the compute function. Inside, call `shard()` to get this node's portion:

```python
--8<-- "guides/05_data_sharding.py:8:22"
```

The function receives the *full* dataset as arguments — `full_x` and `full_y` are the complete arrays. `shard()` divides them into contiguous rank-ordered slices: with 4 nodes and 1000 samples, node 0 gets the first quarter, node 1 the second, and so on. If the length is not divisible by the node count, the slice boundaries are calculated proportionally and sizes can differ by one.

The `shuffle=True` parameter randomizes the order before sharding, with a fixed `seed` ensuring all nodes use the same permutation. Without shuffling, each node gets a contiguous block of the original data order.

## Sharding multiple arrays

When you pass multiple arrays to `shard()`, the same indices are selected from each one — so paired data stays consistent:

```python
--8<-- "guides/05_data_sharding.py:13:13"
```

This is critical for supervised learning: features and labels, inputs and targets, questions and answers. After sharding, `x[i]` still corresponds to `y[i]` because the same positions were selected from both arrays. You can pass any number of arrays to a single `shard()` call, and they'll all be split at the same indices.

## Type preservation

`shard()` returns the same type it receives. Lists produce lists, tuples produce tuples, NumPy arrays produce arrays, PyTorch tensors produce tensors:

```python
--8<-- "guides/05_data_sharding.py:25:42"
```

This means you can shard a tensor and immediately pass it to a model without type conversions or wrapping. The sharding operation is transparent to downstream code — it doesn't know (or care) that it's working with a subset.

## Equal-size shards with `drop_last`

By default, proportional splitting can produce shards of slightly different sizes when the total isn't evenly divisible. If your training loop requires equal shard sizes, use `drop_last=True`:

```python
x, y = sky.shard(x_full, y_full, drop_last=True)
```

This discards the remainder before splitting, guaranteeing every node gets exactly the same number of samples.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python guides/05_data_sharding.py
```

---

**What you learned:**

- **`shard()`** automatically partitions data into a contiguous slice for the current node.
- **Multiple arrays** sharded in a single call stay aligned — same indices selected from each.
- **`shuffle=True` + `seed`** randomize the split deterministically, avoiding bias from data ordering.
- **Type preservation** — lists, tuples, arrays, and tensors all stay their original type after sharding.
- **`drop_last=True`** guarantees equal-size shards by discarding the remainder.
