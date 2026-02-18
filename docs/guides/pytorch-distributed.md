# PyTorch Distributed

Training a neural network across multiple nodes requires coordinating processes that don't share memory. Each node needs to know the cluster topology — who the master is, how many peers exist, what rank it holds — and the processes need to synchronize gradients during backpropagation. PyTorch's DistributedDataParallel (DDP) handles the gradient synchronization, but the environment setup is notoriously manual: setting `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, and calling `init_process_group()` correctly on every node.

Skyward's `@torch` integration does all of this automatically. It reads the cluster topology from `instance_info()`, configures the environment variables, and initializes the process group before your function runs. You write a standard DDP training loop — Skyward handles the distributed plumbing.

## The `@torch` Integration

Add `@sky.integrations.torch` below `@sky.compute`:

```python
--8<-- "examples/guides/06_pytorch_distributed.py:6:8"
```

When this function executes on a remote worker, the decorator reads `instance_info()` and sets `MASTER_ADDR` to the head node's private IP, `MASTER_PORT` to the coordination port, `WORLD_SIZE` to the total number of nodes, and `RANK` to this node's index. It then calls `torch.distributed.init_process_group()` with the configured backend (defaulting to `nccl` for GPU, `gloo` for CPU). By the time your function body runs, the distributed environment is fully initialized.

Decorator order matters: `@sky.compute` must be outermost (it creates the `PendingCompute` wrapper), and `@sky.integrations.torch` goes below it (it runs inside the remote worker, before your function body).

## Model with DDP

Define a standard model and wrap it with `DistributedDataParallel`:

```python
--8<-- "examples/guides/06_pytorch_distributed.py:21:28"
```

DDP replicates the model on each node and synchronizes gradients during `backward()`. Each node trains on its own shard of the data, but the model parameters stay in sync because gradients are averaged across all nodes before each optimizer step. The `if dist.is_initialized()` guard lets the same code work in both single-node and multi-node contexts.

## Distributed Data Loading

Use `DistributedSampler` to ensure each node gets a unique subset of the data:

```python
--8<-- "examples/guides/06_pytorch_distributed.py:30:34"
```

The sampler reads the rank and world size from the process group and partitions the dataset indices accordingly. Unlike `sky.shard()`, which operates on raw data, `DistributedSampler` integrates with PyTorch's `DataLoader` and handles shuffling per-epoch. Call `sampler.set_epoch(epoch)` before each epoch so the shuffling pattern changes — without this, every epoch sees the same order.

Both approaches — `sky.shard()` and `DistributedSampler` — achieve the same goal (each node processes different data), but `DistributedSampler` is the PyTorch-native way and handles edge cases like uneven dataset sizes and drop-last semantics within the DataLoader pipeline.

## Aggregating Metrics

During training, each node computes local metrics (loss, accuracy). To get global metrics — averaged across all nodes — use `all_reduce`:

```python
--8<-- "examples/guides/06_pytorch_distributed.py:61:64"
```

`all_reduce` with `ReduceOp.SUM` sums the tensor across all nodes in-place. After the operation, every node holds the same aggregated values. Dividing by the number of nodes (or total samples) gives you the global average. This is how the head node can log consistent, cluster-wide metrics.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/06_pytorch_distributed.py
```

---

**What you learned:**

- **`@sky.integrations.torch`** configures `MASTER_ADDR`, `WORLD_SIZE`, `RANK`, and calls `init_process_group()` automatically.
- **DDP** synchronizes gradients across nodes — each node trains on different data, but model parameters stay in sync.
- **`DistributedSampler`** partitions data per node — call `set_epoch()` each epoch for proper shuffling.
- **`all_reduce`** aggregates metrics across all nodes — essential for consistent logging.
- **Decorator order** — `@sky.compute` outermost, `@sky.integrations.torch` below it.
