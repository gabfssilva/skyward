# PyTorch Distributed

In this guide you'll train a neural network across **multiple nodes** using PyTorch's DistributedDataParallel (DDP). Skyward handles all the distributed setup — `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, and process group initialization.

## The @torch Integration

Add `@sky.integrations.torch` to your compute function:

```python
--8<-- "examples/guides/06_pytorch_distributed.py:6:8"
```

This decorator automatically configures the PyTorch distributed environment on each node. No manual `init_process_group()` needed.

## Model with DDP

Define a model and wrap it with DDP:

```python
--8<-- "examples/guides/06_pytorch_distributed.py:21:28"
```

DDP synchronizes gradients across nodes during `backward()`. Each node trains on its shard of the data, but the model stays in sync.

## Distributed Data Loading

Use `DistributedSampler` to shard the dataset:

```python
--8<-- "examples/guides/06_pytorch_distributed.py:30:34"
```

The sampler ensures each node gets a unique subset of the data. Call `sampler.set_epoch(epoch)` before each epoch for proper shuffling.

## Aggregating Metrics

Use `all_reduce` to combine metrics across nodes:

```python
--8<-- "examples/guides/06_pytorch_distributed.py:61:64"
```

`all_reduce` sums tensors across all nodes. After reduction, every node has the same aggregated values — enabling consistent logging from the head node.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/06_pytorch_distributed.py
```

---

**What you learned:**

- **`@sky.integrations.torch`** configures distributed PyTorch automatically.
- **DDP** synchronizes gradients across nodes.
- **`DistributedSampler`** shards data — call `set_epoch()` each epoch.
- **`all_reduce`** aggregates metrics across all nodes.
- **`info.is_head`** controls which node logs progress.
