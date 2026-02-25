# Distributed Training

Distributed training across multiple machines requires solving two problems simultaneously. First, the environment: every node needs to know the cluster topology — who the master is, how many peers exist, what rank each process holds. Second, the data: each node should train on a different subset, but the model parameters need to stay synchronized across all of them.

Skyward handles the first problem automatically. When you provision a multi-node pool and broadcast a function with `@`, every worker receives the same function and arguments, but each one sees a different `instance_info()` — its own position in the cluster. Plugins like `sky.plugins.torch()` and `sky.plugins.jax()` read this topology and configure the framework's distributed environment before your function runs. The second problem — data partitioning — is handled either by `sky.shard()` or by the framework's own distributed sampler.

This page explains the concepts. For step-by-step tutorials with runnable code, see the guides: [PyTorch Distributed](guides/pytorch-distributed.md), [Keras Training](guides/keras-training.md), and [HuggingFace Fine-tuning](guides/huggingface-finetuning.md).

## How It Works

When a function is broadcast to a pool with `@`, Skyward sends the same serialized payload to every node. Each node deserializes and executes the function independently. From the framework's perspective, this looks like `N` separate processes running the same script — exactly what tools like `torchrun` or `jax.distributed.initialize()` expect.

The difference is how the environment gets configured. In a traditional setup, you'd write a launch script that sets `MASTER_ADDR`, `WORLD_SIZE`, and `RANK` on each machine, then starts the training process. With Skyward, plugins do this for you. They read the cluster topology from `instance_info()` — which is populated from a `COMPUTE_POOL` environment variable that Skyward injects on each worker — and set the appropriate variables before your function body runs.

```python
@sky.compute
def train():
    import torch.distributed as dist
    # dist.is_initialized() is True — process group already configured
    ...

with sky.ComputePool(
    provider=sky.AWS(),
    nodes=4,
    accelerator="A100",
    plugins=[sky.plugins.torch()],
) as pool:
    results = train() @ pool  # runs on all 4 nodes
```

This is roughly equivalent to running `torchrun --nnodes=4 --nproc_per_node=1 train.py` on a pre-configured cluster — except there's no cluster to pre-configure. Skyward provisions the machines, installs dependencies, configures the distributed environment, runs your function, collects the results, and tears everything down when the `with` block exits.

## Plugins

Each supported framework has its own plugin. They all follow the same pattern: transform the worker image to install dependencies, then configure the distributed runtime at task execution time by reading `instance_info()` and setting environment variables. Plugins are specified on the pool, not on individual functions.

### PyTorch

`sky.plugins.torch()` adds `torch` to the worker's pip dependencies and configures `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, and calls `torch.distributed.init_process_group()`. The backend defaults to `nccl` for GPU nodes and `gloo` for CPU. Once initialized, you wrap your model with `DistributedDataParallel` and PyTorch handles gradient synchronization automatically — each node computes gradients on its own data, and DDP averages them across all nodes before each optimizer step.

The plugin also configures `LOCAL_WORLD_SIZE` and `NODE_RANK` for multi-GPU-per-node setups, though the most common Skyward pattern is one process per node.

See the [PyTorch Distributed guide](guides/pytorch-distributed.md) for a complete training example with DDP, `DistributedSampler`, and metric aggregation.

### Keras 3

`sky.plugins.keras(backend="jax")` sets the `KERAS_BACKEND` environment variable on the worker before Keras is imported — this is critical because Keras reads the backend at import time. When using the JAX backend, combine with `sky.plugins.jax()`:

```python
plugins=[sky.plugins.jax(), sky.plugins.keras(backend="jax")]
```

Keras 3 is backend-agnostic, but Skyward's automatic distribution (`DataParallel` with device discovery) is currently JAX-only. For the `torch` and `tensorflow` backends, the plugin delegates to those frameworks' native distributed init. For data-parallel training where each node trains independently on its shard (the most common pattern), the `keras` plugin alone is sufficient regardless of backend.

See the [Keras Training guide](guides/keras-training.md) for a complete MNIST example with data sharding.

### JAX

`sky.plugins.jax()` adds `jax[cuda12]` to pip and configures `JAX_COORDINATOR_ADDRESS`, `JAX_NUM_PROCESSES`, `JAX_PROCESS_ID`, and `JAX_LOCAL_DEVICE_COUNT`, then calls `jax.distributed.initialize()`. After initialization, JAX sees all devices across all nodes as a single device mesh, and operations like `pmap` and `pjit` distribute computation automatically.

### HuggingFace Transformers

`sky.plugins.huggingface(token="...")` adds `transformers`, `datasets`, and `tokenizers` to pip, sets `HF_TOKEN`, and runs `huggingface-cli login` during bootstrap. For multi-node training, combine with `sky.plugins.torch()`. The HuggingFace `Trainer` auto-detects the distributed setup and handles gradient synchronization, mixed-precision training, and distributed evaluation internally.

For single-node fine-tuning, the `Trainer` manages device placement on its own — the `huggingface` plugin handles authentication and dependencies. For multi-node, combine with `sky.plugins.torch()`.

See the [HuggingFace Fine-tuning guide](guides/huggingface-finetuning.md) for a complete example.

## Data Partitioning

In distributed training, each node should process different data but the same model. There are two approaches, and which one you use depends on the framework.

**`sky.shard()`** is Skyward's built-in data partitioning. It works inside any `@sky.compute` function and is framework-agnostic. You pass the full dataset as an argument, call `shard()` inside the function, and each node gets its portion based on `instance_info()`. The sharding is type-preserving (lists produce lists, tensors produce tensors) and supports synchronized shuffling with a fixed seed. This is the natural choice for Keras, JAX, and any workflow where you load data inside the function.

```python
@sky.compute
def train(x_full, y_full):
    x, y = sky.shard(x_full, y_full, shuffle=True, seed=42)
    # x[i] still corresponds to y[i]
    return fit(x, y)
```

**`DistributedSampler`** is PyTorch's native approach. It integrates with `DataLoader` and handles shuffling per-epoch (via `set_epoch()`), uneven dataset sizes, and drop-last semantics within the DataLoader pipeline. If you're using PyTorch DDP, `DistributedSampler` is the idiomatic choice.

Both approaches achieve the same goal: each node trains on different data. The choice is primarily about which framework's idioms you prefer. For a detailed explanation of sharding mechanics — modulo striding, multi-array alignment, `shuffle`, `drop_last` — see [Data Sharding](guides/data-sharding.md).

## Runtime Context

Inside a `@sky.compute` function, `sky.instance_info()` returns an `InstanceInfo` describing this node's position in the cluster. Plugins use this internally, but you can also use it directly for custom distributed logic — coordinating checkpoints, conditional logging, role-based execution.

```python
@sky.compute
def distributed_task(data):
    info = sky.instance_info()
    print(f"Node {info.node} of {info.total_nodes}")

    if info.is_head:
        coordinate_others()

    return process(data)
```

The key fields are `node` (0 to N-1), `total_nodes`, `is_head` (true for node 0), `head_addr` (private IP of the head node), `head_port` (coordination port), `accelerators` (GPU count on this node), and `peers` (list of all nodes with their addresses). This is the same information that plugins use to set `MASTER_ADDR`, `WORLD_SIZE`, and `RANK` — you can read it directly when building custom coordination logic or when using a framework that Skyward doesn't have a built-in plugin for.

The head node pattern is especially common in distributed training: only the head node saves checkpoints, logs to experiment trackers, or prints progress. Other nodes do the same computation but stay silent. This avoids duplicate writes and noisy output.

## Output Control

In distributed training, having every node print progress is noisy — four nodes produce four copies of every log line. Skyward provides output control decorators that silence stdout or stderr based on the node's identity:

```python
@sky.compute
@sky.stdout(only="head")
def train():
    print(f"Epoch {epoch}: loss={loss:.4f}")  # only head node prints
```

`only="head"` silences all non-head nodes. You can also pass a predicate — `only=lambda info: info.node < 2` — for finer control (for example, printing from only the first two nodes for debugging). `@sky.silent` suppresses both stdout and stderr on all nodes entirely. These decorators are implemented by redirecting output streams to `StringIO()` based on `instance_info()` at function entry.

Output control decorators go below `@sky.compute`:

```python
@sky.compute
@sky.stdout(only="head")
def train():
    ...
```

## Next Steps

- **[PyTorch Distributed](guides/pytorch-distributed.md)** — DDP training with `DistributedSampler` and metric aggregation
- **[Keras Training](guides/keras-training.md)** — MNIST across multiple GPUs with JAX backend
- **[HuggingFace Fine-tuning](guides/huggingface-finetuning.md)** — Transformer fine-tuning on cloud GPUs
- **[Data Sharding](guides/data-sharding.md)** — How `shard()` partitions data across nodes
- **[Plugins](integrations.md)** — Full plugin reference including joblib and scikit-learn
