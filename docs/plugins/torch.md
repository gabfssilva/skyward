# PyTorch

PyTorch's distributed training model is built around `DistributedDataParallel` (DDP). Each process — typically one per node — holds a complete copy of the model. During the forward pass, each process computes gradients on its own data shard. During the backward pass, DDP synchronizes gradients across all processes using a collective communication backend (NCCL for GPUs, gloo for CPUs). The optimizer then steps with identical averaged gradients on every process, keeping the model copies in sync without explicit parameter transfers.

The hard part is the setup. Before `init_process_group()` can be called, every process needs five pieces of information: the address of the rendezvous master (`MASTER_ADDR`), the master port (`MASTER_PORT`), the total number of processes (`WORLD_SIZE`), this process's global rank (`RANK`), and its local rank on the machine (`LOCAL_RANK`). These must be set as environment variables before any distributed operation. In a traditional setup, you write a launch script or use `torchrun` to inject these values. With Skyward, the `torch` plugin reads the cluster topology from `instance_info()` and sets everything before your function body runs.

## What It Does

`sky.plugins.torch()` contributes two hooks to the plugin pipeline: an image transform that installs PyTorch with the correct CUDA wheels on the remote worker, and an `around_app` hook that initializes PyTorch's distributed process group once per worker process.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `"nccl" \| "gloo" \| None` | `None` | Process group backend. Auto-detected if `None`: `nccl` when CUDA is available, `gloo` otherwise. |
| `cuda` | `str` | `"cu128"` | CUDA version suffix for the PyTorch wheel index. Determines which prebuilt wheels are pulled from `download.pytorch.org`. |
| `version` | `str` | `"latest"` | PyTorch version. `"latest"` installs the latest release. A bare version string (e.g. `"2.3.0"`) pins with `==`. Constraint prefixes like `">=2.3"` are passed through as-is. |
| `vision` | `str \| None` | `None` | Torchvision version. Same semantics as `version`. `None` skips installation, `"latest"` installs the latest release. |
| `audio` | `str \| None` | `None` | Torchaudio version. Same semantics as `version`. `None` skips installation, `"latest"` installs the latest release. |

The `cuda` value determines the wheel index URL. When the cluster has a GPU accelerator (one with CUDA support in its metadata), the plugin uses `https://download.pytorch.org/whl/{cuda}` as the pip index. When the accelerator is `None` or does not support CUDA, it falls back to `https://download.pytorch.org/whl/cpu`. This auto-detection happens at image transform time, using the cluster's spec to decide.

## How It Works

### Image Transform

The `transform` hook builds the pip package list and index from the parameters. It assembles the list of PyTorch packages — always `torch`, optionally `torchvision` and `torchaudio` — with their version constraints, then selects the correct pip index based on the cluster's accelerator.

The accelerator detection uses pattern matching on `cluster.spec.accelerator`. If the cluster has an `Accelerator` with CUDA metadata, the CUDA wheel index is used. Otherwise — no accelerator, or an accelerator without CUDA support — the CPU index is used. This means you do not need to manually switch between CUDA and CPU wheels; the plugin reads the cluster configuration and does it for you.

The packages and index are appended to the existing image using `replace()`, preserving any packages and indexes already defined in the `Image` or added by other plugins.

### Worker Lifecycle (`around_app`)

The `around_app` hook initializes PyTorch's distributed process group once per worker process. When the first task arrives, the hook:

1. Imports `torch` and `torch.distributed` (these are remote-only imports — PyTorch does not need to be installed locally).
2. Reads `instance_info()` from the hook's parameter to get the cluster topology.
3. If the cluster has fewer than 2 nodes, yields immediately — no distributed setup needed for single-node pools.
4. Sets the environment variables: `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK` (always `"0"` — Skyward runs one process per node), `LOCAL_WORLD_SIZE` (always `"1"`), and `NODE_RANK`.
5. Selects the backend: if explicitly provided, uses that; otherwise, `"nccl"` when `torch.cuda.is_available()` and `"gloo"` otherwise.
6. Calls `dist.init_process_group(backend=..., init_method="env://")`.
7. Yields to the worker lifecycle — subsequent tasks run with the process group already active.
8. On worker shutdown, calls `dist.destroy_process_group()` in the `finally` block.

The environment variables come from `instance_info()`: `head_addr` becomes `MASTER_ADDR`, `head_port` becomes `MASTER_PORT`, `total_nodes` becomes `WORLD_SIZE`, and `node` becomes `RANK`. These values are populated from the `COMPUTE_POOL` environment variable that Skyward injects on each worker at startup.

`around_app` is the right hook for this because `init_process_group` is a one-time, process-global operation — calling it twice raises an error. The `around_app` lifecycle guarantees it runs exactly once, and its `finally` block ensures `destroy_process_group()` cleans up when the worker shuts down. This is the same pattern used by the JAX plugin for `jax.distributed.initialize()`.

## Usage

### Basic DDP Training

```python
import skyward as sky

@sky.compute
@sky.stdout(only="head")
def train() -> dict:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model = nn.Linear(784, 10).cuda()
    model = DDP(model)

    x = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        sampler.set_epoch(epoch)
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            loss = loss_fn(model(batch_x), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: loss={loss.item():.4f}")

    return {"final_loss": loss.item(), "rank": rank}

with sky.ComputePool(
    provider=sky.AWS(),
    accelerator="A100",
    nodes=4,
    plugins=[sky.plugins.torch()],
) as pool:
    results = train() @ pool
    for r in results:
        print(f"Rank {r['rank']}: loss={r['final_loss']:.4f}")
```

The `@` operator broadcasts `train()` to all 4 nodes. Each node runs the same function, but `dist.get_rank()` returns a different value (0 through 3), and `DistributedSampler` partitions the data accordingly. DDP synchronizes gradients in the backward pass, so all nodes converge on the same model parameters.

`@sky.stdout(only="head")` silences print statements on non-head nodes, so you see one set of epoch logs instead of four.

### With Torchvision and Torchaudio

```python
with sky.ComputePool(
    provider=sky.AWS(),
    accelerator="A100",
    nodes=2,
    plugins=[sky.plugins.torch(vision="latest", audio="latest")],
) as pool:
    results = train() @ pool
```

This installs `torch`, `torchvision`, and `torchaudio` from the CUDA wheel index. Inside the function, you can import `torchvision.models`, `torchvision.transforms`, `torchaudio`, etc.

### Pinning Versions

```python
plugins=[sky.plugins.torch(version="2.3.0", vision="0.18.0", cuda="cu124")]
```

This pins `torch==2.3.0` and `torchvision==0.18.0`, installed from the CUDA 12.4 wheel index. Version pinning is important for reproducibility — different PyTorch versions can produce different training results due to changes in default behaviors, numerical stability, and operator implementations.

### CPU-Only

```python
with sky.ComputePool(
    provider=sky.AWS(),
    nodes=4,
    plugins=[sky.plugins.torch(backend="gloo")],
) as pool:
    results = train() @ pool
```

Without an `accelerator`, the pool uses CPU instances. The plugin detects the absence of a CUDA accelerator and installs the CPU-only PyTorch wheels from `download.pytorch.org/whl/cpu`. The `backend="gloo"` is explicit here — gloo is PyTorch's CPU-compatible collective communication backend.

### Combining with HuggingFace

```python
with sky.ComputePool(
    provider=sky.AWS(),
    accelerator="A100",
    nodes=2,
    plugins=[
        sky.plugins.torch(),
        sky.plugins.huggingface(token="hf_xxx"),
    ],
) as pool:
    results = finetune() @ pool
```

The `torch` plugin handles DDP initialization, and the `huggingface` plugin handles authentication and installs `transformers`, `datasets`, and `tokenizers`. Inside the function, HuggingFace's `Trainer` auto-detects the distributed environment set up by the torch plugin and uses it for distributed training, gradient synchronization, and distributed evaluation.

## Next Steps

- [PyTorch Distributed guide](../guides/pytorch-distributed.md) -- Step-by-step DDP training walkthrough
- [PyTorch Model Roundtrip guide](../guides/torch-model-roundtrip.md) -- Sending models to and from the cloud
- [HuggingFace plugin](huggingface.md) -- Fine-tuning Transformers on multiple nodes
- [What are Plugins?](index.md) -- How the plugin system works
- [JAX plugin](jax.md) -- The JAX equivalent for comparison
