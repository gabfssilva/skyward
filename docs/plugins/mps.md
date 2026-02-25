# NVIDIA MPS

When multiple CUDA processes share a single GPU — concurrent inference servers, multiple workers running independent forward passes, parallel data preprocessing pipelines — the default CUDA behavior is time-slicing. Each process gets exclusive access to the GPU for a time quantum, then yields to the next. Context switches between processes are expensive: the GPU must save and restore execution state, flush caches, and re-establish memory mappings. For workloads that do not saturate the GPU's compute capacity individually, time-slicing wastes significant throughput because the hardware sits idle during context switches and because each process's kernels cannot overlap with another's.

NVIDIA Multi-Process Service (MPS) solves this. It provides a shared GPU context that multiple CUDA processes connect to through a single daemon. Instead of each process owning its own CUDA context and time-slicing, all processes submit work through the MPS daemon, which funnels their kernels into a unified execution stream. The GPU's SM (Streaming Multiprocessor) scheduler can then overlap kernels from different processes, filling compute gaps and improving utilization. The result is higher aggregate throughput and lower latency, especially for workloads where individual processes use only a fraction of the GPU's capacity.

Skyward's `mps` plugin starts the MPS daemon during instance bootstrap and configures the environment variables that CUDA uses to connect to it. Every CUDA process on the worker — including concurrent task threads or processes managed by the `Worker` executor — automatically routes through MPS.

## What It Does

The plugin contributes two hooks:

**Image transform** — Sets environment variables that configure the MPS runtime:

- `CUDA_MPS_PIPE_DIRECTORY` — The directory for the MPS daemon's named pipes (set to `/tmp/nvidia-mps`). All CUDA processes on the instance use this path to communicate with the daemon.
- `CUDA_MPS_LOG_DIRECTORY` — The directory for MPS daemon logs (set to `/tmp/nvidia-mps-log`).
- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` — Optionally limits the percentage of GPU compute threads each MPS client can use. This prevents a single client from monopolizing the GPU.
- `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` — Optionally limits the amount of pinned (page-locked) memory each client can allocate per device.

**Bootstrap** — After the base environment is set up, the plugin runs two commands: `mkdir -p /tmp/nvidia-mps /tmp/nvidia-mps-log` to create the pipe and log directories, then `nvidia-cuda-mps-control -d` to start the MPS daemon in the background. The daemon runs for the lifetime of the instance. Once it is running, any CUDA process that starts on the instance automatically connects to it through the pipe directory.

## When to Use MPS

MPS is not for every workload. It is specifically valuable when multiple independent CUDA processes need to share a GPU concurrently, and each individual process does not fully saturate the GPU on its own.

**High-concurrency inference** is the primary use case. If you are running a model serving workload where many requests arrive simultaneously, each running a forward pass through a relatively small model (ResNet-50, DistilBERT, a small diffusion model), the individual forward passes may only use 10-30% of the GPU's compute capacity. Without MPS, the GPU time-slices between them. With MPS, their kernels overlap on the SMs, and aggregate throughput increases substantially.

**Multiple workers per GPU** is the related pattern. Skyward's `Worker(concurrency=N, executor="process")` runs N worker processes on each node. If the node has one GPU, all N processes share it. Without MPS, they context-switch. With MPS, they share efficiently. This is useful for workloads that are partially GPU-bound and partially CPU-bound — each process can use the GPU for its GPU portion while other processes use it for theirs.

**Batch data preprocessing** that uses GPU-accelerated libraries (cuDF, cuPy, torchvision transforms on GPU) can also benefit, especially when multiple preprocessing pipelines run concurrently.

MPS is generally **not useful** for:

- **Single-process workloads** — If only one process uses the GPU, MPS adds no benefit (and a negligible amount of overhead).
- **Workloads that saturate the GPU** — If a single process uses 90%+ of the GPU's compute, there is nothing to overlap. Multi-node training with DDP, where each node runs one training process that fully utilizes the GPU, does not benefit from MPS.
- **Multi-GPU training** — MPS operates per-GPU. For multi-GPU setups, each GPU has its own MPS daemon. But if the training process already uses all GPUs via NCCL, MPS is irrelevant.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `active_thread_percentage` | `int \| None` | `None` | Maximum percentage of GPU compute threads (1-100) each MPS client can use. Useful for fair scheduling across concurrent processes. |
| `pinned_memory_limit` | `str \| None` | `None` | Per-device pinned memory limit per client (e.g. `"0=2G"` for 2 GB on device 0). Prevents a single client from exhausting pinned memory. |

**`active_thread_percentage`** controls how much of the GPU each MPS client can use. If you have 8 concurrent processes sharing a GPU, setting `active_thread_percentage=12` ensures each one gets roughly 12% of the SMs. Without this, a single client could submit enough work to saturate the GPU, starving others. A common heuristic is `100 // concurrency`.

**`pinned_memory_limit`** restricts page-locked memory allocation per client. Pinned memory enables fast GPU transfers but is a finite resource. If 8 processes each try to pin 4 GB, the system may run out. The format is `"device_id=limit"` — for example, `"0=2G"` limits each client to 2 GB of pinned memory on GPU 0.

## How It Differs from Multi-Node Training

MPS and multi-node distributed training solve fundamentally different problems. Multi-node training (via the `torch` plugin with DDP, or the `jax` plugin) splits a single training job across multiple GPUs on multiple machines, synchronizing gradients between them. Each GPU runs one process that fully utilizes it.

MPS enables multiple independent processes to share a single GPU efficiently. The processes do not coordinate — they each run their own workload, and MPS ensures their GPU operations overlap rather than time-slice.

You can combine both patterns. A multi-node cluster with the `torch` plugin for DDP training does not need MPS (each node's GPU is dedicated to one training process). But a multi-node cluster for inference — where each node handles many concurrent requests — benefits from MPS on each node.

## Usage

### Concurrent Inference

Run multiple inference tasks concurrently on a single GPU node:

```python
import skyward as sky


@sky.compute
def inference(task_id: int, batch_size: int) -> dict:
    import torch
    from torchvision.models import resnet50

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet50(weights=None).to(device).eval()
    dummy = torch.randn(batch_size, 3, 224, 224, device=device)

    with torch.no_grad():
        for _ in range(100):
            model(dummy)
    torch.cuda.synchronize()

    return {"task_id": task_id, "device": device}


CONCURRENCY = 8

with sky.ComputePool(
    provider=sky.AWS(),
    nodes=1,
    accelerator="T4",
    worker=sky.Worker(concurrency=CONCURRENCY, executor="process"),
    plugins=[
        sky.plugins.torch(),
        sky.plugins.mps(active_thread_percentage=100 // CONCURRENCY),
    ],
    image=sky.Image(pip=["torchvision"]),
) as pool:
    tasks = [inference(i, batch_size=1) for i in range(CONCURRENCY * 2)]
    results = list(sky.gather(*tasks, stream=True) >> pool)
```

The `executor="process"` setting means each concurrent task runs in its own process with its own CUDA context. MPS unifies those contexts into a single shared context on the GPU. The `active_thread_percentage=100 // 8 = 12` ensures fair sharing across 8 concurrent processes.

Without MPS, these 8 processes would time-slice on the GPU. With MPS, their kernels overlap, and aggregate throughput is higher. The improvement depends on how much of the GPU each individual process utilizes — smaller models and smaller batch sizes see greater relative improvement because there is more idle compute to fill.

### With Higher Concurrency

For very high concurrency (many small tasks), increase the worker's concurrency and lower the per-client thread percentage:

```python
with sky.ComputePool(
    provider=sky.AWS(),
    nodes=1,
    accelerator="A100",
    worker=sky.Worker(concurrency=32, executor="process"),
    plugins=[
        sky.plugins.torch(),
        sky.plugins.mps(
            active_thread_percentage=3,
            pinned_memory_limit="0=1G",
        ),
    ],
) as pool:
    ...
```

With 32 concurrent processes, each gets 3% of the GPU's compute threads and at most 1 GB of pinned memory. This is appropriate for very lightweight inference tasks (small models, single-sample batches) where the goal is maximum throughput from a single GPU.

## Next Steps

- [PyTorch Distributed](../guides/pytorch-distributed.md) — Multi-node training with DDP (complementary to MPS)
- [Worker Executors](../guides/worker-executors.md) — Thread vs process executors and when to use each
- [What are Plugins?](index.md) — How the plugin system works
