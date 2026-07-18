# NVIDIA MPS

When several CUDA processes share one GPU, the default behaviour is time-slicing: each gets the device for a quantum, then yields. The context switches are expensive — state saved and restored, caches flushed — and kernels from different processes never overlap. For workloads that do not individually saturate the GPU, that wastes most of it.

NVIDIA Multi-Process Service gives those processes one shared CUDA context, funnelled through a daemon. The SM scheduler can then overlap their kernels, filling the gaps.

`sky.plugins.Mps()` brings that daemon up in the worker process, before any task or subprocess exists, and puts the variables CUDA reads to find it in the environment every child inherits.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `active_thread_percentage` | `int \| None` | `None` | Ceiling on the share of GPU compute one client may use, 1–100. Left to the MPS default when unset |
| `pinned_memory_limit` | `str \| None` | `None` | Per-device pinned memory limit, e.g. `"0=2G"` for 2 GB on device 0 |

**`active_thread_percentage`** is fair-share scheduling. With 8 concurrent processes, `100 // 8 = 12` gives each roughly an eighth of the SMs; without it, one client can submit enough work to starve the rest.

**`pinned_memory_limit`** caps page-locked allocation per client. Pinned memory makes transfers fast and is finite; eight processes each pinning 4 GB will exhaust it. The format is `"device=limit"`.

## How it works

The plugin has exactly one hook: `setup`, entered once in the worker process before it takes its first task. It:

1. Creates `/tmp/nvidia-mps` and `/tmp/nvidia-mps-log`.
2. Runs `nvidia-cuda-mps-control -d` to start the daemon.
3. Sets `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY`, plus `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` and `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` when those parameters are given.

There is no `image` hook and no `bootstrap` hook. MPS ships with the CUDA driver, so there is nothing to install, and the image's `env` reaches only the bootstrap shell — which has exited by the time the worker starts. The daemon and its variables have to be put up where the tasks actually run.

**Starting the daemon is best-effort.** On a machine without the control binary the call fails and is swallowed: a worker that cannot share its GPU should still run the task on the whole one, not refuse to start.

## When to use it

MPS earns its keep when several independent CUDA processes share a GPU and none of them saturates it alone.

- **High-concurrency inference.** Many small forward passes (ResNet-50, DistilBERT) that each use 10–30% of the card. Without MPS they time-slice; with it their kernels overlap.
- **Several task slots per GPU.** `executor=sky.Executor(type="process", concurrency=N)` runs N processes on the node. If it has one GPU, they all share it.
- **GPU-accelerated preprocessing** (cuDF, cuPy, GPU transforms) running in parallel pipelines.

It is not useful for a single-process workload, for anything that already saturates the GPU — DDP training, where each node runs one process that owns the card — or as a substitute for multi-GPU work.

## Usage

### Concurrent inference

```python
import skyward as sky


@sky.function
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

with sky.Compute(
    provider=sky.AWS(),
    nodes=1,
    accelerator=sky.accelerators.T4(),
    executor=sky.Executor(type="process", concurrency=CONCURRENCY),
    image=sky.Image(pip=["torch", "torchvision"]),
    plugins=[sky.plugins.Mps(active_thread_percentage=100 // CONCURRENCY)],
) as compute:
    tasks = [inference(i, batch_size=1) for i in range(CONCURRENCY * 2)]
    results = list(sky.gather(*tasks) >> compute)
```

`type="process"` means each concurrent task has its own CUDA context; MPS unifies them into one on the card. The gain scales with how much of the GPU each process leaves idle — smaller models and smaller batches benefit most.

### Higher concurrency

```python
with sky.Compute(
    provider=sky.AWS(),
    nodes=1,
    accelerator=sky.accelerators.A100(),
    executor=sky.Executor(type="process", concurrency=32),
    plugins=[
        sky.plugins.Mps(
            active_thread_percentage=3,
            pinned_memory_limit="0=1G",
        ),
    ],
) as compute:
    ...
```

## How it differs from MIG

MPS shares in software: one context, no memory isolation, any number of clients, works on any CUDA GPU. [MIG](mig.md) partitions in hardware: dedicated SMs, memory and L2 per slice, fixed at bootstrap, and only on supported datacenter cards.

Use MPS for flexible sharing between trusted processes; MIG when you need guaranteed isolation and predictable per-partition performance.

## Next steps

- [NVIDIA MIG](mig.md) — hardware partitioning instead
- [Worker Executors](../guides/worker-executors.md) — thread vs process
- [What are plugins?](index.md) — the hook model
