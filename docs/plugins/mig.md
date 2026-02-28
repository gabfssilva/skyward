# NVIDIA MIG

A single A100 or H100 GPU has enormous compute capacity — often more than a single workload needs. Running a ResNet-50 inference or a small fine-tuning job on an 80 GB A100 leaves most of the hardware idle: hundreds of SMs without work, gigabytes of memory unaddressed, L2 cache serving a fraction of its bandwidth. You're paying for the full card but using a sliver of it.

NVIDIA Multi-Instance GPU (MIG) solves this at the hardware level. It partitions one physical GPU into multiple isolated instances — each with its own dedicated compute units (SMs), its own memory controller, and its own L2 cache. These are not virtual devices sharing resources through time-slicing or software scheduling. They are physically isolated execution environments carved out of the silicon. Two processes running on two MIG partitions cannot interfere with each other: one cannot access the other's memory, steal its compute cycles, or compete for cache bandwidth. Each partition behaves like a smaller, independent GPU.

The operational burden of MIG is what keeps most teams from using it. You need to enable MIG mode (which requires a GPU reset), create GPU instances with the right profile via `nvidia-smi`, create compute instances inside those GPU instances, enumerate the resulting MIG UUIDs, and assign each process to its partition by setting `CUDA_VISIBLE_DEVICES` to the correct UUID. Get any step wrong — mismatched profiles, wrong UUID indexing, MIG mode not enabled — and CUDA sees the wrong device or no device at all.

Skyward's `mig` plugin handles the full lifecycle: enables MIG mode during bootstrap, creates the requested partitions, and pins each subprocess to its own device before any task executes. Your `@sky.compute` functions see a normal CUDA device — they don't know MIG exists.

## What it does

The plugin configures container-level GPU visibility, enables MIG mode and creates partitions during bootstrap, and assigns each subprocess its own MIG device at runtime.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `profile` | `str` | — | MIG profile name (e.g. `"3g.40gb"`, `"1g.10gb"`, `"7g.80gb"`). Determines the size and maximum count of partitions. |

The profile string is passed directly to `nvidia-smi mig -cgi`. Available profiles depend on the GPU model — an A100 80GB supports seven profiles (from `1g.10gb` to `7g.80gb`), while an H100 supports a different set, and an A30 supports fewer still. The first number indicates compute slices (groups of SMs), not a fraction of the GPU: `3g.40gb` gets about 3/7 of the SMs, which is roughly 42 streaming multiprocessors on an A100. The second number is dedicated memory.

See [NVIDIA's supported GPUs page](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html) for the full profile matrix per GPU, or run `nvidia-smi mig -lgip` on a MIG-capable node to list what the hardware supports.

## How it works

### Image transform

The `transform` hook sets a single environment variable: `NVIDIA_VISIBLE_DEVICES=all`. This tells the NVIDIA container runtime to expose every MIG partition to the worker process. Without it, a container environment might present only a subset of devices — or a single device that maps to the whole GPU, bypassing the MIG partitions entirely. The variable is merged into the existing image environment using `replace()`, preserving any variables already defined in the `Image` or added by other plugins.

### Bootstrap

The `bootstrap` hook generates the shell commands that partition the GPU. It runs after the standard bootstrap phases (apt, pip, Python setup) and produces two kinds of commands:

First, it enables MIG mode:

```
nvidia-smi -mig 1
```

This is a mode switch on the GPU — it reconfigures the hardware to support partitioning. On a freshly booted cloud instance, MIG mode is typically off by default. Enabling it does not require a GPU reset on supported drivers (R470+), but it must happen before any GPU instances are created.

Then, for each worker subprocess (determined by `cluster.spec.worker.concurrency`), it creates one partition:

```
nvidia-smi mig -cgi <profile> -C
```

The `-cgi` flag creates a GPU Instance with the given profile, and `-C` immediately creates a Compute Instance inside it. For `concurrency=2` and `profile="3g.40gb"`, the bootstrap produces:

```
nvidia-smi -mig 1
nvidia-smi mig -cgi 3g.40gb -C
nvidia-smi mig -cgi 3g.40gb -C
```

The number of partitions created equals the worker concurrency. If the GPU does not support that many instances for the given profile — for example, requesting three `3g.40gb` partitions on an A100, which only supports two — `nvidia-smi` exits with an error and the bootstrap fails. This is intentional: the concurrency and profile must agree, and the failure happens early (during provisioning) rather than silently at runtime.

### Process lifecycle (`around_process`)

The `around_process` hook runs once per subprocess, before the first task executes. It solves the last piece of the MIG puzzle: assigning each subprocess to its specific partition.

The hook:

1. Calls `nvidia-smi -L` to list all GPU devices and their MIG instances. The output includes MIG UUIDs in the format `MIG-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.
2. Extracts all MIG UUIDs using a regex match.
3. Reads `instance_info().worker` to determine this subprocess's index (0, 1, 2, ...).
4. Sets `CUDA_VISIBLE_DEVICES` to the UUID at that index.

After this, CUDA presents the assigned partition as the only available device. `torch.device("cuda")` resolves to this partition. `torch.cuda.device_count()` returns 1. The subprocess has no way to access other partitions — the isolation is enforced by both the environment variable and the hardware.

Each subprocess runs this independently in its own process. Worker 0 gets the first MIG UUID, worker 1 gets the second, and so on. Because the hook runs exactly once per process (not per task), the device assignment is stable for the lifetime of the subprocess — subsequent tasks on the same worker reuse the same partition without re-running the hook.

## When to use MIG

MIG is specifically valuable when you have a high-end GPU and workloads that don't need its full capacity.

**Independent training runs** are the primary use case. If you're running hyperparameter sweeps, architecture comparisons, or ablation studies where each run is a separate training job, MIG lets you run multiple jobs on a single card. Each job gets guaranteed resources — no interference, no contention, predictable performance.

**Concurrent inference** works well when each model fits within a partition's memory. Two models serving requests on two `3g.40gb` partitions of an A100 each get dedicated compute and memory — better isolation than MPS, with no risk of one model's memory allocation starving the other.

**Development and experimentation** benefits from MIG when you have a powerful GPU but your experiments are small. Instead of wasting 70 GB of memory while fine-tuning a small model, partition the card and run several experiments simultaneously.

MIG is generally **not useful** for:

- **Workloads that saturate the GPU** — If your training loop uses all SMs and all memory, partitioning reduces performance. A single `7g.80gb` partition on an A100 has fewer SMs than the full card.
- **Multi-GPU distributed training** — DDP and FSDP expect each rank to own a full GPU. MIG partitions are not designed for gradient synchronization across them. Use the `torch` plugin for distributed training instead.
- **Unsupported GPUs** — Consumer GPUs do not support MIG. See [Requirements](#requirements) for the full constraint list.
- **Dynamic workloads with varying resource needs** — MIG partitions are fixed at setup time. If your workload needs more memory for some tasks and less for others, MPS offers more flexibility.

## How it differs from MPS

MIG and MPS both allow multiple processes to share a GPU, but the isolation model is fundamentally different.

**MIG** provides hardware-level isolation. Each partition has its own dedicated SMs, memory, and L2 cache. One partition cannot access another's memory or steal its compute cycles. The trade-off is that partitions are fixed — you choose a profile at setup time, and all partitions on a GPU must use the same profile. MIG is only available on [supported datacenter and professional GPUs](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html).

**MPS** provides software-level sharing. All processes submit kernels through a shared daemon, and the GPU scheduler overlaps their work. There is no memory isolation — a misbehaving process can consume all available memory. The benefit is flexibility: any number of processes can share the GPU without predefined partitions, and MPS works on any CUDA GPU.

Use MIG when you need guaranteed isolation and predictable performance per partition. Use MPS when you need flexible sharing and the processes are trusted. The two are mutually exclusive on the same GPU — enabling MIG mode disables MPS-style sharing within each partition, though MPS can be used within a single MIG partition if needed.

## Usage

### Independent training on partitions

The most common pattern: run independent training jobs on separate partitions of a single GPU.

```python
import skyward as sky


@sky.compute
def train_on_partition(epochs: int, lr: float) -> dict:
    import os

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    info = sky.instance_info()
    device = torch.device("cuda")

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    x = torch.randn(5000, 784, device=device)
    y = torch.randint(0, 10, (5000,), device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (output.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)

    return {
        "worker": info.worker,
        "partition": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "final_loss": round(epoch_loss / len(loader), 4),
        "accuracy": round(100.0 * correct / total, 1),
    }


PARTITIONS = 2
PROFILE = "3g.40gb"

with sky.ComputePool(
    provider=sky.Verda(),
    nodes=1,
    accelerator=sky.accelerators.A100(),
    worker=sky.Worker(concurrency=PARTITIONS, executor="process"),
    image=sky.Image(pip=["torch"]),
    plugins=[sky.plugins.mig(profile=PROFILE)],
) as pool:
    tasks = [train_on_partition(epochs=10, lr=1e-3) for _ in range(PARTITIONS)]
    results = list(sky.gather(*tasks, stream=True) >> pool)
```

The concurrency and profile must agree. A `3g.40gb` profile on an A100 supports exactly two partitions, so `concurrency=2`. Setting concurrency to three would fail during bootstrap because the GPU cannot create a third instance of that profile.

Notice that PyTorch is installed via `Image(pip=["torch"])`, not via `sky.plugins.torch()`. The torch plugin is designed for multi-node distributed training — it calls `init_process_group()`, which MIG partitions don't need. MIG partitions are independent workloads, not a distributed cluster.

### Maximum partitions

For lightweight workloads — small model inference, quick evaluations, data preprocessing with GPU-accelerated libraries — you can maximize the number of partitions with a smaller profile:

```python
with sky.ComputePool(
    provider=sky.AWS(),
    nodes=1,
    accelerator=sky.accelerators.A100(memory="80GB"),
    worker=sky.Worker(concurrency=7, executor="process"),
    image=sky.Image(pip=["torch"]),
    plugins=[sky.plugins.mig(profile="1g.10gb")],
) as pool:
    tasks = [evaluate(model_id=i) for i in range(7)]
    results = list(sky.gather(*tasks, stream=True) >> pool)
```

Seven partitions from a single A100 80GB, each with ~10 GB of memory and 14 SMs. Each partition can run a small model (DistilBERT, ResNet-18, a lightweight diffusion decoder) independently. This is seven times the throughput of running them sequentially on the full card — at the cost of reduced per-partition compute.

The `1g.10gb` profile is the smallest available on the A100. Smaller profiles mean more partitions but less compute and memory per partition. If your model needs more than 10 GB, step up to `2g.20gb` (three partitions) or `3g.40gb` (two partitions).

### Hyperparameter sweep

Different configurations running simultaneously on separate partitions — each partition explores a different point in the hyperparameter space:

```python
configs = [
    {"epochs": 20, "lr": 1e-3},
    {"epochs": 20, "lr": 3e-4},
]

with sky.ComputePool(
    provider=sky.Verda(),
    nodes=1,
    accelerator=sky.accelerators.A100(),
    worker=sky.Worker(concurrency=len(configs), executor="process"),
    image=sky.Image(pip=["torch"]),
    plugins=[sky.plugins.mig(profile="3g.40gb")],
) as pool:
    tasks = [train_on_partition(**cfg) for cfg in configs]
    results = list(sky.gather(*tasks, stream=True) >> pool)

    best = max(results, key=lambda r: r["accuracy"])
    print(f"Best: worker {best['worker']} with acc={best['accuracy']}%")
```

Both configurations run simultaneously with hardware-enforced isolation. Neither run can affect the other's performance, so the results are directly comparable — no noise from resource contention.

### Multi-node with MIG

MIG works per-GPU, not per-cluster. On a multi-node pool, each node independently partitions its own GPU:

```python
with sky.ComputePool(
    provider=sky.AWS(),
    nodes=3,
    accelerator=sky.accelerators.A100(),
    worker=sky.Worker(concurrency=2, executor="process"),
    image=sky.Image(pip=["torch"]),
    plugins=[sky.plugins.mig(profile="3g.40gb")],
) as pool:
    # 3 nodes * 2 partitions = 6 independent workers
    tasks = [train_on_partition(epochs=10, lr=lr) for lr in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]]
    results = list(sky.gather(*tasks, stream=True) >> pool)
```

Each of the 3 nodes gets its own A100 split into two `3g.40gb` partitions, giving you 6 independent workers total. Tasks are dispatched round-robin across all 6 workers. This is not distributed training — there is no gradient synchronization between partitions. Each task runs independently, which is exactly what you want for sweeps, evaluations, and embarrassingly parallel workloads.

## Requirements

- **MIG-capable GPU** — Supported on datacenter and professional GPUs such as A100, H100, and B200. Consumer GPUs do not support MIG. See [NVIDIA's supported GPUs page](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html) for the current list.
- **Process executor** — `Worker(executor="process")` is required. MIG device assignment works by setting `CUDA_VISIBLE_DEVICES` per subprocess. The thread executor shares a single process (and a single `CUDA_VISIBLE_DEVICES`), so all threads would see the same partition.
- **Concurrency matches profile** — The `concurrency` value must not exceed the number of partitions the GPU supports for the given profile. An A100 80GB supports 2 partitions for `3g.40gb`, 3 for `2g.20gb`, and 7 for `1g.10gb`.
- **Single GPU per node** — The current implementation assumes one GPU per node. Multi-GPU nodes with per-GPU MIG partitioning are not yet supported.

## Next steps

- [NVIDIA MIG Guide](../guides/nvidia-mig.md) — Step-by-step walkthrough with a training example
- [NVIDIA MPS](mps.md) — Software-level GPU sharing (complementary approach)
- [Worker Executors](../guides/worker-executors.md) — Thread vs process executors and when to use each
- [What are Plugins?](index.md) — How the plugin system works
