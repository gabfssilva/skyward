# NVIDIA MIG

A single A100 or H100 has more compute than many workloads need. Running a small fine-tune on an 80 GB A100 leaves most of the card idle — hundreds of SMs without work, gigabytes unaddressed — and you pay for all of it.

NVIDIA Multi-Instance GPU partitions one physical GPU into isolated instances, each with its own SMs, memory controller and L2 cache. These are not time-sliced virtual devices: two processes on two MIG partitions cannot touch each other's memory, steal each other's cycles, or compete for cache bandwidth.

The operational burden is what stops most teams using it. Enable MIG mode, create GPU instances with the right profile, create compute instances inside them, enumerate the resulting UUIDs, and point each process at its own through `CUDA_VISIBLE_DEVICES`. Get any of it wrong and CUDA sees the wrong device, or none.

`sky.plugins.Mig()` does the whole sequence. Your `@sky.function` sees a normal CUDA device and does not know MIG exists.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `profile` | `str` | — (required) | The MIG profile every slice is cut to, e.g. `"3g.40gb"` or `"1g.10gb"` |

The string is passed straight to `nvidia-smi mig -cgi`. Available profiles depend on the GPU: an A100 80GB supports seven, from `1g.10gb` to `7g.80gb`. The first number is compute slices (groups of SMs), not a fraction of the card; the second is dedicated memory.

See [NVIDIA's supported GPUs page](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html), or run `nvidia-smi mig -lgip` on a MIG-capable node.

## Requirements

The plugin's contract only holds under one executor configuration:

```python
executor=sky.Executor(type="process", concurrency=N, reuse=True)
```

Each concurrent slot has to be a distinct, long-lived child with a stable `info.worker`, because slice *k* is claimed by child *k* for that child's life. Under the thread executor every task shares one process and one `info.worker` of zero — they would all pin the same slice. Without `reuse`, a child dies after its task and the pinning buys nothing.

Beyond that:

- **A MIG-capable GPU** — datacenter and professional cards (A100, H100, B200). Consumer GPUs do not support MIG.
- **Concurrency the profile can satisfy** — an A100 80GB fits 2 partitions at `3g.40gb`, 3 at `2g.20gb`, 7 at `1g.10gb`. Asking for more fails during bootstrap, which is the right time for it to fail.
- **One GPU per node** — the implementation reads one flat list of slice UUIDs and indexes it by worker.

## How it works

### `bootstrap`

The `bootstrap` hook returns two shell phases, appended after the image's own. The first enables MIG mode and cuts the card, once per concurrent slot:

```
nvidia-smi -mig 1
nvidia-smi mig -cgi 3g.40gb -C
nvidia-smi mig -cgi 3g.40gb -C
```

`-cgi` creates a GPU Instance with the profile; `-C` immediately creates a Compute Instance inside it. The number of partitions is the executor's concurrency, which the hook is handed for exactly this reason.

The second phase installs NVIDIA DCGM (`datacenter-gpu-manager`) and starts `nv-hostengine` if it is not already up. Both are best-effort — the phase does not fail the bootstrap when the package is unavailable.

There is no `image` hook: the plugin adds no packages and sets no image environment.

### `run`

On the first task in each process, the plugin runs `nvidia-smi -L`, extracts the MIG UUIDs by regex, and sets `CUDA_VISIBLE_DEVICES` to the one at `info.worker`.

It has to be the process that will import the GPU library and run the task — the child, not the worker that spawned it — which is why this is a `run` hook and not `setup`. A module-global flag under a lock keeps it to once per process; the pin is then stable for that child's life.

Afterwards CUDA presents the assigned partition as the only device: `torch.device("cuda")` resolves to it, `torch.cuda.device_count()` returns 1, and the isolation is enforced by the hardware as much as the variable.

## When to use MIG

**Independent runs** are the main case — hyperparameter sweeps, architecture comparisons, ablations, where each run is its own job. Each gets guaranteed resources, so results are directly comparable with no contention noise.

**Concurrent inference** works when each model fits a partition's memory: better isolation than MPS, with no risk of one model's allocation starving another.

MIG is **not** useful for workloads that saturate the GPU (a `7g.80gb` slice has fewer SMs than the whole card), for distributed training (DDP and FSDP expect each rank to own a full GPU — use [`Torch`](torch.md)), on consumer GPUs, or for workloads whose resource needs vary, since partitions are fixed at bootstrap. For that last case, [MPS](mps.md) is the flexible alternative.

## Usage

### Independent training on partitions

```python
import skyward as sky

PARTITIONS = 2
PROFILE = "3g.40gb"


@sky.function
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
        nn.Linear(256, 10),
    ).to(device)

    x = torch.randn(5000, 784, device=device)
    y = torch.randint(0, 10, (5000,), device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        epoch_loss = 0.0
        correct = total = 0
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


with sky.Compute(
    provider=sky.Verda(),
    nodes=1,
    accelerator=sky.accelerators.A100(),
    executor=sky.Executor(type="process", concurrency=PARTITIONS, reuse=True),
    image=sky.Image(pip=["torch"]),
    plugins=[sky.plugins.Mig(profile=PROFILE)],
) as compute:
    tasks = [train_on_partition(epochs=10, lr=1e-3) for _ in range(PARTITIONS)]
    results = list(sky.gather(*tasks, stream=True) >> compute)
```

The concurrency and the profile must agree: `3g.40gb` on an A100 supports exactly two partitions, so `concurrency=2`.

PyTorch comes from `Image(pip=["torch"])`, not from `sky.plugins.Torch()`. The torch plugin forms a process group, which independent MIG partitions have no use for — they are separate workloads, not a distributed job.

### Maximum partitions

```python
with sky.Compute(
    provider=sky.AWS(),
    nodes=1,
    accelerator=sky.accelerators.A100(),
    executor=sky.Executor(type="process", concurrency=7, reuse=True),
    image=sky.Image(pip=["torch"]),
    plugins=[sky.plugins.Mig(profile="1g.10gb")],
) as compute:
    tasks = [evaluate(model_id=i) for i in range(7)]
    results = list(sky.gather(*tasks, stream=True) >> compute)
```

Seven slices of ~10 GB each from one A100 80GB. Enough for a small model apiece; less compute per slice than the whole card would give one of them.

### Multi-node

MIG is per-GPU, not per-compute. On several nodes, each partitions its own card:

```python
with sky.Compute(
    provider=sky.AWS(),
    nodes=3,
    accelerator=sky.accelerators.A100(),
    executor=sky.Executor(type="process", concurrency=2, reuse=True),
    image=sky.Image(pip=["torch"]),
    plugins=[sky.plugins.Mig(profile="3g.40gb")],
) as compute:
    tasks = [train_on_partition(epochs=10, lr=lr) for lr in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]]
    results = list(sky.gather(*tasks, stream=True) >> compute)
```

Six independent slots — three nodes, two slices each. There is no gradient synchronization between them; this is for sweeps and embarrassingly parallel work.

## Next steps

- [NVIDIA MIG guide](../guides/nvidia-mig.md) — walkthrough with a training example
- [NVIDIA MPS](mps.md) — software sharing, the complementary approach
- [Worker Executors](../guides/worker-executors.md) — thread vs process
- [What are plugins?](index.md) — the hook model
