# PyTorch

PyTorch's distributed training model is built around `DistributedDataParallel`. Each process — one per node here — holds a complete copy of the model, computes gradients on its own shard, and DDP synchronizes them across processes through a collective backend (NCCL on GPUs, gloo on CPUs). The optimizer then steps with identical averaged gradients everywhere, keeping the copies in sync without explicit parameter transfers.

The hard part is the setup. Before `init_process_group()` can be called, every process needs the rendezvous address and port, the world size, and its own rank, all as environment variables. Normally you write a launch script or use `torchrun` to inject them. `sky.plugins.Torch()` reads the topology from the node and sets them before your function body runs.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `"nccl" \| "gloo"` | `"nccl"` | `nccl` on GPUs, `gloo` on CPUs |
| `cuda` | `str` | `"cu128"` | The CUDA build to install torch from, as a `download.pytorch.org/whl` suffix. Ignored for `gloo`, which takes the CPU wheel |
| `version` | `str \| None` | `None` | Pin, if the code needs one. Otherwise whatever the index has |

The `cuda` default is pinned rather than left to PyPI's, because PyPI tracks the newest CUDA and the newest CUDA outruns the driver the GPU images ship. A torch built for a CUDA the driver cannot load hangs on the first collective — which is exactly where it looks like a network fault and is not.

Note that the wheel index follows `backend`, not the compute's accelerator: `backend="gloo"` installs from `download.pytorch.org/whl/cpu`, and anything else from `download.pytorch.org/whl/{cuda}`.

`Torch` is a **collective** plugin: a compute running one cannot be resized, because removing a rank does not shrink the job, it hangs it at the next all-reduce.

## How it works

### `image`

Appends `torch` (optionally pinned) to the image's pip list, and the matching wheel index scoped to the `torch` package. Both are appended to whatever the image and the earlier plugins already asked for.

### `run`

On the first task in each process, the plugin sets `MASTER_ADDR` (rank zero's address), `MASTER_PORT` (29500), `RANK`, `WORLD_SIZE`, `NODE_RANK`, and `LOCAL_RANK`/`LOCAL_WORLD_SIZE` fixed at `0`/`1` — Skyward runs one rank per node — then calls `dist.init_process_group(backend=..., rank=..., world_size=...)`.

It happens on the first task rather than at worker startup because `init_process_group` is a collective, and the process that blocks in it has to be the one that runs the collective code afterwards. Under `executor="process"` that is the child, not the worker. A module-global flag under a lock keeps it to once per process; every task after calls straight through.

Unlike [`Accelerate`](accelerate.md), this hook does not check whether a group already exists. If you list both plugins, list `Torch` first.

## Usage

### Basic DDP training

```python
import skyward as sky


@sky.function
@sky.stdout(only="head")
def train() -> dict:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model = DDP(nn.Linear(784, 10).cuda())

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


with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.A100(),
    nodes=4,
    plugins=[sky.plugins.Torch()],
) as compute:
    results = train() @ compute
    for r in results:
        print(f"Rank {r['rank']}: loss={r['final_loss']:.4f}")
```

`@` broadcasts to all four nodes. Each runs the same function, but `dist.get_rank()` differs, and `DistributedSampler` partitions accordingly. `@sky.stdout(only="head")` silences the prints on every node but rank zero, so you see one set of epoch logs instead of four.

### Extra packages

The plugin installs `torch` only. Torchvision, torchaudio and anything else go in the image:

```python
with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.A100(),
    nodes=2,
    image=sky.Image(
        pip=["torchvision", "torchaudio"],
        pip_indexes=[
            sky.PipIndex(
                url="https://download.pytorch.org/whl/cu128",
                packages=["torchvision", "torchaudio"],
            ),
        ],
    ),
    plugins=[sky.plugins.Torch()],
) as compute:
    results = train() @ compute
```

### Pinning

```python
plugins=[sky.plugins.Torch(version="2.6.0", cuda="cu124")]
```

Installs `torch==2.6.0` from the CUDA 12.4 index. Pinning matters for reproducibility: different PyTorch versions differ in default behaviours, numerical stability and operator implementations.

### CPU only

```python
with sky.Compute(
    provider=sky.AWS(),
    nodes=4,
    plugins=[sky.plugins.Torch(backend="gloo")],
) as compute:
    results = train() @ compute
```

`gloo` selects the CPU wheel index and PyTorch's CPU-compatible collective backend.

### Combining with HuggingFace

```python
with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.A100(),
    nodes=2,
    image=sky.Image(pip=["transformers", "datasets"]),
    plugins=[
        sky.plugins.Torch(),
        sky.plugins.HuggingFace(token=os.environ["HF_TOKEN"]),
    ],
) as compute:
    results = finetune() @ compute
```

`Torch` forms the process group; `HuggingFace` installs `huggingface_hub` and puts the token you pass into the worker's environment as `HF_TOKEN`. The token is not read from your environment for you — pass it. It does **not** install `transformers` or `datasets` — those go in the image. Inside the function, `Trainer` detects the distributed environment the torch plugin set up.

## Next steps

- [PyTorch Distributed guide](../guides/pytorch-distributed.md) — a DDP walkthrough
- [PyTorch Model Roundtrip guide](../guides/torch-model-roundtrip.md) — sending models to and from the cloud
- [Accelerate](accelerate.md) — FSDP and DeepSpeed on top of the same group
- [What are plugins?](index.md) — the hook model
