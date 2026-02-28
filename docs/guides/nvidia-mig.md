# NVIDIA MIG

A single A100 or H100 GPU has enormous compute capacity — often more than a single workload needs. NVIDIA Multi-Instance GPU (MIG) solves this by partitioning one physical GPU into multiple isolated instances, each with its own compute units, memory, and L2 cache. Two inference servers, two training runs, or two independent benchmarks can share the same card without any interference. No time-slicing, no memory contention — each partition behaves like a smaller, dedicated GPU.

The catch is that setting up MIG is manual and finicky. You need to enable MIG mode via `nvidia-smi`, create GPU instances with the right profile, create compute instances inside them, and then assign each process to its partition through `CUDA_VISIBLE_DEVICES` with MIG-specific UUIDs. Skyward's `mig` plugin handles all of this: it enables MIG during bootstrap, creates the partitions, and assigns each subprocess to its own device automatically.

## How MIG partitioning works

A MIG-capable GPU (A100, A30, H100) can be sliced into profiles like `1g.10gb`, `3g.40gb`, or `7g.80gb`. The first number indicates how many compute slices the partition gets; the second is its dedicated memory. An A100 80GB can be split into two `3g.40gb` partitions (42 SMs and ~40GB each) or seven `1g.10gb` partitions (14 SMs and ~10GB each). The profiles are fixed — you choose one when creating instances, and all partitions on a GPU must use the same profile.

Each partition gets its own MIG UUID, visible through `nvidia-smi -L`. When a process sets `CUDA_VISIBLE_DEVICES` to a MIG UUID, CUDA presents that partition as the only available device. The process sees a GPU with fewer SMs and less memory, but the isolation is hardware-enforced — there's no way for one partition to access another's memory or steal its compute cycles.

## The `mig` plugin

Add `sky.plugins.mig()` to your pool and set the worker concurrency to match the number of partitions:

```python
--8<-- "examples/guides/18_nvidia_mig.py:5:6"
```

```python
--8<-- "examples/guides/18_nvidia_mig.py:61:68"
```

Three things happen here. First, the `mig` plugin's `profile` parameter tells Skyward which MIG profile to create — in this case, `3g.40gb`, which splits the GPU into two partitions. Second, `Worker(concurrency=2, executor="process")` creates two loky subprocesses on the node, one per partition. Third, `Image(pip=["torch"])` installs PyTorch on the worker — we're not using `sky.plugins.torch()` here because the torch plugin is designed for multi-node distributed training (DDP), and MIG partitions on a single GPU are independent workloads, not a distributed cluster.

The concurrency and the profile must agree: if a profile creates two partitions, concurrency should be two. If you set concurrency to three but the GPU only supports two instances of your profile, the bootstrap will fail.

## What the plugin does

The plugin handles the full MIG lifecycle so your `@sky.compute` function doesn't need to know about partitioning at all. When the node boots, the plugin enables MIG mode and creates the requested partitions. When a subprocess picks up its first task, the plugin detects which partition belongs to it and sets `CUDA_VISIBLE_DEVICES` accordingly. From that point on, CUDA sees only that partition — your code calls `torch.device("cuda")` and lands on the right slice of the GPU automatically.

For the technical details on how each hook works, see the [MIG plugin reference](../plugins/mig.md).

## Training on a partition

The function itself doesn't know about MIG — it just sees a CUDA device:

```python
--8<-- "examples/guides/18_nvidia_mig.py:9:57"
```

There are no `if torch.cuda.is_available()` guards — this code runs on a GPU node with MIG partitions, so CUDA is always present. `torch.device("cuda")` resolves to the MIG partition assigned by the plugin. The function trains a small feedforward network on synthetic data and returns the final metrics along with the worker index and partition UUID.

`instance_info().worker` returns the subprocess index (0 or 1), which the plugin already used to assign the correct MIG partition. Each subprocess runs independently in its own loky process with its own CUDA context, so there's no shared state between the two training runs.

## Running both partitions in parallel

Dispatch one task per partition using `gather()`:

```python
--8<-- "examples/guides/18_nvidia_mig.py:69:70"
```

Both tasks execute simultaneously — worker 0 on partition 0, worker 1 on partition 1. Since MIG provides hardware-level isolation, neither task impacts the other's performance. The wall time should be close to the time of a single task, not the sum.

## Choosing the right profile

The profile determines the trade-off between partition count and per-partition resources. On an A100 80GB, a `3g.40gb` profile splits the GPU into two partitions with ~40 GB and 42 SMs each, while a `1g.10gb` profile fits seven partitions with ~10 GB and 14 SMs each. Other GPUs have different profile sets — an RTX PRO 6000 supports four profiles instead of seven. Throughput scales roughly linearly with profile size. The first number in the profile name indicates compute slices, not a fraction of the GPU: `3g.40gb` gets about 3/7 of the SMs, not half.

See [NVIDIA's supported GPUs page](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html) for the full list of profiles per GPU, or run `nvidia-smi mig -lgip` on a MIG-capable node to see what's available.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/18_nvidia_mig.py
```

---

**What you learned:**

- **`sky.plugins.mig(profile="3g.40gb")`** — enables MIG mode, creates partitions during bootstrap, and assigns each subprocess to its own MIG device.
- **`Worker(concurrency=N, executor="process")`** — one loky subprocess per partition; concurrency must match the number of partitions the profile supports.
- **Hardware isolation** — each partition has dedicated compute units, memory, and L2 cache; no time-slicing or contention.
- **Transparent to `@sky.compute`** — functions see a normal CUDA device; the plugin sets `CUDA_VISIBLE_DEVICES` per subprocess via the `around_process` hook.
- **Profile choice** — more partitions means less compute per partition; pick based on your workload's memory and throughput needs.
