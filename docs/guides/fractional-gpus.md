# Fractional GPUs

Not every workload needs a full GPU. Running DistilBERT inference on an 80GB A100 is like renting a freight truck to deliver a letter — technically works, but you're paying for capacity you'll never use. Fractional GPUs solve this by giving you a slice of a physical GPU — a quarter of an L4, half an A16 — with proportionally less memory and a proportionally lower price. The workload sees a GPU device with reduced VRAM, but otherwise behaves identically to a full card.

Cloud providers implement fractional GPUs at the hypervisor level. AWS offers the G6f instance family, which partitions NVIDIA L4 GPUs into 1/8, 1/4, and 1/2 slices. Vultr exposes fractional plans for GPUs like the A16, where the instance receives a portion of the card's VRAM. In both cases, the partition is transparent to CUDA — your code calls `torch.device("cuda")` and lands on the assigned slice without any special configuration.

## Requesting a fractional GPU

Pass a fractional `count` to any accelerator factory:

```python
--8<-- "examples/guides/20_fractional_gpus.py:1:4"
```

```python
--8<-- "examples/guides/20_fractional_gpus.py:39:46"
```

`sky.accelerators.L4(count=0.5)` tells Skyward to find an instance with half an L4 — 12 GB of the L4's 24 GB VRAM. On AWS, this resolves to a `g6f.4xlarge` instance. The `count` parameter accepts any float: `0.125` for 1/8 of a GPU, `0.25` for a quarter, `0.5` for half. When a provider doesn't have a plan that matches the exact fraction, Skyward skips it — there's no rounding up to a full GPU.

Fractional GPU instances use NVIDIA vGPU, which requires a GRID driver instead of the standard datacenter driver. For providers that don't ship the GRID driver by default, Skyward handles this automatically — the provider injects a bootstrap plugin that downloads and installs the correct driver. On AWS, this pulls the GRID driver from `s3://ec2-linux-nvidia-drivers` and compiles the proprietary kernel module. The `provision_timeout=600` accounts for the extra bootstrap time (~4 minutes for driver compilation).

Vultr works the same way. The provider computes the fractional count by dividing the plan's reported VRAM by the GPU's full memory. A Vultr plan offering 8 GB on an A16 (which has 16 GB total) becomes `count=0.5` automatically:

```python
sky.Compute(
    provider=sky.Vultr(),
    accelerator=sky.accelerators.A16(count=0.5),
    image=sky.Image(pip=["torch", "transformers"]),
)
```

## Running inference on a slice

The function itself doesn't know or care that it's running on a fractional GPU. CUDA presents the slice as the only visible device:

```python
--8<-- "examples/guides/20_fractional_gpus.py:6:29"
```

`torch.cuda.get_device_properties(0).total_memory` reports the VRAM available to this slice, not the full physical GPU. On a `g6f.4xlarge` (half L4), this is approximately 12 GB. The HuggingFace pipeline loads the model, tokenizes the input, runs forward passes on the GPU, and returns predictions — identical to running on a full card, just with less memory headroom.

`instance_info().accelerators` returns the fractional count (0.5 in this case), so your code can introspect how much GPU it was allocated if needed.

```python
--8<-- "examples/guides/20_fractional_gpus.py:32:36"
```

```python
--8<-- "examples/guides/20_fractional_gpus.py:46:51"
```

## Fractional GPUs vs MIG

Both fractional GPUs and [NVIDIA MIG](nvidia-mig.md) give you a portion of a physical GPU, but they operate at different levels.

**Fractional GPUs are provider-managed.** The cloud provider partitions the GPU before you see the instance. You request `count=0.5` and receive a VM with half a GPU already carved out. Skyward installs the NVIDIA GRID driver during bootstrap — the only driver that supports vGPU devices. Your code sees a normal CUDA device; the partitioning is invisible.

**MIG is user-managed.** You rent a full GPU (A100, H100) and partition it yourself using `sky.plugins.mig(profile="3g.40gb")`. The plugin enables MIG mode during bootstrap, creates the partitions, and assigns each subprocess to its own device. This gives you hardware-enforced isolation with dedicated streaming multiprocessors, memory controllers, and L2 cache per partition — stronger guarantees than hypervisor-level slicing.

| | Fractional GPU | MIG |
|---|---|---|
| Setup | Automatic — Skyward installs GRID driver | Plugin enables MIG mode during bootstrap |
| Isolation | Hypervisor / vGPU level | Hardware level (dedicated SMs, memory, L2 cache) |
| GPU models | L4 (AWS), A16 (Vultr) | A100, A30, H100, H200, B200 |
| Granularity | Provider-defined fractions (1/8, 1/4, 1/2) | MIG profiles (1g.10gb through 7g.80gb) |
| Multiple workloads per GPU | One workload per VM | Multiple subprocesses via `Worker(concurrency=N)` |
| Use case | Cost optimization for small inference workloads | Running multiple isolated workloads on a single large GPU |

If your workload fits in 3-12 GB of VRAM and you want the simplest possible setup, fractional GPUs are the right choice. If you need hardware isolation, want to run multiple workloads on the same card, or need fine-grained control over the partition profile, use MIG.

## When to use fractional GPUs

Fractional GPUs make sense when your workload's memory footprint is well below a full GPU. Inference with small to medium models (DistilBERT, GPT-2, ViT-base) rarely needs more than a few gigabytes. Development and debugging — iterating on training code with a small data sample — benefits from having a GPU available without paying full price. Batch jobs that process items sequentially can use the reduced throughput of a smaller slice without bottlenecking.

The trade-off is capacity. A 1/8 L4 slice has about 3 GB of VRAM — enough for small transformers, not enough for anything with more than ~500M parameters. A 1/2 L4 gives you 12 GB, which handles most medium models but won't fit a 7B parameter model. If your model doesn't load, there's no fallback — you need a larger fraction or a full GPU.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/20_fractional_gpus.py
```

---

**What you learned:**

- **`sky.accelerators.L4(count=0.5)`** — requests half a GPU; the `count` parameter accepts any float for the fraction you need.
- **Automatic GRID driver** — Skyward detects fractional GPU instances on AWS and installs the NVIDIA GRID driver during bootstrap; no manual configuration needed.
- **Transparent to CUDA** — fractional GPUs appear as regular CUDA devices with reduced VRAM; existing code works without modification.
- **AWS G6f instances** — offer 1/8, 1/4, and 1/2 fractions of NVIDIA L4 GPUs; use `provision_timeout=600` to account for driver installation.
- **Vultr fractional plans** — auto-detected from VRAM ratios; request via fractional `count` on the accelerator.
- **vs MIG** — fractional GPUs are simpler but offer weaker isolation; MIG provides hardware-enforced partitioning on larger GPUs.
