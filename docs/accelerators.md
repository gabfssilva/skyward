# Accelerators

Every cloud provider names GPU instances differently. AWS calls an A100 machine a `p4d.24xlarge`. RunPod uses a `gpuTypeId`. Vast.ai filters by GPU model, and spells the same card `NVIDIA H100 80GB SXM5`, `H100-80G-PCIe`, or `1x H100` depending on the host. The `accelerator` parameter abstracts all of it: you describe the hardware you want, and the provider figures out how to get it.

That abstraction is a single canonical catalog. A provider's raw name is parsed once into a canonical accelerator plus its VRAM, which is what makes "the cheapest H100 across my accounts" a question with an answer. Form factor (SXM, PCIe, NVL) is deliberately not part of the canonical name — it is what the raw string and the VRAM already tell you, and folding it in is what splits `h100` from `h100-sxm` in the first place.

This page is both a practical guide for choosing hardware and a technical reference for the accelerators themselves.

## Choosing an accelerator

### By workload

| Workload | Recommended | Why |
|----------|------------|-----|
| Development/prototyping | T4 | Cheapest GPU on any provider; 16 GB is enough for small models |
| Inference (small models) | L4 | 242 TFLOPS FP8 at 72W |
| Inference (large models) | L40S, RTX 4090 | 48/24 GB memory, strong FP8 |
| Fine-tuning (LoRA) | RTX 4090, A100 | 24-40 GB fits adapters for 7-13B models |
| Full fine-tuning | A100, H100 | Need full model + optimizer in memory |
| Pre-training | 8x H100, 8x B200 | Maximum compute + NVLink for gradient sync |
| Budget training | RTX 3090, RTX 4090 | Best TFLOPS/$ on marketplace providers |
| Maximum memory (single GPU) | MI300X | 192 GB fits 70B in FP16 without sharding |

### By model size

| Parameters | Minimum | Recommended | Notes |
|------------|---------|-------------|-------|
| < 1B | T4 (16 GB) | L4 (24 GB) | FP8 inference on L4 is fast and cheap |
| 1-7B | A10G (24 GB) | RTX 4090 (24 GB) | LoRA fits on 24 GB; full fine-tune needs 40+ GB |
| 7-13B | A100 (80 GB) | A100 (80 GB) | Optimizer states double memory needs |
| 13-70B | A100 (80 GB) | 2x H100 or MI300X | MI300X fits 70B on one card |
| 70B+ | 4x H100 | 8x H100 or 8x B200 | Tensor parallelism across NVLink |

Price is the axis these tables deliberately leave out, because it moves. Ask your own accounts:

```bash
sky offers list --accelerator H100 --min-vram 80 --limit 10
sky offers summary --accelerator A100
```

## Requesting an accelerator

Use a factory under `sky.accelerators`:

```python
import skyward as sky

with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.A100(),
) as compute:
    result = train(data) >> compute
```

Each factory returns a frozen `Accelerator(name, count)`. There is no hand-written factory behind each name — the attribute is resolved against the catalog, so a name that exists is a name a provider can be asked for. Factory attributes use Python identifiers for catalog names containing hyphens: `RTX_4090`, `H100_NVL`, `TPU_V5P_8`, `RTX_PRO_6000`.

`count` is the number of accelerators per node:

```python
sky.accelerators.H100(count=4)
sky.accelerators.RTX_4090(count=1)
sky.accelerators.TPU_V5P(count=8)
```

`count` is an integer, and it is independent of `nodes` — the number of machines:

```python
with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.H100(count=4),
    nodes=2,
) as compute:
    ...  # two machines, four H100s each, eight in total
```

Memory and form-factor variants are separate catalog entries rather than arguments: `H100()` and `H100_NVL()`, not `H100(memory=...)`. For CPU-only compute, leave `accelerator=None`.

### Detecting at runtime

`Info` describes the compute's topology — rank, peers, worker slots — and says nothing about the hardware. Ask the machine itself:

```python
@sky.function
def check_gpu():
    import torch

    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "rank": sky.instance_info().rank,
    }
```

## At a glance

Performance numbers are **dense tensor/matrix core** throughput — the metric that matters for ML. Structured sparsity (2:4) doubles these figures, but most training workloads don't benefit from it. Memory is the catalog's VRAM figure, which is what `--min-vram` filters on.

### NVIDIA datacenter

| Factory | BF16 | FP8 | Memory | Bandwidth | TDP |
|---|---|---|---|---|---|
| `B300()` | — | — | 288 GB HBM3e | — | — |
| `B200()` | 2,250 | 4,500 | 192 GB HBM3e | 7,700 GB/s | 1,000W |
| `H200()` | 989 | 1,979 | 141 GB HBM3e | 4,800 GB/s | 700W |
| `H100_NVL()` | 989 | 1,979 | 94 GB HBM3 | 3,900 GB/s | 400W |
| `H100()` | 989 | 1,979 | 80 GB HBM3 | 3,350 GB/s | 700W |
| `GH200()` | 989 | 1,979 | 96 GB HBM3 | 4,000 GB/s | 1,000W |
| `A100()` | 312 | — | 80 GB HBM2e | 2,039 GB/s | 400W |
| `L40S()` | 362 | 733 | 48 GB GDDR6 | 864 GB/s | 350W |
| `A40()` | 150 | — | 48 GB GDDR6 | 696 GB/s | 300W |
| `L4()` | 121 | 242 | 24 GB GDDR6 | 300 GB/s | 72W |
| `A10G()` | 125 | — | 24 GB GDDR6 | 600 GB/s | 150W |
| `V100()` | — | — | 32 GB HBM2 | 900 GB/s | 300W |
| `T4()` | 65* | — | 16 GB GDDR6 | 320 GB/s | 70W |

<small>All values in TFLOPS (dense tensor core). *T4 and V100 predate BF16 — the T4 figure is FP16. "—" means the hardware does not support that precision, or the figure is not published.</small>

### NVIDIA consumer

| Factory | BF16 | FP8 | Memory | Bandwidth | TDP |
|---|---|---|---|---|---|
| `RTX_5090()` | 210 | 419 | 32 GB GDDR7 | 1,792 GB/s | 575W |
| `RTX_4090()` | 165 | 330 | 24 GB GDDR6X | 1,008 GB/s | 450W |
| `RTX_5080()` | 113 | 225 | 16 GB GDDR7 | 960 GB/s | 360W |
| `RTX_4080SUPER()` | 105 | 209 | 16 GB GDDR6X | 736 GB/s | 320W |
| `RTX_3090()` | 71 | — | 24 GB GDDR6X | 936 GB/s | 350W |

The RTX 3090 is the cheapest way to get 24 GB of VRAM. It has no FP8 and no BF16 tensor path worth using, so mixed precision means explicit FP16. For LoRA fine-tuning and inference on models up to ~13B it is hard to beat on price.

### AMD, Google, Intel, AWS

| Factory | BF16 | FP8 | Memory | Bandwidth | TDP |
|---|---|---|---|---|---|
| `MI355X()` | 2,300 | 4,600 | 288 GB HBM3e | 8,000 GB/s | 1,400W |
| `MI325X()` | 1,307 | 2,615 | 256 GB HBM3e | 6,000 GB/s | 1,000W |
| `MI300X()` | 1,307 | 2,615 | 192 GB HBM3 | 5,300 GB/s | 750W |
| `MI250X()` | 383 | — | 128 GB HBM2e | 3,277 GB/s | 500W |
| `GAUDI3()` | 1,835 | 1,835 | 128 GB HBM2e | 3,700 GB/s | 900W |
| `TPU_V5P()` | 459 | 459 | 95 GB HBM2e | 2,765 GB/s | ~250W |
| `TPU_V5E()` | 197 | — | 16 GB HBM2e | 819 GB/s | ~120W |
| `TRAINIUM2()` | 667 | 1,299 | 64 GB HBM3 | 2,900 GB/s | ~500W |
| `INFERENTIA2()` | 190 | — | 32 GB HBM2e | 820 GB/s | ~175W |

<small>AMD values are matrix engine TFLOPS. Gaudi 3 values are MME throughput. TPU and Trainium TDP are estimates — official figures are not published. TPUs are GCP-only; Trainium and Inferentia are AWS-only.</small>

## Hardware reference

### Hopper and Blackwell

The workhorses of large-scale training and inference. Blackwell roughly doubles Hopper's tensor throughput and memory bandwidth while adding FP4.

| Spec | H100 SXM | H200 | B200 |
|---|---|---|---|
| **Architecture** | Hopper | Hopper | Blackwell |
| **Process** | TSMC 4N | TSMC 4N | TSMC 4NP |
| **CUDA cores** | 16,896 | 16,896 | 18,432 |
| **Tensor cores** | 4th gen (528) | 4th gen (528) | 5th gen (592) |
| **Compute capability** | 9.0 | 9.0 | 10.0 |
| **FP32** | 67 TFLOPS | 67 TFLOPS | 75 TFLOPS |
| **BF16 tensor** | 989 / 1,979 | 989 / 1,979 | 2,250 / 4,500 |
| **FP8 tensor** | 1,979 / 3,958 | 1,979 / 3,958 | 4,500 / 9,000 |
| **FP4 tensor** | — | — | 9,000 / 18,000 |
| **Memory** | 80 GB HBM3 | 141 GB HBM3e | 192 GB HBM3e |
| **Mem bandwidth** | 3,350 GB/s | 4,800 GB/s | 7,700 GB/s |
| **NVLink** | 4th gen, 900 GB/s | 4th gen, 900 GB/s | 5th gen, 1,800 GB/s |
| **PCIe** | Gen 5 | Gen 5 | Gen 5 |
| **TDP** | 700W | 700W | 1,000W |

<small>Tensor values shown as dense / sparse.</small>

The H200 is compute-identical to the H100 — same die, same clocks. The difference is memory: HBM3e at 141 GB instead of 80 GB HBM3, with 43% more bandwidth. That matters for inference of large models, where capacity and bandwidth are the bottleneck rather than raw compute. If your model fits in 80 GB, the H200 buys you nothing and costs more.

The B200 uses a dual-die design: two dies connected by a 10 TB/s internal link. From software's perspective it is a single GPU.

### Ampere and Ada Lovelace

Ampere (A100, A10G) introduced TF32 and structural sparsity. Ada Lovelace (L40S, L4) added FP8 through 4th-gen tensor cores and moved to TSMC 4N — the same process as Hopper.

| Spec | A100 80GB | L40S | L4 | A10G |
|---|---|---|---|---|
| **Architecture** | Ampere | Ada Lovelace | Ada Lovelace | Ampere |
| **Process** | TSMC 7nm | TSMC 4N | TSMC 4N | Samsung 8nm |
| **CUDA cores** | 6,912 | 18,176 | 7,424 | 9,216 |
| **Tensor cores** | 3rd gen (432) | 4th gen (568) | 4th gen (232) | 3rd gen (288) |
| **Compute capability** | 8.0 | 8.9 | 8.9 | 8.6 |
| **FP32** | 19.5 TFLOPS | 91.6 TFLOPS | 30.3 TFLOPS | 31.2 TFLOPS |
| **BF16 tensor** | 312 / 624 | 362 / 733 | 121 / 242 | 125 / 250 |
| **FP8 tensor** | — | 733 / 1,466 | 242 / 485 | — |
| **INT8 tensor** | 624 / 1,248 | 733 / 1,466 | 242 / 485 | 250 / 500 |
| **Memory** | 80 GB HBM2e | 48 GB GDDR6 | 24 GB GDDR6 | 24 GB GDDR6 |
| **Mem bandwidth** | 2,039 GB/s | 864 GB/s | 300 GB/s | 600 GB/s |
| **NVLink** | 3rd gen, 600 GB/s | — | — | — |
| **TDP** | 400W | 350W | 72W | 150W |

<small>Tensor values shown as dense / sparse. A100 and A10G lack FP8 (3rd-gen tensor cores). L40S and L4 lack NVLink.</small>

The A100 remains the price/performance sweet spot for training, and its 80 GB HBM2e fits most 7-13B parameter models. The L40S has 3x more raw FP32 compute but less than half the memory bandwidth — better for inference and mixed workloads than for large-scale training. The L4 at 72W is the best value for inference: its FP8 throughput (242 TFLOPS) rivals the A100's BF16 (312 TFLOPS) at a fraction of the cost and power.

The absence of NVLink on L40S and L4 matters for multi-GPU training. Gradient synchronization falls back to PCIe, which is roughly an order of magnitude slower. For data-parallel training across several cards in one node, that difference dominates.

### T4

| Spec | T4 |
|---|---|
| **Architecture** | Turing |
| **Process** | TSMC 12nm |
| **CUDA cores** | 2,560 |
| **Tensor cores** | 2nd gen (320) |
| **Compute capability** | 7.5 |
| **FP32** | 8.1 TFLOPS |
| **FP16 tensor** | 65 / 130 |
| **INT8 tensor** | 130 / 260 |
| **Memory** | 16 GB GDDR6 |
| **Mem bandwidth** | 320 GB/s |
| **TDP** | 70W |

The T4 is the cheapest GPU available across Skyward's providers, and at 70W it is also the least power-hungry. Its 16 GB is enough for inference on models up to roughly 3B parameters and for development work. Turing's 2nd-gen tensor cores have no BF16 and no FP8, so mixed precision means explicit FP16 casting.

## The catalog

The canonical catalog carries the normalized name, VRAM, manufacturer, architecture, and CUDA compatibility range for over 140 accelerators — the ones above plus the long tail of consumer and workstation cards that marketplace providers list. `sky.accelerators` is generated from that catalog, so a name it accepts is a name the offer normalizer recognizes. Do not maintain a second list of provider spellings in application code.

To see what your accounts can actually get:

```bash
sky offers list --accelerator RTX_4090 --refresh
sky offers list --min-vram 80 --max-price 3
```

Offer rows carry the normalized accelerator, VRAM, CPU, memory, region, provider, and the provider's billing unit. `--refresh` asks the daemon to refetch stale data before answering; if a refresh fails, the stale rows remain and the provider records the error.

---

## Next steps

- **[Choosing the best provider](choosing-a-provider.md)** — which provider to rent which GPU from
- **[Compare accelerators](compare.md)** — interactive comparison with live pricing
- **[Using accelerators](guides/using-accelerators.md)** — a walkthrough with runnable code
- **[Fractional GPUs](guides/fractional-gpus.md)** — MIG and MPS for sharing one card
- **[Accelerator API](reference/accelerators.md)** — the generated factory reference
