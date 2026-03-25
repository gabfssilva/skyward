# Accelerators

Every cloud provider names GPU instances differently. AWS calls an A100 machine a `p4d.24xlarge`. RunPod uses a `gpuTypeId`. VastAI filters by GPU model. The `accelerator` parameter on `Compute` abstracts this: you describe the hardware you want, and the provider figures out how to get it.

This page serves as both a practical guide for choosing hardware in Skyward and a technical reference for ML accelerators. The comparison tables cover 18 accelerators across NVIDIA, AMD, Google, Intel, and AWS — from the $0.11/hr T4 (VastAI) to the $6/hr+ B200 (Verda).

## Choosing an accelerator

### By workload

| Workload | Recommended | Why |
|----------|------------|-----|
| Development/prototyping | T4 | $0.13/hr on VastAI, 16 GB is enough for small models |
| Inference (small models) | L4 | 242 TFLOPS FP8 at 72W and $0.22/hr on RunPod |
| Inference (large models) | L40S, RTX 4090 | 48/24 GB memory, strong FP8 |
| Fine-tuning (LoRA) | RTX 4090, A100-40GB | 24-40 GB fits adapters for 7-13B models |
| Full fine-tuning | A100-80GB, H100 | Need full model + optimizer in memory |
| Pre-training | 8x H100, 8x B200 | Maximum compute + NVLink for gradient sync |
| Budget training | RTX 3090, RTX 4090 | Best TFLOPS/$ on marketplace providers |
| Maximum memory (single GPU) | MI300X | 192 GB fits 70B in FP16 without sharding |

### By model size

| Parameters | Minimum | Recommended | Notes |
|------------|---------|-------------|-------|
| < 1B | T4 (16 GB) | L4 (24 GB) | FP8 inference on L4 is fast and cheap |
| 1-7B | A10G (24 GB) | RTX 4090 (24 GB) | LoRA fits on 24 GB; full fine-tune needs 40+ GB |
| 7-13B | A100-40GB | A100-80GB | Optimizer states double memory needs |
| 13-70B | A100-80GB | 2x H100 or MI300X | MI300X fits 70B on one card |
| 70B+ | 4x H100 | 8x H100 or 8x B200 | Tensor parallelism across NVLink |

### By efficiency

| Metric | Best choice | Value | Price | Provider |
|--------|-----------|-------|-------|----------|
| **TFLOPS/$** (BF16, spot) | L40S | 1,097 TFLOPS/$ | $0.33/hr | RunPod |
| **TFLOPS/$** (BF16, consumer) | RTX 4090 | 688 TFLOPS/$ | $0.24/hr | RunPod |
| **TFLOPS/$** (BF16, datacenter) | A100 | 495 TFLOPS/$ | $0.63/hr | VastAI |
| **TFLOPS/W** (BF16) | L4 | 1.68 TFLOPS/W | $0.22/hr | RunPod |
| **GB/$** (memory per dollar) | RTX 3090 | 141 GB/$ | $0.17/hr | RunPod |
| **GB/W** (memory per watt) | L4 | 0.33 GB/W | $0.22/hr | RunPod |

<small>TFLOPS/$ computed as BF16 dense TFLOPS / cheapest spot $/hr. Higher is better.</small>

## At a glance

Performance numbers below are **dense tensor/matrix core** throughput — the metric that matters for ML. Structured sparsity (2:4) doubles these figures, but most training workloads don't benefit from it. Prices are the cheapest spot rate across Skyward providers at time of writing.

### NVIDIA datacenter

| Accelerator | BF16 | FP8 | Memory | Bandwidth | TDP | From |
|---|---|---|---|---|---|---|
| B200 | 2,250 | 4,500 | 180 GB HBM3e | 7,700 GB/s | 1,000W | $6.26/hr (Verda) |
| H200 | 989 | 1,979 | 141 GB HBM3e | 4,800 GB/s | 700W | $2.29/hr (RunPod) |
| H100 SXM | 989 | 1,979 | 80 GB HBM3 | 3,350 GB/s | 700W | $1.50/hr (VastAI) |
| A100 80GB | 312 | — | 80 GB HBM2e | 2,039 GB/s | 400W | $0.62/hr (VastAI) |
| L40S | 362 | 733 | 48 GB GDDR6 | 864 GB/s | 350W | $0.33/hr (RunPod) |
| L4 | 121 | 242 | 24 GB GDDR6 | 300 GB/s | 72W | $0.22/hr (RunPod) |
| A10G | 125 | — | 24 GB GDDR6 | 600 GB/s | 150W | $2.01/hr (AWS) |
| T4 | 65* | — | 16 GB GDDR6 | 320 GB/s | 70W | $0.11/hr (VastAI) |

<small>All values in TFLOPS (dense tensor core). *T4 uses FP16 — Turing lacks native BF16. "—" in spec columns = hardware does not support this precision. "—" in price column = no offers currently listed on Skyward providers.</small>

### NVIDIA consumer

| Accelerator | BF16 | FP8 | Memory | Bandwidth | TDP | From |
|---|---|---|---|---|---|---|
| RTX 5090 | 210 | 419 | 32 GB GDDR7 | 1,792 GB/s | 575W | $0.53/hr (RunPod) |
| RTX 4090 | 165 | 330 | 24 GB GDDR6X | 1,008 GB/s | 450W | $0.24/hr (RunPod) |
| RTX 4080 S | 105 | 209 | 16 GB GDDR6X | 736 GB/s | 320W | $0.17/hr (RunPod) |
| RTX 3090 | 71 | — | 24 GB GDDR6X | 936 GB/s | 350W | $0.17/hr (RunPod) |

### AMD, Google, Intel, AWS

| Accelerator | BF16 | FP8 | Memory | Bandwidth | TDP | From |
|---|---|---|---|---|---|---|
| MI300X | 1,307 | 2,615 | 192 GB HBM3 | 5,300 GB/s | 750W | — |
| MI250X | 383 | — | 128 GB HBM2e | 3,277 GB/s | 500W | — |
| TPU v5p | 459 | 459 | 95 GB HBM2e | 2,765 GB/s | ~250W | GCP |
| TPU v5e | 197 | — | 16 GB HBM2e | 819 GB/s | ~120W | GCP |
| Gaudi 3 | 1,835 | 1,835 | 128 GB HBM2e | 3,700 GB/s | 900W | AWS |
| Trainium2 | 667 | 1,299 | 96 GB HBM3 | 2,900 GB/s | ~500W | AWS |

<small>AMD values are matrix engine TFLOPS. Gaudi 3 values are MME (Matrix Multiplication Engine) throughput. TPU/Trainium TDP are estimates — official figures not published.</small>

## Using accelerators

### Factory functions

Use the factory functions under `sky.accelerators`:

```python
sky.Compute(provider=sky.AWS(), accelerator=sky.accelerators.A100())
sky.Compute(provider=sky.AWS(), accelerator=sky.accelerators.H100(count=4))
sky.Compute(provider=sky.AWS(), accelerator=sky.accelerators.A100(memory="40GB"))
```

Each factory returns an `Accelerator` dataclass — frozen, immutable, with `name`, `memory`, `count`, and optional `metadata` (CUDA versions, form factors). The factory populates defaults from an internal catalog, so `sky.accelerators.H100()` already knows it has 80GB of HBM3 without you specifying it.

### Memory and form factor variants

Some accelerators ship in multiple configurations. Use keyword arguments to select the variant:

```python
# Memory variants
sky.accelerators.A100()              # 80GB (default)
sky.accelerators.A100(memory="40GB") # 40GB PCIe variant
sky.accelerators.V100(memory="32GB") # 32GB variant

# Form factor variants
sky.accelerators.H100()                        # Default
sky.accelerators.H100(form_factor="SXM")       # High-bandwidth SXM
sky.accelerators.H100(form_factor="NVL")       # NVLink 2-GPU module
```

### Multi-GPU

Pass `count` to request multiple accelerators per node:

```python
sky.accelerators.H100(count=4)    # 4x H100 per node
sky.accelerators.A100(count=8)    # 8x A100 per node
```

### Custom accelerators

For hardware not in the catalog — experimental chips, private clouds, or overriding defaults:

```python
import skyward as sky

my_gpu = sky.accelerators.Custom("My-GPU", memory="48GB")
my_gpu = sky.accelerators.Custom("H100-Custom", memory="80GB", count=8, cuda_min="12.0")
```

### Detecting at runtime

Inside a `@sky.function` function, `sky.instance_info()` reports what hardware the function is running on:

```python
@sky.function
def check_gpu():
    import torch

    info = sky.instance_info()

    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "accelerators": info.accelerators,
        "accelerator_info": info.accelerator,
    }
```

## Hardware reference

### NVIDIA datacenter

#### Hopper and Blackwell (current generation)

The workhorses of large-scale training and inference. Blackwell roughly doubles Hopper's tensor throughput and memory bandwidth while adding FP4 support.

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
| **Memory** | 80 GB HBM3 | 141 GB HBM3e | 180 GB HBM3e |
| **Mem bandwidth** | 3,350 GB/s | 4,800 GB/s | 7,700 GB/s |
| **NVLink** | 4th gen, 900 GB/s | 4th gen, 900 GB/s | 5th gen, 1,800 GB/s |
| **PCIe** | Gen 5 | Gen 5 | Gen 5 |
| **TDP** | 700W | 700W | 1,000W |

<small>Tensor values shown as dense / sparse.</small>

The H200 is compute-identical to the H100 — same die, same clocks. The difference is memory: HBM3e at 141 GB (vs 80 GB HBM3) with 43% more bandwidth (4.8 TB/s vs 3.35 TB/s). This matters most for inference of large models where memory capacity and bandwidth are the bottleneck, not raw compute.

The B200 uses a dual-die design: two GB100 dies connected by a 10 TB/s internal link (NV-HBI). From software's perspective it appears as a single GPU with 208 billion transistors.

| Factory | Memory | Notes |
|---------|--------|-------|
| `H100()` | 40/80GB HBM3 | Flagship training GPU. FP8 support. |
| `H100(form_factor="SXM")` | 80GB | High-bandwidth SXM variant. |
| `H100(form_factor="NVL")` | 94GB | NVLink 2-GPU module. |
| `H200()` | 141GB HBM3e | 1.4-1.9x inference speedup vs H100. |
| `B100()` | 192GB HBM3e | FP4 support, 2nd-gen transformer engine. |
| `B200()` | 192GB HBM3e | Flagship Blackwell. ~2.5x inference vs H100. |

#### Ampere and Ada Lovelace

Ampere (A100, A10G) introduced TF32 and structural sparsity. Ada Lovelace (L40S, L4) added FP8 via 4th-gen tensor cores and moved to TSMC 4N — the same process as Hopper.

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
| **PCIe** | Gen 4 | Gen 4 | Gen 4 | Gen 4 |
| **TDP** | 400W | 350W | 72W | 150W |

<small>Tensor values shown as dense / sparse. A100 and A10G lack FP8 (3rd-gen tensor cores). L40S and L4 lack NVLink.</small>

The A100 remains the price/performance sweet spot for training at $0.62/hr spot on VastAI. Its 80 GB HBM2e fits most 7-13B parameter models. The L40S offers 3x more raw FP32 compute but half the memory bandwidth — better suited for inference and mixed workloads than large-scale training. The L4 at 72W and $0.22/hr on RunPod is the best value for inference: its FP8 throughput (242 TFLOPS) rivals the A100's BF16 (312 TFLOPS) at less than half the cost.

| Factory | Memory | Notes |
|---------|--------|-------|
| `A100()` | 80GB HBM2e | Default. First GPU with TF32 and structural sparsity. |
| `A100(memory="40GB")` | 40GB | PCIe variant. |
| `A100(count=8)` | 8x 80GB | Multi-GPU. |
| `A800()` | 80GB | China-specific A100 variant. |
| `A40()` | 48GB GDDR6 | Professional visualization + compute. |
| `A10G()` | 24GB GDDR6 | AWS g5 instances. |
| `L4()` | 24GB GDDR6 | Ada Lovelace. Replaces T4 for inference. |
| `L40S()` | 48GB GDDR6 | Ada Lovelace. Compute-optimized. |

#### Legacy (still widely available)

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

<small>The T4 uses Turing's 2nd-gen tensor cores — no BF16 or FP8 support. FP16 is the highest-precision tensor format available.</small>

The T4 at $0.13/hr on VastAI and 70W is the cheapest GPU available on Skyward. Its 16 GB is enough for inference on models up to ~3B parameters and development work. No BF16 means you need explicit FP16 casting for mixed precision.

| Factory | Memory | Notes |
|---------|--------|-------|
| `T4()` | 16GB GDDR6 | Cheapest option for dev/inference. |
| `T4G()` | 16GB | ARM64 variant for AWS Graviton. |
| `V100()` | 16/32GB HBM2 | Volta. `V100(memory="32GB")` for the larger variant. |
| `P100()` | 16GB HBM2 | Pascal. First HBM GPU for deep learning. |

### NVIDIA consumer

Available primarily on marketplace providers like VastAI, RunPod, and TensorDock. Consumer cards lack NVLink (except the RTX 3090) and ECC memory, but deliver exceptional price/performance for single-GPU workloads.

| Spec | RTX 5090 | RTX 4090 | RTX 4080 S | RTX 3090 |
|---|---|---|---|---|
| **Architecture** | Blackwell (GB203) | Ada Lovelace (AD102) | Ada Lovelace (AD103) | Ampere (GA102) |
| **Process** | TSMC 4N | TSMC 4N | TSMC 4N | Samsung 8nm |
| **CUDA cores** | 21,760 | 16,384 | 10,240 | 10,496 |
| **Tensor cores** | 5th gen (680) | 4th gen (512) | 4th gen (320) | 3rd gen (328) |
| **Compute capability** | 12.0 | 8.9 | 8.9 | 8.6 |
| **FP32** | 105 TFLOPS | 82.6 TFLOPS | 52.2 TFLOPS | 35.6 TFLOPS |
| **BF16 tensor** | 210 / 419 | 165 / 330 | 105 / 209 | 71 / 142 |
| **FP8 tensor** | 419 / 838 | 330 / 661 | 209 / 418 | — |
| **INT8 tensor** | 838 / 1,676 | 661 / 1,321 | 418 / 836 | 285 / 569 |
| **Memory** | 32 GB GDDR7 | 24 GB GDDR6X | 16 GB GDDR6X | 24 GB GDDR6X |
| **Mem bandwidth** | 1,792 GB/s | 1,008 GB/s | 736 GB/s | 936 GB/s |
| **NVLink** | — | — | — | 3rd gen, 112.5 GB/s |
| **PCIe** | Gen 5 | Gen 4 | Gen 4 | Gen 4 |
| **TDP** | 575W | 450W | 320W | 350W |

<small>Tensor values shown as dense / sparse. RTX 3090 lacks FP8 (3rd-gen tensor cores, Ampere). RTX 3090 is the last consumer card with NVLink support — NVIDIA removed it from the 40 and 50 series. RTX 5090 moves to GDDR7, delivering 78% more bandwidth than the 4090.</small>

The RTX 4090 at $0.24/hr spot on RunPod is the most popular consumer GPU for ML — 165 TFLOPS BF16 with 24 GB of memory handles LoRA fine-tuning of 7B models comfortably. The RTX 3090 offers the same 24 GB at similar pricing on RunPod but without FP8 support. The RTX 5090 at $0.53/hr on RunPod brings 32 GB GDDR7 and PCIe Gen 5 but costs ~2x more.

```python
# RTX 50 series (Blackwell)
sky.accelerators.RTX_5090()    # 32GB
sky.accelerators.RTX_5080()    # 16GB

# RTX 40 series (Ada Lovelace)
sky.accelerators.RTX_4090()    # 24GB — popular for fine-tuning
sky.accelerators.RTX_4080()    # 16GB

# RTX 30 series (Ampere)
sky.accelerators.RTX_3090()    # 24GB
sky.accelerators.RTX_3080()    # 10GB

# Older generations also available: RTX 20xx, GTX 16xx, GTX 10xx
```

Workstation GPUs like `RTX_A6000()` (48GB), `RTX_6000_Ada()`, and `RTX_PRO_6000()` are also supported.

### AMD Instinct

AMD's datacenter accelerators use CDNA architecture with matrix cores (AMD's equivalent of tensor cores). The MI300X's 192 GB HBM3 makes it the highest-memory single-GPU available — enough to fit a 70B model in FP16 without sharding.

| Spec | MI300X | MI250X |
|---|---|---|
| **Architecture** | CDNA 3 | CDNA 2 |
| **Process** | TSMC 5nm + 6nm | TSMC 6nm |
| **Compute units** | 304 (8 XCDs) | 220 (2 GCDs) |
| **FP32 (matrix)** | 163 TFLOPS | 47.9 TFLOPS |
| **BF16 (matrix)** | 1,307 TFLOPS | 383 TFLOPS |
| **FP8 (matrix)** | 2,615 TFLOPS | — |
| **INT8 (matrix)** | 2,615 TOPS | 383 TOPS |
| **Memory** | 192 GB HBM3 | 128 GB HBM2e |
| **Mem bandwidth** | 5,300 GB/s | 3,277 GB/s |
| **Interconnect** | Infinity Fabric 4th gen, 896 GB/s (8-GPU) | Infinity Fabric 3rd gen |
| **TDP** | 750W | 500W |

<small>The MI300X uses 3.5D chiplet packaging: 8 XCD compute dies (5nm) + 4 I/O dies (6nm) with 256 MB Infinity Cache. The MI250X is a dual-GCD MCM — each GCD appears as a separate device to software. MI250X powered the Frontier exascale supercomputer.</small>

```python
pool = sky.Compute(
    provider=sky.VastAI(),
    accelerator=sky.accelerators.MI300X(),
)
```

| Factory | Memory | Architecture | Notes |
|---------|--------|-------------|-------|
| `MI300X()` | 192GB HBM3 | CDNA 3 | Designed for LLMs. |
| `MI300A()` | 128GB | CDNA 3 (APU) | Integrated CPU+GPU. |
| `MI250X()` | 128GB HBM2e | CDNA 2 | HPC workloads. |
| `MI250()` | 128GB HBM2e | CDNA 2 | Training. |
| `MI210()` | 64GB HBM2e | CDNA 2 | Training. |
| `MI100()` | 32GB HBM2 | CDNA 1 | Compute. |

### Google TPUs

Tensor Processing Units are Google's custom ASICs, available exclusively through GCP. TPUs are architecturally different from GPUs — they use systolic arrays optimized for BF16/INT8 matrix multiplication and connect via a dedicated inter-chip interconnect (ICI) in torus topologies.

| Spec | TPU v5p | TPU v5e |
|---|---|---|
| **Type** | Training-optimized | Cost-optimized |
| **BF16** | 459 TFLOPS | 197 TFLOPS |
| **FP8** | 459 TFLOPS | — |
| **INT8** | 918 TOPS | 393 TOPS |
| **Memory** | 95 GB HBM2e | 16 GB HBM2e |
| **Mem bandwidth** | 2,765 GB/s | 819 GB/s |
| **ICI bandwidth** | 1,200 GB/s (3D torus) | 400 GB/s (2D torus) |
| **TDP** | ~250W (liquid cooled) | ~120W (air cooled) |
| **Max pod size** | 8,960 chips | 256 chips |

<small>TPU v5p has 2 TensorCores + 4 SparseCores (2nd-gen) per chip for embedding-heavy workloads. TPU v5e has 1 TensorCore with 4 MXUs (128x128 systolic arrays). Google does not publish FP32 or TDP figures. ICI uses optical circuit switches for flexible topology.</small>

The v5p targets large-scale training (up to 8,960-chip pods). The v5e is cost-optimized for inference and training of models up to ~200B parameters, offering 2.3x price/performance over TPU v4.

```python
# Single TPU chip
sky.accelerators.TPUv5p()

# Multi-chip slices
sky.accelerators.TPUv5p_8()    # 8-chip slice
sky.accelerators.TPUv4_64()    # 64-chip slice
sky.accelerators.TPUv3_32()    # 32-chip slice
```

| Factory | Generation | Notes |
|---------|-----------|-------|
| `TPUv6()` | 6th gen (2024) | Latest. |
| `TPUv5p()` | 5th gen perf | Training-optimized. |
| `TPUv5e()` | 5th gen eff | Inference-optimized. |
| `TPUv4()` | 4th gen | General. |
| `TPUv3()` / `TPUv2()` | Legacy | Still available. |

### Intel Gaudi and AWS Trainium

Two non-GPU accelerators designed specifically for deep learning, each with distinct architectural bets.

| Spec | Gaudi 3 | Trainium2 |
|---|---|---|
| **Vendor** | Intel (Habana Labs) | AWS (Annapurna Labs) |
| **Process** | TSMC 5nm | 5nm |
| **Compute engines** | 8 MMEs + 64 TPCs | 8 NeuronCores-v3 |
| **BF16** | 1,835 TFLOPS (MME) | 667 TFLOPS |
| **FP8** | 1,835 TFLOPS (MME) | 1,299 TFLOPS |
| **FP8 sparse** | — | 2,563 TFLOPS |
| **Memory** | 128 GB HBM2e | 96 GB HBM3 |
| **Mem bandwidth** | 3,700 GB/s | 2,900 GB/s |
| **On-chip SRAM** | 96 MB (12.8 TB/s) | 224 MB scratchpad |
| **Interconnect** | 24x 200GbE RoCEv2 (4.8 Tbps) | NeuronLink-v3 (1,280 GB/s) |
| **TDP** | 900W (OAM) | ~500W |

<small>Gaudi 3 is unique in having 24 on-die RDMA/Ethernet ports — no external NIC needed. It uses a dual-chiplet design (two 5nm dies). Trainium2 has 16 dedicated Collective Communication cores for distributed training. A Trn2 instance has 16 chips (20.8 PFLOPS FP8); UltraServer packs 64 chips (83.2 PFLOPS).</small>

Gaudi 3 achieves identical BF16 and FP8 throughput (1,835 TFLOPS) — unusual, as most accelerators halve throughput going from FP8 to BF16. Its integrated Ethernet eliminates the need for InfiniBand, reducing infrastructure cost. Trainium2's NeuronLink interconnect enables tight coupling of up to 64 chips, and its configurable FP8 (cFP8) format allows custom exponent/mantissa splits.

```python
# Gaudi on AWS DL2 instances
pool = sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.Gaudi3(),
)

# Trainium on AWS Trn2 instances
pool = sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.Trainium2(),
    image=sky.Image(pip=["torch-neuronx"]),
)
```

| Factory | Memory | Notes |
|---------|--------|-------|
| `Gaudi3()` | 128GB HBM2e | Latest generation. |
| `Gaudi2()` | 96GB HBM2e | 2x performance vs Gaudi. |
| `Gaudi()` | — | First gen. |
| `Trainium2()` | 64GB | 4x performance vs v1. |
| `Trainium1()` | 32GB | First gen. |
| `Trainium3()` | 128GB | Latest generation. |

### AWS Inferentia

Custom silicon for cost-effective inference:

```python
pool = sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.Inferentia2(),
    image=sky.Image(pip=["torch-neuronx"]),
)
```

| Factory | Memory | Instance | Notes |
|---------|--------|----------|-------|
| `Inferentia1()` | 8GB | inf1.* | Single model serving. |
| `Inferentia2()` | 32GB | inf2.* | High throughput. |

---

## Related topics

- **[Providers](providers.md)** — AWS, RunPod, VastAI, Verda, and Container configuration
- **[Distributed Training](distributed-training.md)** — multi-node training guides
- **[API Reference](reference/accelerators.md)** — complete accelerator API
