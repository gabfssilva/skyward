# Accelerators

Every cloud provider has its own naming scheme for GPU instances. AWS calls an A100 machine a `p4d.24xlarge`. RunPod uses a `gpuTypeId`. VastAI filters marketplace offers by GPU model. The `accelerator` parameter on `ComputePool` is Skyward's answer to this fragmentation: you describe the hardware you want, and the provider figures out how to get it.

## Two Ways to Specify

The simplest form is a string:

```python
sky.ComputePool(provider=sky.AWS(), accelerator="A100")
```

This works, but it carries no metadata — no memory size, no count, no type safety. The richer form uses the factory functions under `sky.accelerators`:

```python
sky.ComputePool(provider=sky.AWS(), accelerator=sky.accelerators.A100())
sky.ComputePool(provider=sky.AWS(), accelerator=sky.accelerators.H100(count=4))
sky.ComputePool(provider=sky.AWS(), accelerator=sky.accelerators.A100(memory="40GB"))
```

Each factory returns an `Accelerator` dataclass — frozen, immutable, with `name`, `memory`, `count`, and optional `metadata` (CUDA versions, form factors). The factory populates defaults from an internal catalog, so `sky.accelerators.H100()` already knows it has 80GB of HBM3 without you specifying it.

## NVIDIA Datacenter GPUs

These are the workhorses of ML training and inference — available on AWS, RunPod, Verda, and VastAI.

### Hopper and Blackwell (Current Generation)

| Factory | Memory | Architecture | Notes |
|---------|--------|-------------|-------|
| `H100()` | 40/80GB HBM3 | Hopper | Flagship training GPU. FP8 support. |
| `H100(form_factor="SXM")` | 80GB | Hopper | High-bandwidth SXM variant. |
| `H100(form_factor="NVL")` | 94GB | Hopper | NVLink 2-GPU module. |
| `H200()` | 141GB HBM3e | Hopper | 1.4-1.9x inference speedup vs H100. |
| `GH200()` | 96GB HBM3 | Grace Hopper | CPU+GPU superchip via NVLink-C2C. |
| `B100()` | 192GB HBM3e | Blackwell | FP4 support, 2nd-gen transformer engine. |
| `B200()` | 192GB HBM3e | Blackwell | Flagship Blackwell. 2.5x inference vs H100. |
| `GB200()` | 384GB HBM3e | Grace Blackwell | Grace CPU + 2x Blackwell GPUs. |

### Ampere and Ada Lovelace

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

### Legacy (Still Widely Available)

| Factory | Memory | Notes |
|---------|--------|-------|
| `T4()` | 16GB GDDR6 | Turing. Cheapest option for dev/inference. |
| `T4G()` | 16GB | ARM64 variant for AWS Graviton. |
| `V100()` | 16/32GB HBM2 | Volta. `V100(memory="32GB")` for the larger variant. |
| `P100()` | 16GB HBM2 | Pascal. First HBM GPU for deep learning. |

## Consumer GPUs

Available primarily on marketplace providers like VastAI and RunPod. Useful for cost-effective inference and fine-tuning:

```python
# RTX 50 series (Blackwell)
sky.accelerators.RTX_5090()    # 32GB
sky.accelerators.RTX_5080()    # 16GB

# RTX 40 series (Ada Lovelace)
sky.accelerators.RTX_4090()    # 24GB — popular for fine-tuning
sky.accelerators.RTX_4080()
sky.accelerators.RTX_4070_Ti()

# RTX 30 series (Ampere)
sky.accelerators.RTX_3090()    # 24GB
sky.accelerators.RTX_3080()

# Older generations also available: RTX 20xx, GTX 16xx, GTX 10xx
```

Workstation GPUs like `RTX_A6000()` (48GB), `RTX_6000_Ada()`, and `RTX_PRO_6000()` are also supported.

## AMD Instinct

AMD's datacenter compute accelerators. Supported through VastAI and other marketplace providers:

```python
pool = sky.ComputePool(
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

## AWS Trainium

Custom silicon designed for training. Requires the NeuronX SDK:

```python
pool = sky.ComputePool(
    provider=sky.AWS(),
    accelerator=sky.accelerators.Trainium2(),
    image=sky.Image(pip=["torch-neuronx"]),
)
```

| Factory | Memory | Instance | Notes |
|---------|--------|----------|-------|
| `Trainium1()` | 32GB | trn1.* | First gen. |
| `Trainium2()` | 64GB | trn2.* | 4x performance vs v1. |
| `Trainium3()` | 128GB | — | Latest generation. |

## AWS Inferentia

Custom silicon for cost-effective inference:

```python
pool = sky.ComputePool(
    provider=sky.AWS(),
    accelerator=sky.accelerators.Inferentia2(),
    image=sky.Image(pip=["torch-neuronx"]),
)
```

| Factory | Memory | Instance | Notes |
|---------|--------|----------|-------|
| `Inferentia1()` | 8GB | inf1.* | Single model serving. |
| `Inferentia2()` | 32GB | inf2.* | High throughput. |

## Google TPUs

Tensor Processing Units for JAX and TensorFlow workloads. Available as individual chips or pre-configured slices:

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

## Habana Gaudi

Intel's Habana accelerators for deep learning:

| Factory | Memory | Notes |
|---------|--------|-------|
| `Gaudi3()` | 128GB HBM2e | Latest generation. |
| `Gaudi2()` | 96GB HBM2e | 2x performance vs Gaudi. |
| `Gaudi()` | — | First gen. |

## Custom Accelerators

For hardware not in the catalog — experimental chips, private clouds, or overriding defaults:

```python
import skyward as sky

my_gpu = sky.accelerators.Custom("My-GPU", memory="48GB")
my_gpu = sky.accelerators.Custom("H100-Custom", memory="80GB", count=8, cuda_min="12.0")
```

## Selection Guide

### By Workload

| Workload | Recommended | Why |
|----------|------------|-----|
| Inference (small models) | T4, L4 | Cost-effective, good throughput |
| Inference (large models) | L40S, A10G | More memory |
| Fine-tuning (LoRA) | A100-40GB | Sufficient for adapters |
| Full fine-tuning | A100-80GB, H100 | Need full model in memory |
| Pre-training | 8x H100 | Maximum compute |
| Development | T4 | Cheapest option |

### By Model Size

| Parameters | Minimum | Recommended |
|------------|---------|-------------|
| < 1B | T4 (16GB) | T4 |
| 1-7B | A10G (24GB) | A100-40GB |
| 7-13B | A100-40GB | A100-80GB |
| 13-70B | A100-80GB | 2x H100 |
| 70B+ | 4x H100 | 8x H100 |

## Detecting Accelerators at Runtime

Inside a `@sky.compute` function, `sky.instance_info()` reports what hardware the function is running on:

```python
@sky.compute
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

---

## Related Topics

- [Providers](providers.md) — AWS, RunPod, VastAI, Verda, and Container configuration
- [Distributed Training](distributed-training.md) — Multi-node training guides
- [API Reference](reference/accelerators.md) — Complete accelerator API
