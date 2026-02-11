# Accelerators

Comprehensive guide to accelerator selection in Skyward.

## Accelerator Selection

### String Shorthand

```python
import skyward as sky

# Simple string
pool = sky.ComputePool(provider=sky.AWS(), accelerator="T4")
pool = sky.ComputePool(provider=sky.AWS(), accelerator="A100")
pool = sky.ComputePool(provider=sky.AWS(), accelerator="H100")
```

### Accelerator Factory

```python
import skyward as sky

# Full control
pool = sky.ComputePool(
    provider=sky.AWS(),
    accelerator=sky.AcceleratorSpec.NVIDIA.H100(),
)

# Multiple GPUs
pool = sky.ComputePool(
    provider=sky.AWS(),
    accelerator=sky.AcceleratorSpec.NVIDIA.A100(count=4),
)

# Memory variant
pool = sky.ComputePool(
    provider=sky.AWS(),
    accelerator=sky.AcceleratorSpec.NVIDIA.A100(memory="40GB"),
)
```

### Literal Type

```python
import skyward as sky

# Type-safe literal
pool = sky.ComputePool(provider=sky.AWS(), accelerator=sky.accelerators.A100())
```

## NVIDIA GPUs

### Available GPUs

| GPU | Memory | Use Case | AWS Instance |
|-----|--------|----------|--------------|
| **T4** | 16GB | Inference, development | g4dn.* |
| **L4** | 24GB | Inference, light training | g6.* |
| **L40S** | 48GB | Training, inference | g6e.* |
| **A10G** | 24GB | Training, inference | g5.* |
| **A100-40GB** | 40GB | Training | p4de.* |
| **A100-80GB** | 80GB | Large models | p4d.*, p4de.* |
| **H100-80GB** | 80GB | Latest generation | p5.* |
| **H200** | 141GB | Maximum memory | p5e.* |

### Factory Methods

```python
import skyward as sky

# Basic usage
sky.AcceleratorSpec.NVIDIA.T4()  # 1x T4
sky.AcceleratorSpec.NVIDIA.A100()  # 1x A100 80GB
sky.AcceleratorSpec.NVIDIA.H100()  # 1x H100 80GB

# Multiple GPUs
sky.AcceleratorSpec.NVIDIA.A100(count=8)  # 8x A100

# Memory variant
sky.AcceleratorSpec.NVIDIA.A100(memory="40GB")
sky.AcceleratorSpec.NVIDIA.V100(memory="32GB")

# Form factor
sky.AcceleratorSpec.NVIDIA.H100(form_factor="SXM")
sky.AcceleratorSpec.NVIDIA.H100(form_factor="PCIe")
sky.AcceleratorSpec.NVIDIA.H100(form_factor="NVL")
```

## Accelerator Selection Guide

### By Workload

| Workload | Recommended | Why |
|----------|-----------------|-----|
| Inference (small) | T4, L4 | Cost-effective, good throughput |
| Inference (large) | L40S, A10G | More memory for larger models |
| Fine-tuning (LoRA) | A100-40GB | Sufficient for adapters |
| Full fine-tuning | A100-80GB, H100 | Need full model in memory |
| Pre-training | 8x H100 | Maximum compute |
| Development | T4 | Cheapest option |

### By Model Size

| Model Size | Minimum | Recommended |
|------------|-------------|-------------|
| < 1B params | T4 (16GB) | T4 |
| 1-7B params | A10G (24GB) | A100-40GB |
| 7-13B params | A100-40GB | A100-80GB |
| 13-70B params | A100-80GB | 2x H100 |
| 70B+ params | 4x H100 | 8x H100 |

### Cost Optimization

| Strategy | Savings | Use Case |
|----------|---------|----------|
| Spot instances | 60-90% | Fault-tolerant training |
| T4 over A100 | 90% | Development, small models |

## AWS Trainium

AWS custom silicon for training:

```python
import skyward as sky

pool = sky.ComputePool(
    provider=sky.AWS(),
    accelerator=sky.AcceleratorSpec.AWS.Trainium(version=2),
    image=sky.Image(pip=["torch-neuronx"]),
)
```

### Trainium Instances

| Instance | Trainium | Memory | Use Case |
|----------|----------|--------|----------|
| `trn1.2xlarge` | 1x Trn1 | 32GB | Development |
| `trn1.32xlarge` | 16x Trn1 | 512GB | Large training |
| `trn2.48xlarge` | 16x Trn2 | 1TB | Latest generation |

## AWS Inferentia

AWS custom silicon for inference:

```python
import skyward as sky

pool = sky.ComputePool(
    provider=sky.AWS(),
    accelerator=sky.AcceleratorSpec.AWS.Inferentia(version=2),
    image=sky.Image(pip=["torch-neuronx"]),
)
```

### Inferentia Instances

| Instance | Inferentia | Memory | Use Case |
|----------|------------|--------|----------|
| `inf2.xlarge` | 1x Inf2 | 32GB | Single model |
| `inf2.48xlarge` | 12x Inf2 | 384GB | High throughput |

## Future Accelerator Support

Support for additional accelerators is planned:

- **Google TPUs**: Cloud TPU v5p, TPU slices
- **AMD GPUs**: MI300X, MI250X, Radeon Pro

Check the repository for updates on accelerator support.

## Detecting Accelerators at Runtime

```python
import skyward as sky

@sky.compute
def check_gpu():
    import torch

    info = sky.instance_info()

    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "accelerators": info.accelerators,
    }
```

## Helper Functions

```python
import skyward as sky

# Check accelerator type
sky.is_nvidia("H100")      # True
sky.is_nvidia("MI300X")    # False
sky.is_trainium("Trainium2")  # True

# Get current accelerator in compute context
@sky.compute
def my_function():
    acc = sky.current_accelerator()
    print(f"Running on: {acc}")
```

## Troubleshooting

### "No instances with accelerator X available"

1. Check if the accelerator is available in your region
2. Try a different region
3. Request a service quota increase

### "CUDA out of memory"

1. Reduce batch size
2. Use gradient checkpointing
3. Use a larger accelerator

---

## Related Topics

- [Providers](providers.md) — AWS, RunPod, VastAI, and Verda configuration
- [Distributed Training](distributed-training.md) — Multi-node training guides
- [API Reference](reference/pool.md) — Complete API documentation