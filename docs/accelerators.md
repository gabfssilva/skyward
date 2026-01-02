# GPU Accelerators

Comprehensive guide to GPU selection and MIG partitioning in Skyward.

## GPU Selection

### String Shorthand

```python
from skyward import ComputePool, AWS

# Simple string
pool = ComputePool(provider=AWS(), accelerator="T4")
pool = ComputePool(provider=AWS(), accelerator="A100")
pool = ComputePool(provider=AWS(), accelerator="H100")
```

### Accelerator Factory

```python
from skyward import Accelerator

# Full control
pool = ComputePool(
    provider=AWS(),
    accelerator=Accelerator.NVIDIA.H100(),
)

# Multiple GPUs
pool = ComputePool(
    provider=AWS(),
    accelerator=Accelerator.NVIDIA.A100(count=4),
)

# Memory variant
pool = ComputePool(
    provider=AWS(),
    accelerator=Accelerator.NVIDIA.A100(memory="40GB"),
)
```

### Literal Type

```python
from skyward import NVIDIA

# Type-safe literal
pool = ComputePool(provider=AWS(), accelerator=NVIDIA.A100)
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
from skyward import Accelerator

# Basic usage
Accelerator.NVIDIA.T4()              # 1x T4
Accelerator.NVIDIA.A100()            # 1x A100 80GB
Accelerator.NVIDIA.H100()            # 1x H100 80GB

# Multiple GPUs
Accelerator.NVIDIA.A100(count=8)     # 8x A100

# Memory variant
Accelerator.NVIDIA.A100(memory="40GB")
Accelerator.NVIDIA.V100(memory="32GB")

# Form factor
Accelerator.NVIDIA.H100(form_factor="SXM")
Accelerator.NVIDIA.H100(form_factor="PCIe")
Accelerator.NVIDIA.H100(form_factor="NVL")
```

## MIG (Multi-Instance GPU)

MIG allows partitioning a single GPU into multiple isolated instances.

### Supported GPUs

- H100-80GB
- A100-80GB
- A100-40GB

### MIG Profiles

**H100/A100 (80GB):**

| Profile | Workers | Memory | Compute |
|---------|---------|--------|---------|
| `1g.10gb` | 7 | 10GB | 1/7 GPU |
| `1g.20gb` | 4 | 20GB | 1/4 GPU |
| `2g.20gb` | 3 | 20GB | 2/7 GPU |
| `3g.40gb` | 2 | 40GB | 3/7 GPU |
| `4g.40gb` | 1 | 40GB | 4/7 GPU |
| `7g.80gb` | 1 | 80GB | Full GPU |

**A100 (40GB):**

| Profile | Workers | Memory | Compute |
|---------|---------|--------|---------|
| `1g.5gb` | 7 | 5GB | 1/7 GPU |
| `2g.10gb` | 3 | 10GB | 2/7 GPU |
| `3g.20gb` | 2 | 20GB | 3/7 GPU |
| `4g.20gb` | 1 | 20GB | 4/7 GPU |

### Using MIG

```python
from skyward import Accelerator, ComputePool, AWS

# Single MIG profile - creates 2 workers from 1 GPU
pool = ComputePool(
    provider=AWS(),
    accelerator=Accelerator.NVIDIA.H100(mig="3g.40gb"),
)

# Execute on both MIG partitions
results = train() @ pool  # Returns 2 results
```

### Multiple MIG Partitions

```python
# Multiple partitions on one GPU
pool = ComputePool(
    provider=AWS(),
    accelerator=Accelerator.NVIDIA.A100(mig=["3g.40gb", "3g.40gb"]),
)
```

### MIG Benefits

1. **Cost sharing**: Run multiple jobs on one GPU
2. **Isolation**: Each partition has dedicated memory
3. **Fault tolerance**: One partition crashing doesn't affect others

### MIG Limitations

- Not all GPUs support MIG
- Profile must match GPU memory
- Cannot change partitions without restart

## GPU Selection Guide

### By Workload

| Workload | Recommended GPU | Why |
|----------|-----------------|-----|
| Inference (small) | T4, L4 | Cost-effective, good throughput |
| Inference (large) | L40S, A10G | More memory for larger models |
| Fine-tuning (LoRA) | A100-40GB | Sufficient for adapters |
| Full fine-tuning | A100-80GB, H100 | Need full model in memory |
| Pre-training | 8x H100 | Maximum compute |
| Development | T4 | Cheapest GPU option |

### By Model Size

| Model Size | Minimum GPU | Recommended |
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
| MIG partitions | 50-85% | Multi-tenant inference |
| T4 over A100 | 90% | Development, small models |

## AWS Trainium

AWS custom silicon for training:

```python
from skyward import Accelerator, ComputePool, AWS

pool = ComputePool(
    provider=AWS(),
    accelerator=Accelerator.AWS.Trainium(version=2),
    pip=["torch-neuronx"],
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
pool = ComputePool(
    provider=AWS(),
    accelerator=Accelerator.AWS.Inferentia(version=2),
    pip=["torch-neuronx"],
)
```

### Inferentia Instances

| Instance | Inferentia | Memory | Use Case |
|----------|------------|--------|----------|
| `inf2.xlarge` | 1x Inf2 | 32GB | Single model |
| `inf2.48xlarge` | 12x Inf2 | 384GB | High throughput |

## Google TPUs

```python
from skyward import Accelerator

# Single TPU
Accelerator.Google.TPU(version="v5p")

# TPU slice (pod)
Accelerator.Google.TPUSlice("v5p-8")
```

## AMD GPUs

```python
from skyward import Accelerator

# MI series
Accelerator.AMD.MI("300X")     # MI300X
Accelerator.AMD.MI("250X")     # MI250X

# Radeon Pro
Accelerator.AMD.RadeonPro("V710")
```

## Detecting GPUs at Runtime

```python
from skyward import compute, instance_info

@compute
def check_gpu():
    import torch

    pool = instance_info()

    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "accelerators": pool.accelerators,
    }
```

## Helper Functions

```python
from skyward import is_nvidia, is_trainium, current_accelerator

# Check accelerator type
is_nvidia("H100")      # True
is_nvidia("MI300X")    # False
is_trainium("Trainium2")  # True

# Get current accelerator in compute context
@compute
def my_function():
    acc = current_accelerator()
    print(f"Running on: {acc}")
```

## Troubleshooting

### "No instances with accelerator X available"

1. Check if the GPU is available in your region
2. Try a different region
3. Request a service quota increase

### "CUDA out of memory"

1. Reduce batch size
2. Use gradient checkpointing
3. Use a larger GPU or MIG with more memory

### "MIG partition failed"

1. Ensure GPU supports MIG (H100, A100 only)
2. Check profile compatibility with GPU memory
3. Verify no other processes are using the GPU
