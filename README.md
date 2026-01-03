# Skyward

Cloud servers are a hassle. Skyward makes them ephemeral — spin up cloud GPUs, run your code, tear them down. No servers to maintain, no clusters to babysit.

```python
from skyward import compute, ComputePool, AWS

@compute
def train(data):
    import torch
    model = create_model().cuda()
    return model.fit(data)

with ComputePool(provider=AWS(), accelerator="A100") as pool:
    result = train(my_data) >> pool
```

## Why Ephemeral?

Training ML models doesn't require servers that run forever — it requires GPUs that appear when you need them and vanish when training ends.

Traditional cloud workflow:
1. Provision GPU instance
2. SSH in, install PyTorch, configure NCCL
3. Run training
4. Forget to terminate
5. $2,400 bill for an idle p4d.24xlarge over the weekend

Skyward workflow:
```python
with ComputePool(provider=AWS(), accelerator="A100") as pool:
    model = train(data) >> pool
# GPUs terminated. Impossible to forget.
```

| | Traditional | Ephemeral (Skyward) |
|--|-------------|---------------------|
| Setup time | Hours (Terraform, Ansible, SSH) | Seconds (Python decorator) |
| Idle costs | $1-40/hour until you remember | Zero — auto-terminates |
| Environment | Drifts over time | Fresh every run |
| Cleanup | Manual, error-prone | Automatic |

You're a researcher, not a cloud administrator. Spend your time on models, not infrastructure.

## Features

- **Simple API**: `@compute` decorator makes any function remotely executable
- **Multi-Cloud**: AWS, DigitalOcean, and Verda providers
- **GPU Acceleration**: Request T4, A100, H100, and more with MIG support
- **Distributed Training**: Built-in support for PyTorch DDP, Keras 3, JAX, and HuggingFace
- **Type-Safe Parallelism**: Execute multiple functions concurrently with full type inference
- **Cost Optimization**: Spot instances, real-time cost tracking, and auto-shutdown

## Installation

```bash
# Using uv (recommended)
uv add skyward

# Using pip
pip install skyward
```

## Quick Start

### Hello World

```python
from skyward import compute, ComputePool, AWS

@compute
def hello() -> str:
    return "Hello from the cloud!"

with ComputePool(provider=AWS()) as pool:
    result = hello() >> pool
    print(result)  # "Hello from the cloud!"
```

### GPU Computation

```python
from skyward import compute, ComputePool, AWS, Image

@compute
def matrix_multiply(size: int) -> float:
    import torch
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    return torch.matmul(a, b).sum().item()

with ComputePool(
    provider=AWS(),
    accelerator="T4",
    image=Image(pip=["torch"]),
    spot="always",
) as pool:
    result = matrix_multiply(4096) >> pool
```

### Parallel Execution

```python
from skyward import compute, ComputePool, AWS, gather

@compute
def process(x: int) -> int:
    return x * 2

with ComputePool(provider=AWS()) as pool:
    # Using gather()
    results = gather(process(1), process(2), process(3)) >> pool
    print(results)  # (2, 4, 6)

    # Using & operator (type-safe)
    a, b, c = (process(10) & process(20) & process(30)) >> pool
    print(a, b, c)  # 20 40 60
```

### Multi-Node Broadcasting

```python
from skyward import compute, ComputePool, AWS, shard, instance_info

@compute
def process_partition(data: list[int]) -> dict:
    pool = instance_info()
    local_data = shard(data)  # Automatic data partitioning
    return {
        "node": pool.node,
        "sum": sum(local_data),
    }

with ComputePool(provider=AWS(), nodes=4, accelerator="T4") as pool:
    # @ broadcasts to ALL nodes
    results = process_partition(list(range(1000))) @ pool
    total = sum(r["sum"] for r in results)
```

### Distributed Training (PyTorch)

```python
from skyward import compute, ComputePool, AWS, NVIDIA, DistributedSampler, Image

@compute
def train(epochs: int) -> dict:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = MyModel().cuda()
    if dist.is_initialized():
        model = DDP(model)

    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, sampler=sampler)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        train_epoch(model, loader)

    return {"loss": final_loss}

with ComputePool(
    provider=AWS(),
    nodes=2,
    accelerator=NVIDIA.A100,
    image=Image(pip=["torch"]),
) as pool:
    results = train(epochs=10) @ pool
```

## Execution Operators

| Operator | Syntax | Description |
|----------|--------|-------------|
| `>>` | `fn() >> pool` | Execute on single worker, return result |
| `@` | `fn() @ pool` | Broadcast to ALL workers, return tuple |
| `&` | `fn1() & fn2() >> pool` | Parallel execution, return tuple |
| `gather()` | `gather(fn1(), fn2()) >> pool` | Type-safe parallel execution |

## Providers

| Provider | GPUs | Spot | Regions |
|----------|------|------|---------|
| `AWS()` | H100, A100, T4, L4, etc. | Yes | 20+ |
| `DigitalOcean()` | CPU only | No | 9 |
| `Verda()` | H100, A100 | Yes | 3 |

## GPU Accelerators

```python
from skyward import Accelerator, NVIDIA

# String shorthand
ComputePool(accelerator="T4")
ComputePool(accelerator="A100")

# Factory methods
ComputePool(accelerator=Accelerator.NVIDIA.H100())
ComputePool(accelerator=Accelerator.NVIDIA.A100(count=4))

# MIG partitioning
ComputePool(accelerator=Accelerator.NVIDIA.H100(mig="3g.40gb"))

# Literal type
ComputePool(accelerator=NVIDIA.A100)
```

## Documentation

- [Documentation Index](docs/index.md) — Complete documentation overview
- [Getting Started](docs/getting-started.md) — Installation and first steps
- [Core Concepts](docs/concepts.md) — Understanding the programming model
- [API Reference](docs/api-reference.md) — Complete API documentation
- [Distributed Training](docs/distributed-training.md) — PyTorch, Keras, JAX, HuggingFace guides
- [Providers](docs/providers.md) — AWS, DigitalOcean, Verda setup
- [Accelerators](docs/accelerators.md) — GPU types and MIG partitioning
- [Examples](docs/examples.md) — All examples explained
- [Troubleshooting](docs/troubleshooting.md) — Common issues and solutions
- [FAQ](docs/faq.md) — Frequently asked questions
- [Architecture](docs/architecture.md) — Internal design

## Examples

The `examples/` directory contains complete working examples:

| Example | Description |
|---------|-------------|
| `0_providers.py` | Discover available instances |
| `1_hello.py` | Basic remote execution |
| `2_parallel_execution.py` | `gather()` and `&` operator |
| `3_gpu_accelerators.py` | GPU detection and benchmarks |
| `4_digitalocean_provider.py` | DigitalOcean CPU workloads |
| `5_broadcast.py` | `@` operator for all nodes |
| `6_data_sharding.py` | `shard()` and `DistributedSampler` |
| `7_cluster_coordination.py` | `instance_info()`, node roles |
| `8_s3_volumes.py` | Mount S3 buckets |
| `9_event_monitoring.py` | Events and callbacks |
| `10_pytorch_distributed.py` | PyTorch DDP training |
| `11_keras_training.py` | Keras 3 Vision Transformer |
| `12_huggingface_finetuning.py` | Transformers fine-tuning |
| `13_multi_pool.py` | Parallel pool provisioning |
| `14_distributed_scikit_grid_search.py` | Distributed sklearn GridSearchCV |
| `15_concurrency.py` | Concurrent task execution |
| `16_joblib_concurrency.py` | Joblib integration |

## Requirements

- Python 3.13+
- Cloud provider credentials (AWS, DigitalOcean, or Verda)

## License

MIT
