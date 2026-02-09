# Skyward

[![CI](https://github.com/gabfssilva/skyward/actions/workflows/python-package.yml/badge.svg)](https://github.com/gabfssilva/skyward/actions/workflows/python-package.yml)
[![PyPI](https://img.shields.io/pypi/v/skyward)](https://pypi.org/project/skyward/)
[![Python](https://img.shields.io/pypi/pyversions/skyward)](https://pypi.org/project/skyward/)
[![License](https://img.shields.io/github/license/gabfssilva/skyward)](https://github.com/gabfssilva/skyward/blob/main/LICENSE)

Ephemeral cloud GPUs for Python. Spin up, run code, tear down. No infrastructure to manage, no idle costs.

```python
import skyward as sky

@sky.compute
def train(epochs: int) -> dict:
    import torch
    model = torch.nn.Linear(100, 10).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        loss = model(torch.randn(32, 100, device="cuda")).sum()
        loss.backward()
        optimizer.step()

    return {"final_loss": loss.item()}

@sky.pool(provider=sky.AWS(), accelerator="T4", image=sky.Image(pip=["torch"]))
def main():
    return train(epochs=100) >> sky

print(main())
```

## Why Ephemeral?

Training jobs have a defined end. Your infrastructure should too.

| | Traditional | Skyward |
|--|-------------|---------|
| Setup | Hours (Terraform, SSH, CUDA) | Seconds (Python decorator) |
| Idle costs | $1-40/hour until you remember | Zero — auto-terminates |
| Environment | Drifts over time | Fresh every run |
| Cleanup | Manual, error-prone | Automatic |

One forgotten `p4d.24xlarge` over the weekend = $1,500 wasted. With Skyward, that's impossible.

## Install

```bash
uv add skyward
```

## Credentials

```bash
# AWS (standard AWS credential resolution)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

# DigitalOcean
export DIGITALOCEAN_TOKEN=...

# Verda
export VERDA_API_KEY=...
```

## Execution Operators

| Operator | Syntax | Description |
|----------|--------|-------------|
| `>>` | `fn() >> sky` | Execute on one worker, return result |
| `@` | `fn() @ sky` | Broadcast to all workers, return tuple |
| `&` | `fn1() & fn2() >> sky` | Parallel execution with type inference |
| `gather()` | `gather(*fns) >> sky` | Dynamic parallel execution |

```python
@sky.pool(provider=sky.AWS(), nodes=4)
def main():
    # Parallel execution
    a, b, c = (process(1) & process(2) & process(3)) >> sky

    # Broadcast to all nodes
    results = init_model() @ sky  # Runs on every node

    # Dynamic parallelism
    results = sky.gather(*[task(x) for x in data]) >> sky
```

## Providers & GPUs

| Provider | GPUs | Spot | Regions |
|----------|------|------|---------|
| `AWS()` | T4, L4, A10G, A100, H100, H200 | Yes | 20+ |
| `DigitalOcean()` | CPU only | No | 9 |
| `Verda()` | A100, H100 | Yes | 3 |

### Multi-Provider Fallback

```python
# Try AWS first, fallback to Verda if unavailable
@sky.pool(
    provider=[sky.AWS(), sky.Verda()],
    selection="cheapest",  # or "first", "available", custom callable
    accelerator="A100",
)
def main():
    return train() >> sky
```

## ComputePool Options

```python
sky.ComputePool(
    provider=sky.AWS(region="us-east-1"),

    # Resources
    nodes=4,                              # Multi-node cluster
    accelerator="A100",                   # Or: Accelerator.NVIDIA.H100(count=8)

    # Environment
    image=sky.Image(
        pip=["torch", "transformers"],
        apt=["ffmpeg"],
        env={"CUDA_VISIBLE_DEVICES": "0"},
    ),

    # Cost optimization
    allocation="always-spot",             # 60-90% cheaper, or "on-demand", "cheapest"
    timeout=3600,                         # Auto-shutdown after 1 hour

    # Storage
    volume=[
        sky.S3Volume(bucket="data", mount_path="/data", read_only=True),
    ],
)
```

## Distributed Training

Built-in support for PyTorch DDP, Keras 3, JAX, and HuggingFace Transformers.

```python
@sky.integrations.torch(backend="nccl")
@sky.compute
def train_distributed():
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = DDP(MyModel().cuda())
    # dist.is_initialized() == True
    # MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE all set
    ...

@sky.pool(provider=sky.AWS(), nodes=4, accelerator="A100")
def main():
    results = train_distributed() @ sky  # Runs on all 4 nodes
```

Other frameworks:

```python
@sky.integrations.keras(backend="jax")    # Keras 3 with JAX backend
@sky.integrations.jax()                   # Pure JAX distributed
@sky.integrations.transformers()          # HuggingFace Trainer
```

## Data Sharding

Automatic data partitioning across workers:

```python
@sky.compute
def process(full_data: list[int]) -> int:
    local_data = sky.shard(full_data)  # Each worker gets its portion
    return sum(local_data)

@sky.pool(provider=sky.AWS(), nodes=4)
def main():
    results = process(list(range(10000))) @ sky
    total = sum(results)
```

## Cluster Information

```python
@sky.compute
def worker_info() -> dict:
    info = sky.instance_info()
    return {
        "node": info.node,
        "total": info.total_nodes,
        "is_head": info.is_head,
        "accelerators": info.accelerators,
    }
```

## Scikit-learn & Joblib Integration

Distribute `GridSearchCV`, `cross_val_score`, or any joblib workload:

```python
from sklearn.model_selection import GridSearchCV

with sky.integrations.ScikitLearnPool(
    provider=sky.AWS(),
    nodes=4,
    concurrency=4,
) as pool:
    grid = GridSearchCV(estimator, param_grid, n_jobs=-1)
    grid.fit(X, y)  # Distributed across 16 workers
```

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Installation and first steps |
| [Core Concepts](docs/concepts.md) | Programming model and ephemeral compute |
| [API Reference](docs/api-reference.md) | Complete API documentation |
| [Distributed Training](docs/distributed-training.md) | PyTorch, Keras, JAX, HuggingFace guides |
| [Providers](docs/providers.md) | AWS, DigitalOcean, Verda setup |
| [Accelerators](docs/accelerators.md) | GPU types and MIG partitioning |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |

## Requirements

- Python 3.13+
- Cloud provider credentials

## License

MIT
