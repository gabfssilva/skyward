# Skyward Documentation

Skyward is a Python library for ephemeral GPU compute. Spin up cloud GPUs, run your ML training code, and tear them down automatically. No infrastructure to manage.

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | Installation, credentials setup, and first examples |
| [Core Concepts](concepts.md) | Understanding the programming model and ephemeral compute |
| [API Reference](api-reference.md) | Complete API documentation |
| [Distributed Training](distributed-training.md) | Multi-GPU training with PyTorch, Keras, JAX, and HuggingFace |
| [Providers](providers.md) | AWS, DigitalOcean, and Verda configuration |
| [Accelerators](accelerators.md) | GPU selection and MIG partitioning |
| [Examples](examples.md) | All 16 examples explained |
| [Architecture](architecture.md) | Internal design and extension points |

## Quick Links

- [Your First Remote Function](getting-started.md#your-first-remote-function)
- [GPU Selection Guide](accelerators.md#gpu-selection-guide)
- [PyTorch DDP Training](distributed-training.md#pytorch-distributed-training)
- [Troubleshooting](troubleshooting.md)
- [FAQ](faq.md)

## Quick Example

```python
import skyward as sky

@sky.compute
def train(data):
    import torch
    model = create_model().cuda()
    return model.fit(data)

with sky.ComputePool(provider=sky.AWS(), accelerator="A100") as pool:
    result = train(my_data) >> pool
# GPU terminated automatically
```

## Execution Operators

| Operator | Syntax | Description |
|----------|--------|-------------|
| `>>` | `fn() >> pool` | Execute on single worker |
| `@` | `fn() @ pool` | Broadcast to ALL workers |
| `&` | `fn1() & fn2() >> pool` | Parallel execution |
| `gather()` | `gather(fn1(), fn2()) >> pool` | Dynamic parallel execution |

## Requirements

- Python 3.13+
- Cloud provider credentials (AWS, DigitalOcean, or Verda)

## Getting Help

- [Troubleshooting Guide](troubleshooting.md) — Common issues and solutions
- [FAQ](faq.md) — Frequently asked questions
- [GitHub Issues](https://github.com/gabfssilva/skyward/issues) — Report bugs or request features
