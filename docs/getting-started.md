# Getting Started

This guide will help you install Skyward and run your first cloud computation.

## Prerequisites

- Python 3.13 or higher
- Cloud provider credentials (AWS, DigitalOcean, or Verda)

## Installation

### Using uv (Recommended)

```bash
uv add skyward
```

### Using pip

```bash
pip install skyward
```

### Optional Dependencies

```bash
# PyTorch support
uv add skyward[pytorch]

# HuggingFace support
uv add skyward[huggingface]

# AWS type hints
uv add skyward[aws]

# All extras
uv add skyward[all]
```

## Provider Credentials

### AWS

Skyward uses standard AWS credential resolution. Set up credentials using any of these methods:

**Environment Variables:**

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

**AWS CLI:**

```bash
aws configure
```

**Credentials File (`~/.aws/credentials`):**

```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
```

Required IAM permissions:
- `ec2:*` - Instance management
- `iam:PassRole` - For instance profiles
- `ssm:*` - For Session Manager connectivity (optional but recommended)

### DigitalOcean

```bash
export DIGITALOCEAN_TOKEN=your_api_token
```

Create a token at: https://cloud.digitalocean.com/account/api/tokens

### Verda

```bash
export VERDA_API_KEY=your_api_key
```

## Your First Remote Function

Create a file `hello.py`:

```python
import skyward as sky

@sky.compute
def hello() -> str:
    """This function runs on a remote EC2 instance."""
    import socket
    return f"Hello from {socket.gethostname()}!"

@sky.pool(provider=sky.AWS())
def main():
    result = hello() >> sky
    print(result)

if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python hello.py
```

**What happens:**

1. Skyward provisions an EC2 instance
2. Installs Python and dependencies
3. Serializes your function and sends it to the instance
4. Executes the function remotely
5. Returns the result to your local machine
6. Terminates the instance

## Your First GPU Job

```python
import skyward as sky

@sky.compute
def gpu_info() -> dict:
    """Get GPU information from the remote instance."""
    import torch
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

@sky.pool(
    provider=sky.AWS(),
    accelerator="T4",                 # Request a T4 GPU
    image=sky.Image(pip=["torch"]),   # Install PyTorch
    allocation="always-spot",         # Use spot instances (cheaper)
)
def main():
    info = gpu_info() >> sky
    print(f"GPU: {info['device_name']}")
    print(f"CUDA devices: {info['device_count']}")

if __name__ == "__main__":
    main()
```

## Understanding the Output

When you run a Skyward job, you'll see events like:

```
[InfraCreating] Creating infrastructure...
[InstanceLaunching] Launching instance i-0abc123...
[InstanceProvisioned] Instance ready (52.1.2.3)
[BootstrapStarting] Installing dependencies...
[BootstrapProgress] Installing torch...
[BootstrapCompleted] Setup complete (45s)
[PoolStarted] Pool ready with 1 worker(s)
Hello from ip-172-31-0-1!
[InstanceStopping] Terminating i-0abc123...
[CostFinal] Total cost: $0.12 (5 minutes @ $1.44/hr)
```

## Common Patterns

### Parallel Execution

Execute multiple functions concurrently:

```python
import skyward as sky

@sky.compute
def square(x: int) -> int:
    return x * x

@sky.pool(provider=sky.AWS())
def main():
    # Method 1: gather()
    results = sky.gather(square(1), square(2), square(3)) >> sky
    print(results)  # (1, 4, 9)

    # Method 2: & operator
    a, b, c = (square(4) & square(5) & square(6)) >> sky
    print(a, b, c)  # 16 25 36
```

### Multi-Node Clusters

Scale to multiple instances:

```python
import skyward as sky

@sky.compute
def worker_info() -> dict:
    info = sky.instance_info()
    return {
        "node": info.node,
        "total": info.total_nodes,
        "is_head": info.is_head,
    }

@sky.pool(provider=sky.AWS(), nodes=4)
def main():
    # @ broadcasts to ALL nodes
    results = worker_info() @ sky
    for r in results:
        print(f"Node {r['node']}/{r['total']} (head={r['is_head']})")
```

### Allocation Strategies

Save up to 70% with different allocation strategies:

```python
import skyward as sky

# Try spot, fallback to on-demand (default)
sky.ComputePool(provider=sky.AWS(), allocation="spot-if-available")

# Always use spot (fails if unavailable) - maximum savings
sky.ComputePool(provider=sky.AWS(), allocation="always-spot")

# Always on-demand (for critical workloads)
sky.ComputePool(provider=sky.AWS(), allocation="on-demand")

# Compare prices, pick cheapest option
sky.ComputePool(provider=sky.AWS(), allocation="cheapest")
```

## Troubleshooting

For common issues and solutions, see the [Troubleshooting Guide](troubleshooting.md).

Quick fixes for the most common issues:

- **"No instances available"**: Try a different region or use `allocation="spot-if-available"`
- **"Permission denied"**: Check your IAM/API permissions
- **"Bootstrap timeout"**: Increase timeout with `timeout=7200`
- **Connection issues**: Verify SSM access is enabled (AWS)

## Next Steps

- [Core Concepts](concepts.md) — Understand the programming model
- [API Reference](api-reference.md) — Complete API documentation
- [Distributed Training](distributed-training.md) — Train models across multiple GPUs
- [Examples](examples.md) — More example code
- [FAQ](faq.md) — Frequently asked questions