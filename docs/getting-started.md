# Getting Started

This guide will help you install Skyward and run your first cloud computation.

## Prerequisites

- Python 3.12 or higher
- Cloud provider credentials (AWS, RunPod, VastAI, or Verda)

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

### Verda

```bash
export VERDA_CLIENT_ID=your_client_id
export VERDA_CLIENT_SECRET=your_client_secret
```

### RunPod

```bash
export RUNPOD_API_KEY=your_api_key
```

Get your API key from **Settings > API Keys** at [runpod.io](https://www.runpod.io/).

### VastAI

```bash
pip install vastai
vastai set api-key YOUR_API_KEY
```

Get your API key at: https://cloud.vast.ai/account/

## Your First Remote Function

Create a file `hello.py`:

```python
import skyward as sky

@sky.compute
def hello() -> str:
    """This function runs on a remote EC2 instance."""
    import socket
    return f"Hello from {socket.gethostname()}!"

with sky.ComputePool(provider=sky.AWS()) as pool:
    result = hello() >> pool
    print(result)
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

## Your First Accelerator Job

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

with sky.ComputePool(
    provider=sky.AWS(),
    accelerator=sky.accelerators.T4(),  # Request a T4 accelerator
    image=sky.Image(pip=["torch"]),   # Install PyTorch
    allocation="always-spot",         # Use spot instances (cheaper)
) as pool:
    info = gpu_info() >> pool
    print(f"GPU: {info['device_name']}")
    print(f"CUDA devices: {info['device_count']}")
```

## Understanding the Output

When you run a Skyward job, you'll see events like:

```
[ClusterProvisioned] Cluster ready in us-east-1
[InstanceLaunched] Launching instance i-0abc123...
[InstanceRunning] Instance running (52.1.2.3)
[InstanceProvisioned] Instance provisioned, starting bootstrap
[BootstrapPhase] Phase 'apt' started
[BootstrapPhase] Phase 'apt' completed (12s)
[BootstrapPhase] Phase 'pip' started
[BootstrapPhase] Phase 'pip' completed (45s)
[InstanceBootstrapped] Bootstrap complete
[NodeReady] Node 0 ready
[ClusterReady] Cluster ready with 1 node(s)
Hello from ip-172-31-0-1!
[TaskCompleted] Task completed in 2.3s
[ClusterDestroyed] Cluster terminated
```

## Common Patterns

### Parallel Execution

Execute multiple functions concurrently:

```python
import skyward as sky

@sky.compute
def square(x: int) -> int:
    return x * x

with sky.ComputePool(provider=sky.AWS()) as pool:
    # Method 1: gather()
    results = sky.gather(square(1), square(2), square(3)) >> pool
    print(results)  # (1, 4, 9)

    # Method 2: & operator
    a, b, c = (square(4) & square(5) & square(6)) >> pool
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

with sky.ComputePool(provider=sky.AWS(), nodes=4) as pool:
    # @ broadcasts to ALL nodes
    results = worker_info() @ pool
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

Quick fixes for the most common issues:

- **"No instances available"**: Try a different region or use `allocation="spot-if-available"`
- **"Permission denied"**: Check your IAM/API permissions (AWS needs `ec2:*`, `iam:PassRole`)
- **"Bootstrap timeout"**: Increase timeout with `timeout=7200`, or reduce pip dependencies
- **Connection issues (AWS)**: Verify SSM access is enabled and the instance has outbound internet

### Test Locally First

Always test your function locally before running remotely:

```python
result = my_function.local(test_data)  # Runs locally, no cloud
```

### Enable Verbose Logging

```python
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="DEBUG")
```

## Next Steps

- [Core Concepts](concepts.md) — Understand the programming model
- [Providers](providers.md) — Configure AWS, RunPod, VastAI, or Verda
- [Distributed Training](distributed-training.md) — Train models across multiple nodes
- [Integrations](integrations.md) — PyTorch, Keras, JAX, joblib integrations