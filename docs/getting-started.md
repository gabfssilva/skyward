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
from skyward import compute, ComputePool, AWS

@compute
def hello() -> str:
    """This function runs on a remote EC2 instance."""
    import socket
    return f"Hello from {socket.gethostname()}!"

if __name__ == "__main__":
    with ComputePool(provider=AWS()) as pool:
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

## Your First GPU Job

```python
from skyward import compute, ComputePool, AWS, Image

@compute
def gpu_info() -> dict:
    """Get GPU information from the remote instance."""
    import torch
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

if __name__ == "__main__":
    with ComputePool(
        provider=AWS(),
        accelerator="T4",           # Request a T4 GPU
        image=Image(pip=["torch"]), # Install PyTorch
        spot="always",              # Use spot instances (cheaper)
    ) as pool:
        info = gpu_info() >> pool
        print(f"GPU: {info['device_name']}")
        print(f"CUDA devices: {info['device_count']}")
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
from skyward import compute, ComputePool, AWS, gather

@compute
def square(x: int) -> int:
    return x * x

with ComputePool(provider=AWS()) as pool:
    # Method 1: gather()
    results = gather(square(1), square(2), square(3)) >> pool
    print(results)  # (1, 4, 9)

    # Method 2: & operator
    a, b, c = (square(4) & square(5) & square(6)) >> pool
    print(a, b, c)  # 16 25 36
```

### Multi-Node Clusters

Scale to multiple instances:

```python
from skyward import compute, ComputePool, AWS, instance_info

@compute
def worker_info() -> dict:
    pool = instance_info()
    return {
        "node": pool.node,
        "total": pool.total_nodes,
        "is_head": pool.is_head,
    }

with ComputePool(provider=AWS(), nodes=4) as pool:
    # @ broadcasts to ALL nodes
    results = worker_info() @ pool
    for r in results:
        print(f"Node {r['node']}/{r['total']} (head={r['is_head']})")
```

### Spot Instances

Save up to 70% with spot pricing:

```python
from skyward import ComputePool, AWS, Spot

# Always use spot (fails if unavailable)
ComputePool(provider=AWS(), spot="always")
ComputePool(provider=AWS(), spot=Spot.Always())

# Try spot, fallback to on-demand
ComputePool(provider=AWS(), spot="if-available")
ComputePool(provider=AWS(), spot=Spot.IfAvailable())

# Never use spot
ComputePool(provider=AWS(), spot="never")
ComputePool(provider=AWS(), spot=Spot.Never)

# At least 80% spot instances
ComputePool(provider=AWS(), spot=0.8)
ComputePool(provider=AWS(), spot=Spot.Percent(0.8))
```

## Troubleshooting

### "No instances available"

Your requested configuration isn't available in the region. Try:
- Different accelerator type
- Different region
- `spot="if-available"` instead of `spot="always"`

### "Permission denied"

Check your AWS IAM permissions. Skyward needs:
- `ec2:RunInstances`, `ec2:TerminateInstances`, `ec2:DescribeInstances`
- `ec2:CreateSecurityGroup`, `ec2:AuthorizeSecurityGroupIngress`
- `iam:PassRole` (for instance profiles)

### "Bootstrap timeout"

The instance took too long to set up. This can happen with:
- Large pip dependencies
- Slow network in certain regions
- Instance type limitations

Try increasing the timeout:

```python
ComputePool(provider=AWS(), timeout=7200)  # 2 hours
```

### Connection issues

AWS uses SSM (Session Manager) by default for reliable connectivity without SSH key management. If you experience connection issues:

1. Ensure your AWS account has SSM access enabled
2. Verify the instance has outbound internet access for SSM endpoint communication
3. Check that IAM permissions include `AmazonSSMManagedInstanceCore`

## Next Steps

- [Core Concepts](concepts.md) - Understand the programming model
- [API Reference](api-reference.md) - Complete API documentation
- [Distributed Training](distributed-training.md) - Train models across multiple GPUs
- [Examples](examples.md) - More example code
