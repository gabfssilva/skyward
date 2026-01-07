# Cloud Providers

Skyward supports multiple cloud providers with a unified API.

## Provider Comparison

| Feature | AWS | DigitalOcean | Verda | VastAI |
|---------|-----|--------------|-------|--------|
| **GPUs** | H100, A100, T4, L4, etc. | H100, H200, L40S, MI300X | H100, A100, H200, GB200 | Marketplace (varies) |
| **Spot Instances** | Yes (60-90% savings) | No | Yes | Yes (bid-based) |
| **Regions** | 20+ | 9 | 3 | Global marketplace |
| **MIG Support** | Yes | Yes | Yes | No |
| **SSH Connectivity** | Yes | Yes | Yes | Yes |
| **Trainium/Inferentia** | Yes | No | No | No |

## AWS

### Setup

**Environment Variables:**

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

**Or use AWS CLI:**

```bash
aws configure
```

### Usage

```python
import skyward as sky

pool = sky.ComputePool(
    provider=sky.AWS(region="us-east-1"),
    accelerator="A100",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `region` | `str` | `"us-east-1"` | AWS region |
| `ami` | `str` | `None` | Custom AMI ID (auto-detected if None) |
| `subnet_id` | `str` | `None` | VPC subnet (auto-created if None) |
| `allocation_strategy` | `str` | `"price-capacity-optimized"` | Spot allocation strategy |

### Available Regions

| Region | GPUs | Trainium |
|--------|------|----------|
| `us-east-1` | H100, A100, T4, L4 | Yes |
| `us-east-2` | A100, T4 | No |
| `us-west-2` | H100, A100, T4, L4 | Yes |
| `eu-west-1` | A100, T4 | No |
| `ap-northeast-1` | T4, L4 | No |

### Instance Types

| Instance | GPUs | Memory | Use Case |
|----------|------|--------|----------|
| `p5.48xlarge` | 8x H100 | 640GB | Large model training |
| `p4d.24xlarge` | 8x A100 | 320GB | Distributed training |
| `g5.xlarge` | 1x A10G | 24GB | Inference |
| `g4dn.xlarge` | 1x T4 | 16GB | Development |
| `trn1.32xlarge` | 16x Trainium | 512GB | NeuronX training |

### SSM Connectivity (Default)

AWS uses Systems Manager (SSM) by default for all connectivity. This provides:
- No SSH key management required
- More reliable connections through AWS infrastructure
- Works with private subnets (no public IP needed)

SSM requires:
- IAM role with `AmazonSSMManagedInstanceCore` policy (auto-created by Skyward)
- VPC with SSM endpoints or outbound internet access

### Required IAM Permissions

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:RunInstances",
                "ec2:TerminateInstances",
                "ec2:DescribeInstances",
                "ec2:DescribeInstanceTypes",
                "ec2:DescribeImages",
                "ec2:CreateSecurityGroup",
                "ec2:AuthorizeSecurityGroupIngress",
                "ec2:DescribeSecurityGroups",
                "ec2:CreateKeyPair",
                "ec2:DescribeKeyPairs"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": "arn:aws:iam::*:role/*"
        }
    ]
}
```

## DigitalOcean

### Setup

```bash
export DIGITALOCEAN_TOKEN=your_api_token
```

Create a token at: https://cloud.digitalocean.com/account/api/tokens

### Usage

```python
import skyward as sky

pool = sky.ComputePool(
    provider=sky.DigitalOcean(region="nyc1"),
    cpu=4,
    memory="8GB",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `region` | `str` | `"nyc1"` | DO region |

### Available Regions

| Region | Location |
|--------|----------|
| `nyc1` | New York 1 |
| `nyc3` | New York 3 |
| `sfo2` | San Francisco 2 |
| `sfo3` | San Francisco 3 |
| `ams3` | Amsterdam 3 |
| `sgp1` | Singapore 1 |
| `lon1` | London 1 |
| `fra1` | Frankfurt 1 |
| `tor1` | Toronto 1 |

### Droplet Sizes

| Size | vCPUs | Memory | Use Case |
|------|-------|--------|----------|
| `s-1vcpu-1gb` | 1 | 1GB | Testing |
| `s-2vcpu-4gb` | 2 | 4GB | Light workloads |
| `s-4vcpu-8gb` | 4 | 8GB | Development |
| `c-8` | 8 | 16GB | CPU compute |
| `c-32` | 32 | 64GB | Heavy compute |

### Notes

- **GPU support**: H100, H200, L40S, and MI300X available in select regions (nyc3, sfo3, tor1)
- **No spot instances**: All instances are on-demand with per-second billing
- **Good for**: GPU inference, development, CPU workloads

## Verda

### Setup

```bash
export VERDA_API_KEY=your_api_key
```

### Usage

```python
import skyward as sky

pool = sky.ComputePool(
    provider=sky.Verda(),  # Auto-discovers region
    accelerator="H100",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `region` | `str` | `None` | Region (auto-discovers if None) |

### Available Regions

| Region | Location | GPUs |
|--------|----------|------|
| `FIN-01` | Finland | H100, A100, H200, GB200 |
| `ICL-01` | Iceland | H100, A100 |
| `ISR-01` | Israel | H100, A100 |

### Auto Region Discovery

Verda automatically selects the best available region:

```python
import skyward as sky

# Auto-discovers region with requested GPU
pool = sky.ComputePool(
    provider=sky.Verda(),
    accelerator="H100",
)
```

### Features

- **Auto region discovery**: Finds regions with available GPUs
- **Spot instances**: Significant cost savings
- **MIG support**: Multi-Instance GPU partitioning

## VastAI

VastAI is a GPU marketplace offering competitive pricing from independent providers worldwide.

### Setup

```bash
pip install vastai
vastai set api-key YOUR_API_KEY
```

Get your API key at: https://cloud.vast.ai/account/

### Usage

```python
import skyward as sky

pool = sky.ComputePool(
    provider=sky.VastAI(geolocation="US"),
    accelerator="RTX 4090",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geolocation` | `str` | `None` | Filter by location (e.g., "US", "EU") |
| `min_reliability` | `float` | `0.9` | Minimum provider reliability score (0-1) |
| `bid_multiplier` | `float` | `1.0` | Multiplier for spot bidding |
| `auto_shutdown` | `int` | `None` | Auto-terminate after N seconds |

### Features

- **Dynamic marketplace**: Real-time pricing from global providers
- **Reliability filtering**: Filter by provider track record
- **Geolocation**: Target specific regions for latency
- **Overlay networks**: Multi-node NCCL support for distributed training
- **Interruptible instances**: Significant cost savings with spot-like pricing

### Overlay Networks

VastAI supports overlay networks for multi-node distributed training:

```python
import skyward as sky

pool = sky.ComputePool(
    provider=sky.VastAI(geolocation="US", min_reliability=0.95),
    accelerator="RTX 4090",
    nodes=4,  # Automatically creates overlay network
)
```

See [VastAI Provider](providers/vastai.md) for detailed overlay network configuration.

## Choosing a Provider

### Use AWS When:
- You need specific GPU types (H100, Trainium)
- Spot instances are important for cost savings
- Enterprise-grade reliability and support
- You're already in the AWS ecosystem

### Use DigitalOcean When:
- Simple setup without complex IAM
- Per-second billing for short jobs
- H100/H200/L40S for inference workloads
- Development and testing environments

### Use Verda When:
- European data residency (Finland, Iceland, Israel)
- H100/A100/GB200 availability
- Automatic region selection
- Competitive GPU pricing with spot support

### Use VastAI When:
- Maximum cost savings (marketplace pricing)
- Consumer GPUs (RTX 4090, 3090, etc.)
- Flexible compute requirements
- Multi-node training with overlay networks

## Multi-Provider Selection

Use multiple providers with automatic fallback:

```python
import skyward as sky

@sky.pool(
    provider=[sky.AWS(), sky.Verda(), sky.VastAI()],
    selection="cheapest",
    accelerator="A100",
)
def main():
    return train() >> sky
```

### Selection Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `"first"` | Use first provider in list | Explicit priority |
| `"cheapest"` | Compare spot/on-demand prices | Cost optimization |
| `"available"` | First with matching instances | Availability priority |
| Custom callable | Your selection logic | Complex requirements |

### Automatic Fallback

If the selected provider fails (no capacity, provisioning error), Skyward automatically tries the next:

```python
import skyward as sky

# AWS first, fallback to Verda if AWS fails
pool = sky.ComputePool(
    provider=[sky.AWS(), sky.Verda()],
    selection="first",
    accelerator="H100",
)
```

### Custom Selection

```python
import skyward as sky

def prefer_us_east(providers, spec):
    """Prefer providers with us-east region."""
    for p in providers:
        if hasattr(p.config, 'region') and "us-east" in (p.config.region or ""):
            return p
    return providers[0]

pool = sky.ComputePool(
    provider=[sky.AWS(), sky.Verda(), sky.DigitalOcean()],
    selection=prefer_us_east,
    accelerator="A100",
)
```

### String Shortcuts

```python
import skyward as sky

# Instead of sky.AWS(), you can use strings:
pool = sky.ComputePool(
    provider=["aws", "verda"],
    selection="cheapest",
    accelerator="A100",
)
```

### Sequential Multi-Provider Workflows

You can also use different providers for different stages:

```python
import skyward as sky

# Development on DigitalOcean (cheap CPU)
with sky.ComputePool(provider=sky.DigitalOcean(), cpu=4) as dev_pool:
    preprocess_data() >> dev_pool

# Training on AWS (H100 GPUs)
with sky.ComputePool(provider=sky.AWS(), accelerator="H100") as train_pool:
    train_model() @ train_pool

# Inference on Verda (cost-effective)
with sky.ComputePool(provider=sky.Verda(), accelerator="A100") as infer_pool:
    run_inference() >> infer_pool
```

## Common Issues

### AWS: "No instances available"

1. Try a different region
2. Use `allocation="spot-if-available"` to fallback to on-demand
3. Request a service quota increase

### DigitalOcean: "Authentication failed"

1. Verify your token at https://cloud.digitalocean.com/account/api
2. Check token permissions (read/write access needed)

### Verda: "Region not available"

1. Remove the `region` parameter to enable auto-discovery
2. Check your account's region access

### VastAI: "No offers available"

1. Lower `min_reliability` threshold (e.g., 0.8)
2. Expand `geolocation` filter or remove it
3. Try a different accelerator type
4. Check marketplace availability at https://cloud.vast.ai/

---

## Related Topics

- [Getting Started](getting-started.md) — Installation and credentials setup
- [Accelerators](accelerators.md) — GPU selection and MIG partitioning
- [Troubleshooting](troubleshooting.md) — Common issues and solutions
- [API Reference](api-reference.md) — Complete API documentation

### Provider-Specific Documentation

- [AWS Provider](providers/aws.md) — Detailed AWS configuration
- [DigitalOcean Provider](providers/digital_ocean.md) — DigitalOcean setup
- [Verda Provider](providers/verda.md) — Verda configuration
- [VastAI Provider](providers/vastai.md) — VastAI and overlay networks