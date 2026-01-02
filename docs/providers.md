# Cloud Providers

Skyward supports multiple cloud providers with a unified API.

## Provider Comparison

| Feature | AWS | DigitalOcean | Verda |
|---------|-----|--------------|-------|
| **GPUs** | H100, A100, T4, L4, etc. | None (CPU only) | H100, A100 |
| **Spot Instances** | Yes (60-90% savings) | No | Yes |
| **Regions** | 20+ | 9 | 3 |
| **MIG Support** | Yes | N/A | Yes |
| **SSM Connectivity** | Yes | No | No |
| **Trainium/Inferentia** | Yes | No | No |

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
from skyward import AWS, ComputePool

pool = ComputePool(
    provider=AWS(region="us-east-1"),
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
from skyward import DigitalOcean, ComputePool

pool = ComputePool(
    provider=DigitalOcean(region="nyc1"),
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

- **No GPU support**: DigitalOcean doesn't offer GPU instances
- **No spot instances**: All instances are on-demand
- **Good for**: CPU workloads, testing, development

## Verda

### Setup

```bash
export VERDA_API_KEY=your_api_key
```

### Usage

```python
from skyward import Verda, ComputePool

pool = ComputePool(
    provider=Verda(),  # Auto-discovers region
    accelerator="H100",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `region` | `str` | `None` | Region (auto-discovers if None) |

### Available Regions

| Region | GPUs |
|--------|------|
| `us-east` | H100, A100 |
| `us-west` | H100, A100 |
| `eu-west` | A100 |

### Auto Region Discovery

Verda automatically selects the best available region:

```python
# Auto-discovers region with requested GPU
pool = ComputePool(
    provider=Verda(),
    accelerator="H100",
)
```

### Features

- **Auto region discovery**: Finds regions with available GPUs
- **Spot instances**: Significant cost savings
- **MIG support**: Multi-Instance GPU partitioning

## Choosing a Provider

### Use AWS When:
- You need specific GPU types (H100, Trainium)
- Spot instances are important for cost savings
- You need SSM connectivity for reliability
- You're already in the AWS ecosystem

### Use DigitalOcean When:
- CPU-only workloads
- Cost-effective development/testing
- Simple setup without complex IAM

### Use Verda When:
- H100/A100 availability is priority
- You want automatic region selection
- Competitive GPU pricing

## Multi-Provider Workflows

You can use different providers for different stages:

```python
from skyward import AWS, DigitalOcean, Verda

# Development on DigitalOcean (cheap CPU)
with ComputePool(provider=DigitalOcean(), cpu=4) as dev_pool:
    preprocess_data() >> dev_pool

# Training on AWS (H100 GPUs)
with ComputePool(provider=AWS(), accelerator="H100") as train_pool:
    train_model() @ train_pool

# Inference on Verda (cost-effective)
with ComputePool(provider=Verda(), accelerator="A100") as infer_pool:
    run_inference() >> infer_pool
```

## Common Issues

### AWS: "No instances available"

1. Try a different region
2. Use `spot="if-available"` to fallback to on-demand
3. Request a service quota increase

### DigitalOcean: "Authentication failed"

1. Verify your token at https://cloud.digitalocean.com/account/api
2. Check token permissions (read/write access needed)

### Verda: "Region not available"

1. Remove the `region` parameter to enable auto-discovery
2. Check your account's region access
