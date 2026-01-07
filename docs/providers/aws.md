# AWS Provider

Amazon Web Services (AWS) is the most feature-complete provider in Skyward, offering access to a wide range of GPU instances, spot pricing, and AWS-specific accelerators like Trainium and Inferentia.

## When to Use AWS

- Production workloads requiring reliability and scale
- Access to H100, A100, Trainium, or Inferentia accelerators
- Significant cost savings via spot instances (60-90%)
- Multi-region deployments
- Integration with existing AWS infrastructure

## Setup

### Environment Variables

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1  # optional, can be set in code
```

### Using AWS CLI

Alternatively, configure credentials via AWS CLI:

```bash
aws configure
```

This stores credentials in `~/.aws/credentials`.

### Basic Usage

```python
import skyward as sky

pool = sky.ComputePool(
    provider=sky.AWS(region="us-east-1"),
    accelerator="H100",
)
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `region` | `str` | `"us-east-1"` | AWS region for instance deployment |
| `ami` | `str \| None` | `None` | Custom AMI ID (auto-detected if None) |
| `subnet_id` | `str \| None` | `None` | VPC subnet (auto-created if None) |
| `security_group_id` | `str \| None` | `None` | Security group (auto-created if None) |
| `instance_profile_arn` | `str \| None` | `None` | IAM instance profile (auto-created if None) |
| `username` | `str \| None` | `None` | Override SSH username |
| `instance_timeout` | `int` | `300` | Bootstrap timeout in seconds |
| `allocation_strategy` | `str` | `"price-capacity-optimized"` | Spot allocation strategy |

## Automatic Infrastructure

Skyward automatically creates and manages the following AWS resources:

### S3 Bucket

- Created per region automatically
- Used exclusively for S3 volume support
- Cached and reused across clusters

### IAM Role and Instance Profile

- Role name: `skyward-role`
- Instance profile: `skyward-instance-profile`
- Attached policies:
  - S3 access for volume operations

### Security Group

- Name: `skyward-sg`
- Inbound rules:
  - SSH (port 22) from `0.0.0.0/0`
  - All traffic between instances in the same group
- Outbound rules:
  - All traffic allowed

### Subnet Discovery

- Auto-discovers available subnets across all Availability Zones
- Selects subnets with public IP auto-assignment enabled

## Required IAM Permissions

The IAM user or role running Skyward needs these permissions:

> **Note**: `ssm:GetParameter` is used to look up AMI IDs from AWS public SSM parameters, not for SSM connectivity. Skyward uses SSH for all instance communication.

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
                "ec2:DescribeSubnets",
                "ec2:DescribeSecurityGroups",
                "ec2:CreateSecurityGroup",
                "ec2:AuthorizeSecurityGroupIngress",
                "ec2:AuthorizeSecurityGroupEgress",
                "ec2:CreateFleet",
                "ec2:DescribeFleetInstances",
                "ec2:CreateTags"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ssm:GetParameter"
            ],
            "Resource": "arn:aws:ssm:*:*:parameter/aws/service/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:CreateBucket",
                "s3:PutObject",
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::skyward-*",
                "arn:aws:s3:::skyward-*/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:CreateRole",
                "iam:CreateInstanceProfile",
                "iam:AddRoleToInstanceProfile",
                "iam:AttachRolePolicy",
                "iam:PassRole",
                "iam:GetRole",
                "iam:GetInstanceProfile"
            ],
            "Resource": [
                "arn:aws:iam::*:role/skyward-*",
                "arn:aws:iam::*:instance-profile/skyward-*"
            ]
        }
    ]
}
```

## AMI Selection

Skyward automatically selects the appropriate AMI based on instance type:

| Instance Type | AMI | Notes |
|---------------|-----|-------|
| GPU instances | Deep Learning AMI (DLAMI) | Pre-installed NVIDIA drivers |
| Fractional GPU (G6f) | Ubuntu 22.04 + GRID driver | GRID driver installed at bootstrap |
| CPU instances | Ubuntu 22.04 | Minimal base image |

### Architecture Support

- **x86_64**: Standard AMD/Intel processors
- **arm64**: Graviton processors (limited GPU support)

### Custom AMI

Override automatic selection with a custom AMI:

```python
pool = sky.ComputePool(
    provider=sky.AWS(
        region="us-east-1",
        ami="ami-0123456789abcdef0",  # Your custom AMI
    ),
    accelerator="A100",
)
```

## Instance Types

### GPU Instances

| Instance | GPUs | GPU Memory | vCPUs | RAM | Use Case |
|----------|------|------------|-------|-----|----------|
| `p5.48xlarge` | 8x H100 | 640 GB | 192 | 2 TB | Large model training |
| `p5n.48xlarge` | 8x H200 | 1.1 TB | 192 | 2 TB | Largest models |
| `p4d.24xlarge` | 8x A100 (40GB) | 320 GB | 96 | 1.1 TB | Distributed training |
| `p4de.24xlarge` | 8x A100 (80GB) | 640 GB | 96 | 1.1 TB | Large batch training |
| `g5.xlarge` | 1x A10G | 24 GB | 4 | 16 GB | Inference |
| `g5.48xlarge` | 8x A10G | 192 GB | 192 | 768 GB | Multi-GPU inference |
| `g6.xlarge` | 1x L4 | 24 GB | 4 | 16 GB | Cost-effective inference |
| `g4dn.xlarge` | 1x T4 | 16 GB | 4 | 16 GB | Development |

### Fractional GPU (G6f)

G6f instances provide fractional access to A10G GPUs:

| Instance | GPU Fraction | GPU Memory | vCPUs | RAM |
|----------|--------------|------------|-------|-----|
| `g6f.xlarge` | 0.25x A10G | 6 GB | 4 | 16 GB |
| `g6f.2xlarge` | 0.5x A10G | 12 GB | 8 | 32 GB |
| `g6f.4xlarge` | 1x A10G | 24 GB | 16 | 64 GB |
| `g6f.8xlarge` | 2x A10G | 48 GB | 32 | 128 GB |

### Trainium/Inferentia

| Instance | Accelerators | Use Case |
|----------|--------------|----------|
| `trn1.2xlarge` | 1x Trainium1 | Small model training |
| `trn1.32xlarge` | 16x Trainium1 | Distributed training |
| `trn1n.32xlarge` | 16x Trainium1 | High-bandwidth training |
| `trn2.48xlarge` | 16x Trainium2 | Latest generation training |
| `inf2.xlarge` | 1x Inferentia2 | Single-model inference |
| `inf2.48xlarge` | 12x Inferentia2 | High-throughput inference |

## Supported Accelerators

### NVIDIA GPUs

```python
# H100 variants
sky.AcceleratorSpec.NVIDIA.H100()  # 1x H100-80GB
sky.AcceleratorSpec.NVIDIA.H100(count=8)  # 8x H100-80GB
sky.AcceleratorSpec.NVIDIA.H100(form_factor="SXM")  # H100-SXM
sky.AcceleratorSpec.NVIDIA.H100(form_factor="PCIe")  # H100-PCIe
sky.AcceleratorSpec.NVIDIA.H100(form_factor="NVL")  # H100-NVL

# A100 variants
sky.AcceleratorSpec.NVIDIA.A100()  # 1x A100-80GB
sky.AcceleratorSpec.NVIDIA.A100(memory="40GB")  # A100-40GB
sky.AcceleratorSpec.NVIDIA.A100(count=8)  # 8x A100

# Other GPUs
sky.AcceleratorSpec.NVIDIA.L4()
sky.AcceleratorSpec.NVIDIA.L40S()
sky.AcceleratorSpec.NVIDIA.T4()
sky.AcceleratorSpec.NVIDIA.V100()
```

### MIG Support

Multi-Instance GPU (MIG) is supported on H100 and A100:

```python
# H100 MIG profiles
sky.AcceleratorSpec.NVIDIA.H100(mig="1g.10gb")  # 1 compute slice, 10GB
sky.AcceleratorSpec.NVIDIA.H100(mig="3g.40gb")  # 3 compute slices, 40GB
sky.AcceleratorSpec.NVIDIA.H100(mig="7g.80gb")  # 7 compute slices, 80GB

# A100-80GB MIG profiles
sky.AcceleratorSpec.NVIDIA.A100(mig="1g.10gb")
sky.AcceleratorSpec.NVIDIA.A100(mig="2g.20gb")
sky.AcceleratorSpec.NVIDIA.A100(mig="3g.40gb")
sky.AcceleratorSpec.NVIDIA.A100(mig="7g.80gb")

# A100-40GB MIG profiles
sky.AcceleratorSpec.NVIDIA.A100(memory="40GB", mig="1g.5gb")
sky.AcceleratorSpec.NVIDIA.A100(memory="40GB", mig="3g.20gb")
```

### AWS Trainium/Inferentia

```python
# Trainium for training
sky.AcceleratorSpec.AWS.Trainium(version=1)
sky.AcceleratorSpec.AWS.Trainium(version=2)

# Inferentia for inference
sky.AcceleratorSpec.AWS.Inferentia(version=2)
```

## Spot Instances

Spot instances offer 60-90% savings over on-demand pricing.

### Allocation Strategies

| Strategy | Description |
|----------|-------------|
| `"price-capacity-optimized"` | Balance between price and capacity (default) |
| `"capacity-optimized"` | Prioritize capacity availability |
| `"lowest-price"` | Select cheapest instances |
| `"diversified"` | Spread across instance types |

### Usage

```python
pool = sky.ComputePool(
    provider=sky.AWS(
        region="us-east-1",
        allocation_strategy="price-capacity-optimized",
    ),
    accelerator="A100",
    allocation="spot",  # Request spot instances
)
```

### Spot with Fallback

```python
pool = sky.ComputePool(
    provider=sky.AWS(region="us-east-1"),
    accelerator="A100",
    allocation="spot-if-available",  # Fallback to on-demand
)
```

## Available Regions

| Region | Location | H100 | A100 | T4/L4 | Trainium |
|--------|----------|------|------|-------|----------|
| `us-east-1` | N. Virginia | Yes | Yes | Yes | Yes |
| `us-east-2` | Ohio | No | Yes | Yes | No |
| `us-west-2` | Oregon | Yes | Yes | Yes | Yes |
| `eu-west-1` | Ireland | No | Yes | Yes | No |
| `eu-central-1` | Frankfurt | No | Yes | Yes | No |
| `ap-northeast-1` | Tokyo | No | Yes | Yes | No |
| `ap-southeast-1` | Singapore | No | Yes | Yes | No |

## SSH Connectivity

AWS uses standard SSH for all instance connectivity:

- **Public IP**: Instances are assigned public IPs via the auto-created subnet
- **SSH Key**: Your local SSH key (`~/.ssh/id_ed25519.pub` or `~/.ssh/id_rsa.pub`) is automatically injected
- **Port**: Standard SSH port 22

### SSH Username

| AMI Type | Default Username |
|----------|------------------|
| Deep Learning AMI | `ubuntu` |
| Ubuntu base | `ubuntu` |
| Amazon Linux | `ec2-user` |

Override with the `username` parameter if needed.

## EBS Volumes

AWS supports EBS volume mounting for persistent storage:

```python
pool = sky.ComputePool(
    provider=sky.AWS(region="us-east-1"),
    accelerator="A100",
    volumes=[
        sky.Volume(size_gb=500, mount_path="/data"),
    ],
)
```

Volumes are automatically:
- Created in the same AZ as instances
- Attached and mounted at the specified path
- Formatted with ext4 filesystem (if new)

## Troubleshooting

### "No instances available"

1. **Try a different region**: GPU capacity varies by region

```python
# Try us-west-2 instead of us-east-1
provider = sky.AWS(region="us-west-2")
```

2. **Use spot-if-available**: Fall back to on-demand

```python
pool = sky.ComputePool(
    provider=sky.AWS(region="us-east-1"),
    accelerator="A100",
    allocation="spot-if-available",
)
```

3. **Request quota increase**: Check your service quotas in AWS Console

### "Access Denied" Errors

- Verify IAM permissions match the policy above
- Ensure `iam:PassRole` is allowed for skyward roles
- Check that your credentials are correctly configured

### Instance Bootstrap Timeout

- Default timeout is 300 seconds
- Increase with `instance_timeout` parameter:

```python
provider = sky.AWS(
    region="us-east-1",
    instance_timeout=600,  # 10 minutes
)
```

### Subnet/VPC Issues

If auto-created infrastructure fails:

```python
# Use existing VPC resources
provider = sky.AWS(
    region="us-east-1",
    subnet_id="subnet-0123456789abcdef0",
    security_group_id="sg-0123456789abcdef0",
)
```

---

## Related Topics

- [Accelerators](../accelerators.md) - GPU selection and MIG partitioning
- [Getting Started](../getting-started.md) - Installation and setup
- [Troubleshooting](../troubleshooting.md) - Common issues and solutions
