# Verda Provider

Verda is a cloud provider focused on GPU compute, offering competitive pricing and automatic region discovery. It provides access to a wide range of NVIDIA and AMD accelerators with full spot instance support.

## When to Use Verda

- European data residency requirements (Finland, Iceland, Israel)
- Cost-effective GPU compute via spot instances
- Automatic region discovery for best availability
- Access to latest GPUs including GB200
- AMD MI250X and MI300X workloads

## Setup

### API Credentials

Obtain your API credentials from the Verda dashboard.

### Environment Variables

```bash
export VERDA_CLIENT_ID=your_client_id
export VERDA_CLIENT_SECRET=your_client_secret
```

### Basic Usage

```python
import skyward as sky

# With explicit region
pool = sky.ComputePool(
    provider=sky.Verda(region="FIN-01"),
    accelerator="H100",
)

# With auto-region discovery
pool = sky.ComputePool(
    provider=sky.Verda(),  # Auto-discovers region with availability
    accelerator="H100",
)
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `region` | `str` | `"FIN-01"` | Verda datacenter region |
| `client_id` | `str \| None` | `None` | API client ID (uses env var if None) |
| `client_secret` | `str \| None` | `None` | API client secret (uses env var if None) |
| `ssh_key_id` | `str \| None` | `None` | Pre-registered SSH key ID |
| `instance_timeout` | `int` | `300` | Bootstrap timeout in seconds |

## SSH Key Management

Skyward automatically manages SSH keys on Verda:

1. **Auto-detection**: Searches for local SSH keys:
   - `~/.ssh/id_ed25519.pub`
   - `~/.ssh/id_rsa.pub`
   - `~/.ssh/id_ecdsa.pub`

2. **Fingerprint matching**: Checks existing keys by fingerprint

3. **Auto-registration**: Creates the key on Verda and stores the ID

### Using a Pre-registered Key

```python
provider = sky.Verda(
    region="FIN-01",
    ssh_key_id="key-abc123",
)
```

## Available Regions

| Region | Location | Notes |
|--------|----------|-------|
| `FIN-01` | Finland | Default region, European data residency |
| `ICL-01` | Iceland | Green energy, cool climate |
| `ISR-01` | Israel | Middle East presence |

### Auto-Region Fallback

If your requested region lacks capacity, Verda automatically finds another region with availability:

```python
# Will automatically use another region if FIN-01 is full
pool = sky.ComputePool(
    provider=sky.Verda(region="FIN-01"),
    accelerator="H100",
    nodes=8,
)
```

### Auto-Region Discovery

Omit the region to let Skyward find the best available region:

```python
pool = sky.ComputePool(
    provider=sky.Verda(),  # No region specified
    accelerator="A100",
    nodes=4,
)
```

## Instance Types

Verda dynamically fetches available instance types from the API. GPU configurations are parsed from descriptions like `"8x H100 SXM5 80GB"`.

### Common Configurations

| GPU | Count Options | Memory per GPU |
|-----|--------------|----------------|
| V100 | 1, 2, 4, 8 | 16 GB, 32 GB |
| A100 | 1, 2, 4, 8 | 40 GB, 80 GB |
| H100 | 1, 2, 4, 8 | 80 GB |
| H100-SXM | 8 | 80 GB |
| H200 | 8 | 141 GB |
| L40S | 1, 2, 4, 8 | 48 GB |
| GB200 | 2 | 192 GB |
| MI250X | 1, 2, 4, 8 | 128 GB |
| MI300X | 1, 2, 4, 8 | 192 GB |

Availability varies by region and demand.

## Supported Accelerators

### NVIDIA GPUs

```python
# V100
sky.Accelerator.NVIDIA.V100()
sky.Accelerator.NVIDIA.V100(memory="32GB")

# A100
sky.Accelerator.NVIDIA.A100()                  # 80GB default
sky.Accelerator.NVIDIA.A100(memory="40GB")
sky.Accelerator.NVIDIA.A100(count=8)

# H100
sky.Accelerator.NVIDIA.H100()
sky.Accelerator.NVIDIA.H100(count=8)
sky.Accelerator.NVIDIA.H100(form_factor="SXM")

# H200
sky.Accelerator.NVIDIA.H200()
sky.Accelerator.NVIDIA.H200(count=8)

# L40S
sky.Accelerator.NVIDIA.L40S()
sky.Accelerator.NVIDIA.L40S(count=4)

# GB200 (latest generation)
sky.Accelerator.NVIDIA.GB200()
```

### AMD GPUs

```python
# MI250X
sky.Accelerator.AMD.MI("250X")
sky.Accelerator.AMD.MI("250X", count=8)

# MI300X
sky.Accelerator.AMD.MI("300X")
sky.Accelerator.AMD.MI("300X", count=8)
```

### MIG Support

Full MIG support on H100 and A100:

```python
# H100 MIG profiles
sky.Accelerator.NVIDIA.H100(mig="1g.10gb")
sky.Accelerator.NVIDIA.H100(mig="2g.20gb")
sky.Accelerator.NVIDIA.H100(mig="3g.40gb")
sky.Accelerator.NVIDIA.H100(mig="4g.40gb")
sky.Accelerator.NVIDIA.H100(mig="7g.80gb")

# A100-80GB MIG profiles
sky.Accelerator.NVIDIA.A100(mig="1g.10gb")
sky.Accelerator.NVIDIA.A100(mig="3g.40gb")
sky.Accelerator.NVIDIA.A100(mig="7g.80gb")

# A100-40GB MIG profiles
sky.Accelerator.NVIDIA.A100(memory="40GB", mig="1g.5gb")
sky.Accelerator.NVIDIA.A100(memory="40GB", mig="3g.20gb")
```

## Spot Instances

Verda offers full spot instance support with significant cost savings.

### Requesting Spot Instances

```python
pool = sky.ComputePool(
    provider=sky.Verda(region="FIN-01"),
    accelerator="H100",
    allocation="spot",
)
```

### Spot with Fallback

```python
pool = sky.ComputePool(
    provider=sky.Verda(region="FIN-01"),
    accelerator="A100",
    allocation="spot-if-available",  # Falls back to on-demand
)
```

### Availability Checking

Verda's API provides real-time availability for both spot and on-demand instances. Skyward automatically checks availability before launching.

## Billing

Verda uses 10-minute billing increments:

- **Minimum charge**: 10 minutes
- **Billing increment**: 10 minutes
- **Spot pricing**: Significantly lower than on-demand
- **No long-term commitments**: Pay only for what you use

## Connectivity

### SSH Access

- **Protocol**: SSH on port 22
- **Public IP**: All instances receive a public IP
- **Username**: `root`

### Example SSH Connection

```bash
ssh root@<instance-ip>
```

## Examples

### Auto-Region GPU Training

```python
import skyward as sky

# Skyward finds the best region with H100 availability
pool = sky.ComputePool(
    provider=sky.Verda(),  # Auto-discover region
    accelerator=sky.Accelerator.NVIDIA.H100(count=8),
    image=sky.Image(
        pip=["torch", "transformers", "accelerate"],
    ),
)

@sky.compute
def train_model(config: dict) -> dict:
    # Training code here
    return {"loss": 0.01, "accuracy": 0.99}

with pool:
    results = train_model(training_config)
```

### Spot Instance Batch Processing

```python
import skyward as sky

pool = sky.ComputePool(
    provider=sky.Verda(region="ICL-01"),
    accelerator="A100",
    allocation="spot",
    nodes=4,
)

@sky.compute
def process_batch(data: list) -> list:
    import torch
    # GPU processing
    return processed_data

with pool:
    results = sky.gather(
        process_batch(batch) for batch in batches
    )
```

### AMD MI300X for Large Models

```python
import skyward as sky

pool = sky.ComputePool(
    provider=sky.Verda(region="FIN-01"),
    accelerator=sky.Accelerator.AMD.MI("300X", count=8),
    image=sky.Image(
        pip=["torch", "transformers"],
        env={"HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
    ),
)
```

### MIG Partitioning

```python
import skyward as sky

# Run multiple workloads on partitioned GPU
pool = sky.ComputePool(
    provider=sky.Verda(region="FIN-01"),
    accelerator=sky.Accelerator.NVIDIA.H100(mig="3g.40gb"),
)
```

## Troubleshooting

### "Region not available"

1. **Use auto-discovery**: Remove the `region` parameter

```python
# Let Skyward find an available region
provider = sky.Verda()  # No region specified
```

2. **Try a different region**:

```python
# Try ICL-01 or ISR-01
provider = sky.Verda(region="ICL-01")
```

### "Authentication failed"

1. **Check environment variables**:

```bash
echo $VERDA_CLIENT_ID
echo $VERDA_CLIENT_SECRET
```

2. **Verify credentials**: Log into Verda dashboard and confirm client ID/secret

3. **Regenerate credentials**: If expired, create new API credentials

### "No instances available"

1. **Enable spot fallback**:

```python
pool = sky.ComputePool(
    provider=sky.Verda(),
    accelerator="H100",
    allocation="spot-if-available",
)
```

2. **Reduce node count**: Start with fewer nodes

3. **Try different GPU**: Check availability of alternative accelerators

### "SSH connection refused"

1. **Wait for bootstrap**: Instance may still be initializing

2. **Check instance status**: Ensure instance is in `running` state

3. **Verify SSH key**: Confirm your local SSH key was registered

### Bootstrap Timeout

Increase timeout for complex images:

```python
provider = sky.Verda(
    region="FIN-01",
    instance_timeout=600,  # 10 minutes
)
```

### Instance Polling

Verda uses exponential backoff (2-10 seconds) when polling for instance status. Long provisioning times are normal for high-demand GPUs.

---

## Related Topics

- [Accelerators](../accelerators.md) - GPU selection and MIG partitioning
- [Getting Started](../getting-started.md) - Installation and setup
- [Troubleshooting](../troubleshooting.md) - Common issues and solutions
