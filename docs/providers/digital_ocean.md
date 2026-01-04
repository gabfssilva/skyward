# DigitalOcean Provider

DigitalOcean offers a straightforward cloud experience with simple pricing and easy-to-use GPU instances. It provides access to NVIDIA H100, H200, L40S GPUs and AMD MI300X accelerators.

## When to Use DigitalOcean

- Simple setup without complex IAM configuration
- Access to H100, H200, L40S, and AMD MI300X GPUs
- Per-second billing for cost efficiency
- Development and testing environments
- CPU-only workloads at competitive pricing

## Setup

### API Token

Create a DigitalOcean API token at: https://cloud.digitalocean.com/account/api/tokens

The token needs **Read** and **Write** permissions.

### Environment Variable

```bash
export DIGITALOCEAN_TOKEN=your_api_token
```

### Basic Usage

```python
import skyward as sky

# GPU workload
pool = sky.ComputePool(
    provider=sky.DigitalOcean(region="nyc3"),
    accelerator="H100",
)

# CPU workload
pool = sky.ComputePool(
    provider=sky.DigitalOcean(region="nyc1"),
    cpu=4,
    memory="8GB",
)
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `region` | `str` | `"nyc3"` | DigitalOcean datacenter region |
| `token` | `str \| None` | `None` | API token (uses env var if None) |
| `ssh_key_fingerprint` | `str \| None` | `None` | Pre-registered SSH key fingerprint |
| `instance_timeout` | `int` | `300` | Bootstrap timeout in seconds |

## SSH Key Management

Skyward automatically manages SSH keys on DigitalOcean:

1. **Auto-detection**: Searches for local SSH keys in order:
   - `~/.ssh/id_ed25519.pub`
   - `~/.ssh/id_rsa.pub`
   - `~/.ssh/id_ecdsa.pub`

2. **Fingerprint matching**: Checks if the key already exists on DigitalOcean by fingerprint

3. **Auto-registration**: Creates the key on DigitalOcean if not found

### Using a Pre-registered Key

```python
provider = sky.DigitalOcean(
    region="nyc3",
    ssh_key_fingerprint="ab:cd:ef:12:34:56:78:90:...",
)
```

## Droplet Types

### CPU Droplets

| Size | vCPUs | Memory | Disk | Use Case |
|------|-------|--------|------|----------|
| `s-1vcpu-1gb` | 1 | 1 GB | 25 GB | Testing |
| `s-2vcpu-4gb` | 2 | 4 GB | 80 GB | Light workloads |
| `s-4vcpu-8gb` | 4 | 8 GB | 160 GB | Development |
| `c-2` | 2 | 4 GB | 25 GB | CPU-optimized |
| `c-8` | 8 | 16 GB | 100 GB | CPU compute |
| `c-32` | 32 | 64 GB | 400 GB | Heavy compute |

### GPU Droplets

| Type | GPUs | GPU Memory | vCPUs | RAM | Image |
|------|------|------------|-------|-----|-------|
| H100 1x | 1x H100 | 80 GB | 20 | 240 GB | `gpu-h100x1-base` |
| H100 8x | 8x H100 | 640 GB | 160 | 1.9 TB | `gpu-h100x8-base` |
| H200 | 8x H200 | 1.1 TB | 192 | 2 TB | `gpu-h200-base` |
| L40S | 1-8x L40S | 48 GB each | varies | varies | `gpu-l40s-base` |
| MI300X | 8x MI300X | 1.5 TB | 192 | 1.5 TB | `gpu-amd-base` |

## Supported Accelerators

### NVIDIA GPUs

```python
# H100
sky.Accelerator.NVIDIA.H100()           # 1x H100-80GB
sky.Accelerator.NVIDIA.H100(count=8)    # 8x H100-80GB

# H200
sky.Accelerator.NVIDIA.H200()
sky.Accelerator.NVIDIA.H200(count=8)

# L40S
sky.Accelerator.NVIDIA.L40S()
sky.Accelerator.NVIDIA.L40S(count=4)
```

### AMD GPUs

```python
# MI300X
sky.Accelerator.AMD.MI("300X")
sky.Accelerator.AMD.MI("300X", count=8)
```

### MIG Support

MIG is supported on H100 and A100 GPUs:

```python
sky.Accelerator.NVIDIA.H100(mig="3g.40gb")
sky.Accelerator.NVIDIA.H100(mig="7g.80gb")
```

## Available Regions

| Region | Location | GPUs Available |
|--------|----------|----------------|
| `nyc1` | New York 1 | CPU only |
| `nyc3` | New York 3 | H100, L40S |
| `sfo2` | San Francisco 2 | CPU only |
| `sfo3` | San Francisco 3 | H100, MI300X |
| `ams3` | Amsterdam 3 | CPU only |
| `sgp1` | Singapore 1 | CPU only |
| `lon1` | London 1 | CPU only |
| `fra1` | Frankfurt 1 | CPU only |
| `tor1` | Toronto 1 | H100 |

GPU availability varies by region. Use `nyc3`, `sfo3`, or `tor1` for GPU workloads.

## Connectivity

### SSH Access

- **Protocol**: SSH on port 22
- **Public IP**: All droplets receive a public IPv4 address
- **Username**: Depends on droplet type

| Droplet Type | SSH Username |
|--------------|--------------|
| CPU droplets | `root` |
| GPU droplets | `ubuntu` |

### Example SSH Connection

```bash
# For CPU droplet
ssh root@<droplet-ip>

# For GPU droplet
ssh ubuntu@<droplet-ip>
```

## Billing

DigitalOcean uses per-second billing:

- **Minimum charge**: 60 seconds
- **Billing increment**: 1 second after minimum
- **No spot instances**: All instances are on-demand
- **Predictable pricing**: No surge pricing or auctions

### Pricing Comparison

| Resource | Approximate Hourly Cost |
|----------|------------------------|
| H100 1x | ~$3.50/hour |
| H100 8x | ~$28/hour |
| c-8 (CPU) | ~$0.12/hour |
| s-4vcpu-8gb | ~$0.048/hour |

Check [DigitalOcean Pricing](https://www.digitalocean.com/pricing) for current rates.

## Examples

### CPU Workload

```python
import skyward as sky

pool = sky.ComputePool(
    provider=sky.DigitalOcean(region="nyc1"),
    cpu=4,
    memory="8GB",
    nodes=3,
)

@sky.compute
def process_data(chunk: list) -> list:
    return [item.upper() for item in chunk]

with pool:
    results = sky.gather(
        process_data(chunk) for chunk in data_chunks
    )
```

### GPU Inference

```python
import skyward as sky

pool = sky.ComputePool(
    provider=sky.DigitalOcean(region="nyc3"),
    accelerator="H100",
    image=sky.Image(
        pip=["torch", "transformers"],
    ),
)

@sky.compute
def run_inference(prompts: list[str]) -> list[str]:
    from transformers import pipeline
    pipe = pipeline("text-generation", device="cuda")
    return [pipe(p)[0]["generated_text"] for p in prompts]

with pool:
    outputs = run_inference(my_prompts)
```

### AMD MI300X for Training

```python
import skyward as sky

pool = sky.ComputePool(
    provider=sky.DigitalOcean(region="sfo3"),
    accelerator=sky.Accelerator.AMD.MI("300X", count=8),
    image=sky.Image(
        pip=["torch", "torchvision"],
    ),
)
```

## Troubleshooting

### "Authentication failed"

1. **Verify token**: Check your token at https://cloud.digitalocean.com/account/api
2. **Check permissions**: Token needs Read and Write access
3. **Regenerate token**: If compromised, create a new token

```bash
# Verify token is set
echo $DIGITALOCEAN_TOKEN

# Test API access
curl -X GET \
  -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
  "https://api.digitalocean.com/v2/account"
```

### "SSH key not found"

1. **Check local keys exist**:

```bash
ls -la ~/.ssh/id_*.pub
```

2. **Generate a new key**:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

3. **Use explicit fingerprint**:

```python
provider = sky.DigitalOcean(
    region="nyc3",
    ssh_key_fingerprint="your-key-fingerprint",
)
```

### "Droplet size unavailable"

- GPU droplets are only available in specific regions
- Try `nyc3`, `sfo3`, or `tor1` for GPU workloads
- CPU droplets are available in all regions

### Bootstrap Timeout

Increase timeout for slow bootstraps:

```python
provider = sky.DigitalOcean(
    region="nyc3",
    instance_timeout=600,  # 10 minutes
)
```

### Rate Limiting

DigitalOcean API has rate limits. If you encounter `429` errors:

1. Wait a few minutes before retrying
2. Reduce parallel operations
3. Check your API usage in the DigitalOcean dashboard

---

## Related Topics

- [Accelerators](../accelerators.md) - GPU selection and MIG partitioning
- [Getting Started](../getting-started.md) - Installation and setup
- [Troubleshooting](../troubleshooting.md) - Common issues and solutions
