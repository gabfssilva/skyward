# VastAI Provider

VastAI is a GPU marketplace offering competitive pricing from independent providers worldwide. Unlike traditional cloud providers with fixed instance types, VastAI offers dynamic pricing based on real-time supply and demand.

## When to Use VastAI

- Maximum cost savings through marketplace pricing
- Consumer GPUs (RTX 4090, 3090, etc.) at low prices
- Flexible compute requirements
- Multi-node distributed training with overlay networks
- Short-term experimentation and development

## Setup

### Install VastAI CLI

The VastAI provider requires the `vastai` CLI for overlay network functionality:

```bash
pip install vastai
```

### Configure API Key

Get your API key at: https://cloud.vast.ai/account/

**Option 1: CLI Configuration (Recommended)**

```bash
vastai set api-key YOUR_API_KEY
```

This stores the key in `~/.config/vastai/vast_api_key`.

**Option 2: Environment Variable**

```bash
export VAST_API_KEY=your_api_key
```

**Option 3: Direct Parameter**

```python
provider = sky.VastAI(api_key="your_api_key")
```

### Basic Usage

```python
import skyward as sky

pool = sky.ComputePool(
    provider=sky.VastAI(
        geolocation="US",
        min_reliability=0.95,
    ),
    accelerator="RTX 4090",
    allocation="spot-if-available",
)
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | `None` | API key (uses CLI config or env var if None) |
| `min_reliability` | `float` | `0.95` | Minimum host reliability score (0.0-1.0) |
| `geolocation` | `str \| None` | `None` | Filter by location (e.g., "US", "EU", "DE") |
| `bid_multiplier` | `float` | `1.2` | Multiplier for spot bidding (1.0 = minimum bid) |
| `instance_timeout` | `int` | `300` | Auto-shutdown timeout in seconds |
| `docker_image` | `str \| None` | `None` | Custom Docker image (auto-detected if None) |
| `disk_gb` | `int` | `100` | Requested disk space in GB |
| `use_overlay` | `bool` | `True` | Enable overlay networking for multi-node |
| `overlay_timeout` | `int` | `120` | Timeout for overlay operations in seconds |

## Marketplace Dynamics

Unlike traditional cloud providers, VastAI is a dynamic marketplace:

- **Offers change frequently**: Hosts come and go, prices fluctuate
- **Reliability varies**: Each host has a reliability score based on uptime history
- **No fixed regions**: Offers are global, filterable by geolocation
- **Docker-based**: Instances are containers, not VMs

### Pricing Model

VastAI offers two pricing modes:

**Interruptible (Spot-like):**
- Bid-based pricing using `min_bid` from offers
- `bid_multiplier` controls how much above minimum you bid
- Can be interrupted if outbid
- Significant cost savings (often 50-70% less)

**On-Demand:**
- Fixed `dph_total` (dollars per hour) pricing
- Not interruptible
- Higher cost but guaranteed availability

```python
# Interruptible pricing (default)
pool = sky.ComputePool(
    provider=sky.VastAI(bid_multiplier=1.3),  # Bid 30% above minimum
    accelerator="RTX 4090",
    allocation="spot-if-available",
)

# On-demand pricing
pool = sky.ComputePool(
    provider=sky.VastAI(),
    accelerator="RTX 4090",
    allocation="on-demand",
)
```

## Reliability Filtering

VastAI tracks host reliability based on historical uptime. Use `min_reliability` to filter hosts:

```python
# High reliability (fewer options, more stable)
provider = sky.VastAI(min_reliability=0.98)

# Moderate reliability (more options, lower cost)
provider = sky.VastAI(min_reliability=0.90)

# Low threshold (most options, highest risk)
provider = sky.VastAI(min_reliability=0.80)
```

**Reliability Guidelines:**
- `0.98+`: Production workloads, critical training runs
- `0.95`: Default, good balance of cost and stability
- `0.90`: Development, fault-tolerant workloads
- `0.80`: Experimentation, checkpoint-heavy training

## Geolocation Filtering

Filter offers by geographic location:

```python
# United States only
provider = sky.VastAI(geolocation="US")

# Europe
provider = sky.VastAI(geolocation="EU")

# Specific country
provider = sky.VastAI(geolocation="DE")  # Germany
provider = sky.VastAI(geolocation="CA")  # Canada

# Any location (default)
provider = sky.VastAI(geolocation=None)
```

## Supported Accelerators

VastAI offers a wide range of GPUs from the marketplace:

### Consumer GPUs

```python
# GeForce RTX Series
sky.Accelerator("RTX 4090")
sky.Accelerator("RTX 4080")
sky.Accelerator("RTX 3090")
sky.Accelerator("RTX 3080")
sky.Accelerator("RTX 3080 Ti")
```

### Professional/Datacenter GPUs

```python
# NVIDIA A-Series
sky.Accelerator("A100")
sky.Accelerator("A100", count=8)
sky.Accelerator("A6000")
sky.Accelerator("A40")

# NVIDIA H-Series
sky.Accelerator("H100")
sky.Accelerator("H100", count=8)

# NVIDIA L-Series
sky.Accelerator("L40S")
sky.Accelerator("L4")
```

**Note:** GPU availability varies based on marketplace supply. Use reliability and geolocation filters to find suitable offers.

## Overlay Networks

Overlay networks create virtual LANs enabling direct communication between instances on all ports. This is **required** for NCCL-based distributed training.

### Automatic Overlay Setup

When `nodes > 1`, Skyward automatically:

1. Searches for offers in physical clusters (`cluster_id != null`)
2. Creates an overlay network on the selected cluster
3. Joins all instances to the overlay
4. Uses overlay IPs for inter-node communication

```python
import skyward as sky

# Multi-node automatically creates overlay
pool = sky.ComputePool(
    provider=sky.VastAI(
        geolocation="US",
        min_reliability=0.95,
    ),
    accelerator="RTX 4090",
    nodes=4,  # Automatically creates overlay network
)
```

### How Overlay Networks Work

1. **Physical Clusters**: VastAI groups machines into physical clusters with fast local networking
2. **Overlay Creation**: A virtual network is created on top of the physical cluster
3. **Instance Joining**: Each instance gets an overlay IP (typically on `eth0`)
4. **Direct Communication**: Instances can communicate on all ports without NAT

### Overlay Network Requirements

- **Multi-node only**: Single-node pools don't need overlay
- **Physical cluster**: Offers must have `cluster_id` (machines in same datacenter)
- **VastAI CLI**: Required for overlay management commands

### Disabling Overlay (Not Recommended)

If you don't need NCCL communication, you can disable overlay:

```python
provider = sky.VastAI(use_overlay=False)
```

**Warning:** Without overlay, NCCL-based distributed training will fail. Only disable for non-distributed workloads.

### Manual Overlay Commands (Advanced)

The VastAI CLI provides direct overlay management:

```bash
# List existing overlays
vastai show overlays

# Create overlay on cluster
vastai create overlay CLUSTER_ID OVERLAY_NAME

# Join instance to overlay
vastai join overlay OVERLAY_NAME INSTANCE_ID

# Delete overlay
vastai delete overlay OVERLAY_NAME
```

## Docker Images

VastAI uses Docker containers. Skyward automatically selects an appropriate CUDA image based on your accelerator requirements:

```python
# Automatic image selection (recommended)
pool = sky.ComputePool(
    provider=sky.VastAI(),
    accelerator="RTX 4090",
)
# Uses: nvcr.io/nvidia/cuda:12.4.0-runtime-ubuntu22.04 (auto-detected)

# Custom image
pool = sky.ComputePool(
    provider=sky.VastAI(docker_image="nvcr.io/nvidia/pytorch:24.01-py3"),
    accelerator="RTX 4090",
)
```

### Image Selection Logic

1. If `docker_image` is specified, use it
2. If accelerator has CUDA metadata, derive from requirements
3. Fallback: `nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04`

## Examples

### Single Node GPU Training

```python
import skyward as sky

@sky.compute
def train_model(config: dict) -> dict:
    import torch
    # Training code here
    return {"loss": 0.01}

@sky.pool(
    provider=sky.VastAI(
        geolocation="US",
        min_reliability=0.95,
    ),
    accelerator="RTX 4090",
    image=sky.Image(pip=["torch", "transformers"]),
    allocation="spot-if-available",
)
def main():
    result = train_model({"epochs": 10}) >> sky
    print(f"Training result: {result}")

if __name__ == "__main__":
    main()
```

### Multi-Node Distributed Training

```python
import skyward as sky

@sky.compute
@sky.integrations.torch()
def train_distributed(data_shard):
    import torch
    import torch.distributed as dist

    # NCCL initialized via overlay network
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Distributed training logic
    return {"rank": rank, "world_size": world_size}

@sky.pool(
    provider=sky.VastAI(
        geolocation="US",
        min_reliability=0.98,  # Higher for distributed
    ),
    accelerator=sky.Accelerator("RTX 4090", count=1),
    nodes=4,  # Creates overlay network automatically
    image=sky.Image(pip=["torch"]),
)
def main():
    # Broadcast to all nodes
    results = train_distributed(None) @ sky
    for r in results:
        print(f"Node {r['rank']}/{r['world_size']} completed")

if __name__ == "__main__":
    main()
```

### Cost-Optimized Batch Processing

```python
import skyward as sky

@sky.compute
def process_batch(batch_id: int) -> dict:
    # Processing logic
    return {"batch": batch_id, "status": "complete"}

@sky.pool(
    provider=sky.VastAI(
        min_reliability=0.90,  # Lower for fault-tolerant work
        bid_multiplier=1.1,    # Aggressive bidding
    ),
    accelerator="RTX 3080",
    image=sky.Image(pip=["numpy", "scipy"]),
)
def main():
    # Process multiple batches in parallel
    results = sky.gather(
        process_batch(i) for i in range(10)
    ) >> sky
    print(f"Processed {len(results)} batches")

if __name__ == "__main__":
    main()
```

### High-Reliability Production Run

```python
import skyward as sky

@sky.compute
def production_inference(data):
    # Critical inference workload
    return predictions

pool = sky.ComputePool(
    provider=sky.VastAI(
        min_reliability=0.99,   # Maximum reliability
        geolocation="US",       # Consistent latency
    ),
    accelerator="A100",
    allocation="on-demand",     # No interruptions
    image=sky.Image(pip=["torch", "transformers"]),
)
```

## Troubleshooting

### "No offers available"

The marketplace may not have matching offers. Try:

1. **Lower reliability threshold:**
   ```python
   provider = sky.VastAI(min_reliability=0.85)
   ```

2. **Remove geolocation filter:**
   ```python
   provider = sky.VastAI(geolocation=None)
   ```

3. **Try different GPU:**
   ```python
   # RTX 4090 unavailable? Try RTX 3090
   pool = sky.ComputePool(provider=provider, accelerator="RTX 3090")
   ```

4. **Check marketplace:** Visit https://cloud.vast.ai/ to see current availability

### "No physical cluster found"

Multi-node requires offers in physical clusters for overlay networking:

1. **Reduce node count:**
   ```python
   pool = sky.ComputePool(provider=provider, nodes=2)  # Instead of 4
   ```

2. **Try different GPU with cluster availability:**
   ```python
   pool = sky.ComputePool(provider=provider, accelerator="RTX 3090")
   ```

3. **Disable overlay (not recommended for NCCL):**
   ```python
   provider = sky.VastAI(use_overlay=False)
   ```

### "VastAI CLI not found"

Install the CLI:

```bash
pip install vastai
vastai set api-key YOUR_API_KEY
```

### "Overlay network setup failed"

1. **Check CLI authentication:**
   ```bash
   vastai show instances  # Should list your instances
   ```

2. **Increase overlay timeout:**
   ```python
   provider = sky.VastAI(overlay_timeout=180)
   ```

3. **Check instance status:**
   ```bash
   vastai show instances
   ```

### SSH Connection Issues

VastAI uses non-standard SSH ports. Skyward handles this automatically, but for manual debugging:

```bash
# Get instance info
vastai show instance INSTANCE_ID

# SSH with correct port
ssh -p PORT root@HOST
```

### Instance Interrupted

Interruptible instances may be reclaimed if outbid:

1. **Increase bid multiplier:**
   ```python
   provider = sky.VastAI(bid_multiplier=1.5)  # 50% above minimum
   ```

2. **Use on-demand pricing:**
   ```python
   pool = sky.ComputePool(provider=provider, allocation="on-demand")
   ```

3. **Checkpoint frequently:** Save model state regularly for fault tolerance

---

## Related Topics

- [Accelerators](../accelerators.md) - GPU selection
- [Distributed Training](../distributed-training.md) - Multi-node training patterns
- [Getting Started](../getting-started.md) - Installation and setup
- [Troubleshooting](../troubleshooting.md) - Common issues and solutions
