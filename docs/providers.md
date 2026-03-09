# Cloud providers

Skyward supports eight providers. Seven are cloud services — AWS, GCP, Hyperstack, RunPod, TensorDock, Verda, VastAI — and one is local containers for development and CI. All implement the same `Provider` protocol, so the orchestration layer (actor system, SSH tunnels, bootstrap, task dispatch) works identically regardless of which provider you choose. The difference is in how instances are provisioned, what hardware is available, and how authentication works.

Provider configs are lightweight frozen dataclasses. They hold configuration — region, API keys, disk sizes — but don't import any cloud SDK at module level. The SDK is loaded lazily when the pool starts, so `import skyward` stays fast regardless of which providers are installed.

## Provider comparison

| Feature | AWS | GCP | Hyperstack | RunPod | TensorDock | Verda | VastAI | Container |
|---------|-----|-----|------------|--------|------------|-------|--------|-----------|
| **GPUs** | H100, A100, T4, L4, Trainium, Inferentia | H100, A100, T4, L4, V100, H200 | A100, H100, RTX series | H100, A100, A40, RTX series | H100, A100, L40, RTX series, V100 | H100, A100, H200, GB200 | Marketplace (varies) | None (CPU) |
| **Spot Instances** | Yes (60-90% savings) | Yes (preemptible/spot) | No (on-demand only) | Yes | No (on-demand only) | Yes | Yes (bid-based) | N/A |
| **Regions** | 20+ | 40+ zones | Canada, Norway, US | Global (Secure + Community) | 100+ locations, 20+ countries | FIN, ICL, ISR | Global marketplace | Local |
| **Auth** | AWS credentials | Application Default Credentials | API key | API key | API key + token | Client ID + Secret | API key | None |
| **Billing** | Per-second | Per-second | Per-second | Per-second | Per-second | Per-second | Per-minute | Free |

## AWS

AWS uses EC2 Fleet for provisioning, with automatic spot-to-on-demand fallback. Instances are launched in a VPC with security groups managed by Skyward (or you can provide your own). SSH keys are created per cluster and cleaned up on teardown.

AMI resolution happens automatically via SSM Parameter Store — Skyward looks up the latest Ubuntu AMI for your chosen version and architecture. You can override this with a custom AMI.

### Setup

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

Or use the AWS CLI:

```bash
aws configure
```

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.AWS(region="us-east-1"),
    accelerator=sky.accelerators.A100(),
    nodes=2,
) as compute:
    result = train(data) >> compute
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `region` | `str` | `"us-east-1"` | AWS region |
| `ami` | `str or None` | `None` | Custom AMI ID. Auto-resolved via SSM if not set. |
| `ubuntu_version` | `str` | `"24.04"` | Ubuntu LTS version for auto-resolved AMIs |
| `subnet_id` | `str or None` | `None` | VPC subnet. Uses default VPC if not set. |
| `security_group_id` | `str or None` | `None` | Security group. Auto-created if not set. |
| `instance_profile_arn` | `str or None` | `None` | IAM instance profile. Auto-created if not set. |
| `username` | `str or None` | `None` | SSH user. Auto-detected from AMI if not set. |
| `instance_timeout` | `int` | `300` | Safety timeout in seconds (auto-shutdown timer) |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds |
| `allocation_strategy` | `str` | `"price-capacity-optimized"` | EC2 Fleet spot allocation strategy |
| `exclude_burstable` | `bool` | `False` | Exclude burstable instances (t3, t4g) |

### Required IAM permissions

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
                "ec2:DescribeKeyPairs",
                "ec2:CreateFleet",
                "ec2:DescribeFleets",
                "ssm:GetParameter"
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

## GCP

GCP uses Compute Engine with instance templates and `bulk_insert` for fleet-style provisioning. Skyward resolves the best machine type dynamically — for GPUs like T4 and V100, it uses N1 machines with guest accelerators; for A100 and H100, it picks the matching A2/A3 machine family with built-in GPUs. Spot instances use the `SPOT` provisioning model with automatic deletion on preemption.

SSH keys are injected via instance metadata. The project is auto-detected from Application Default Credentials or `GOOGLE_CLOUD_PROJECT`. GCP API calls use sync clients dispatched to a dedicated thread pool (configurable via `thread_pool_size`).

Skyward creates an instance template and a firewall rule per cluster, both cleaned up on teardown. Instances use Google's Deep Learning VM images (CUDA 12.x, NVIDIA 570 drivers) for GPU workloads.

### Setup

```bash
gcloud auth application-default login
```

Or set the project explicitly:

```bash
export GOOGLE_CLOUD_PROJECT=your_project_id
```

!!! warning "GPU Quotas"
    Listing available accelerator types does not mean you have quota. Check your quotas with:
    ```bash
    gcloud compute regions describe <region> --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep GPU
    ```
    Request quota increases in the [Cloud Console](https://console.cloud.google.com/iam-admin/quotas).

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.GCP(zone="us-central1-a"),
    accelerator=sky.accelerators.T4(),
    nodes=2,
) as compute:
    result = train(data) >> compute
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | `str or None` | `None` | GCP project ID. Auto-detected from ADC or `GOOGLE_CLOUD_PROJECT`. |
| `zone` | `str` | `"us-central1-a"` | Compute Engine zone |
| `network` | `str` | `"default"` | VPC network name |
| `subnet` | `str or None` | `None` | Specific subnet. Uses auto-mode subnet if not set. |
| `disk_size_gb` | `int` | `200` | Boot disk size in GB |
| `disk_type` | `str` | `"pd-balanced"` | Boot disk type |
| `instance_timeout` | `int` | `300` | Safety timeout in seconds (self-destruction timer) |
| `service_account` | `str or None` | `None` | GCE service account email |
| `thread_pool_size` | `int` | `8` | Thread pool size for blocking GCP API calls |

### Required permissions

The authenticated principal needs the following roles (or equivalent permissions):

- `compute.instances.create`, `compute.instances.delete`, `compute.instances.list`, `compute.instances.get`
- `compute.instanceTemplates.create`, `compute.instanceTemplates.delete`
- `compute.firewalls.create`, `compute.firewalls.delete`, `compute.firewalls.get`
- `compute.machineTypes.list`, `compute.acceleratorTypes.list`
- `compute.images.getFromFamily`

The simplest approach is the **Compute Admin** role (`roles/compute.admin`).

### Install

```bash
uv add "skyward[gcp]"
```

## RunPod

RunPod offers GPU pods in two tiers: **Secure Cloud** (enterprise-grade, dedicated hardware) and **Community Cloud** (lower-cost, peer-hosted). Skyward provisions pods via RunPod's GraphQL API, configures SSH access, and manages the full lifecycle.

SSH keys are auto-detected from `~/.ssh/id_ed25519.pub` or `~/.ssh/id_rsa.pub` and registered on your RunPod account.

### Setup

```bash
export RUNPOD_API_KEY=your_api_key
```

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.RunPod(),
    accelerator=sky.accelerators.A100(),
    nodes=2,
) as compute:
    result = train(data) >> compute
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cluster_mode` | `ClusterMode` | `"individual"` | Cluster mode (`"instant"` or `"individual"`) |
| `global_networking` | `bool or None` | `None` | Enable global networking |
| `api_key` | `str or None` | `None` | API key (falls back to `RUNPOD_API_KEY` env var) |
| `cloud_type` | `CloudType` | `SECURE` | `CloudType.SECURE` or `CloudType.COMMUNITY` |
| `ubuntu` | `str` | `"newest"` | Ubuntu version (`"20.04"`, `"22.04"`, `"24.04"`, `"newest"`) |
| `container_disk_gb` | `int` | `50` | Container disk size in GB |
| `volume_gb` | `int` | `20` | Persistent volume size in GB |
| `volume_mount_path` | `str` | `"/workspace"` | Volume mount path |
| `data_center_ids` | `tuple or "global"` | `"global"` | Preferred data centers or `"global"` for auto-selection |
| `ports` | `tuple[str, ...]` | `("22/tcp",)` | Port mappings |
| `provision_timeout` | `float` | `300.0` | Instance provision timeout in seconds |
| `bootstrap_timeout` | `float` | `600.0` | Bootstrap timeout in seconds |
| `instance_timeout` | `int` | `300` | Auto-shutdown safety timeout in seconds |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds |
| `cpu_clock` | `str` | `"3c"` | CPU clock tier (`"3c"` or `"5c"`) |
| `bid_multiplier` | `float` | `1` | Multiplier for spot bid price |
| `registry_auth` | `str or None` | `"docker hub"` | Container registry credential name. `None` to skip. |

## Verda

Verda is a GPU cloud with data centers in Europe and the Middle East. It uses OAuth2 authentication with a client ID and secret — not a single API key.

SSH keys are auto-detected and registered on Verda if needed. If `region` is not specified (the default is `"FIN-01"`), Verda uses its default region. The provider also supports auto-region discovery: if the requested GPU isn't available in the configured region, Skyward finds another region with availability.

### Setup

```bash
export VERDA_CLIENT_ID=your_client_id
export VERDA_CLIENT_SECRET=your_client_secret
```

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.Verda(),
    accelerator=sky.accelerators.H100(),
    nodes=4,
) as compute:
    results = train() @ compute
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `region` | `str` | `"FIN-01"` | Preferred region |
| `client_id` | `str or None` | `None` | OAuth2 client ID (falls back to `VERDA_CLIENT_ID`) |
| `client_secret` | `str or None` | `None` | OAuth2 client secret (falls back to `VERDA_CLIENT_SECRET`) |
| `ssh_key_id` | `str or None` | `None` | Specific SSH key ID to use |
| `instance_timeout` | `int` | `300` | Safety timeout in seconds |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds |

### Available regions

| Region | Location | GPUs |
|--------|----------|------|
| `FIN-01` | Finland | H100, A100, H200, GB200 |
| `ICL-01` | Iceland | H100, A100 |
| `ISR-01` | Israel | H100, A100 |

## VastAI

VastAI is a GPU marketplace — instances are Docker containers running on hosts from independent providers worldwide. Pricing is dynamic, and reliability varies by host. Skyward filters offers by reliability score, CUDA version, and optional geolocation, then provisions containers via the VastAI API.

SSH keys are auto-detected from `~/.ssh/id_ed25519.pub` or `~/.ssh/id_rsa.pub` and registered on VastAI if needed. For multi-node clusters, VastAI supports overlay networks for NCCL communication between instances.

### Setup

```bash
export VAST_API_KEY=your_api_key
```

Get your API key at: [https://cloud.vast.ai/account/](https://cloud.vast.ai/account/)

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.VastAI(geolocation="US"),
    accelerator=sky.accelerators.RTX_4090(),
    nodes=2,
) as compute:
    result = train(data) >> compute
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str or None` | `None` | API key (falls back to `VAST_API_KEY`) |
| `min_reliability` | `float` | `0.95` | Minimum host reliability score (0.0-1.0) |
| `verified_only` | `bool` | `True` | Only select offers from verified hosts |
| `min_cuda` | `float` | `12.0` | Minimum CUDA version |
| `geolocation` | `str or None` | `None` | Filter by region/country (e.g., `"US"`, `"EU"`) |
| `bid_multiplier` | `float` | `1.2` | Multiplier for spot bid price |
| `instance_timeout` | `int` | `300` | Auto-shutdown safety timeout in seconds |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds |
| `docker_image` | `str or None` | `None` | Base Docker image for containers |
| `disk_gb` | `int` | `100` | Disk space in GB |
| `use_overlay` | `bool` | `True` | Enable overlay networking for multi-node clusters |
| `overlay_timeout` | `int` | `120` | Timeout for overlay operations in seconds |
| `require_direct_port` | `bool` | `False` | Only select offers with direct port access |

VastAI also provides a helper for building NVIDIA CUDA base images:

```python
image_name = sky.VastAI.ubuntu(version="24.04", cuda="12.9.1")
# → "nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04"
```

## Hyperstack

Hyperstack provides bare-metal GPU instances via their InfraHub API. Resources are organized into environments that group VMs, keypairs, and volumes within a region. Environments are created per cluster and cascade-deleted on teardown. All instances are on-demand — no spot pricing.

### Setup

```bash
export HYPERSTACK_API_KEY=your_api_key
```

Get your API key at the [Hyperstack Console](https://infrahub.nexgencloud.com/).

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.Hyperstack(region="CANADA-1"),
    accelerator=sky.accelerators.A100(),
    nodes=2,
) as compute:
    result = train(data) >> compute
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str or None` | `None` | API key (falls back to `HYPERSTACK_API_KEY` env var) |
| `region` | `str or tuple or None` | `None` | Deployment region(s). A single string (e.g. `"CANADA-1"`), a tuple (e.g. `("CANADA-1", "NORWAY-1")`), or `None` to search all regions. |
| `image` | `str or None` | `None` | OS image name override. Auto-selects newest Ubuntu + CUDA image if not set. |
| `network_optimised` | `bool` | `False` | Require network-optimised environments with SR-IOV support |
| `network_optimised_regions` | `tuple[str, ...]` | `("CANADA-1", "US-1")` | Regions known to support network-optimised environments |
| `object_storage_region` | `str` | `"CANADA-1"` | Region for S3-compatible object storage (volume mounts) |
| `object_storage_endpoint` | `str` | `"https://ca1.obj.nexgencloud.io"` | Endpoint URL for S3-compatible object storage |
| `instance_timeout` | `int` | `300` | Auto-shutdown safety timeout in seconds |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds |
| `teardown_timeout` | `int` | `120` | Timeout for teardown operations in seconds |
| `teardown_poll_interval` | `float` | `2.0` | Poll interval during teardown in seconds |

### Available regions

| Region | Location |
|--------|----------|
| `CANADA-1` | Canada |
| `NORWAY-1` | Norway |
| `US-1` | United States |

## TensorDock

TensorDock is a GPU marketplace with bare-metal VMs across 100+ locations in 20+ countries. Per-second billing, on-demand only (no spot). Skyward queries available hostnodes, selects the cheapest matching your GPU requirements, and deploys VMs with cloud-init for SSH key injection.

SSH keys are injected per-instance via cloud-init (TensorDock has no SSH key registration API). The SSH user is `user` (not root). Port forwarding maps internal ports to random external ports — Skyward handles this automatically.

### Setup

```bash
export TENSORDOCK_API_KEY=your_api_key
export TENSORDOCK_API_TOKEN=your_api_token
```

Get your credentials at: [https://console.tensordock.com/api](https://console.tensordock.com/api)

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.TensorDock(location="us"),
    accelerator=sky.accelerators.RTX_4090(),
    nodes=2,
) as compute:
    result = train(data) >> compute
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str or None` | `None` | API key (falls back to `TENSORDOCK_API_KEY`) |
| `api_token` | `str or None` | `None` | API token (falls back to `TENSORDOCK_API_TOKEN`) |
| `location` | `str or None` | `None` | Country filter (e.g., `"United States"`, `"Germany"`). Global if not set. |
| `tier` | `int or None` | `None` | Hostnode tier (0-4). `None` for any tier. |
| `storage_gb` | `int` | `100` | Disk storage per VM in GB |
| `operating_system` | `str` | `"ubuntu2404"` | OS image ID (e.g., `"ubuntu2404"`, `"ubuntu2204"`) |
| `instance_timeout` | `int` | `300` | Auto-shutdown in seconds |
| `request_timeout` | `int` | `120` | HTTP request timeout in seconds |
| `min_ram_gb` | `int or None` | `None` | Minimum RAM per VM in GB |
| `min_vcpus` | `int or None` | `None` | Minimum vCPUs per VM |

!!! note "Port forwarding"
    TensorDock maps internal ports to random external ports. SSH is never on port 22 externally. Skyward reads the port mapping from the deploy response and configures SSH tunnels accordingly — no manual port configuration needed.

## Container

The Container provider runs compute nodes as local containers — Docker, podman, nerdctl, or Apple's container CLI. No cloud credentials, no costs. Useful for development, CI testing, and validating your code before deploying to real hardware.

Containers are launched with SSH access, joined to a shared network, and bootstrapped the same way cloud instances are. From the pool's perspective, they look like any other nodes.

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.Container(),
    nodes=2,
    image=sky.Image(pip=["numpy"]),
) as compute:
    result = train(data) >> compute
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `str` | `"ghcr.io/gabfssilva/skyward:py{python_version}"` | Docker image (Python version auto-detected) |
| `ssh_user` | `str` | `"root"` | SSH user inside the container |
| `binary` | `str` | `"docker"` | Container runtime (`"docker"`, `"podman"`, `"nerdctl"`) |
| `container_prefix` | `str or None` | `None` | Prefix for container names |
| `network` | `str or None` | `None` | Docker network name. Auto-created if not set. |

## Choosing a provider

**AWS** — When you need specific hardware (H100, Trainium, Inferentia), spot instance savings, or enterprise reliability. Best if you're already in the AWS ecosystem.

**GCP** — Deep integration with Google Cloud. Deep Learning VM images with pre-installed CUDA drivers, dynamic machine type resolution, fleet-style provisioning via `bulk_insert`. Supports T4, L4, V100, A100, H100, H200.

**RunPod** — Fast provisioning, competitive pricing, minimal setup. Both Secure Cloud (dedicated) and Community Cloud (cheaper) tiers. Good for A100/H100/RTX workloads.

**Hyperstack** — Bare-metal GPU cloud with environment-scoped resource management. On-demand only, regions in Canada, Norway, and US.

**Verda** — European data residency (Finland, Iceland, Israel). H100/A100/H200/GB200 availability with automatic region selection.

**TensorDock** — Bare-metal VMs across 100+ locations with per-second billing. Good for RTX 4090, A100, H100 workloads without spot complexity. On-demand only.

**VastAI** — Maximum cost savings through marketplace pricing. Consumer GPUs (RTX 4090, 3090) available alongside datacenter hardware. Overlay networks for multi-node training.

**Container** — Local development and CI. Zero cost, instant provisioning. Validates your code end-to-end before deploying to a real provider.

## Common issues

### GCP: "No GCP accelerator matches"

1. Check available accelerators in your zone: `gcloud compute accelerator-types list --filter="zone:us-central1-a"`
2. Try a different zone — GPU availability varies by zone
3. Request GPU quota increases in the [Cloud Console](https://console.cloud.google.com/iam-admin/quotas)

### GCP: "Quota exceeded"

1. Check current quotas: `gcloud compute regions describe <region> | grep -A2 GPU`
2. Request increases for the specific GPU type (e.g., `NVIDIA_T4_GPUS`, `NVIDIA_L4_GPUS`)
3. Both on-demand and preemptible quotas are separate — check both

### AWS: "No instances available"

1. Try a different region
2. Use `allocation="spot-if-available"` (the default) to fall back to on-demand
3. Request a service quota increase in the AWS console

### Verda: "Region not available"

1. The default region is `"FIN-01"` — try a different one or let auto-discovery find capacity
2. Check your account's region access

### TensorDock: "No hostnodes available"

1. Try a different location or remove the `location` filter
2. Try a different GPU type — hostnode availability is dynamic
3. Check marketplace availability at [https://marketplace.tensordock.com](https://marketplace.tensordock.com)

### VastAI: "No offers available"

1. Lower `min_reliability` (e.g., 0.8)
2. Expand or remove the `geolocation` filter
3. Try a different accelerator type
4. Check marketplace availability at [https://cloud.vast.ai/](https://cloud.vast.ai/)

---

## Related topics

- [Getting Started](getting-started.md) — Installation and credentials setup
- [Accelerators](accelerators.md) — Accelerator selection guide
- [API Reference](reference/pool.md) — Complete API documentation
