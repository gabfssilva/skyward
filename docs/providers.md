# Cloud providers

Skyward supports fifteen providers. Fourteen are cloud services — AWS, GCP, Hyperstack, JarvisLabs, Lambda, Massed Compute, Novita, RunPod, Salad, Scaleway, TensorDock, Vast.ai, Verda, Vultr — and one runs local containers for development and CI. All implement the same provider protocol, so the orchestration layer (offer selection, SSH, bootstrap, task dispatch) works identically regardless of which one you choose. The difference is in how machines are provisioned, what hardware is available, and how authentication works.

A provider has two identities. The **kind** — `aws`, `runpod` — selects the adapter. The **account** is a named set of credentials and non-secret configuration, and it is what you write:

```python
import skyward as sky

with sky.Compute(
    provider=sky.AWS(region="us-east-1"),
    accelerator=sky.accelerators.A100(),
    nodes=2,
) as compute:
    result = train(data) >> compute
```

Accounts are frozen structs. They hold configuration and import no cloud SDK at module level — the SDK loads when the daemon provisions, so `import skyward` stays fast regardless of which providers are installed. Three adapters need an SDK of their own (`skyward[aws]`, `skyward[gcp]`, `skyward[salad]`, or `skyward[providers]` for all three); the other twelve speak plain HTTP.

`name` is the account alias, and it defaults to the kind. Give two accounts of the same kind two names when you need both:

```python
production = sky.AWS(name="production", region="us-east-1")
research = sky.AWS(name="research", region="eu-west-1")
```

Credentials are resolved in your process — from the argument you passed, then from the environment or credential file listed in each section below — and stored by the daemon on the provider row. The daemon never reads your environment, and the API serves an account's configuration back but never its credentials.

### Disk size

Set disk size uniformly across providers with `disk_gb` on `Spec` or directly on `Compute`:

```python
sky.Compute(provider=sky.AWS(), disk_gb=500)
```

When set, `disk_gb` overrides the provider's own default. When omitted, each provider uses its built-in default (100 GB for AWS, 200 GB for GCP, 50 GB for RunPod). Providers where disk is determined by the instance plan ignore it.

## Provider comparison

| Feature | AWS | GCP | Hyperstack | JarvisLabs | Lambda | Massed Compute | Novita | RunPod | Salad | Scaleway | TensorDock | Vast.ai | Verda | Vultr | Container |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Spot** | Yes | Yes | No | No | No | Yes | Yes | Yes | No | No | No | Yes (bid) | Yes | No | N/A |
| **Private network** | VPC | VPC | Yes | Yes | Yes | Yes | Yes | Opt-in | No | Yes | Yes | Overlay | Yes | Yes | Bridge |
| **Auth** | Access key + secret | Service-account JSON | API key | API key | API key | API key | API key | API key | API key | Secret key + project | Key + token | API key | Client ID + secret | API key | None |
| **Catalog TTL** | 30 min | 6 h | 1 h | 15 min | 5 min | 10 min | 15 min | 10 min | 10 min | 10 min | 5 min | 2 min | 15 min | 6 h | 1 day |
| **SDK extra** | `aws` | `gcp` | — | — | — | — | — | — | `salad` | — | — | — | — | — | — |

The catalog TTL is how long the daemon keeps that provider's offers before refetching. A marketplace whose capacity turns over in minutes declares a shorter one than a fixed instance catalog.

For what each provider is actually good at — and which ones I use — see [Choosing the best provider](choosing-a-provider.md).

## AWS

AWS uses EC2 Fleet for provisioning, with automatic spot-to-on-demand fallback. Instances launch in a VPC with security groups managed by Skyward, or your own. SSH keys are created per compute and cleaned up on teardown.

AMI resolution happens automatically via SSM Parameter Store — Skyward looks up the latest Ubuntu AMI for your chosen version and architecture. Override it with `ami`.

Only static keys resolve from `~/.aws/credentials`. SSO, assume-role, and process credentials need botocore's full chain, which the SDK deliberately does not depend on — pass those keys explicitly.

### Setup

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

Or use the AWS CLI:

```bash
aws configure
```

### Install

```bash
uv add "skyward[aws]"
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

`region` also takes a sequence, which widens the offer search:

```python
sky.AWS(region=["us-east-1", "us-west-2"])
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `access_key_id` | `str or None` | `None` | Access key. Falls back to `AWS_ACCESS_KEY_ID`, then `~/.aws/credentials`. |
| `secret_access_key` | `str or None` | `None` | Secret key. Falls back to `AWS_SECRET_ACCESS_KEY`. |
| `session_token` | `str or None` | `None` | Session token for temporary credentials. Falls back to `AWS_SESSION_TOKEN`. |
| `region` | `str or Sequence[str]` | `"us-east-1"` | Region, or several to search across. |
| `ami` | `str or None` | `None` | Custom AMI ID. Auto-resolved via SSM if not set. |
| `ubuntu_version` | `str` | `"24.04"` | Ubuntu LTS version for auto-resolved AMIs. |
| `subnet_id` | `str or None` | `None` | VPC subnet. Uses the default VPC if not set. |
| `security_group_id` | `str or None` | `None` | Security group. Auto-created if not set. |
| `instance_profile_arn` | `str or None` | `None` | IAM instance profile. Auto-created if not set. |
| `username` | `str or None` | `None` | SSH user. Auto-detected from the AMI if not set. |
| `disk_gb` | `int` | `100` | Root volume size in GB. |
| `instance_timeout` | `int` | `300` | Safety timeout in seconds (auto-shutdown timer). |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds. |
| `allocation_strategy` | `str` | `"price-capacity-optimized"` | EC2 Fleet spot allocation strategy. Also `"capacity-optimized"`, `"lowest-price"`. |
| `exclude_burstable` | `bool` | `False` | Exclude burstable instances (t3, t4g). |

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

GCP uses Compute Engine with instance templates and `bulk_insert` for fleet-style provisioning. Skyward resolves the machine type dynamically — for GPUs like T4 and V100 it uses N1 machines with guest accelerators; for A100 and H100 it picks the matching A2/A3 machine family with built-in GPUs. Spot instances use the `SPOT` provisioning model with automatic deletion on preemption.

SSH keys are injected via instance metadata. GCP API calls are sync clients dispatched to a dedicated thread pool, sized by `thread_pool_size`. Skyward creates an instance template and a firewall rule per compute, both cleaned up on teardown. Instances use Google's Deep Learning VM images for GPU workloads.

`GOOGLE_APPLICATION_CREDENTIALS` names a file; its *contents* are what travel to the daemon.

### Setup

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=your_project_id
```

Listing available accelerator types does not mean you have quota. Check yours before provisioning:

```bash
gcloud compute regions describe <region> --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep GPU
```

Request increases in the [Cloud Console](https://console.cloud.google.com/iam-admin/quotas). On-demand and preemptible quotas are separate.

### Install

```bash
uv add "skyward[gcp]"
```

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
| `service_account_json` | `str or None` | `None` | Service-account JSON contents. Read from the file `GOOGLE_APPLICATION_CREDENTIALS` names. |
| `project` | `str or None` | `None` | Project ID. Falls back to `GOOGLE_CLOUD_PROJECT`. |
| `zone` | `str or Sequence[str]` | `"us-central1-a"` | Zone, or several to search across. |
| `network` | `str` | `"default"` | VPC network name. |
| `subnet` | `str or None` | `None` | Specific subnet. Uses the auto-mode subnet if not set. |
| `disk_size_gb` | `int` | `200` | Boot disk size in GB. |
| `disk_type` | `str` | `"pd-balanced"` | Boot disk type. Also `"pd-ssd"`, `"pd-standard"`, `"pd-extreme"`, `"hyperdisk-balanced"`. |
| `instance_timeout` | `int` | `300` | Safety timeout in seconds (self-destruction timer). |
| `service_account` | `str or None` | `None` | GCE service account email. |
| `thread_pool_size` | `int` | `8` | Thread pool size for blocking GCP API calls. |

### Required permissions

The authenticated principal needs:

- `compute.instances.create`, `compute.instances.delete`, `compute.instances.list`, `compute.instances.get`
- `compute.instanceTemplates.create`, `compute.instanceTemplates.delete`
- `compute.firewalls.create`, `compute.firewalls.delete`, `compute.firewalls.get`
- `compute.machineTypes.list`, `compute.acceleratorTypes.list`
- `compute.images.getFromFamily`

The simplest approach is the **Compute Admin** role (`roles/compute.admin`).

## RunPod

RunPod offers GPU pods in two tiers: **Secure Cloud** (dedicated hardware) and **Community Cloud** (lower-cost, peer-hosted). Skyward provisions pods through RunPod's REST API, configures SSH access, and manages the lifecycle.

`data_center_ids` takes `"global"` as a mode rather than as a member — it means the whole fleet, not a data center named global.

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

### Standalone mode

RunPod's individual pods don't share a private network — each pod gets its own IP, but pods can't reach each other directly. RunPod therefore defaults to standalone mode, so multi-node workloads that don't need inter-node communication (hyperparameter sweeps, batch inference) work without extra configuration:

```python
with sky.Compute(
    provider=sky.RunPod(),
    accelerator=sky.accelerators.A100(),
    nodes=4,
) as compute:
    results = sky.gather(*tasks) >> compute
```

The daemon reaches each worker independently over SSH. Distributed collections and distributed training are not available in this mode. For cluster mode you need RunPod's global networking, which requires Secure Cloud — the daemon refuses cluster formation on a Community Cloud offer:

```python
sky.Compute(
    provider=sky.RunPod(cloud_type="secure", global_networking=True),
    accelerator=sky.accelerators.A100(),
    nodes=4,
    options=sky.Options(cluster=True),
)
```

See the [Standalone workers](guides/standalone-workers.md) guide for details.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str or None` | `None` | API key. Falls back to `RUNPOD_API_KEY`. |
| `cluster_mode` | `"individual" or "cluster"` | `"individual"` | Whether pods are provisioned as an interconnected cluster. |
| `cloud_type` | `"secure" or "community"` | `"secure"` | Dedicated hardware, or cheaper peer-hosted. |
| `base_image` | `"nvidia", "runpod_base", "runpod_pytorch"` | `"nvidia"` | Which prebuilt base image family to launch from. |
| `container_image` | `str or None` | `None` | An explicit image tag, overriding `base_image`. |
| `ubuntu` | `str` | `"newest"` | Ubuntu version (`"20.04"`, `"22.04"`, `"24.04"`, `"newest"`). |
| `container_disk_gb` | `int` | `50` | Container disk size in GB. |
| `volume_gb` | `int` | `20` | Persistent volume size in GB. |
| `volume_mount_path` | `str` | `"/workspace"` | Where the persistent volume mounts. |
| `data_center_ids` | `str or Sequence[str]` | `"global"` | Preferred data centers, or `"global"` for the whole fleet. |
| `country_codes` | `str or Sequence[str] or None` | `None` | Restrict to these countries. |
| `exclude_country_codes` | `str or Sequence[str]` | `()` | Exclude these countries. |
| `ports` | `Sequence[str]` | `("22/tcp",)` | Port mappings requested from RunPod. |
| `bid_multiplier` | `float` | `1.0` | Multiplier over the minimum spot bid. |
| `registry_auth` | `str or None` | `"docker hub"` | Container registry credential name. `None` to skip. |
| `min_inet_down` | `float or None` | `None` | Minimum download bandwidth in Mbps. |
| `min_inet_up` | `float or None` | `None` | Minimum upload bandwidth in Mbps. |
| `global_networking` | `bool or None` | `None` | Enable RunPod global networking. Required for cluster mode. |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds. |
| `cpu_clock` | `str` | `"3c"` | CPU clock tier (`"3c"` or `"5c"`). |

## Vast.ai

Vast.ai is a GPU marketplace — instances are Docker containers running on hosts from independent providers worldwide. Pricing is dynamic and reliability varies by host. Skyward filters offers by reliability score, CUDA version, and optional geolocation, then provisions containers through the Vast.ai API.

For multi-node clusters, Vast.ai uses overlay networks for inter-node communication.

### Setup

```bash
export VAST_API_KEY=your_api_key
```

Get your API key at [cloud.vast.ai/account](https://cloud.vast.ai/account/).

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
| `api_key` | `str or None` | `None` | API key. Falls back to `VAST_API_KEY`. |
| `min_reliability` | `float` | `0.95` | Minimum host reliability score (0.0-1.0). |
| `verified_only` | `bool` | `True` | Only select offers from verified hosts. |
| `min_cuda` | `float` | `12.0` | Minimum CUDA version. |
| `geolocation` | `str or None` | `None` | Filter by region or country (e.g. `"US"`, `"EU"`). |
| `bid_multiplier` | `float` | `1.2` | Multiplier over the minimum spot bid. |
| `instance_timeout` | `int` | `300` | Auto-shutdown safety timeout in seconds. |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds. |
| `docker_image` | `str or None` | `None` | Base Docker image for containers. |
| `disk_gb` | `float` | `100.0` | Disk space in GB. |
| `overlay_timeout` | `int` | `120` | Timeout for overlay network operations in seconds. |
| `require_direct_port` | `bool` | `False` | Only select offers with direct port access. |
| `min_inet_down` | `float or None` | `None` | Minimum download bandwidth in Mbps. |
| `min_inet_up` | `float or None` | `None` | Minimum upload bandwidth in Mbps. |
| `limit` | `int` | `500` | How many offers to fetch per catalog refresh. |

## Hyperstack

Hyperstack provides GPU instances through NexGen Cloud's InfraHub API. Resources are organized into environments that group VMs, keypairs, and volumes within a region. Environments are created per compute and cascade-deleted on teardown. All instances are on-demand — no spot pricing.

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
| `api_key` | `str or None` | `None` | API key. Falls back to `HYPERSTACK_API_KEY`. |
| `region` | `str or Sequence[str] or None` | `None` | Region(s) to deploy in. `None` searches all regions. |
| `image` | `str or None` | `None` | OS image name override. Auto-selects the newest Ubuntu + CUDA image if not set. |
| `network_optimised` | `bool` | `False` | Require network-optimised environments with SR-IOV support. |
| `network_optimised_regions` | `Sequence[str]` | `("CANADA-1", "US-1")` | Regions known to support network-optimised environments. |
| `object_storage_region` | `str` | `"CANADA-1"` | Region for S3-compatible object storage (volume mounts). |
| `object_storage_endpoint` | `str` | `"https://ca1.obj.nexgencloud.io"` | Endpoint URL for that storage. |
| `instance_timeout` | `int` | `300` | Auto-shutdown safety timeout in seconds. |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds. |
| `teardown_timeout` | `float` | `120.0` | Timeout for teardown operations in seconds. |
| `teardown_poll_interval` | `float` | `2.0` | Poll interval during teardown in seconds. |

## Verda

Verda is a GPU cloud with data centers in Europe and the Middle East. It uses OAuth2 authentication — a client ID and secret, not a single API key.

SSH keys are auto-registered if needed. If the requested GPU isn't available in the configured region, Skyward looks for another region with availability.

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
| `client_id` | `str or None` | `None` | OAuth2 client ID. Falls back to `VERDA_CLIENT_ID`. |
| `client_secret` | `str or None` | `None` | OAuth2 client secret. Falls back to `VERDA_CLIENT_SECRET`. |
| `region` | `str` | `"FIN-01"` | Preferred region. |
| `ssh_key_id` | `str or None` | `None` | A specific registered SSH key to use. |
| `image` | `str or None` | `None` | OS image override. |
| `cuda` | `str` | `"13.0"` | CUDA version for the selected image. |
| `instance_timeout` | `int` | `300` | Safety timeout in seconds. |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds. |

## Lambda

Lambda Cloud offers on-demand GPU instances with a straightforward API. Instances run Ubuntu with NVIDIA drivers pre-installed. SSH keys are auto-registered and cleaned up by Skyward.

If no region is given, Lambda auto-selects the first region with available capacity.

### Setup

```bash
export LAMBDA_API_KEY=your_api_key
```

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.Lambda(),
    accelerator=sky.accelerators.H100(),
    nodes=2,
) as compute:
    result = train(data) >> compute
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str or None` | `None` | API key. Falls back to `LAMBDA_API_KEY`. |
| `region` | `str or None` | `None` | Preferred region (e.g. `"us-east-3"`). Auto-selects capacity if not set. |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds. |

## Salad

Salad runs GPU workloads as containers on a distributed network of consumer machines, reached through its own container gateway. There is no private network between containers, so a Salad compute is always standalone: task dispatch works, distributed collections and distributed training do not.

`organization` and `project` are the account's namespace. They are not optional to Salad — only to the constructor, where the environment may answer for them.

### Setup

```bash
export SALAD_API_KEY=your_api_key
export SALAD_ORGANIZATION=your_org
export SALAD_PROJECT=your_project
```

### Install

```bash
uv add "skyward[salad]"
```

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.Salad(priority="batch"),
    accelerator=sky.accelerators.RTX_4090(),
    nodes=4,
) as compute:
    results = sky.gather(*tasks) >> compute
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str or None` | `None` | API key. Falls back to `SALAD_API_KEY`. |
| `organization` | `str or None` | `None` | Organization name. Falls back to `SALAD_ORGANIZATION`. |
| `project` | `str or None` | `None` | Project name. Falls back to `SALAD_PROJECT`. |
| `priority` | `"high", "medium", "low", "batch"` | `"low"` | Allocation priority. Higher is faster to place and more expensive. |
| `country_codes` | `str or Sequence[str] or None` | `None` | Restrict placement to these countries. |
| `image` | `str or None` | `None` | Container image override. |
| `storage_gb` | `int` | `50` | Container storage in GB. |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds. |
| `allocation_timeout` | `float` | `300.0` | Seconds to wait for a container to be placed. |
| `poll_interval` | `float` | `2.0` | Seconds between allocation status checks. |

## Novita

Novita is a GPU cloud where instances are Docker containers with configurable GPU count and root filesystem size. SSH access goes through Novita's proxy — no openssh-server or key injection inside the container. Skyward reads the connection details from the instance metadata and connects through the proxy automatically.

Novita resolves CUDA compatibility dynamically: Skyward queries the instance's maximum supported CUDA version and tries descending versions until it finds a host with availability. A custom `docker_image` is used as-is.

### Setup

```bash
export NOVITA_API_KEY=your_api_key
```

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.Novita(),
    accelerator=sky.accelerators.A100(),
    nodes=2,
) as compute:
    result = train(data) >> compute
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str or None` | `None` | API key. Falls back to `NOVITA_API_KEY`. |
| `cluster_id` | `str or None` | `None` | Target cluster or region ID. `None` auto-selects. |
| `rootfs_size` | `int` | `50` | Root filesystem size in GB. |
| `docker_image` | `str or None` | `None` | Base Docker image. Defaults to an NVIDIA CUDA runtime image. |
| `min_cuda_version` | `str or None` | `None` | Minimum CUDA version (e.g. `"12.4"`). |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds. |

## TensorDock

TensorDock is a GPU marketplace with bare-metal VMs across 100+ locations. Per-second billing, on-demand only. Skyward queries available hostnodes, selects the cheapest matching your requirements, and deploys VMs with cloud-init for SSH key injection.

TensorDock has no SSH key registration API, so keys are injected per instance through cloud-init. The SSH user is `user`, not root. Port forwarding maps internal ports to random external ports — SSH is never on port 22 externally. Skyward reads the mapping from the deploy response and configures its tunnels accordingly; no manual port configuration is needed.

### Setup

```bash
export TENSORDOCK_API_KEY=your_api_key
export TENSORDOCK_API_TOKEN=your_api_token
```

Get your credentials at [console.tensordock.com/api](https://console.tensordock.com/api).

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.TensorDock(location="United States"),
    accelerator=sky.accelerators.RTX_4090(),
    nodes=2,
) as compute:
    result = train(data) >> compute
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str or None` | `None` | API key. Falls back to `TENSORDOCK_API_KEY`. |
| `api_token` | `str or None` | `None` | API token. Falls back to `TENSORDOCK_API_TOKEN`. |
| `location` | `str or None` | `None` | Country filter (e.g. `"United States"`, `"Germany"`). Global if not set. |
| `tier` | `int or None` | `None` | Hostnode tier (0-4). `None` for any tier. |
| `storage_gb` | `int` | `100` | Disk storage per VM in GB. |
| `operating_system` | `str` | `"ubuntu2404"` | OS image ID (e.g. `"ubuntu2404"`, `"ubuntu2204"`). |
| `instance_timeout` | `int` | `300` | Auto-shutdown in seconds. |
| `request_timeout` | `int` | `120` | HTTP request timeout in seconds. |
| `min_ram_gb` | `int or None` | `None` | Minimum RAM per VM in GB. |
| `min_vcpus` | `int or None` | `None` | Minimum vCPUs per VM. |

## JarvisLabs

JarvisLabs offers instances in India (IN1, IN2) and Finland (EU1). Per-minute billing with a prepaid wallet model. SSH keys are auto-registered. Provisioning is unusually fast because instances start from prebuilt framework images rather than a bare OS.

EU1 supports H100 and H200 only, with either 1 or 8 GPUs, and requires at least 100 GB of storage.

### Setup

```bash
export JL_API_KEY=your_api_token
```

Get your token from [jarvislabs.ai/settings/api-keys](https://jarvislabs.ai/settings/api-keys).

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.JarvisLabs(region="IN2"),
    accelerator=sky.accelerators.L4(),
    nodes=2,
) as compute:
    result = train(data) >> compute
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str or None` | `None` | API token. Falls back to `JL_API_KEY`. |
| `region` | `str or None` | `None` | Region: `IN1`, `IN2`, `EU1`. Auto-selects if not set. |
| `template` | `str` | `"pytorch"` | Framework template: `pytorch`, `tensorflow`, `jax`, `vm`. |
| `storage_gb` | `int` | `50` | Disk storage in GB. Minimum 100 for EU1 and `vm`. |
| `instance_timeout` | `int` | `300` | Auto-shutdown safety timer in seconds. |
| `thread_pool_size` | `int` | `8` | Max threads for blocking SDK calls. |

## Massed Compute

Massed Compute is a bare-metal GPU cloud across US data centers. Instances run Ubuntu with NVIDIA drivers pre-installed, SSH access via key or password, and all ports open by default — no firewall configuration. SSH keys are auto-registered and cleaned up by Skyward.

Spot instances are available on select GPU types. Region is auto-placed: Massed Compute assigns the best available data center.

### Setup

```bash
export MASSED_API_KEY=your_api_key
```

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.MassedCompute(),
    accelerator=sky.accelerators.RTX_A6000(),
    nodes=2,
) as compute:
    result = train(data) >> compute
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str or None` | `None` | API key. Falls back to `MASSED_API_KEY`. |
| `image_id` | `int` | `184` | OS image ID. `184` is Ubuntu 24.04, `84` is Ubuntu 22.04 with drivers. |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds. |

## Scaleway

Scaleway provides GPU instances in European data centers — Paris, Amsterdam, and Warsaw. On-demand only, per-hour billing. Useful when EU data residency is a requirement.

Leaving `zone` unset searches all nine GPU zones.

### Setup

```bash
export SCW_SECRET_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
export SCW_DEFAULT_PROJECT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.Scaleway(zone="fr-par-2"),
    accelerator=sky.accelerators.H100(),
    nodes=2,
) as compute:
    result = train(data) >> compute
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `secret_key` | `str or None` | `None` | API secret key. Falls back to `SCW_SECRET_KEY`. |
| `project_id` | `str or None` | `None` | Project ID. Falls back to `SCW_DEFAULT_PROJECT_ID`. |
| `zone` | `str or Sequence[str] or None` | `None` | Zone(s). `None` searches all GPU zones. |
| `image` | `str or None` | `None` | OS image UUID override. Auto-selects an Ubuntu GPU image if not set. |
| `instance_timeout` | `int` | `300` | Auto-shutdown safety timeout in seconds. |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds. |

Available zones: `fr-par-1`, `fr-par-2`, `fr-par-3`, `nl-ams-1`, `nl-ams-2`, `nl-ams-3`, `pl-waw-1`, `pl-waw-2`, `pl-waw-3`.

## Vultr

Vultr offers GPU instances in two modes. **Cloud GPU** gives virtual instances with vGPU or passthrough, faster provisioning, and fractional GPU support. **Bare metal** gives dedicated physical servers with no virtualization overhead. Cloud GPU is the default.

### Setup

```bash
export VULTR_API_KEY=your_api_key
```

Generate an API key from the [Vultr customer portal](https://my.vultr.com/settings/#settingsapi).

### Usage

```python
import skyward as sky

with sky.Compute(
    provider=sky.Vultr(region="ewr"),
    accelerator=sky.accelerators.A100(),
    nodes=2,
) as compute:
    result = train(data) >> compute
```

Bare metal is the same account with one field changed:

```python
sky.Vultr(mode="bare_metal", region="ewr")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str or None` | `None` | API key. Falls back to `VULTR_API_KEY`. |
| `mode` | `"cloud" or "bare_metal"` | `"cloud"` | Virtual instances, or dedicated servers. |
| `region` | `str or None` | `None` | Vultr region ID (e.g. `"ewr"`, `"ord"`, `"dfw"`). `None` searches all. |
| `os_id` | `int` | `2284` | OS image ID. The default is Ubuntu 24.04. |
| `instance_timeout` | `int` | `300` | Safety timeout in seconds. |
| `request_timeout` | `int` | `30` | HTTP request timeout in seconds. |

## Container

The Container provider runs nodes as local containers — Docker, podman, or nerdctl. No cloud credentials, no cost. Use it for development, CI, and validating your code before it touches real hardware.

Containers are launched with SSH access, joined to a shared network, and bootstrapped exactly the way cloud instances are. From the compute's perspective they are ordinary nodes.

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
| `image` | `str` | `"ghcr.io/gabfssilva/skyward:py{python_version}"` | Container image. The Python version is substituted automatically. |
| `ssh_user` | `str` | `"root"` | SSH user inside the container. |
| `binary` | `str` | `"docker"` | Container runtime (`"docker"`, `"podman"`, `"nerdctl"`). |
| `container_prefix` | `str or None` | `None` | Prefix for container names. |
| `network` | `str or None` | `None` | Network name. Auto-created if not set. |

## Inspecting the catalog

The daemon keeps one hardware catalog per registered account and refreshes it when its rows go stale. Query it before committing to a provider:

```bash
sky providers list
sky providers check
sky offers list --accelerator A100 --min-vram 40 --max-price 3
sky offers summary --accelerator H100
sky offers fetch --refresh
```

A failed refresh does not erase the previous rows: the stale offers stay available and the account records the error, which `sky providers check` reports. Offer selection uses this same cache, so a `Compute` with several `Spec` alternatives does not call every provider separately for each decision.

## Common issues

### GCP: "No GCP accelerator matches"

1. Check available accelerators in your zone: `gcloud compute accelerator-types list --filter="zone:us-central1-a"`
2. Try a different zone — GPU availability varies by zone.
3. Request GPU quota increases in the [Cloud Console](https://console.cloud.google.com/iam-admin/quotas).

### GCP: "Quota exceeded"

1. Check current quotas: `gcloud compute regions describe <region> | grep -A2 GPU`
2. Request increases for the specific GPU type (e.g. `NVIDIA_T4_GPUS`, `NVIDIA_L4_GPUS`).
3. On-demand and preemptible quotas are separate — check both.

### AWS: "No instances available"

1. Try a different region, or pass several to `region`.
2. Use `allocation="spot_if_available"` (the default) to fall back to on-demand.
3. Request a service quota increase in the AWS console.

### Verda: "Region not available"

1. The default region is `"FIN-01"` — try another, or let auto-discovery find capacity.
2. Check your account's region access.

### TensorDock: "No hostnodes available"

1. Try a different location, or remove the `location` filter.
2. Try a different GPU type — hostnode availability is dynamic.
3. Check availability at [marketplace.tensordock.com](https://marketplace.tensordock.com).

### Vast.ai: "No offers available"

1. Lower `min_reliability` (e.g. 0.8), or set `verified_only=False`.
2. Expand or remove the `geolocation` filter.
3. Raise `limit` — the catalog fetch is capped at 500 offers by default.
4. Check availability at [cloud.vast.ai](https://cloud.vast.ai/).

### RunPod: "does not allow cluster formation"

Cluster mode needs `global_networking=True` and Secure Cloud. Community Cloud offers cannot form a cluster — either drop `cluster=True` and run standalone, or set `cloud_type="secure"`.

---

## Next steps

- **[Choosing the best provider](choosing-a-provider.md)** — an opinionated guide to which one to reach for
- **[Getting started](getting-started.md)** — installation and credential setup
- **[Accelerators](accelerators.md)** — accelerator selection and hardware specs
- **[Multi-provider selection](guides/multi-provider.md)** — several providers with automatic fallback
- **[Compute and task dispatch](reference/pool.md)** — the `Spec` and `Compute` API
- **[Provider API reference](reference/providers/aws.md)** — the generated signature of every account struct
