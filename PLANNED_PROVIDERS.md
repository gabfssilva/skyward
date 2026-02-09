# Planned GPU Cloud Providers

This document lists potential cloud providers for future Skyward integration, organized by SDK/API complexity.

**Already implemented:** AWS, Vast.ai, DigitalOcean, Verda

---

## TIER 1: Native Async SDK (Best fit for Skyward)

Providers with native `asyncio` support in their Python SDK.

| Provider | H100 $/hr | A100 $/hr | SDK Package | Auth Method | Docs |
|----------|-----------|-----------|-------------|-------------|------|
| **Nebius** | $2.00 | $1.30 | `nebius` | Bearer Token / Service Account JWT | [docs.nebius.com](https://docs.nebius.com/compute/) |
| **Azure** | $3.40 | $3.67 | `azure-mgmt-compute` | OAuth2 (DefaultAzureCredential) | [learn.microsoft.com](https://learn.microsoft.com/en-us/python/api/overview/azure/mgmt-compute-readme) |
| **FluidStack** | $2.10 | $1.30 | `fluidstack` | Bearer Token (API Key) | [docs.fluidstack.io](https://docs.fluidstack.io/sdks/python/overview) |
| **Salad** | N/A | N/A | `salad-cloud-sdk` | API Key (Salad-Api-Key header) | [docs.salad.com](https://docs.salad.com/reference/api-usage) |

### Details

#### Nebius (HIGH PRIORITY)
- **Async Client:** `nebius` SDK is asyncio-first (gRPC-based)
- **GPUs:** H200, H100, L40S, A100, L4, B200, GB200
- **Notes:** Yandex spin-off. EU data centers (Finland). Competitive pricing.

#### Azure (MEDIUM PRIORITY)
- **Async Client:** `azure.mgmt.compute.aio.ComputeManagementClient`
- **GPUs:** H100, A100, A10, T4, V100, MI300X (AMD)
- **Notes:** Full async support. Enterprise-focused. Spot VMs available.

#### FluidStack (MEDIUM PRIORITY)
- **Async Client:** `AsyncFluidStack`
- **GPUs:** GB200, B200, H200, H100, A100, L40S
- **Notes:** InfiniBand clusters. 3.2 Tbps interconnect.

#### Salad (MEDIUM PRIORITY)
- **Async Client:** `SaladCloudSdkAsync`
- **GPUs:** RTX 5090, 5080, 4090, 4080, 3090, 3080, 3070, 3060
- **Notes:** Container-based (no SSH/VMs). 60k+ consumer GPUs. Very cheap.

---

## TIER 2: Simple HTTP API (Easy to implement with httpx)

Providers with straightforward REST APIs. No SDK needed - use `httpx` directly.

| Provider | H100 $/hr | A100 $/hr | Auth Method | Docs |
|----------|-----------|-----------|-------------|------|
| **RunPod** | $1.99 | $1.19 | Bearer Token | [docs.runpod.io](https://docs.runpod.io/api-reference/overview) |
| **Lambda Labs** | $2.99 | $1.29 | HTTP Basic Auth (API key as username) | [docs.lambda.ai](https://docs.lambda.ai/public-cloud/cloud-api/) |
| **Thunder Compute** | $1.89 | $0.66 | API Key (Authorization header) | [thundercompute.com/docs](https://www.thundercompute.com/docs) |
| **Hyperstack** | $1.90 | $1.35 | API Key (api_key header) | [docs.hyperstack.cloud](https://docs.hyperstack.cloud/docs/api-reference/) |
| **TensorDock** | $2.25 | $0.75 | Bearer Token | [tensordock API](https://dashboard.tensordock.com/api/docs/getting-started) |
| **Shadeform** | varies | varies | API Key (X-API-KEY header) | [docs.shadeform.ai](https://docs.shadeform.ai/getting-started/introduction) |
| **Jarvis Labs** | $2.99 | $1.29 | Bearer Token | [docs.jarvislabs.ai](https://docs.jarvislabs.ai/) |
| **Northflank** | $2.74 | $1.76 | Bearer Token (JWT) | [northflank.com/docs](https://northflank.com/docs/v1/api/introduction) |

### Details

#### RunPod (HIGH PRIORITY)
- **Base URL:** `https://rest.runpod.io/v1`
- **GPUs:** B200, H200, H100, A100, L40S, L40, L4, RTX 6000 Ada, RTX 5090, RTX 4090, RTX 3090
- **Notes:** Per-second billing. Huge GPU variety. Very popular in ML community.

#### Lambda Labs (HIGH PRIORITY)
- **Base URL:** `https://cloud.lambdalabs.com/api/v1`
- **GPUs:** B200, H100, GH200, A100, A10, A6000, V100
- **Notes:** Simple REST. Reference in deep learning. Lambda Stack pre-installed.

#### Thunder Compute (MEDIUM PRIORITY)
- **Base URL:** `https://api.thundercompute.com:8443/v1`
- **GPUs:** H100, A100, T4
- **Notes:** Claims cheapest A100 ($0.66/hr). OpenAPI spec available.

#### Hyperstack (MEDIUM PRIORITY)
- **Base URL:** `https://infrahub-api.nexgencloud.com/v1`
- **GPUs:** H200, H100, A100, L40
- **Notes:** 100% renewable energy. Has sync SDK (alpha) but prefer httpx.

#### TensorDock (MEDIUM PRIORITY)
- **Base URL:** `https://dashboard.tensordock.com/api/v2`
- **GPUs:** H100, A100, L40, RTX 6000 Ada, RTX A6000, RTX 4090, RTX 3090, V100
- **Notes:** Marketplace model. 100 req/min rate limit. Spot pricing available.

#### Shadeform (HIGH PRIORITY - Aggregator)
- **Base URL:** `https://api.shadeform.ai/v1`
- **GPUs:** All major GPUs via 32+ providers
- **Notes:** **AGGREGATOR** - One implementation covers AWS, Azure, GCP, Lambda, Nebius, CoreWeave, RunPod, Hyperstack, etc. No markup. Standardized `shade_instance_type`.

#### Jarvis Labs (LOW PRIORITY)
- **Base URL:** Not fully documented (use SDK)
- **GPUs:** H200, H100, A100, RTX 6000 Ada, A6000, A5000
- **Notes:** Good for beginners. Minute-level billing.

#### Northflank (LOW PRIORITY)
- **Base URL:** `https://api.northflank.com/v1/`
- **GPUs:** B200, GB300, H200, H100, GH200, A100, L40S, L40, L4, MI300X
- **Notes:** PaaS with GPU support. No Python SDK (JS only). 1000 req/hr rate limit.

---

## TIER 3: Sync SDK (Wrap in asyncio.to_thread)

Providers with synchronous-only SDKs. Wrap calls in `asyncio.to_thread()`.

| Provider | H100 $/hr | A100 $/hr | SDK Package | Auth Method | Docs |
|----------|-----------|-----------|-------------|-------------|------|
| **GCP** | $3.20 | $3.67 | `google-cloud-compute` | Service Account (OAuth2) / ADC | [cloud.google.com](https://cloud.google.com/python/docs/reference/compute/latest) |
| **OCI (Oracle)** | $3.50 | $2.50 | `oci` | API Key Signing (like AWS SigV4) | [docs.oracle.com](https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/pythonsdk.htm) |

### Details

#### GCP (HIGH PRIORITY)
- **Client:** `google.cloud.compute_v1.InstancesClient`
- **GPUs:** GB300, GB200, B200, H200, H100, A100, L4, T4, V100, P100
- **Notes:** No async client for Compute Engine. Flexible CPU/GPU/storage combos.

#### OCI - Oracle (LOW PRIORITY)
- **Client:** `oci.core.ComputeClient`
- **GPUs:** B200, B300, GB200, GB300, H200, H100, A100, L40S, A10, V100, MI300X, MI355X (AMD)
- **Notes:** Superclusters up to 131k B200 GPUs. Request signing required for REST.

---

## TIER 4: Kubernetes-Native

Requires `kubernetes-asyncio` and understanding of Kubernetes CRDs.

| Provider | H100 $/hr | A100 $/hr | Auth Method | Docs |
|----------|-----------|-----------|-------------|------|
| **CoreWeave** | $6.16 | $2.21 | Bearer Token / Kubernetes RBAC | [docs.coreweave.com](https://docs.coreweave.com/) |

### Details

#### CoreWeave (LOW PRIORITY)
- **GPUs:** GB300, GB200, B200, H200, H100, RTX Pro 6000, L40S, L40, GH200, A100
- **Notes:** Kubernetes-native only. VMs via KubeVirt CRDs. Higher complexity. Enterprise-focused.

---

## NOT RECOMMENDED

| Provider | Issue | Docs |
|----------|-------|------|
| **Paperspace** | Core API deprecated July 2024. New GraphQL API under development. | [docs.digitalocean.com](https://docs.digitalocean.com/reference/paperspace/api-reference/) |

---

## Recommended Implementation Order

### Phase 1 (High Priority)
1. **Lambda Labs** - Simple REST, popular, good GPU selection
2. **RunPod** - Per-second billing, huge GPU variety, popular
3. **Nebius** - Native async SDK, competitive pricing ($2/hr H100)
4. **Shadeform** - Aggregator: one implementation = 32+ providers

### Phase 2 (Medium Priority)
5. **GCP** - Major hyperscaler, wrap sync SDK
6. **TensorDock** - Budget marketplace, spot pricing
7. **Thunder Compute** - Cheapest A100 ($0.66/hr)
8. **FluidStack** - Native async, InfiniBand clusters
9. **Azure** - Native async SDK, enterprise

### Phase 3 (Lower Priority)
10. **Hyperstack** - Green energy focus
11. **Salad** - Consumer GPU containerized workloads
12. **OCI** - Enterprise, AMD MI300X support
13. **CoreWeave** - Kubernetes-native (more complex)
14. **Jarvis Labs** - Simpler use cases
15. **Northflank** - PaaS model

---

## Quick Reference: Cheapest Options

| GPU | Provider | Price |
|-----|----------|-------|
| **A100 80GB** | Thunder Compute | $0.66/hr |
| **A100 80GB** | TensorDock (spot) | $0.67/hr |
| **H100** | Thunder Compute | $1.89/hr |
| **H100** | Hyperstack | $1.90/hr |
| **H100** | RunPod | $1.99/hr |
| **RTX 4090** | Salad | $0.16/hr |
| **RTX 4090** | RunPod | $0.34/hr |
