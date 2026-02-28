# Thunder Compute Provider Implementation Plan

## 1. API Research Summary

**Base URL**: `https://api.thundercompute.com:8443/v1`

**Authentication**: Bearer token — `Authorization: Bearer <token>`. Tokens created at `console.thundercompute.com/settings?tab=tokens`. Env var: `TNR_API_TOKEN`.

**Location**: Canada (Quebec) only. No region selection.

**Billing**: Per-minute, on-demand only. No spot instances.

**Key Endpoints**:

| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| `POST` | `/instances/create` | Launch instance | Yes |
| `GET` | `/instances/list` | List all instances | Yes |
| `POST` | `/instances/{id}/delete` | Terminate instance | Yes |
| `POST` | `/instances/{id}/add_key` | Add SSH key to instance | Yes |
| `POST` | `/keys/add` | Register SSH key to org | Yes |
| `GET` | `/keys/list` | List org SSH keys | Yes |
| `DELETE` | `/keys/{id}` | Revoke SSH key | Yes |
| `GET` | `/pricing` | Fetch hourly rates | No |
| `GET` | `/thunder-templates` | List instance templates | No |

**Instance Create Request**:
```json
{
  "cpu_cores": 18,
  "disk_size_gb": 100,
  "gpu_type": "h100",
  "num_gpus": 1,
  "mode": "production",
  "template": "base",
  "public_key": "ssh-ed25519 AAAA..."
}
```

**Instance Create Response**: `{"identifier": 123, "uuid": "...", "key": "..."}`

**Instance List Response** — map of UUID → instance info:
- `name`, `uuid`, `status`, `ip`, `port` (int — dynamic SSH port)
- `cpuCores`, `memory`, `numGpus`, `gpuType`, `template`, `mode`, `storage`

**Instance Statuses**: `CREATING`, `RESTORING`, `RUNNING`, `DELETING`

**SSH**: user `ubuntu`, dynamic port (from `port` field), IP from `ip` field

**GPU Types & Pricing**:

| GPU | VRAM | Prototyping $/hr | Production $/GPU/hr | Max GPUs |
|-----|------|-------------------|---------------------|----------|
| A6000 | 48GB | $0.27 | N/A | — |
| A100 80GB | 80GB | $0.78 | $1.79 | 8 |
| H100 | 80GB | $1.38 | $2.49 | 8 |

**Modes**: `"prototyping"` (cheaper, limited CUDA) vs `"production"` (full CUDA, multi-GPU)

**Software Stack**: CUDA 13.0, Driver 580, PyTorch 2.9.0 pre-installed

---

## 2. Config Class Design

File: `skyward/providers/thunder/config.py`

```python
@dataclass(frozen=True, slots=True)
class ThunderCompute(ProviderConfig):
    api_token: str | None = None            # Falls back to TNR_API_TOKEN env var
    mode: Literal["prototyping", "production"] = "production"
    template: str = "base"                  # "base", "ollama", "comfy-ui", or snapshot ID
    disk_size_gb: int = 100
    cpu_cores: int = 18
    request_timeout: int = 30

    async def create_provider(self) -> ThunderComputeProvider:
        from skyward.providers.thunder.provider import ThunderComputeProvider
        return await ThunderComputeProvider.create(self)

    @property
    def type(self) -> str: return "thunder"
```

- `mode` defaults to `"production"` for full CUDA + multi-GPU
- No `region` field — Quebec only
- No spot/allocation support — on-demand only

---

## 3. Provider-Specific State

```python
@dataclass(frozen=True, slots=True)
class ThunderSpecific:
    ssh_key_id: str
    ssh_public_key: str
    mode: str
    template: str
    disk_size_gb: int
    cpu_cores: int
    pricing: dict[str, float]    # Cached from /pricing
```

---

## 4. Client Design

File: `skyward/providers/thunder/client.py`

- **Auth**: `BearerAuth(token)` from `skyward.infra.http`
- **Rate limiting**: `@throttle(max_concurrent=3, interval=0.5)` (safety measure)

```python
THUNDER_API_BASE = "https://api.thundercompute.com:8443/v1"

class ThunderComputeError(Exception): ...

class ThunderComputeClient:
    async def create_instance(self, cpu_cores, disk_size_gb, gpu_type, num_gpus, mode, template, public_key) -> InstanceCreateResponse
    async def list_instances(self) -> dict[str, InstanceListItem]
    async def delete_instance(self, uuid: str) -> None
    async def add_key_to_instance(self, uuid: str, public_key: str) -> None
    async def add_ssh_key(self, name: str, public_key: str) -> SSHKeyResponse
    async def list_ssh_keys(self) -> list[SSHKeyResponse]
    async def get_pricing(self) -> dict[str, float]
    async def get_templates(self) -> list[dict]
```

**API token resolution**: config → `TNR_API_TOKEN` env → `~/.thunder/token` → error

---

## 5. Types

File: `skyward/providers/thunder/types.py`

```python
class InstanceCreateResponse(TypedDict):
    identifier: int
    uuid: str
    key: NotRequired[str]

class InstanceListItem(TypedDict):
    name: str
    uuid: str
    status: str           # CREATING, RUNNING, RESTORING, DELETING
    ip: str
    port: int             # Dynamic SSH port
    cpuCores: int
    memory: int           # MB
    numGpus: int
    gpuType: str
    template: str
    mode: str
    storage: int          # GB
    createdAt: str

class SSHKeyResponse(TypedDict):
    id: str
    name: str
    public_key: str
    fingerprint: str

class PricingResponse(TypedDict):
    pricing: dict[str, float]

class ErrorResponse(TypedDict):
    code: int
    error: str
    message: str
```

---

## 6. Provider Implementation

### GPU Type Mapping

```python
_GPU_TYPE_MAP: dict[str, str] = {
    "H100": "h100",
    "A100": "a100xl",      # Thunder uses "a100xl" for A100 80GB
    "A6000": "a6000",
    "RTX A6000": "a6000",
}

_GPU_VRAM: dict[str, int] = {"h100": 80, "a100xl": 80, "a6000": 48}
_GPU_DISPLAY: dict[str, str] = {"h100": "H100", "a100xl": "A100 80GB", "a6000": "RTX A6000"}
```

### `offers(spec)`
1. Call `get_pricing()` (no auth)
2. Map `spec.accelerator_name` to Thunder GPU type
3. Look up price, build `Offer` with `spot_price=None`, `billing_unit="minute"`

### `prepare(spec, offer)`
1. Get local SSH key, register globally on Thunder org
2. Fetch + cache pricing in `ThunderSpecific`
3. Return `Cluster` with `ssh_user="ubuntu"`, `use_sudo=True`

### `provision(cluster, count)`
1. For each instance, call `create_instance()` with `public_key` (belt-and-suspenders with org key)
2. Return `Instance(id=uuid, status="provisioning")`

### `get_instance(cluster, id)`
1. Call `list_instances()` (no single-instance endpoint!)
2. Find by UUID, map status
3. Set `ssh_port=info["port"]` (dynamic port!)

### `terminate(cluster, ids)`
1. `asyncio.gather` over `delete_instance(uuid)`

### `teardown(cluster)`
1. No-op — no cluster-level resources

---

## 7. Status Mapping

| Thunder Status | Skyward Status | Notes |
|---------------|---------------|-------|
| `CREATING` | `"provisioning"` | Being created |
| `RESTORING` | `"provisioning"` | Snapshot restore |
| `RUNNING` + has IP | `"provisioned"` | Ready for SSH |
| `RUNNING` without IP | `"provisioning"` | IP not yet assigned |
| `DELETING` | return `None` | Being destroyed |
| Not found | return `None` | Gone |

---

## 8. Registration Changes

1. `skyward/providers/__init__.py` — add `from .thunder.config import ThunderCompute`
2. `skyward/config.py` — add `"thunder": ThunderCompute` to `_get_provider_map()`
3. `skyward/__init__.py` — add `ThunderCompute` to imports and `__all__`
4. `pyproject.toml` — add `"thunder: Thunder Compute provider tests"` marker
5. `Taskfile.yml` — add `test:sanity:thunder`
6. `docs/providers.md` and `CLAUDE.md` — update provider lists

No optional dependencies needed — Thunder has no SDK, uses `skyward.infra.http.HttpClient`

---

## 9. Sanity Test

```python
@pytest.mark.sanity
@pytest.mark.thunder
@pytest.mark.timeout(TIMEOUT)
@pytest.mark.xdist_group("thunder")
class TestThunderComputeSanity:
    @pytest.fixture(scope="class")
    def pool(self):
        with sky.App(console=False), sky.ComputePool(
            provider=sky.ThunderCompute(mode="production"),
            accelerator=sky.accelerators.A100(),
            nodes=NODES,
            image=sky.Image(pip=["torch"]),
        ) as p:
            yield p

    def test_single_dispatch(self, pool): ...
    def test_broadcast(self, pool): ...
```

---

## 10. Open Questions / Risks

1. **No single-instance GET endpoint** — `get_instance()` must call `list_instances()` and filter. Acceptable for small instance counts
2. **Dynamic SSH port** — SSH port is returned in `port` field, NOT port 22. Must set `Instance.ssh_port` from response
3. **No spot instances** — on-demand only. `allocation="spot"` silently falls back
4. **Single region (Quebec)** — no multi-region failover possible
5. **Limited GPU selection** — only A6000, A100, H100. Unknown accelerators yield empty offers
6. **Pre-installed CUDA 13.0** — bootstrap must not conflict with existing stack
7. **No private networking** — multi-node communication over public IPs, may affect NCCL performance
8. **`gpu_type` API values** — need runtime verification (lowercase `"h100"` vs `"H100"`)
9. **Prototyping mode limitations** — no custom CUDA kernels, max 2 H100 GPUs
10. **API token also readable from `~/.thunder/token`** — support CLI config file as fallback

## Implementation Sequence

1. `types.py` → 2. `client.py` → 3. `config.py` → 4. `provider.py` → 5. `__init__.py` → 6. Registration points → 7. Tests → 8. Docs
