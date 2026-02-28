# Lambda Cloud Provider Implementation Plan

## 1. API Research Summary

**Base URL**: `https://cloud.lambdalabs.com/api/v1`

**Authentication**: Bearer token — `Authorization: Bearer <API_KEY>`

**Rate Limiting**:
- General endpoints: 1 request/second
- Launch endpoint: 1 request/12 seconds (5 per minute)

**Response Envelope**: `{"data": <payload>}` on success, `{"error": {"code": "...", "message": "...", "suggestion": "..."}}` on error.

**Endpoints**:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/instance-types` | List available instance types with specs and regional availability |
| `GET` | `/instances` | List running instances |
| `GET` | `/instances/{id}` | Get single instance details |
| `POST` | `/instance-operations/launch` | Launch instances (supports `quantity` for batch) |
| `POST` | `/instance-operations/terminate` | Terminate instances (batch) |
| `POST` | `/instance-operations/restart` | Restart instances |
| `GET` | `/ssh-keys` | List SSH keys |
| `POST` | `/ssh-keys` | Add SSH key (name + public_key) |
| `DELETE` | `/ssh-keys/{id}` | Delete SSH key |
| `GET` | `/file-systems` | List persistent filesystems |

**Launch Request Body**:
```json
{
  "region_name": "us-west-1",
  "instance_type_name": "gpu_8x_a100",
  "ssh_key_names": ["my-key"],
  "file_system_names": [],
  "name": "my-instance",
  "quantity": 1,
  "user_data": "<cloud-init script>"
}
```

**Instance Statuses**: `active`, `booting`, `terminating`, `unhealthy`, `terminated`

**Key Observations**:
- Bare-metal VM provider (not Docker-based)
- Lambda Stack pre-installed (NVIDIA drivers, CUDA, frameworks)
- SSH on port 22, user is `ubuntu`
- **On-demand only** — no spot pricing
- `user_data` supports cloud-init (max 1MB)
- `quantity` param allows batch launching in a single API call
- Pricing in `price_cents_per_hour`

---

## 2. Config Class Design

File: `skyward/providers/lambda_cloud/config.py`

```python
@dataclass(frozen=True, slots=True)
class Lambda(ProviderConfig):
    api_key: str | None = None      # Falls back to LAMBDA_API_KEY env var
    region: str | None = None       # If None, auto-select from regions with capacity
    request_timeout: int = 30

    async def create_provider(self) -> LambdaProvider:
        from skyward.providers.lambda_cloud.provider import LambdaProvider
        return await LambdaProvider.create(self)

    @property
    def type(self) -> str: return "lambda"
```

- Minimal config: Lambda Cloud is simple (no spot, no Docker, no complex networking)
- `region` optional — auto-selects from regions with capacity if `None`

---

## 3. Provider-Specific State

```python
@dataclass(frozen=True, slots=True)
class LambdaSpecific:
    ssh_key_name: str           # Lambda launch API takes key NAMES, not IDs
    instance_type_name: str     # e.g., "gpu_8x_a100"
    region: str
    price_cents_per_hour: int
    gpu_description: str
    gpu_count: int
    vcpus: int
    memory_gib: int
    storage_gib: int
```

---

## 4. Client Design

File: `skyward/providers/lambda_cloud/client.py`

- **Auth**: `BearerAuth(token)` from `skyward.infra.http`
- **Rate limiting**: `@throttle(interval=1.0)` general, `@throttle(interval=12.0)` for launch
- **Response unwrapping**: All responses wrapped in `{"data": ...}` envelope

```python
class LambdaError(Exception):
    def __init__(self, code: str, message: str, suggestion: str | None = None): ...

class LambdaClient:
    async def list_instance_types(self) -> dict[str, InstanceTypeEntry]
    async def launch_instances(self, region, type_name, ssh_keys, quantity, name, user_data) -> list[str]
    async def list_instances(self) -> list[InstanceResponse]
    async def get_instance(self, instance_id: str) -> InstanceResponse | None
    async def terminate_instances(self, instance_ids: list[str]) -> list[TerminatedInstanceResponse]
    async def list_ssh_keys(self) -> list[SSHKeyResponse]
    async def add_ssh_key(self, name: str, public_key: str) -> SSHKeyResponse
    async def delete_ssh_key(self, key_id: str) -> None
```

---

## 5. Types

File: `skyward/providers/lambda_cloud/types.py`

```python
class RegionResponse(TypedDict):
    name: str
    description: str

class InstanceSpecsResponse(TypedDict):
    vcpus: int
    memory_gib: int
    storage_gib: int

class InstanceTypeInfo(TypedDict):
    name: str
    price_cents_per_hour: int
    description: str
    gpu_description: NotRequired[str]
    specs: InstanceSpecsResponse

class InstanceTypeEntry(TypedDict):
    instance_type: InstanceTypeInfo
    regions_with_capacity_available: list[RegionResponse]

class InstanceResponse(TypedDict):
    id: str
    name: NotRequired[str]
    ip: NotRequired[str]
    private_ip: NotRequired[str]
    status: str
    hostname: NotRequired[str]
    ssh_key_names: list[str]
    region: RegionResponse
    instance_type: InstanceTypeInfo
    is_reserved: NotRequired[bool]

class SSHKeyResponse(TypedDict):
    id: str
    name: str
    public_key: str

class LaunchResponse(TypedDict):
    instance_ids: list[str]

class ErrorResponse(TypedDict):
    code: str
    message: str
    suggestion: NotRequired[str]
```

Helper: `parse_gpu_from_type_name("gpu_8x_a100") -> ("A100", 8)`

---

## 6. Provider Implementation

### `offers(spec)`
1. Call `list_instance_types()`
2. Filter by accelerator name (fuzzy), region, vcpus, memory, max_hourly_cost
3. Sort by price (cheapest first)
4. Yield `Offer` with `spot_price=None`, `on_demand_price=cents/100`

### `prepare(spec, offer)`
1. Get local SSH key, ensure registered on Lambda Cloud (by fingerprint/name match)
2. Resolve region (from config or first with capacity)
3. Parse GPU count/name from instance type name
4. Return `Cluster[LambdaSpecific]` with `ssh_user="ubuntu"`, `use_sudo=True`

### `provision(cluster, count)`
1. Single API call with `quantity=count` (batch launch)
2. Pass `user_data` for cloud-init bootstrap
3. Return instances with `status="provisioning"`

### `get_instance(cluster, id)`
1. Call `get_instance(id)`, map status

### `terminate(cluster, ids)`
1. Single batch call to terminate endpoint

### `teardown(cluster)`
1. No cluster-level resources — no-op

---

## 7. Status Mapping

| Lambda Status | Skyward Status | Notes |
|---------------|---------------|-------|
| `booting` | `"provisioning"` | Starting up |
| `active` + has IP | `"provisioned"` | Ready |
| `unhealthy` | `"provisioning"` | May recover |
| `terminating` | return `None` | Being destroyed |
| `terminated` | return `None` | Gone |

---

## 8. Registration Changes

1. `skyward/providers/__init__.py` — add `from .lambda_cloud.config import Lambda`
2. `skyward/config.py` — add `"lambda": Lambda` to `_get_provider_map()`
3. `skyward/__init__.py` — add `Lambda` to imports and `__all__`
4. `pyproject.toml` — add `"lambda_cloud: Lambda Cloud provider tests"` marker
5. `Taskfile.yml` — add `test:sanity:lambda`
6. `docs/providers.md` and `CLAUDE.md` — update provider lists

---

## 9. Sanity Test

```python
@pytest.mark.sanity
@pytest.mark.lambda_cloud
@pytest.mark.timeout(TIMEOUT)
@pytest.mark.xdist_group("lambda_cloud")
class TestLambdaSanity:
    @pytest.fixture(scope="class")
    def pool(self):
        with sky.App(console=False), sky.ComputePool(
            provider=sky.Lambda(),
            accelerator=sky.accelerators.A10(),
            nodes=NODES,
            image=sky.Image(pip=["torch"]),
        ) as p:
            yield p

    def test_single_dispatch(self, pool): ...
    def test_broadcast(self, pool): ...
```

---

## 10. Open Questions / Risks

1. **No spot pricing** — `allocation` param ignored, always on-demand
2. **Shutdown command** — Lambda warns against `sudo shutdown -h now` (causes Alert status, billing continues). Use API-based self-termination via curl with injected `LAMBDA_API_KEY` and instance ID
3. **Rate limiting on launch** — 12s rate limit, but `quantity` param allows batch launch (single call for N instances)
4. **Lambda Stack pre-installed** — bootstrap must be compatible with existing environment
5. **No persistent storage integration** — Lambda filesystems are NFS/region-locked, skip `Mountable` for now
6. **GPU naming** — parse from type name: `gpu_1x_a100` → (A100, 1), `gpu_8x_h100_sxm5` → (H100 SXM5, 8)

## Implementation Sequence

1. `types.py` → 2. `client.py` → 3. `config.py` → 4. `provider.py` → 5. `__init__.py` → 6. Registration points → 7. Tests → 8. Docs
