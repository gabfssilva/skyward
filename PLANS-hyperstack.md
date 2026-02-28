# Hyperstack Provider Implementation Plan

## 1. API Research Summary

**Base URL**: `https://infrahub-api.nexgencloud.com/v1`

**Authentication**: Custom header `api_key: <YOUR_API_KEY>` (NOT Bearer, NOT Authorization header). Needs custom Auth implementation.

**Rate Limit**: 500 requests per minute per IP.

**Regions**: `CANADA-1`, `NORWAY-1`, `US-1`

**Key Concepts**:
- **Environments** — organizational containers grouping resources (VMs, keypairs, volumes) within a region. VMs must belong to an environment.
- **Flavors** — hardware configs: `n{gen}-{GPU}x{count}` (e.g., `n3-A100x1`)
- **Images** — pre-built OS images, some with NVIDIA drivers + CUDA (e.g., `Ubuntu Server 22.04 LTS R535 CUDA 12.2`)
- **Keypairs** — SSH keys scoped to environments
- VMs support `user_data` (cloud-init) and `assign_floating_ip` for public IP

**Core Endpoints**:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/core/flavors?region={region}` | List hardware flavors |
| `GET` | `/core/stocks` | GPU stock availability by region |
| `GET` | `/core/images?region={region}` | List OS images |
| `POST` | `/core/environments` | Create environment |
| `GET` | `/core/environments` | List environments |
| `DELETE` | `/core/environments/{id}` | Delete environment |
| `POST` | `/core/keypairs` | Import SSH keypair (requires `environment_name`) |
| `GET` | `/core/keypairs` | List keypairs |
| `DELETE` | `/core/keypair/{id}` | Delete keypair |
| `POST` | `/core/virtual-machines` | Create VMs (supports `count`) |
| `GET` | `/core/virtual-machines` | List VMs |
| `GET` | `/core/virtual-machines/{vm_id}` | Get VM details |
| `DELETE` | `/core/virtual-machines/{vm_id}` | Delete VM |
| `GET` | `/pricebook` | Hourly pricing |

**VM Create Request**:
```json
{
  "name": "skyward-abc123-0",
  "environment_name": "skyward-env-abc123",
  "image_name": "Ubuntu Server 22.04 LTS R535 CUDA 12.2",
  "flavor_name": "n3-A100x1",
  "key_name": "skyward-user-id_ed25519",
  "assign_floating_ip": true,
  "user_data": "<cloud-init>",
  "count": 1
}
```

**VM Statuses**: `CREATING`, `ACTIVE`, `SHUTOFF`, `HIBERNATED`, `ERROR`, `DELETING`, `HIBERNATING`, `RESTORING`

---

## 2. Config Class Design

File: `skyward/providers/hyperstack/config.py`

```python
@dataclass(frozen=True, slots=True)
class Hyperstack(ProviderConfig):
    api_key: str | None = None       # Falls back to HYPERSTACK_API_KEY env var
    region: str = "CANADA-1"         # CANADA-1, NORWAY-1, US-1
    disk_gb: int = 100
    request_timeout: int = 30
    instance_timeout: int = 300

    async def create_provider(self) -> HyperstackProvider:
        from skyward.providers.hyperstack.provider import HyperstackProvider
        return await HyperstackProvider.create(self)

    @property
    def type(self) -> str: return "hyperstack"
```

---

## 3. Provider-Specific State

```python
@dataclass(frozen=True, slots=True)
class HyperstackSpecific:
    environment_name: str
    environment_id: int
    key_name: str
    flavor_name: str
    image_name: str
    region: str
```

- Environment created in `prepare()`, cleaned up in `teardown()`
- Keypairs scoped to environment (auto-cleanup on env deletion)

---

## 4. Client Design

File: `skyward/providers/hyperstack/client.py`

**Custom Auth** (Hyperstack uses `api_key` header, not `Authorization: Bearer`):

```python
class HyperstackAuth:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def headers(self) -> dict[str, str]:
        return {"api_key": self._api_key, "Accept": "application/json", "Content-Type": "application/json"}

    async def on_401(self) -> None:
        pass
```

**Methods**:
```python
class HyperstackClient:
    # Environments
    async def create_environment(self, name: str, region: str) -> EnvironmentResponse
    async def list_environments(self) -> list[EnvironmentResponse]
    async def delete_environment(self, env_id: int) -> None

    # Keypairs
    async def import_keypair(self, env_name: str, name: str, public_key: str) -> KeypairResponse
    async def list_keypairs(self) -> list[KeypairResponse]

    # Flavors & Images
    async def list_flavors(self, region: str | None = None) -> list[FlavorResponse]
    async def list_images(self, region: str | None = None) -> list[ImageResponse]

    # VMs
    async def create_vms(self, payload: CreateVMPayload) -> CreateVMResponse
    async def get_vm(self, vm_id: int) -> VMResponse | None
    async def list_vms(self) -> list[VMResponse]
    async def delete_vm(self, vm_id: int) -> None

    # Pricing & Stock
    async def get_pricebook(self) -> list[PricebookEntry]
    async def get_gpu_stock(self) -> StockResponse
```

Rate limiting: `@throttle(max_concurrent=5, interval=0.12)` (~500 req/min)

---

## 5. Types

File: `skyward/providers/hyperstack/types.py`

```python
class FlavorResponse(TypedDict):
    id: int
    name: str           # e.g., "n3-A100x1"
    cpu: int
    ram: float          # GB
    disk: int
    gpu: str            # e.g., "A100-80G-PCIe"
    gpu_count: int

class ImageResponse(TypedDict):
    id: int
    name: str           # e.g., "Ubuntu Server 22.04 LTS R535 CUDA 12.2"

class EnvironmentResponse(TypedDict):
    id: int
    name: str
    region: str

class KeypairResponse(TypedDict):
    id: int
    name: str
    fingerprint: NotRequired[str]
    public_key: NotRequired[str]

class CreateVMPayload(TypedDict, total=False):
    name: str
    environment_name: str
    image_name: str
    flavor_name: str
    key_name: str
    assign_floating_ip: bool
    user_data: str
    count: int

class VMResponse(TypedDict):
    id: int
    name: NotRequired[str]
    status: str
    fixed_ip: NotRequired[str]
    floating_ip: NotRequired[str]
    floating_ip_status: NotRequired[str]
    flavor: NotRequired[FlavorInVM]
    image: NotRequired[ImageInVM]

class FlavorInVM(TypedDict):
    name: NotRequired[str]
    cpu: NotRequired[int]
    ram: NotRequired[float]
    gpu: NotRequired[str]
    gpu_count: NotRequired[int]

class CreateVMResponse(TypedDict):
    status: NotRequired[bool]
    message: NotRequired[str]
    instances: NotRequired[list[VMResponse]]

class PricebookEntry(TypedDict):
    gpu: NotRequired[str]
    price_per_gpu_hr: NotRequired[float]
    region: NotRequired[str]
```

---

## 6. Provider Implementation

### `offers(spec)`
1. Call `list_flavors(region=config.region)` + `get_pricebook()`
2. Filter flavors by accelerator name (fuzzy: `"A100"` matches `"A100-80G-PCIe"`)
3. Match accelerator count, vcpus, memory if specified
4. Yield `Offer` with pricing from pricebook

### `prepare(spec, offer)`
1. **Create environment**: `skyward-{uuid[:8]}` in config region
2. **Import SSH keypair** into environment
3. **Resolve image**: prefer Ubuntu with CUDA drivers
4. **Resolve flavor** from offer
5. Return `Cluster[HyperstackSpecific]` with `ssh_user="ubuntu"`, `use_sudo=True`

### `provision(cluster, count)`
1. Call `create_vms()` with `assign_floating_ip=True`, `count=min(count, 20)`
2. Batch into multiple calls if count > 20
3. Return instances with `status="provisioning"`

### `get_instance(cluster, id)`
1. Call `get_vm(int(id))`
2. Map status — `ACTIVE` + floating_ip → `"provisioned"`

### `terminate(cluster, ids)`
1. `asyncio.gather` over `delete_vm(int(id))`

### `teardown(cluster)`
1. `delete_environment(specific.environment_id)` — cascades keypairs + remaining VMs

---

## 7. Status Mapping

| Hyperstack Status | Skyward Status | Notes |
|-------------------|---------------|-------|
| `CREATING` / `BUILD` | `"provisioning"` | VM starting up |
| `ACTIVE` + floating_ip | `"provisioned"` | Ready for SSH |
| `ACTIVE` without floating_ip | `"provisioning"` | IP not yet assigned |
| `SHUTOFF` / `HIBERNATED` / `ERROR` / `DELETING` | return `None` | Gone or unusable |

---

## 8. Registration Changes

1. `skyward/providers/__init__.py` — add `from .hyperstack.config import Hyperstack`
2. `skyward/config.py` — add `"hyperstack": Hyperstack` to `_get_provider_map()`
3. `skyward/__init__.py` — add `Hyperstack` to imports and `__all__`
4. `pyproject.toml` — add `"hyperstack: Hyperstack provider tests"` marker
5. `Taskfile.yml` — add `test:sanity:hyperstack`
6. `docs/providers.md` and `CLAUDE.md` — update provider lists

---

## 9. Sanity Test

```python
@pytest.mark.sanity
@pytest.mark.hyperstack
@pytest.mark.timeout(TIMEOUT)
@pytest.mark.xdist_group("hyperstack")
class TestHyperstackSanity:
    @pytest.fixture(scope="class")
    def pool(self):
        with sky.App(console=False), sky.ComputePool(
            provider=sky.Hyperstack(),
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

1. **Custom auth header** — needs custom `HyperstackAuth` class (not `BearerAuth`). Must implement the `Auth` protocol from `skyward.infra.http`
2. **Floating IP timing** — delay between `ACTIVE` status and floating IP assignment. Polling must handle this
3. **Environment cleanup on failure** — if `provision()` fails, `teardown()` still cleans up the environment
4. **No spot instances** — all on-demand. `allocation="spot"` silently falls back
5. **cloud-init format** — may need base64 encoding or escaped newlines. Verify exact format
6. **VM count limit** — max 20 VMs per API call, batch if needed
7. **Security rules / firewall** — verify default rules allow SSH (port 22). May need to create firewall rules post-provisioning
8. **Port randomization** — Hyperstack has `enable_port_randomization` field. Ensure it's disabled or read actual SSH port from response

## Implementation Sequence

1. `types.py` → 2. `config.py` → 3. `client.py` (with custom auth) → 4. `provider.py` → 5. `__init__.py` → 6. Registration points → 7. Tests → 8. Docs
