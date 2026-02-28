# TensorDock Provider Implementation Plan

## 1. API Research Summary

**TensorDock** is a GPU marketplace with hosts across 100+ locations in 20+ countries. Bare-metal VMs (not Docker) with cloud-init support.

**API**: v0 Marketplace API (well-documented, battle-tested)

**Base URL**: `https://marketplace.tensordock.com`

**Authentication**: `api_key` + `api_token` passed as POST form data or GET query parameters. Get at `console.tensordock.com/api`. Env vars: `TENSORDOCK_API_KEY`, `TENSORDOCK_API_TOKEN`.

**Billing**: Per-second.

**Key Endpoints**:

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/api/v0/client/deploy/hostnodes` | GET | No | List hostnodes with GPU stock, pricing, locations |
| `/api/v0/client/deploy/single` | POST | Yes | Deploy a VM on a specific hostnode |
| `/api/v0/client/list` | POST | Yes | List all deployed VMs |
| `/api/v0/client/get/single` | POST | Yes | Get single VM details |
| `/api/v0/client/delete/single` | POST | Yes | Delete a VM |
| `/api/v0/client/start/single` | GET | Yes | Start a stopped VM |
| `/api/v0/client/stop/single` | GET | Yes | Stop a running VM |
| `/api/v0/auth/test` | POST | Yes | Verify credentials |

**Deploy Parameters**:
```
password, name, gpu_count, gpu_model, vcpus, ram,
external_ports, internal_ports, hostnode, storage,
operating_system, cloudinit_script
```

**Deploy Response**:
```json
{
  "success": true,
  "ip": "1.2.3.4",
  "port_forwards": {"22": 12345, "25520": 12346},
  "server": "uuid-of-vm"
}
```

**Critical: Port Forwarding** — TensorDock maps internal ports to random external ports. SSH is NOT on port 22 externally.

**VM Statuses**: `running`, `stopped`, `starting`, `stopping`, `deploying`

**SSH**: user `user` (not root), key injected via cloud-init

**GPU Model Identifiers** (lowercase-hyphenated with interface/memory suffix):

| Display Name | API ID |
|---|---|
| H100 SXM5 80GB | `h100-sxm5-80gb` |
| A100 SXM4 80GB | `a100-sxm4-80gb` |
| A100 PCIe 80GB | `a100-pcie-80gb` |
| L40 48GB | `l40-pcie-48gb` |
| RTX A6000 48GB | `rtxa6000-pcie-48gb` |
| RTX 4090 24GB | `geforcertx4090-pcie-24gb` |
| RTX 3090 24GB | `geforcertx3090-pcie-24gb` |
| V100 SXM2 16GB | `v100-sxm2-16gb` |

---

## 2. Config Class Design

File: `skyward/providers/tensordock/config.py`

```python
@dataclass(frozen=True, slots=True)
class TensorDock(ProviderConfig):
    api_key: str | None = None        # Falls back to TENSORDOCK_API_KEY
    api_token: str | None = None      # Falls back to TENSORDOCK_API_TOKEN
    location: str | None = None       # Country code filter (e.g., "us", "de", "gb")
    storage_gb: int = 100             # Min 100GB
    operating_system: str = "Ubuntu 22.04 LTS"
    instance_timeout: int = 300
    request_timeout: int = 30
    min_ram_gb: int | None = None
    min_vcpus: int | None = None

    async def create_provider(self) -> TensorDockProvider:
        from skyward.providers.tensordock.provider import TensorDockProvider
        return await TensorDockProvider.create(self)

    @property
    def type(self) -> str: return "tensordock"
```

- **Two credentials** (`api_key` + `api_token`) — TensorDock uses a key+token pair
- `location` filter applied to hostnode geographic data
- No spot pricing

---

## 3. Provider-Specific State

```python
@dataclass(frozen=True, slots=True)
class TensorDockSpecific:
    ssh_public_key: str
    password: str              # Required by TensorDock, random-generated
    location: str | None = None
    operating_system: str = "Ubuntu 22.04 LTS"
```

- SSH keys injected per-instance via cloud-init (no SSH key API)
- Password required by deploy API — generate `secrets.token_urlsafe(24)`

---

## 4. Client Design

File: `skyward/providers/tensordock/client.py`

**Auth approach**: v0 API passes `api_key` + `api_token` as query params (all requests). Deploy uses form-encoded POST body. Does NOT fit standard `BearerAuth` — needs direct `aiohttp.ClientSession` usage.

```python
TENSORDOCK_API_BASE = "https://marketplace.tensordock.com"

class TensorDockError(Exception): ...

class TensorDockClient:
    def __init__(self, api_key: str, api_token: str, config: TensorDock | None = None) -> None:
        self._api_key = api_key
        self._api_token = api_token
        # Uses aiohttp directly (like RunPod's GraphQL client)

    def _auth_params(self) -> dict[str, str]:
        return {"api_key": self._api_key, "api_token": self._api_token}

    async def list_hostnodes(self) -> dict[str, HostnodeResponse]
    async def deploy_vm(self, hostnode, gpu_model, gpu_count, vcpus, ram, storage,
                        password, name, os, internal_ports, external_ports,
                        cloudinit_script) -> DeployResponse
    async def get_vm(self, server_id: str) -> VmDetails | None
    async def list_vms(self) -> dict[str, VmDetails]
    async def delete_vm(self, server_id: str) -> None
    async def start_vm(self, server_id: str) -> None
    async def stop_vm(self, server_id: str) -> None
    async def test_auth(self) -> bool
    async def get_billing(self) -> BillingResponse
```

Rate limiting: `@throttle(interval=1.0)` (~1 req/sec as recommended)

---

## 5. Types

File: `skyward/providers/tensordock/types.py`

```python
class HostnodeGpu(TypedDict):
    amount: int

class HostnodeSpecs(TypedDict):
    gpu: dict[str, HostnodeGpu]
    ram: dict[str, int]
    cpu: dict[str, int]
    storage: dict[str, int]

class HostnodeLocation(TypedDict):
    country: str
    city: str
    region: str

class HostnodeResponse(TypedDict):
    id: str
    specs: HostnodeSpecs
    location: HostnodeLocation

class DeployResponse(TypedDict):
    success: bool
    ip: NotRequired[str]
    port_forwards: NotRequired[dict[str, int]]   # {"22": 12345, "25520": 12346}
    server: NotRequired[str]

class VmDetails(TypedDict):
    id: str
    name: NotRequired[str]
    ip: NotRequired[str]
    status: NotRequired[str]
    port_forwards: NotRequired[dict[str, int]]
    gpu_model: NotRequired[str]
    gpu_count: NotRequired[int]
    vcpus: NotRequired[int]
    ram: NotRequired[int]
    storage: NotRequired[int]
    cost: NotRequired[CostInfo]

class CostInfo(TypedDict):
    hour_on: NotRequired[float]
    hour_off: NotRequired[float]

class VmGetResponse(TypedDict):
    success: bool
    virtualmachine: NotRequired[VmDetails]

class VmListResponse(TypedDict):
    success: bool
    virtualmachines: NotRequired[dict[str, VmDetails]]

class BillingResponse(TypedDict):
    success: bool
    balance: NotRequired[float]
    hourly_spending_rate: NotRequired[float]
```

Helper functions:
- `get_ssh_port(vm) -> int` — extract external SSH port from `port_forwards`
- `normalize_gpu_name(td_name) -> str` — `"geforcertx4090-pcie-24gb"` → `"RTX 4090"`
- `get_gpu_memory_gb(td_name) -> int` — extract memory from GPU ID suffix

---

## 6. Provider Implementation

### GPU Name Mapping

```python
_GPU_NAME_MAP: dict[str, str] = {
    "h100-sxm5-80gb": "H100",
    "a100-sxm4-80gb": "A100",
    "a100-pcie-80gb": "A100",
    "l40-pcie-48gb": "L40",
    "rtxa6000-pcie-48gb": "RTX A6000",
    "geforcertx4090-pcie-24gb": "RTX 4090",
    "geforcertx3090-pcie-24gb": "RTX 3090",
    "v100-sxm2-16gb": "V100",
    # ...
}
```

### `offers(spec)`
1. Call `list_hostnodes()` (no auth)
2. Filter by location, GPU model (fuzzy match), GPU count
3. For each matching hostnode+GPU, yield `Offer` with pricing from hostnode data
4. Sort by cheapest price per GPU-hour

### `prepare(spec, offer)`
1. Read local SSH public key
2. Generate random password (`secrets.token_urlsafe(24)`)
3. Verify credentials via `test_auth()`
4. Return `Cluster[TensorDockSpecific]` with `ssh_user="user"`, `use_sudo=True`

### `provision(cluster, count)`
1. For each VM:
   - Query hostnodes again (dynamic availability)
   - Build cloud-init: inject SSH key to `~user/.ssh/authorized_keys`
   - Call `deploy_vm()` with `internal_ports="22,25520"`, `external_ports="22,25520"`
   - Parse `port_forwards` from response
   - Set `Instance.ssh_port = port_forwards["22"]` (external mapped port)
2. Return instances with `status="provisioning"`

### `get_instance(cluster, id)`
1. Call `get_vm(id)`
2. Map status, extract IP + SSH port from `port_forwards`

### `terminate(cluster, ids)`
1. `asyncio.gather` over `delete_vm(id)`

### `teardown(cluster)`
1. No-op — no cluster-level resources (no SSH key API, no VPC, no overlay)

---

## 7. Status Mapping

| TensorDock Status | Skyward Status | Notes |
|-------------------|---------------|-------|
| `running` + IP + SSH port | `"provisioned"` | Ready |
| `deploying` / `starting` | `"provisioning"` | In progress |
| `stopped` / `stopping` / `error` | return `None` | Gone or unusable |

---

## 8. SSH Key Injection via Cloud-Init

No SSH key registration API. Keys injected per-instance:

```yaml
#cloud-config
write_files:
  - path: /home/user/.ssh/authorized_keys
    permissions: '0600'
    owner: user:user
    content: |
      ssh-ed25519 AAAA... user@host
runcmd:
  - chown -R user:user /home/user/.ssh
  - chmod 700 /home/user/.ssh
```

---

## 9. Port Forwarding Handling

TensorDock maps internal → external ports. Critical for Skyward:

1. Request `internal_ports="22,25520"` and `external_ports="22,25520"` at deploy
2. Parse `port_forwards`: `{"22": 34567, "25520": 34568}`
3. Set `Instance.ssh_port = port_forwards["22"]` (e.g., 34567)
4. SSH tunnel: connects to `ip:34567`, then forwards `local → remote:25520` internally — works correctly since tunnel goes through the SSH connection

---

## 10. Registration Changes

1. `skyward/providers/__init__.py` — add `from .tensordock.config import TensorDock`
2. `skyward/config.py` — add `"tensordock": TensorDock` to `_get_provider_map()`
3. `skyward/__init__.py` — add `TensorDock` to imports and `__all__`
4. `pyproject.toml` — add `"tensordock: TensorDock provider tests"` marker
5. `Taskfile.yml` — add `test:sanity:tensordock`
6. `docs/providers.md` and `CLAUDE.md` — update provider lists

---

## 11. Sanity Test

```python
@pytest.mark.sanity
@pytest.mark.tensordock
@pytest.mark.timeout(TIMEOUT)
@pytest.mark.xdist_group("tensordock")
class TestTensorDockSanity:
    @pytest.fixture(scope="class")
    def pool(self):
        with sky.App(console=False), sky.ComputePool(
            provider=sky.TensorDock(),
            accelerator=sky.accelerators.RTX_4090(),
            nodes=NODES,
            image=sky.Image(pip=["torch"]),
        ) as p:
            yield p

    def test_single_dispatch(self, pool): ...
    def test_broadcast(self, pool): ...
```

---

## 12. Open Questions / Risks

1. **Form-encoded POST** — v0 API uses form data (not JSON) for deploy. `HttpClient` sends JSON. Need direct `aiohttp` usage or form-data support
2. **Port forwarding for Casty** — SSH tunnel should work through mapped ports, but needs verification in sanity test
3. **No spot pricing** — on-demand only, `allocation` silently falls back
4. **SSH user is `user`** — not `root`, needs `use_sudo=True`
5. **Hostnode selection strategy** — when multiple match: cheapest price > most available GPUs > location preference
6. **Dynamic hostnode availability** — hostnodes come and go. Re-query at each `provision()` call for autoscaling
7. **cloud-init size limits** — if bootstrap script too large, fall back to SSH-based bootstrap (inject key via cloud-init, bootstrap via SSH)
8. **No preemption** — no spot model, no preemption handling needed
9. **Hostnode pricing format** — exact structure needs runtime verification
10. **NVIDIA driver auto-update** — TensorDock auto-updates drivers on first boot. May need `apt-mark hold` via cloud-init to prevent conflicts

## Implementation Sequence

1. `types.py` → 2. `client.py` (direct aiohttp) → 3. `config.py` → 4. `provider.py` → 5. `__init__.py` → 6. Registration points → 7. Tests → 8. Docs
