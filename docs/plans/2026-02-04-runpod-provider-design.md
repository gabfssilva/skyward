# RunPod Provider Design

## Overview

Implement RunPod GPU Pods as a new provider for Skyward, following the existing event-driven architecture patterns.

## Scope

- **In scope:** GPU Pods (on-demand and spot instances)
- **Out of scope:** Serverless endpoints (different model, no SSH/bootstrap)

## File Structure

```
skyward/providers/runpod/
├── __init__.py      # Exports: RunPod config + RunPodModule
├── config.py        # Frozen dataclass (no SDK deps)
├── client.py        # Async httpx client for REST API
├── handler.py       # @component with event handlers
├── state.py         # RunPodClusterState extends BaseClusterState
└── types.py         # TypedDicts for API responses
```

## Dependencies

No new dependencies. Uses `httpx` (already in project).

## Configuration

```python
@dataclass(frozen=True, slots=True)
class RunPod:
    """RunPod GPU Pods provider configuration."""

    # Auth - fallback to RUNPOD_API_KEY env var
    api_key: str | None = None

    # Instance defaults
    cloud_type: Literal["secure", "community"] = "secure"
    container_disk_gb: int = 50
    volume_gb: int = 20
    volume_mount_path: str = "/workspace"

    # Location - list of IDs or None for auto-select
    data_center_ids: tuple[str, ...] | None = None

    # Networking
    ports: str = "22/tcp"

    # Timeouts
    provision_timeout: float = 300.0
    bootstrap_timeout: float = 600.0
```

Note: Spot/on-demand allocation is controlled via `allocation` parameter in the pool spec, not in the provider config.

## REST API

- **Base URL:** `https://rest.runpod.io/v1`
- **Auth:** Bearer token via `Authorization` header

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /pods | Create pod |
| GET | /pods | List pods |
| GET | /pods/{id} | Get pod |
| POST | /pods/{id}/stop | Stop pod |
| DELETE | /pods/{id} | Terminate pod |

### Pod Status Values

- `RUNNING` - active
- `EXITED` - stopped
- `TERMINATED` - removed

## Client (`client.py`)

```python
@dataclass
class RunPodClient:
    api_key: str

    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, *args) -> None: ...

    # Pod lifecycle
    async def create_pod(self, params: PodCreateParams) -> PodResponse: ...
    async def get_pod(self, pod_id: str) -> PodResponse: ...
    async def list_pods(self) -> list[PodResponse]: ...
    async def stop_pod(self, pod_id: str) -> None: ...
    async def terminate_pod(self, pod_id: str) -> None: ...

    # GPU discovery
    async def get_gpu_types(self) -> list[GpuType]: ...
```

All methods use `@retry` decorator for 429 (rate limit) and 503 errors.

## Handler (`handler.py`)

```python
@component
class RunPodHandler:
    bus: AsyncEventBus
    config: RunPod
    ssh_credentials: SSHCredentialsRegistry

    def __post_init__(self):
        self._clusters: dict[str, RunPodClusterState] = {}
        self._bootstrap_waiters: dict[str, asyncio.Future[bool]] = {}
        self._api_key = self.config.api_key or os.environ.get("RUNPOD_API_KEY")

    @on(ClusterRequested, match=lambda self, e: e.provider == "runpod")
    async def handle_cluster_requested(self, _, event): ...

    @on(ShutdownRequested)
    async def handle_shutdown(self, _, event): ...

    @on(InstanceRequested, match=lambda self, e: e.provider == "runpod")
    async def handle_instance_requested(self, _, event): ...

    @on(InstanceLaunched, match=lambda self, e: e.provider == "runpod")
    async def handle_instance_launched(self, _, event): ...

    @on(BootstrapRequested, match=lambda self, e: e.instance.provider == "runpod")
    async def handle_bootstrap_requested(self, _, event): ...

    @on(BootstrapPhase, audit=False)
    async def handle_bootstrap_phase(self, _, event): ...

    @on(BootstrapFailed, audit=False)
    async def handle_bootstrap_failed(self, _, event): ...
```

## Event Flow

```
ClusterRequested → ClusterProvisioned
    ↓
InstanceRequested → create_pod() → InstanceLaunched
    ↓
InstanceLaunched → poll status → InstanceRunning
    ↓
(InstanceOrchestrator transforms to BootstrapRequested)
    ↓
BootstrapRequested → wait waiter → InstanceBootstrapped
```

## State (`state.py`)

```python
@dataclass
class RunPodClusterState(BaseClusterState):
    cluster_id: str
    spec: PoolSpec

    cloud_type: str = "secure"
    data_center_ids: tuple[str, ...] | None = None

    username: str = "root"  # RunPod pods use root by default
    ssh_key_path: str = ""

    pending_nodes: set[int] = field(default_factory=set)
    pod_ids: dict[int, str] = field(default_factory=dict)  # node_id → pod_id
```

## Integration Points

### `skyward/providers/__init__.py`

```python
from .runpod.config import RunPod
__all__ = [..., "RunPod"]
```

### `skyward/providers/registry.py`

```python
if isinstance(config, RunPod):
    from .runpod.handler import RunPodHandler
    from .runpod import RunPodModule
    return RunPodHandler, RunPodModule, "runpod"
```

### `skyward/providers/runpod/__init__.py`

```python
from .config import RunPod

class RunPodModule(Module):
    @singleton
    @provider
    def provide_client(self, config: RunPod) -> RunPodClient:
        api_key = config.api_key or os.environ.get("RUNPOD_API_KEY", "")
        return RunPodClient(api_key=api_key)

__all__ = ["RunPod", "RunPodModule"]
```

## Usage Example

```python
import skyward as sky

@sky.pool(
    provider=sky.RunPod(data_center_ids=("EU-RO-1",)),
    accelerator="RTX 4090",
    nodes=2,
    allocation="spot",  # spot/on-demand comes from pool spec
)
def main():
    result = train() >> sky
```

## Notes

- RunPod pods run as `root` (unlike AWS which uses `ubuntu` or `ec2-user`)
- `cloud_type` is stored lowercase in config, converted to uppercase when calling API
- SSH key management uses existing `ensure_ssh_key_on_provider` utility
- No new dependencies required
