# RunPod Provider Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement RunPod GPU Pods as a new cloud provider for Skyward.

**Architecture:** Event-driven handler following the existing provider pattern (Verda). Uses httpx for REST API calls with Bearer token auth. Emits standard pipeline events (InstanceLaunched, InstanceRunning) that integrate with InstanceOrchestrator.

**Tech Stack:** httpx (existing), asyncio, TypedDict for API responses

---

## Task 1: Create types.py (API Response Types)

**Files:**
- Create: `skyward/providers/runpod/types.py`

**Step 1: Create the types file**

```python
"""RunPod API response types.

TypedDicts for API responses - no conversion needed.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


# =============================================================================
# Response Types
# =============================================================================


class GpuInfo(TypedDict):
    """GPU configuration from pod response."""

    count: int
    id: NotRequired[str]
    gpuType: NotRequired[str]


class MachineInfo(TypedDict):
    """Machine/host info from pod response."""

    gpuDisplayName: NotRequired[str]
    gpuTypeId: NotRequired[str]
    dataCenterId: NotRequired[str]
    maxDownloadSpeedMbps: NotRequired[int]
    maxUploadSpeedMbps: NotRequired[int]


class PortMapping(TypedDict):
    """Port mapping from pod response."""

    internalPort: int
    externalPort: int
    protocol: str


class PodResponse(TypedDict):
    """Pod response from RunPod API."""

    id: str
    name: NotRequired[str]
    desiredStatus: str  # RUNNING, EXITED, TERMINATED
    publicIp: NotRequired[str | None]
    costPerHr: NotRequired[float]
    adjustedCostPerHr: NotRequired[float]
    machine: NotRequired[MachineInfo]
    gpu: NotRequired[GpuInfo]
    vcpuCount: NotRequired[int]
    memoryInGb: NotRequired[float]
    portMappings: NotRequired[list[PortMapping] | None]
    imageName: NotRequired[str]
    interruptible: NotRequired[bool]


class PodCreateParams(TypedDict, total=False):
    """Parameters for creating a pod."""

    name: str
    imageName: str
    gpuTypeIds: list[str]
    gpuCount: int
    cloudType: str  # SECURE, COMMUNITY
    containerDiskInGb: int
    volumeInGb: int
    volumeMountPath: str
    ports: str
    env: dict[str, str]
    interruptible: bool
    dataCenterIds: list[str]
    minVCPUPerGPU: int
    minRAMPerGPU: int


class GpuTypeResponse(TypedDict):
    """GPU type from RunPod API."""

    id: str
    displayName: str
    memoryInGb: int
    secureCloud: bool
    communityCloud: bool
    lowestPrice: NotRequired[dict[str, float]]


# =============================================================================
# Helper Functions
# =============================================================================


def get_ssh_port(pod: PodResponse) -> int:
    """Extract SSH port from pod port mappings."""
    mappings = pod.get("portMappings")
    if not mappings:
        return 22
    for mapping in mappings:
        if mapping.get("internalPort") == 22:
            return mapping.get("externalPort", 22)
    return 22


def get_gpu_model(pod: PodResponse) -> str:
    """Extract GPU model from pod response."""
    machine = pod.get("machine")
    if machine:
        return machine.get("gpuDisplayName", "")
    gpu = pod.get("gpu")
    if gpu:
        return gpu.get("gpuType", "")
    return ""


def get_gpu_count(pod: PodResponse) -> int:
    """Extract GPU count from pod response."""
    gpu = pod.get("gpu")
    return gpu.get("count", 0) if gpu else 0


__all__ = [
    # Response types
    "GpuInfo",
    "MachineInfo",
    "PortMapping",
    "PodResponse",
    "PodCreateParams",
    "GpuTypeResponse",
    # Helpers
    "get_ssh_port",
    "get_gpu_model",
    "get_gpu_count",
]
```

**Step 2: Verify file was created**

Run: `ls -la skyward/providers/runpod/types.py`

---

## Task 2: Create config.py (Configuration Dataclass)

**Files:**
- Create: `skyward/providers/runpod/config.py`

**Step 1: Create the config file**

```python
"""RunPod provider configuration.

Immutable configuration dataclass for RunPod provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class RunPod:
    """RunPod GPU Pods provider configuration.

    SSH keys are automatically detected from ~/.ssh/id_ed25519.pub or
    ~/.ssh/id_rsa.pub.

    Features:
        - Spot instances: Supports interruptible pricing for cost savings.
        - Auto data center: If not specified, RunPod selects best location.

    Example:
        >>> from skyward.providers.runpod import RunPod
        >>> config = RunPod(spot=True)

    Args:
        api_key: RunPod API key. Falls back to RUNPOD_API_KEY env var.
        cloud_type: Cloud type ("secure" or "community"). Default: secure.
        container_disk_gb: Container disk size in GB. Default: 50.
        volume_gb: Persistent volume size in GB. Default: 20.
        volume_mount_path: Volume mount path. Default: /workspace.
        data_center_ids: Preferred data center IDs. Default: None (auto).
        spot: Use spot/interruptible instances. Default: False.
        ports: Port mappings. Default: "22/tcp".
        provision_timeout: Instance provision timeout in seconds. Default: 300.
        bootstrap_timeout: Bootstrap timeout in seconds. Default: 600.
    """

    api_key: str | None = None
    cloud_type: Literal["secure", "community"] = "secure"
    container_disk_gb: int = 50
    volume_gb: int = 20
    volume_mount_path: str = "/workspace"
    data_center_ids: tuple[str, ...] | None = None
    spot: bool = False
    ports: str = "22/tcp"
    provision_timeout: float = 300.0
    bootstrap_timeout: float = 600.0


# =============================================================================
# Exports
# =============================================================================

__all__ = ["RunPod"]
```

**Step 2: Verify file was created**

Run: `ls -la skyward/providers/runpod/config.py`

---

## Task 3: Create client.py (HTTP Client)

**Files:**
- Create: `skyward/providers/runpod/client.py`

**Step 1: Create the client file**

```python
"""Async HTTP client for RunPod API.

Uses httpx for async HTTP requests with Bearer token authentication.
Returns TypedDicts directly.
"""

from __future__ import annotations

import os
from typing import Any, Self

import httpx

from skyward.retry import on_status_code, retry

from .types import GpuTypeResponse, PodCreateParams, PodResponse

RUNPOD_API_BASE = "https://rest.runpod.io/v1"


class RunPodError(Exception):
    """Error from RunPod API."""


class RunPodClient:
    """Async HTTP client for RunPod API.

    Returns TypedDicts directly from API responses.

    Example:
        async with RunPodClient(api_key="...") as client:
            pod = await client.create_pod({...})
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            base_url=RUNPOD_API_BASE,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client, raising if not in context."""
        if not self._client:
            raise RuntimeError("RunPodClient must be used as async context manager")
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Execute HTTP request and return JSON response."""
        try:
            resp = await self.client.request(method, path, json=json, params=params)
            resp.raise_for_status()
            return resp.json() if resp.content else None
        except httpx.HTTPStatusError as e:
            raise RunPodError(
                f"API error {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise RunPodError(f"Request failed ({type(e).__name__}): {e}") from e

    # =========================================================================
    # Pod Management
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def create_pod(self, params: PodCreateParams) -> PodResponse:
        """Create a new pod."""
        result: PodResponse | None = await self._request("POST", "/pods", json=params)
        if not result:
            raise RunPodError("Failed to create pod: empty response")
        return result

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def get_pod(self, pod_id: str) -> PodResponse | None:
        """Get pod details. Returns None if not found."""
        try:
            result: PodResponse | None = await self._request("GET", f"/pods/{pod_id}")
            return result
        except RunPodError as e:
            if "404" in str(e):
                return None
            raise

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def list_pods(self) -> list[PodResponse]:
        """List all pods."""
        result: list[PodResponse] | None = await self._request("GET", "/pods")
        return result or []

    async def stop_pod(self, pod_id: str) -> None:
        """Stop a pod (pause)."""
        await self._request("POST", f"/pods/{pod_id}/stop")

    async def terminate_pod(self, pod_id: str) -> None:
        """Terminate a pod (destroy)."""
        await self._request("DELETE", f"/pods/{pod_id}")

    # =========================================================================
    # GPU Types
    # =========================================================================

    @retry(on=on_status_code(429, 503), max_attempts=3, base_delay=1.0)
    async def get_gpu_types(self) -> list[GpuTypeResponse]:
        """Get available GPU types."""
        result: list[GpuTypeResponse] | None = await self._request("GET", "/gpu-types")
        return result or []


# =============================================================================
# Utility Functions
# =============================================================================


def get_api_key(config_key: str | None = None) -> str:
    """Get RunPod API key from config or environment.

    Args:
        config_key: API key from config (optional).

    Returns:
        API key string.

    Raises:
        ValueError: If no API key found.
    """
    api_key = config_key or os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise ValueError(
            "RunPod API key not found. Set RUNPOD_API_KEY environment variable "
            "or pass api_key to RunPod config."
        )
    return api_key


__all__ = [
    "RUNPOD_API_BASE",
    "RunPodClient",
    "RunPodError",
    "get_api_key",
]
```

**Step 2: Verify file was created**

Run: `ls -la skyward/providers/runpod/client.py`

---

## Task 4: Create state.py (Cluster State)

**Files:**
- Create: `skyward/providers/runpod/state.py`

**Step 1: Create the state file**

```python
"""RunPod cluster state tracking.

Manages runtime state for RunPod clusters including pod information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skyward.providers.base import BaseClusterState


# =============================================================================
# Cluster State
# =============================================================================


@dataclass
class RunPodClusterState(BaseClusterState):
    """Runtime state for a RunPod cluster.

    Tracks all information needed to manage a cluster's lifecycle
    including launched pods and spec.
    """

    # RunPod-specific fields
    cloud_type: str = "secure"
    data_center_ids: tuple[str, ...] | None = None

    # SSH - RunPod pods use root by default
    username: str = "root"
    ssh_key_path: str = ""

    # Resolved GPU type
    gpu_type_id: str | None = None

    # Pod tracking: node_id -> pod_id
    pod_ids: dict[int, str] = field(default_factory=dict)

    # Pricing info (from pod creation)
    hourly_rate: float = 0.0
    on_demand_rate: float = 0.0

    # Hardware specs
    vcpus: int = 0
    memory_gb: float = 0.0
    gpu_count: int = 0
    gpu_model: str = ""
    gpu_vram_gb: int = 0


# =============================================================================
# Exports
# =============================================================================

__all__ = ["RunPodClusterState"]
```

**Step 2: Verify file was created**

Run: `ls -la skyward/providers/runpod/state.py`

---

## Task 5: Create handler.py (Event Handler)

**Files:**
- Create: `skyward/providers/runpod/handler.py`

**Step 1: Create the handler file**

```python
"""RunPod Provider Handler - event-driven with Event Pipeline.

Uses intermediate events (InstanceLaunched, InstanceRunning) for
decoupled instance lifecycle management.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from dataclasses import field
from typing import TYPE_CHECKING, Any

from loguru import logger

from skyward.app import component, on
from skyward.bus import AsyncEventBus
from skyward.events import (
    BootstrapFailed,
    BootstrapPhase,
    BootstrapRequested,
    ClusterDestroyed,
    ClusterProvisioned,
    ClusterRequested,
    InstanceBootstrapped,
    InstanceLaunched,
    InstanceRequested,
    InstanceRunning,
    ShutdownRequested,
)
from skyward.monitors import SSHCredentialsRegistry
from skyward.providers.ssh_keys import get_ssh_key_path
from skyward.providers.wait import wait_for_ready

from .client import RunPodClient, RunPodError, get_api_key
from .config import RunPod
from .state import RunPodClusterState
from .types import (
    PodCreateParams,
    PodResponse,
    get_gpu_count,
    get_gpu_model,
    get_ssh_port,
)

if TYPE_CHECKING:
    from skyward.spec import PoolSpec


@component
class RunPodHandler:
    """Event-driven RunPod provider using Event Pipeline.

    Flow:
        ClusterRequested -> setup -> ClusterProvisioned
        InstanceRequested -> create_pod -> InstanceLaunched
        InstanceLaunched -> poll running -> InstanceRunning
        BootstrapRequested -> wait bootstrap -> InstanceBootstrapped
        ShutdownRequested -> cleanup -> ClusterDestroyed

    The InstanceOrchestrator handles:
        InstanceRunning -> InstanceProvisioned + BootstrapRequested
    """

    bus: AsyncEventBus
    config: RunPod
    ssh_credentials: SSHCredentialsRegistry

    _clusters: dict[str, RunPodClusterState] = field(default_factory=dict)
    _bootstrap_waiters: dict[str, asyncio.Future[bool]] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Cluster Lifecycle
    # -------------------------------------------------------------------------

    @on(ClusterRequested, match=lambda self, e: e.provider == "runpod")
    async def handle_cluster_requested(self, _: Any, event: ClusterRequested) -> None:
        """Provision RunPod infrastructure for a new cluster."""
        logger.info(f"RunPod: Provisioning cluster for {event.spec.nodes} nodes")

        cluster_id = f"runpod-{uuid.uuid4().hex[:8]}"

        state = RunPodClusterState(
            cluster_id=cluster_id,
            spec=event.spec,
            cloud_type=self.config.cloud_type,
            data_center_ids=self.config.data_center_ids,
        )
        self._clusters[cluster_id] = state

        # Register SSH credentials for EventStreamer
        ssh_key_path = get_ssh_key_path()
        state.ssh_key_path = ssh_key_path
        self.ssh_credentials.register(cluster_id, state.username, ssh_key_path)

        # Resolve GPU type
        gpu_type_id = await self._resolve_gpu_type(event.spec)
        state.gpu_type_id = gpu_type_id

        self.bus.emit(
            ClusterProvisioned(
                request_id=event.request_id,
                cluster_id=cluster_id,
                provider="runpod",
            )
        )

    @on(ShutdownRequested)
    async def handle_shutdown_requested(self, _: Any, event: ShutdownRequested) -> None:
        """Terminate all pods in a cluster."""
        cluster = self._clusters.pop(event.cluster_id, None)
        if not cluster:
            return

        logger.info(f"RunPod: Shutting down cluster {event.cluster_id}")

        api_key = get_api_key(self.config.api_key)
        async with RunPodClient(api_key) as client:
            for pod_id in cluster.pod_ids.values():
                with suppress(Exception):
                    await client.terminate_pod(pod_id)

        self.bus.emit(ClusterDestroyed(cluster_id=event.cluster_id))

    # -------------------------------------------------------------------------
    # Instance Lifecycle - Event Pipeline
    # -------------------------------------------------------------------------

    @on(InstanceRequested, match=lambda self, e: e.provider == "runpod")
    async def handle_instance_requested(self, _: Any, event: InstanceRequested) -> None:
        """Launch RunPod pod and emit InstanceLaunched."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster or not cluster.gpu_type_id:
            return

        logger.info(f"RunPod: Launching pod for node {event.node_id}")

        api_key = get_api_key(self.config.api_key)
        use_spot = self.config.spot or cluster.spec.allocation in (
            "spot",
            "spot-if-available",
        )

        # Build pod creation params
        params: PodCreateParams = {
            "name": f"skyward-{cluster.cluster_id}-{event.node_id}",
            "imageName": self._get_image_name(cluster.spec),
            "gpuTypeIds": [cluster.gpu_type_id],
            "gpuCount": cluster.spec.accelerator_count or 1,
            "cloudType": self.config.cloud_type.upper(),
            "containerDiskInGb": self.config.container_disk_gb,
            "volumeInGb": self.config.volume_gb,
            "volumeMountPath": self.config.volume_mount_path,
            "ports": self.config.ports,
            "interruptible": use_spot,
        }

        if self.config.data_center_ids:
            params["dataCenterIds"] = list(self.config.data_center_ids)

        try:
            async with RunPodClient(api_key) as client:
                pod = await client.create_pod(params)
        except RunPodError as e:
            logger.error(f"RunPod: Failed to create pod: {e}")
            return

        pod_id = pod["id"]
        cluster.pod_ids[event.node_id] = pod_id
        cluster.pending_nodes.add(event.node_id)

        # Emit intermediate event - pod created, waiting for running
        self.bus.emit(
            InstanceLaunched(
                request_id=event.request_id,
                cluster_id=event.cluster_id,
                node_id=event.node_id,
                provider="runpod",
                instance_id=pod_id,
            )
        )

    @on(InstanceLaunched, match=lambda self, e: e.provider == "runpod")
    async def handle_instance_launched(self, _: Any, event: InstanceLaunched) -> None:
        """Wait for pod to be running and emit InstanceRunning."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        api_key = get_api_key(self.config.api_key)
        use_spot = self.config.spot or cluster.spec.allocation in (
            "spot",
            "spot-if-available",
        )

        try:
            async with RunPodClient(api_key) as client:
                pod = await wait_for_ready(
                    poll_fn=lambda: client.get_pod(event.instance_id),
                    ready_check=lambda p: (
                        p is not None
                        and p.get("desiredStatus") == "RUNNING"
                        and bool(p.get("publicIp"))
                    ),
                    terminal_check=lambda p: (
                        p is not None and p.get("desiredStatus") == "TERMINATED"
                    ),
                    timeout=self.config.provision_timeout,
                    interval=5.0,
                    description=f"RunPod pod {event.instance_id}",
                )
        except TimeoutError:
            logger.error(f"RunPod: Pod {event.instance_id} did not become ready")
            return

        if not pod:
            logger.error(f"RunPod: Pod {event.instance_id} not found")
            return

        # Extract info from pod response
        ip = pod.get("publicIp", "")
        ssh_port = get_ssh_port(pod)
        hourly_rate = pod.get("costPerHr", 0.0)
        adjusted_rate = pod.get("adjustedCostPerHr", hourly_rate)

        # Update cluster state with pricing
        cluster.hourly_rate = adjusted_rate
        cluster.on_demand_rate = hourly_rate
        cluster.gpu_count = get_gpu_count(pod)
        cluster.gpu_model = get_gpu_model(pod)
        cluster.vcpus = pod.get("vcpuCount", 0)
        cluster.memory_gb = pod.get("memoryInGb", 0.0)

        # Emit InstanceRunning - InstanceOrchestrator will handle the rest
        self.bus.emit(
            InstanceRunning(
                request_id=event.request_id,
                cluster_id=event.cluster_id,
                node_id=event.node_id,
                provider="runpod",
                instance_id=event.instance_id,
                ip=ip,
                private_ip=None,  # RunPod doesn't provide private IP
                ssh_port=ssh_port,
                spot=use_spot,
                # Pricing info
                hourly_rate=adjusted_rate,
                on_demand_rate=hourly_rate,
                billing_increment=1,  # RunPod bills per-minute
                # Instance details
                instance_type=cluster.gpu_type_id or "",
                gpu_count=cluster.gpu_count,
                gpu_model=cluster.gpu_model,
                # Hardware specs
                vcpus=cluster.vcpus,
                memory_gb=cluster.memory_gb,
                gpu_vram_gb=cluster.gpu_vram_gb,
                # Location
                region=self.config.data_center_ids[0] if self.config.data_center_ids else "",
            )
        )

    @on(BootstrapRequested, match=lambda self, e: e.instance.provider == "runpod")
    async def handle_bootstrap_requested(self, _: Any, event: BootstrapRequested) -> None:
        """Wait for bootstrap completion. EventStreamer handles streaming."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        instance_id = event.instance.id
        logger.debug(f"RunPod: Waiting for bootstrap completion on {instance_id}")

        # Create waiter for bootstrap completion (signaled by BootstrapPhase handler)
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[bool] = loop.create_future()
        self._bootstrap_waiters[instance_id] = waiter

        try:
            # Wait for completion - EventStreamer will emit BootstrapPhase events
            success = await asyncio.wait_for(
                waiter, timeout=self.config.bootstrap_timeout
            )

            if success:
                # Install local skyward wheel if skyward_source == 'local'
                if cluster.spec.image and cluster.spec.image.skyward_source == "local":
                    await self._install_local_skyward(event.instance, cluster)

                # Track instance in cluster state
                cluster.add_instance(event.instance)
                # Emit InstanceBootstrapped - Node will signal NodeReady
                self.bus.emit(InstanceBootstrapped(instance=event.instance))
            else:
                logger.error(f"RunPod: Bootstrap failed on {instance_id}")

        except asyncio.TimeoutError:
            logger.error(f"RunPod: Bootstrap timed out on {instance_id}")
        finally:
            self._bootstrap_waiters.pop(instance_id, None)

    @on(BootstrapPhase, match=lambda self, e: e.instance.provider == "runpod", audit=False)
    async def handle_bootstrap_phase(self, _: Any, event: BootstrapPhase) -> None:
        """Handle bootstrap phase events from EventStreamer."""
        # Only care about bootstrap phase completion/failure
        if event.phase != "bootstrap" or event.event not in ("completed", "failed"):
            return

        instance_id = event.instance.id
        waiter = self._bootstrap_waiters.get(instance_id)
        if waiter and not waiter.done():
            waiter.set_result(event.event == "completed")

    @on(BootstrapFailed, match=lambda self, e: e.instance.provider == "runpod", audit=False)
    async def handle_bootstrap_failed(self, _: Any, event: BootstrapFailed) -> None:
        """Handle bootstrap failure from EventStreamer."""
        instance_id = event.instance.id
        waiter = self._bootstrap_waiters.get(instance_id)
        if waiter and not waiter.done():
            waiter.set_result(False)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    async def _resolve_gpu_type(self, spec: PoolSpec) -> str:
        """Resolve GPU type ID from spec accelerator name."""
        if not spec.accelerator_name:
            raise RuntimeError("RunPod requires accelerator_name in spec")

        api_key = get_api_key(self.config.api_key)
        async with RunPodClient(api_key) as client:
            gpu_types = await client.get_gpu_types()

        # Filter by cloud type
        is_secure = self.config.cloud_type == "secure"
        available = [
            g for g in gpu_types
            if (is_secure and g.get("secureCloud")) or (not is_secure and g.get("communityCloud"))
        ]

        # Match by name (case-insensitive, partial match)
        requested = spec.accelerator_name.upper()
        for gpu in available:
            display_name = gpu.get("displayName", "").upper()
            gpu_id = gpu.get("id", "").upper()
            if requested in display_name or requested in gpu_id:
                logger.info(f"RunPod: Selected GPU type {gpu['id']} ({gpu.get('displayName')})")
                return gpu["id"]

        available_names = [g.get("displayName", g["id"]) for g in available]
        raise RuntimeError(
            f"No GPU type matches '{spec.accelerator_name}'. "
            f"Available: {', '.join(available_names)}"
        )

    def _get_image_name(self, spec: PoolSpec) -> str:
        """Get container image name from spec."""
        # Default to PyTorch CUDA image
        if spec.image and hasattr(spec.image, "container_image"):
            return spec.image.container_image
        return "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

    async def _install_local_skyward(
        self,
        info: Any,
        cluster: RunPodClusterState,
    ) -> None:
        """Install local skyward wheel."""
        from skyward.providers.bootstrap import install_local_skyward, wait_for_ssh

        transport = await wait_for_ssh(
            host=info.ip,
            user=cluster.username,
            key_path=cluster.ssh_key_path,
            port=info.ssh_port,
            timeout=60.0,
            log_prefix="RunPod: ",
        )

        try:
            await install_local_skyward(
                transport=transport,
                info=info,
                log_prefix="RunPod: ",
            )
        finally:
            await transport.close()


__all__ = ["RunPodHandler"]
```

**Step 2: Verify file was created**

Run: `ls -la skyward/providers/runpod/handler.py`

---

## Task 6: Create __init__.py (Module Exports)

**Files:**
- Create: `skyward/providers/runpod/__init__.py`

**Step 1: Create the init file**

```python
"""RunPod GPU Pods provider for Skyward.

RunPod provides GPU cloud computing with both Secure Cloud (dedicated)
and Community Cloud (marketplace) options.

NOTE: Only config classes are imported at package level to avoid deps.
For handlers and modules, import explicitly:

    from skyward.providers.runpod.handler import RunPodHandler
    from skyward.providers.runpod.client import RunPodClient

Environment Variables:
    RUNPOD_API_KEY: API key (required if not passed directly)
"""

# Only config - no heavy dependencies
from .config import RunPod


# Lazy imports for backward compatibility
def __getattr__(name: str):
    if name == "RunPodHandler":
        from .handler import RunPodHandler

        return RunPodHandler
    if name in ("RunPodClient", "RunPodError"):
        from .client import RunPodClient, RunPodError

        if name == "RunPodClient":
            return RunPodClient
        return RunPodError
    if name == "RunPodClusterState":
        from .state import RunPodClusterState

        return RunPodClusterState
    if name == "RunPodModule":
        from injector import Module, provider, singleton

        from .client import RunPodClient, get_api_key

        class RunPodModule(Module):
            """DI module for RunPod provider."""

            @singleton
            @provider
            def provide_runpod_client(self, config: RunPod) -> RunPodClient:
                """Provide RunPod API client."""
                api_key = get_api_key(config.api_key)
                return RunPodClient(api_key)

        return RunPodModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config (always available)
    "RunPod",
    # Lazy (loaded on demand)
    "RunPodClient",
    "RunPodError",
    "RunPodHandler",
    "RunPodClusterState",
    "RunPodModule",
]
```

**Step 2: Verify file was created**

Run: `ls -la skyward/providers/runpod/__init__.py`

---

## Task 7: Update providers/__init__.py

**Files:**
- Modify: `skyward/providers/__init__.py`

**Step 1: Add RunPod import**

Add after the Verda import:

```python
from .runpod.config import RunPod
```

**Step 2: Add to __all__**

Update `__all__` to include "RunPod":

```python
__all__ = [
    # Config classes only - handlers must be imported explicitly
    "AWS",
    "VastAI",
    "Verda",
    "RunPod",
]
```

**Step 3: Verify changes**

Run: `python -c "from skyward.providers import RunPod; print(RunPod)"`
Expected: `<class 'skyward.providers.runpod.config.RunPod'>`

---

## Task 8: Update providers/registry.py

**Files:**
- Modify: `skyward/providers/registry.py`

**Step 1: Add RunPod import**

Add after the Verda import:

```python
from .runpod.config import RunPod
```

**Step 2: Add RunPod case**

Add before the `raise ValueError`:

```python
if isinstance(config, RunPod):
    from .runpod.handler import RunPodHandler
    from .runpod import RunPodModule

    return RunPodHandler, RunPodModule, "runpod"
```

**Step 3: Update error message**

Update the error message to include RunPod:

```python
raise ValueError(
    f"No provider registered for {type(config).__name__}. "
    f"Available providers: AWS, VastAI, Verda, RunPod"
)
```

**Step 4: Verify changes**

Run: `python -c "from skyward.providers.registry import get_provider_for_config; from skyward.providers import RunPod; print(get_provider_for_config(RunPod()))"`
Expected: `(<class '...RunPodHandler'>, <class '...RunPodModule'>, 'runpod')`

---

## Task 9: Verify Full Integration

**Step 1: Test imports work**

Run:
```bash
cd /Users/gabrielfrancisco/workspace/skyward && uv run python -c "
from skyward.providers import RunPod
from skyward.providers.registry import get_provider_for_config

config = RunPod(spot=True, data_center_ids=('EU-RO-1',))
handler_cls, module_cls, name = get_provider_for_config(config)

print(f'Config: {config}')
print(f'Handler: {handler_cls}')
print(f'Module: {module_cls}')
print(f'Name: {name}')
"
```

Expected output:
```
Config: RunPod(api_key=None, cloud_type='secure', container_disk_gb=50, volume_gb=20, volume_mount_path='/workspace', data_center_ids=('EU-RO-1',), spot=True, ports='22/tcp', provision_timeout=300.0, bootstrap_timeout=600.0)
Handler: <class 'skyward.providers.runpod.handler.RunPodHandler'>
Module: <class 'skyward.providers.runpod.RunPodModule'>
Name: runpod
```

**Step 2: Run type checker**

Run: `cd /Users/gabrielfrancisco/workspace/skyward && uv run pyright skyward/providers/runpod/`

Expected: No errors (or only pre-existing ones)

---

## Summary

| Task | File | Action |
|------|------|--------|
| 1 | `skyward/providers/runpod/types.py` | Create |
| 2 | `skyward/providers/runpod/config.py` | Create |
| 3 | `skyward/providers/runpod/client.py` | Create |
| 4 | `skyward/providers/runpod/state.py` | Create |
| 5 | `skyward/providers/runpod/handler.py` | Create |
| 6 | `skyward/providers/runpod/__init__.py` | Create |
| 7 | `skyward/providers/__init__.py` | Modify |
| 8 | `skyward/providers/registry.py` | Modify |
| 9 | (verification) | Test |
