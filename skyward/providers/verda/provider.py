"""Verda Cloud provider for Skyward GPU instances."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from loguru import logger
from verda import VerdaClient
from verda.constants import Actions

from skyward.accelerators import AcceleratorSpec
from skyward.core.callback import emit
from skyward.core.events import (
    InstanceLaunching,
    InstanceProvisioned,
    NetworkReady,
    ProviderName,
    ProvisionedInstance,
    ProvisioningCompleted,
    ProvisioningStarted,
    RegionAutoSelected,
)
from skyward.internal.decorators import audit
from skyward.providers.base import (
    SSHKeyInfo,
    SSHKeyManager,
    get_private_key_path,
    poll_instances,
)
from skyward.providers.ssh import SSHConfig
from skyward.spec.allocation import _AllocationOnDemand, normalize_allocation
from skyward.types import (
    ComputeSpec,
    Instance,
    InstanceSpec,
    Provider,
    parse_memory_mb,
    select_instances,
)

from .discovery import (
    NoAvailableRegionError,
    fetch_available_instances,
    find_available_region,
    get_availability,
)

# =============================================================================
# Internal Types
# =============================================================================

@dataclass
class _VerdaInstance:
    """Internal representation of a Verda instance during provisioning."""

    id: str
    name: str
    ip: str = ""
    private_ip: str = ""
    status: str = ""


# =============================================================================
# SSH Key Management (using SSHKeyManager)
# =============================================================================


def _verda_get_fingerprint(key: Any) -> str:
    """Extract fingerprint from Verda SSH key object."""
    if hasattr(key, "fingerprint"):
        return key.fingerprint
    # Fallback: compute from key content
    if hasattr(key, "key"):
        from skyward.providers.base import compute_fingerprint

        try:
            return compute_fingerprint(key.key.strip())
        except Exception:
            pass
    return ""


def _verda_get_id(key: Any) -> str | None:
    """Extract ID from Verda SSH key object."""
    return getattr(key, "id", None)


# Verda-specific SSHKeyManager configuration
_verda_ssh_key_manager: SSHKeyManager[VerdaClient] = SSHKeyManager(
    list_keys=lambda c: c.ssh_keys.get(),
    create_key=lambda c, name, pub: c.ssh_keys.create(name=name, key=pub),
    get_fingerprint=_verda_get_fingerprint,
    get_id=_verda_get_id,
)


def _get_or_create_ssh_key(client: VerdaClient) -> SSHKeyInfo:
    """Get or create SSH key on Verda using SSHKeyManager."""
    try:
        return _verda_ssh_key_manager.get_or_create(client)
    except Exception as e:
        # Handle "already exists" race condition
        if "already" in str(e).lower():
            return _verda_ssh_key_manager.get_or_create(client)
        raise


# =============================================================================
# Lifecycle Helpers
# =============================================================================


def _wait_for_running(client: VerdaClient, instances: list[_VerdaInstance], timeout: float) -> None:
    """Wait for all instances to become running and get IPs."""

    def fetch_status(vinst: _VerdaInstance) -> tuple[str, dict[str, Any]]:
        all_instances = client.instances.get()
        for inst in all_instances:
            if inst.id == vinst.id:
                ip = getattr(inst, "ip", "") or ""
                # Treat missing IP as not ready yet
                status = inst.status if ip else "pending"
                return status, {"ip": ip, "status": inst.status}
        return "not_found", {}

    def update_instance(vinst: _VerdaInstance, info: dict[str, Any]) -> None:
        vinst.ip = info.get("ip", "")
        vinst.status = info.get("status", "")

    poll_instances(
        instances=instances,
        fetch_status=fetch_status,
        target_status="running",
        update_instance=update_instance,
        timeout=timeout,
    )


# =============================================================================
# Provider
# =============================================================================


@dataclass(frozen=True, slots=True)
class Verda:
    """Verda Cloud provider configuration.

    SSH keys are automatically detected from ~/.ssh/id_ed25519.pub or
    ~/.ssh/id_rsa.pub and registered on Verda if needed.

    Features:
        - Auto-region discovery: If the requested region doesn't have capacity,
          automatically finds another region with availability.
        - Spot instances: Supports spot pricing for cost savings.

    Example:
        from skyward import ComputePool, Verda, compute

        @compute
        def train(data):
            return model.fit(data)

        pool = ComputePool(
            provider=Verda(region="FIN-01"),
            accelerator="H100-80GB",
            pip=["torch"],
        )

        with pool:
            result = train(data) >> pool

    Environment Variables:
        VERDA_CLIENT_ID: API client ID (required if not passed directly)
        VERDA_CLIENT_SECRET: API client secret (required if not passed directly)
    """

    region: str = "FIN-01"
    client_id: str | None = None
    client_secret: str | None = None
    ssh_key_id: str | None = None
    instance_timeout: int = 300

    def build(self) -> VerdaProvider:
        """Build a stateful VerdaProvider from this configuration."""
        return VerdaProvider(
            region=self.region,
            client_id=self.client_id,
            client_secret=self.client_secret,
            ssh_key_id=self.ssh_key_id,
            instance_timeout=self.instance_timeout,
        )


class VerdaProvider(Provider):
    """Stateful Verda provider service.

    This class manages GPU instances and maintains runtime state.
    Created by Verda.build() or automatically by ComputePool.

    Implements the Provider protocol with provision/setup/shutdown lifecycle.
    """

    def __init__(
        self,
        region: str,
        client_id: str | None,
        client_secret: str | None,
        ssh_key_id: str | None,
        instance_timeout: int,
    ) -> None:
        self.region = region
        self.client_id = client_id
        self.client_secret = client_secret
        self.ssh_key_id = ssh_key_id
        self.instance_timeout = instance_timeout

        # Mutable runtime state
        self._resolved_ssh_key_id: str | None = None
        self._username: str = "root"
        self._instances: list[_VerdaInstance] = []
        self._cluster_id: str | None = None

    @property
    def name(self) -> str:
        return "verda"

    @property
    def _client_id(self) -> str:
        """Get client ID from config or environment."""
        client_id = self.client_id or os.environ.get("VERDA_CLIENT_ID")
        if not client_id:
            raise ValueError(
                "Verda client ID not provided. "
                "Set VERDA_CLIENT_ID environment variable or pass client_id to Verda()"
            )
        return client_id

    @property
    def _client_secret(self) -> str:
        """Get client secret from config or environment."""
        client_secret = self.client_secret or os.environ.get("VERDA_CLIENT_SECRET")
        if not client_secret:
            raise ValueError(
                "Verda client secret not provided. "
                "Set VERDA_CLIENT_SECRET environment variable or pass client_secret to Verda()"
            )
        return client_secret

    def _get_client(self) -> VerdaClient:
        """Create authenticated Verda client."""
        return VerdaClient(self._client_id, self._client_secret)

    @audit("Provisioning")
    def provision(self, compute: ComputeSpec) -> tuple[Instance, ...]:
        """Provision Verda GPU instances."""
        cluster_id = str(uuid.uuid4())[:8]
        logger.debug(f"Cluster ID: {cluster_id}, nodes: {compute.nodes}, region: {self.region}")
        emit(ProvisioningStarted())

        client = self._get_client()

        ssh_key_info = _get_or_create_ssh_key(client)
        self._resolved_ssh_key_id = ssh_key_info.id

        emit(NetworkReady(region=self.region))

        acc = AcceleratorSpec.from_value(compute.accelerator)
        allocation = normalize_allocation(compute.allocation)
        prefer_spot = not isinstance(allocation, _AllocationOnDemand)

        if compute.machine is not None:
            # Direct instance type override
            flavor = compute.machine
            # Look up spec for metadata (supported_os)
            specs = self.available_instances()
            spec = next((s for s in specs if s.name == flavor), None)
        else:
            # Infer from resources with availability filtering
            accelerator_type = acc.accelerator if acc else None
            requested_gpu_count = acc.count if acc else 1

            # 1. Get all matching candidates (sorted by price)
            candidates = select_instances(
                self.available_instances(),
                cpu=compute.cpu or 1,
                memory_mb=parse_memory_mb(compute.memory),
                accelerator=accelerator_type,
                accelerator_count=requested_gpu_count,
                prefer_spot=prefer_spot,
            )

            # 2. Get availability for spot/on-demand
            availability = get_availability(client, is_spot=prefer_spot)
            available_types: set[str] = set()
            for region_types in availability.values():
                available_types.update(region_types)

            # 3. Filter to only available candidates
            available_candidates = [c for c in candidates if c.name in available_types]

            if not available_candidates:
                raise NoAvailableRegionError(
                    candidates[0].name if candidates else "unknown",
                    prefer_spot,
                    list(availability.keys()),
                )

            # 4. Use best available (first = cheapest)
            spec = available_candidates[0]
            logger.debug(
                f"Selected {spec.name} from {len(candidates)} candidates "
                f"({len(available_candidates)} available)"
            )

        # Determine instance type name
        instance_type = compute.machine if compute.machine is not None else spec.name

        supported_os = spec.metadata.get("supported_os", ()) if spec else ()
        if compute.accelerator or compute.machine:
            default_cuda_image = "ubuntu-22.04-cuda-12.1"
            os_image = next(
                filter(lambda os: "cuda" in os, supported_os),
                supported_os[0] if supported_os else default_cuda_image,
            )
        else:
            os_image = supported_os[0] if supported_os else "ubuntu-22.04"
        username = "root"
        logger.debug(f"Using image: {os_image}, instance type: {instance_type}")
        self._username = username

        # Generate bootstrap script using Image.bootstrap()
        user_data = compute.image.bootstrap(ttl=compute.timeout)

        script_name = f"skyward-bootstrap-{cluster_id}"
        startup_script = client.startup_scripts.create(name=script_name, script=user_data)

        use_spot = prefer_spot

        # Auto-region discovery
        actual_region = find_available_region(
            client,
            instance_type,
            is_spot=use_spot,
            preferred_region=self.region,
        )

        if actual_region != self.region and spec:
            emit(
                RegionAutoSelected(
                    requested_region=self.region,
                    selected_region=actual_region,
                    spec=spec,
                    provider=ProviderName.Verda,
                )
            )

        emit(
            InstanceLaunching(
                count=compute.nodes,
                candidates=(spec,) if spec else (),
                provider=ProviderName.Verda,
            )
        )

        verda_instances: list[_VerdaInstance] = []
        for i in range(compute.nodes):
            instance_name = f"skyward-{cluster_id}-{i}"
            logger.debug(f"Creating instance {instance_name} (type={instance_type}, spot={use_spot})")

            created = client.instances.create(
                instance_type=instance_type,
                image=os_image,
                ssh_key_ids=[ssh_key_info.id],
                hostname=instance_name,
                description=f"Skyward managed instance - cluster {cluster_id}",
                startup_script_id=startup_script.id,
                location=actual_region,
                is_spot=use_spot,
            )

            verda_instances.append(
                _VerdaInstance(
                    id=created.id,
                    name=instance_name,
                    ip="",
                    status="provisioning",
                )
            )
            logger.debug(f"Instance {instance_name} created with id {created.id}")

        logger.debug(f"Waiting for {len(verda_instances)} instances to become running...")
        _wait_for_running(client, verda_instances, timeout=300)

        self._instances = verda_instances
        self._cluster_id = cluster_id

        instances: list[Instance] = []
        provisioned_instances: list[ProvisionedInstance] = []
        key_path = get_private_key_path()

        def _make_destroy_fn(inst_id: str) -> Callable[[], None]:
            def destroy() -> None:
                with suppress(Exception):
                    self._get_client().instances.action(inst_id, Actions.DELETE)

            return destroy

        for i, vinst in enumerate(verda_instances):
            instance = Instance(
                id=str(vinst.id),
                provider=self,
                ssh=SSHConfig(
                    host=vinst.ip,
                    username=username,
                    key_path=key_path,
                ),
                spot=use_spot,
                private_ip=vinst.private_ip or vinst.ip,
                public_ip=vinst.ip,
                node=i,
                metadata=frozenset(
                    [
                        ("cluster_id", cluster_id),
                        ("instance_id", vinst.id),
                        ("instance_ip", vinst.ip),
                        ("username", username),
                        ("accelerator_count", str(spec.accelerator_count)),
                    ]
                ),
                _destroy_fn=_make_destroy_fn(vinst.id),
            )
            instances.append(instance)

            provisioned = ProvisionedInstance(
                instance_id=str(vinst.id),
                node=i,
                provider=ProviderName.Verda,
                spot=use_spot,
                spec=spec,
                ip=vinst.ip,
            )
            provisioned_instances.append(provisioned)
            emit(InstanceProvisioned(instance=provisioned))

        emit(
            ProvisioningCompleted(
                instances=tuple(provisioned_instances),
                provider=ProviderName.Verda,
                region=actual_region,
            )
        )
        return tuple(instances)

    def discover_peers(self, cluster_id: str) -> tuple[Instance, ...]:
        """Discover peer instances in a cluster via Verda API."""
        from skyward.providers.base import assign_node_indices

        client = self._get_client()
        hostname_pattern = f"skyward-{cluster_id}-"
        all_instances = client.instances.get()
        key_path = get_private_key_path()

        instances = []
        for vinst in all_instances:
            if not hasattr(vinst, "hostname") or not vinst.hostname.startswith(hostname_pattern):
                continue

            if vinst.status != "running":
                continue

            ip = getattr(vinst, "ip", "") or ""

            instances.append(
                Instance(
                    id=str(vinst.id),
                    provider=self,
                    ssh=SSHConfig(host=ip, username=self._username, key_path=key_path),
                    spot=getattr(vinst, "is_spot", False),
                    private_ip=ip,
                    public_ip=ip,
                    node=0,
                    metadata=frozenset([
                        ("cluster_id", cluster_id),
                        ("instance_id", vinst.id),
                        ("instance_ip", ip),
                        ("username", self._username),
                    ]),
                )
            )

        return assign_node_indices(instances)

    def available_instances(self) -> tuple[InstanceSpec, ...]:
        """List all available instance types."""
        client = self._get_client()
        return fetch_available_instances(client)

    def cleanup(self) -> None:
        """No-op cleanup for Verda."""
        pass
