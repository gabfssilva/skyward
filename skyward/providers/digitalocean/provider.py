"""DigitalOcean provider for Skyward GPU instances."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydo import Client

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
    select_instance,
)

from .discovery import fetch_available_instances, get_gpu_image

if TYPE_CHECKING:
    pass


# =============================================================================
# Internal Types
# =============================================================================


@dataclass
class _Droplet:
    """Internal representation of a DigitalOcean Droplet."""

    id: int
    name: str
    ip: str = ""
    private_ip: str = ""


# =============================================================================
# SSH Key Management (using SSHKeyManager)
# =============================================================================

# DigitalOcean-specific SSHKeyManager configuration
_do_ssh_key_manager: SSHKeyManager[Client] = SSHKeyManager(
    list_keys=lambda c: c.ssh_keys.list().get("ssh_keys", []),
    create_key=lambda c, name, pub: c.ssh_keys.create(body={"name": name, "public_key": pub}).get(
        "ssh_key", {}
    ),
    get_fingerprint=lambda k: k.get("fingerprint", ""),
    get_id=lambda k: None,  # DigitalOcean uses fingerprint, not ID
)


def _get_or_create_ssh_key(client: Client) -> SSHKeyInfo:
    """Get or create SSH key on DigitalOcean using SSHKeyManager."""
    try:
        return _do_ssh_key_manager.get_or_create(client)
    except Exception as e:
        # Handle "already been taken" race condition
        if "already been taken" in str(e).lower():
            # Key was created between list and create, re-fetch
            return _do_ssh_key_manager.get_or_create(client)
        raise


# =============================================================================
# Lifecycle Helpers
# =============================================================================


def _wait_for_active(client: Client, droplets: list[_Droplet], timeout: float) -> None:
    """Wait for all droplets to become active and get IPs."""

    def fetch_status(droplet: _Droplet) -> tuple[str, dict[str, Any]]:
        resp = client.droplets.get(droplet_id=droplet.id)
        data = resp["droplet"]
        info: dict[str, Any] = {}
        for network in data["networks"]["v4"]:
            if network["type"] == "public":
                info["ip"] = network["ip_address"]
            elif network["type"] == "private":
                info["private_ip"] = network["ip_address"]
        # Treat missing IP as not ready yet
        status = data["status"] if info.get("ip") else "pending"
        return status, info

    def update_droplet(droplet: _Droplet, info: dict[str, Any]) -> None:
        droplet.ip = info.get("ip", "")
        droplet.private_ip = info.get("private_ip", "")

    poll_instances(
        instances=droplets,
        fetch_status=fetch_status,
        target_status="active",
        update_instance=update_droplet,
        timeout=timeout,
    )


# =============================================================================
# Provider
# =============================================================================


@dataclass(frozen=True, slots=True)
class DigitalOcean:
    """DigitalOcean provider configuration.

    SSH keys are automatically detected from ~/.ssh/id_ed25519.pub or
    ~/.ssh/id_rsa.pub and registered on DigitalOcean if needed.

    Example:
        from skyward import ComputePool, DigitalOcean, compute

        @compute
        def train(data):
            return model.fit(data)

        pool = ComputePool(
            provider=DigitalOcean(region="tor1"),
            accelerator="H100-80GB",
            pip=["torch"],
        )

        with pool:
            result = train(data) >> pool

    Environment Variables:
        DIGITALOCEAN_TOKEN: API token (required if not passed directly)
        DIGITALOCEAN_SSH_KEY_FINGERPRINT: SSH key fingerprint (optional)
    """

    region: str = "nyc3"
    token: str | None = None
    ssh_key_fingerprint: str | None = None
    instance_timeout: int = 300

    def build(self) -> DigitalOceanProvider:
        """Build a stateful DigitalOceanProvider from this configuration."""
        return DigitalOceanProvider(
            region=self.region,
            token=self.token,
            ssh_key_fingerprint=self.ssh_key_fingerprint,
            instance_timeout=self.instance_timeout,
        )


class DigitalOceanProvider(Provider):
    """Stateful DigitalOcean provider service.

    This class manages Droplets and maintains runtime state.
    Created by DigitalOcean.build() or automatically by ComputePool.

    Implements the Provider protocol with provision/setup/shutdown lifecycle.
    """

    def __init__(
        self,
        region: str,
        token: str | None,
        ssh_key_fingerprint: str | None,
        instance_timeout: int,
    ) -> None:
        self.region = region
        self.token = token
        self.ssh_key_fingerprint = ssh_key_fingerprint
        self.instance_timeout = instance_timeout

        # Mutable runtime state
        self._resolved_fingerprint: str | None = None
        self._username: str = "root"
        self._droplets: list[_Droplet] = []

    @property
    def name(self) -> str:
        return "digitalocean"

    @property
    def _token(self) -> str:
        """Get API token from config or environment."""
        token = self.token or os.environ.get("DIGITALOCEAN_TOKEN")
        if not token:
            raise ValueError(
                "DigitalOcean API token not provided. "
                "Set DIGITALOCEAN_TOKEN environment variable or pass token to DigitalOcean()"
            )
        return token

    def _get_client(self) -> Client:
        """Create authenticated pydo client."""
        return Client(token=self._token)

    def _ensure_ssh_key(self) -> str:
        """Get SSH key fingerprint, auto-detecting from local keys if needed."""
        if self.ssh_key_fingerprint:
            return self.ssh_key_fingerprint

        env_fingerprint = os.environ.get("DIGITALOCEAN_SSH_KEY_FINGERPRINT")
        if env_fingerprint:
            return env_fingerprint

        if self._resolved_fingerprint is not None:
            return self._resolved_fingerprint

        client = self._get_client()
        key_info = _get_or_create_ssh_key(client)
        self._resolved_fingerprint = key_info.fingerprint
        return key_info.fingerprint

    @audit("Provisioning")
    def provision(self, compute: ComputeSpec) -> tuple[Instance, ...]:
        """Provision DigitalOcean Droplets."""
        cluster_id = str(uuid.uuid4())[:8]
        logger.debug(f"Cluster ID: {cluster_id}, nodes: {compute.nodes}, region: {self.region}")
        emit(ProvisioningStarted())

        fingerprint = self._ensure_ssh_key()
        emit(NetworkReady(region=self.region))

        client = self._get_client()

        acc = AcceleratorSpec.from_value(compute.accelerator)
        allocation = normalize_allocation(compute.allocation)
        prefer_spot = not isinstance(allocation, _AllocationOnDemand)

        if compute.machine is not None:
            # Direct instance type override
            size = compute.machine
            # Look up spec for metadata (accelerator_count for image selection)
            specs = self.available_instances()
            spec = next((s for s in specs if s.name == size), None)
            if spec:
                accelerator_count = spec.accelerator_count
            elif acc and callable(acc.count):
                accelerator_count = 1  # Default when callable, actual comes from spec
            else:
                accelerator_count = acc.count if acc else 0
        else:
            # Infer from resources (current behavior)
            accelerator_type = acc.accelerator if acc else None
            requested_gpu_count = acc.count if acc else 1

            spec = select_instance(
                self.available_instances(),
                cpu=compute.cpu or 1,
                memory_mb=parse_memory_mb(compute.memory),
                accelerator=accelerator_type,
                accelerator_count=requested_gpu_count,
                prefer_spot=prefer_spot,
            )
            size = spec.name
            accelerator_count = spec.accelerator_count

        if compute.accelerator or (compute.machine and accelerator_count > 0):
            default_acc = AcceleratorSpec("H100", "80GB")
            os_image = get_gpu_image(acc or default_acc, accelerator_count)
            username = "ubuntu"
        else:
            os_image = "ubuntu-24-04-x64"
            username = "root"

        logger.debug(f"Using image: {os_image}, username: {username}")
        self._username = username

        # Generate bootstrap script using Image.bootstrap()
        user_data = compute.image.bootstrap(ttl=self.instance_timeout)

        emit(
            InstanceLaunching(
                count=compute.nodes,
                candidates=(spec,) if spec else (),
                provider=ProviderName.DigitalOcean,
            )
        )

        droplets: list[_Droplet] = []
        for i in range(compute.nodes):
            droplet_name = f"skyward-{cluster_id}-{i}"
            logger.debug(f"Creating droplet {droplet_name} with size {size}")

            create_data: dict[str, Any] = {
                "name": droplet_name,
                "region": self.region,
                "size": size,
                "image": os_image,
                "user_data": user_data,
                "tags": ["skyward", f"skyward-cluster-{cluster_id}"],
                "ssh_keys": [fingerprint],
            }

            resp = client.droplets.create(body=create_data)
            droplet_data = resp["droplet"]
            droplets.append(
                _Droplet(
                    id=droplet_data["id"],
                    name=droplet_name,
                    ip="",
                )
            )
            logger.debug(f"Droplet {droplet_name} created with id {droplet_data['id']}")

        logger.debug(f"Waiting for {len(droplets)} droplets to become active...")
        _wait_for_active(client, droplets, timeout=300)

        self._droplets = droplets

        instances: list[Instance] = []
        provisioned_instances: list[ProvisionedInstance] = []

        key_path = get_private_key_path()

        def _make_destroy_fn(did: int) -> Callable[[], None]:
            def destroy() -> None:
                with suppress(Exception):
                    self._get_client().droplets.destroy(droplet_id=did)

            return destroy

        for i, droplet in enumerate(droplets):
            instance = Instance(
                id=str(droplet.id),
                provider=self,
                ssh=SSHConfig(
                    host=droplet.ip,
                    username=username,
                    key_path=key_path,
                ),
                spot=False,
                private_ip=droplet.private_ip or droplet.ip,
                public_ip=droplet.ip,
                node=i,
                metadata=frozenset(
                    [
                        ("cluster_id", cluster_id),
                        ("droplet_id", droplet.id),
                        ("droplet_ip", droplet.ip),
                        ("username", username),
                        ("accelerator_count", str(accelerator_count)),
                    ]
                ),
                _destroy_fn=_make_destroy_fn(droplet.id),
            )
            instances.append(instance)

            provisioned = ProvisionedInstance(
                instance_id=str(droplet.id),
                node=i,
                provider=ProviderName.DigitalOcean,
                spot=False,
                spec=spec,
                ip=droplet.ip,
            )
            provisioned_instances.append(provisioned)
            emit(InstanceProvisioned(instance=provisioned))

        emit(
            ProvisioningCompleted(
                instances=tuple(provisioned_instances),
                provider=ProviderName.DigitalOcean,
                region=self.region,
            )
        )

        return tuple(instances)

    def discover_peers(self, cluster_id: str) -> tuple[Instance, ...]:
        """Discover peer instances in a cluster via DigitalOcean API."""
        from skyward.providers.base import assign_node_indices

        client = self._get_client()
        tag = f"skyward-cluster-{cluster_id}"
        response = client.droplets.list(tag_name=tag)
        key_path = get_private_key_path()

        instances = []
        for droplet in response.get("droplets", []):
            if droplet["status"] != "active":
                continue

            private_ip = ""
            public_ip = ""
            for network in droplet.get("networks", {}).get("v4", []):
                if network["type"] == "private":
                    private_ip = network["ip_address"]
                elif network["type"] == "public":
                    public_ip = network["ip_address"]

            instances.append(
                Instance(
                    id=str(droplet["id"]),
                    provider=self,
                    ssh=SSHConfig(
                        host=public_ip or private_ip,
                        username=self._username,
                        key_path=key_path,
                    ),
                    spot=False,
                    private_ip=private_ip or public_ip,
                    public_ip=public_ip,
                    node=0,
                    metadata=frozenset([
                        ("cluster_id", cluster_id),
                        ("droplet_id", droplet["id"]),
                        ("droplet_ip", public_ip or private_ip),
                        ("region", self.region),
                    ]),
                )
            )

        return assign_node_indices(instances)

    def available_instances(self) -> tuple[InstanceSpec, ...]:
        """List all available droplet sizes."""
        client = self._get_client()
        return fetch_available_instances(client)

    def cleanup(self) -> None:
        """No-op cleanup for DigitalOcean."""
        pass
