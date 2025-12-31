"""DigitalOcean provider for Skyward GPU instances."""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast, override

from pydo import Client

from skyward.accelerator import Accelerator
from skyward.callback import emit
from skyward.events import (
    BootstrapCompleted,
    BootstrapStarting,
    Error,
    InfraCreated,
    InfraCreating,
    InstanceLaunching,
    InstanceProvisioned,
    InstanceStopping,
    ProviderName,
    ProvisioningCompleted,
)
from skyward.providers.base import (
    SSHKeyInfo,
    SSHKeyManager,
    SSHTransport,
    get_private_key_path,
)
from skyward.providers.common import (
    install_skyward_wheel,
    wait_for_ssh_bootstrap,
)
from skyward.types import (
    ComputeSpec,
    ExitedInstance,
    Instance,
    InstanceSpec,
    Provider,
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
    from tenacity import (
        RetryError,
        retry,
        retry_if_exception_type,
        stop_after_delay,
        wait_fixed,
    )

    class _DropletPendingError(Exception):
        """Droplet not yet active - retry."""

    def poll_droplet(droplet: _Droplet) -> None:
        @retry(
            stop=stop_after_delay(timeout),
            wait=wait_fixed(5),
            retry=retry_if_exception_type(_DropletPendingError),
            reraise=True,
        )
        def check() -> None:
            resp = client.droplets.get(droplet_id=droplet.id)
            data = resp["droplet"]

            if data["status"] != "active":
                raise _DropletPendingError()

            for network in data["networks"]["v4"]:
                if network["type"] == "public":
                    droplet.ip = network["ip_address"]
                elif network["type"] == "private":
                    droplet.private_ip = network["ip_address"]

            if not droplet.ip:
                raise _DropletPendingError()

        try:
            check()
        except RetryError as e:
            raise TimeoutError(f"Droplet {droplet.id} did not become active within {timeout}s") from e

    for droplet in droplets:
        poll_droplet(droplet)


# =============================================================================
# Provider
# =============================================================================


@dataclass(frozen=True)
class DigitalOcean(Provider):
    """Provider for DigitalOcean Droplets.

    Executes functions on DigitalOcean Droplets via SSH.
    Supports GPU Droplets with NVIDIA H100, H200, L40S, and more.

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

    _resolved_fingerprint: str | None = field(default=None, repr=False, compare=False, hash=False)
    _username: str = field(default="root", repr=False, compare=False, hash=False)
    _droplets: list[_Droplet] = field(default_factory=list, repr=False, compare=False, hash=False)

    @property
    @override
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
        object.__setattr__(self, "_resolved_fingerprint", key_info.fingerprint)
        return key_info.fingerprint

    def _get_transport(self, instance: Instance) -> SSHTransport:
        """Get SSHTransport for an instance."""
        ip = instance.get_meta("droplet_ip") or instance.public_ip or instance.private_ip
        username = instance.get_meta("username", self._username)
        key_path = get_private_key_path()
        return SSHTransport(host=ip, username=username, key_path=key_path)

    @override
    def provision(self, compute: ComputeSpec) -> tuple[Instance, ...]:
        """Provision DigitalOcean Droplets."""
        try:
            cluster_id = str(uuid.uuid4())[:8]
            emit(InfraCreating())

            fingerprint = self._ensure_ssh_key()
            emit(InfraCreated(region=self.region))

            client = self._get_client()

            acc = Accelerator.from_value(compute.accelerator)
            accelerator_type = acc.accelerator if acc else None
            requested_gpu_count = acc.count if acc else 1

            spec = select_instance(
                self.available_instances(),
                cpu=1,
                memory_mb=1024,
                accelerator=accelerator_type,
                accelerator_count=requested_gpu_count,
            )

            if compute.accelerator:
                os_image = get_gpu_image(cast(Accelerator, compute.accelerator), spec.accelerator_count)
                username = "ubuntu"
            else:
                os_image = "ubuntu-24-04-x64"
                username = "root"

            object.__setattr__(self, "_username", username)

            # Generate bootstrap script using Image.bootstrap()
            user_data = compute.image.bootstrap(ttl=self.instance_timeout)

            emit(
                InstanceLaunching(
                    count=compute.nodes, instance_type=spec.name, provider=ProviderName.DigitalOcean
                )
            )

            droplets: list[_Droplet] = []
            for i in range(compute.nodes):
                droplet_name = f"skyward-{cluster_id}-{i}"

                create_data: dict[str, Any] = {
                    "name": droplet_name,
                    "region": self.region,
                    "size": spec.name,
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

            _wait_for_active(client, droplets, timeout=300)

            object.__setattr__(self, "_droplets", droplets)

            instances: list[Instance] = []
            for i, droplet in enumerate(droplets):
                instance = Instance(
                    id=str(droplet.id),
                    provider=self,
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
                            ("accelerator_count", str(spec.accelerator_count)),
                        ]
                    ),
                )
                instances.append(instance)

                emit(
                    InstanceProvisioned(
                        instance_id=str(droplet.id),
                        node=i,
                        spot=False,
                        ip=droplet.ip,
                        instance_type=spec.name,
                        provider=ProviderName.DigitalOcean,
                        price_on_demand=spec.price_on_demand,
                        price_spot=spec.price_spot,
                        billing_increment_minutes=spec.billing_increment_minutes,
                    )
                )

            emit(
                ProvisioningCompleted(
                    spot=0,
                    on_demand=len(instances),
                    provider=ProviderName.DigitalOcean,
                    region=self.region,
                    instances=[inst.id for inst in instances],
                )
            )

            return tuple(instances)

        except Exception as e:
            emit(Error(message=f"Provision failed: {e}"))
            raise

    @override
    def setup(self, instances: tuple[Instance, ...], compute: ComputeSpec) -> None:
        """Setup instances (wait for bootstrap to complete)."""
        try:
            for inst in instances:
                emit(BootstrapStarting(instance_id=inst.id))

            def get_ip(inst: Instance) -> str:
                return inst.get_meta("droplet_ip") or inst.public_ip or ""

            wait_for_ssh_bootstrap(instances, get_ip, timeout=300)
            install_skyward_wheel(instances, get_ip)

            for inst in instances:
                emit(BootstrapCompleted(instance_id=inst.id))

        except Exception as e:
            emit(Error(message=f"Setup failed: {e}"))
            raise

    @override
    def shutdown(
        self, instances: tuple[Instance, ...], compute: ComputeSpec
    ) -> tuple[ExitedInstance, ...]:
        """Shutdown Droplets (destroy them)."""
        client = self._get_client()

        exited: list[ExitedInstance] = []
        for inst in instances:
            droplet_id = inst.get_meta("droplet_id")
            if droplet_id:
                emit(InstanceStopping(instance_id=inst.id))
                try:
                    client.droplets.destroy(droplet_id=droplet_id)
                except Exception:
                    pass

            exited.append(
                ExitedInstance(
                    instance=inst,
                    exit_code=0,
                    exit_reason="normal",
                )
            )

        object.__setattr__(self, "_droplets", [])
        return tuple(exited)

    @override
    def create_tunnel(
        self, instance: Instance, remote_port: int = 18861
    ) -> tuple[int, subprocess.Popen[bytes]]:
        """Create SSH tunnel to Droplet using SSHTransport."""
        transport = self._get_transport(instance)
        return transport.create_tunnel(remote_port)

    @override
    def run_command(self, instance: Instance, command: str, timeout: int = 30) -> str:
        """Run shell command on Droplet via SSH using SSHTransport."""
        transport = self._get_transport(instance)
        return transport.run_command(command, timeout)

    @override
    def discover_peers(self, cluster_id: str) -> tuple[Instance, ...]:
        """Discover peer instances in a cluster via DigitalOcean API."""
        client = self._get_client()
        tag = f"skyward-cluster-{cluster_id}"

        response = client.droplets.list(tag_name=tag)

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
                    spot=False,
                    private_ip=private_ip or public_ip,
                    public_ip=public_ip,
                    node=0,
                    metadata=frozenset(
                        [
                            ("cluster_id", cluster_id),
                            ("droplet_id", droplet["id"]),
                            ("droplet_ip", public_ip or private_ip),
                            ("region", self.region),
                        ]
                    ),
                )
            )

        instances.sort(key=lambda i: i.private_ip)

        return tuple(
            Instance(
                id=inst.id,
                provider=inst.provider,
                spot=inst.spot,
                private_ip=inst.private_ip,
                public_ip=inst.public_ip,
                node=idx,
                metadata=inst.metadata,
            )
            for idx, inst in enumerate(instances)
        )

    @override
    def available_instances(self) -> tuple[InstanceSpec, ...]:
        """List all available droplet sizes."""
        client = self._get_client()
        return fetch_available_instances(client)
