"""DigitalOcean provider for Skyward GPU instances."""

from __future__ import annotations

import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast, override

from pydo import Client
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_fixed,
)

from skyward.accelerator import Accelerator
from skyward.cache import cached
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
from skyward.providers.common import (
    SSH_KEY_PATHS,
    SSHKeyInfo,
    compute_fingerprint,
    create_tunnel,
    find_available_port,
    find_local_ssh_key,
    install_skyward_wheel,
    ssh_run,
    wait_for_ssh_bootstrap,
)
from skyward.types import ComputeSpec, ExitedInstance, Instance, InstanceSpec, Provider, select_instance

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
# SSH Key Management
# =============================================================================


def _get_or_create_ssh_key(token: str) -> SSHKeyInfo:
    """Get or create SSH key on DigitalOcean."""
    key_info = find_local_ssh_key()
    if key_info is None:
        raise RuntimeError(
            "No SSH key found. Please create one with:\n"
            "  ssh-keygen -t ed25519 -C 'your@email.com'\n"
            f"Searched locations: {', '.join(SSH_KEY_PATHS)}"
        )

    key_path, public_key = key_info
    fingerprint = compute_fingerprint(public_key)
    key_name = f"skyward-{os.environ.get('USER', 'user')}-{key_path.stem}"

    client = Client(token=token)

    resp = client.ssh_keys.list()
    existing_keys = resp.get("ssh_keys", [])

    for key in existing_keys:
        if key.get("fingerprint") == fingerprint:
            return SSHKeyInfo(
                id=None,
                fingerprint=fingerprint,
                public_key=public_key,
                name=key.get("name", key_name),
            )

    try:
        resp = client.ssh_keys.create(body={"name": key_name, "public_key": public_key})
        created_key = resp.get("ssh_key", {})
        return SSHKeyInfo(
            id=None,
            fingerprint=created_key.get("fingerprint", fingerprint),
            public_key=public_key,
            name=created_key.get("name", key_name),
        )
    except Exception as e:
        if "already been taken" in str(e).lower():
            return SSHKeyInfo(
                id=None,
                fingerprint=fingerprint,
                public_key=public_key,
                name=key_name,
            )
        raise


# =============================================================================
# Instance Type Discovery
# =============================================================================


def _normalize_gpu_model(model: str | None) -> str | None:
    """Normalize DigitalOcean GPU model name."""
    if not model:
        return None

    model = model.lower()
    for prefix in ("nvidia_", "amd_"):
        if model.startswith(prefix):
            model = model[len(prefix):]
            break

    return model.upper()


def _parse_droplet_size(size: dict[str, Any]) -> InstanceSpec:
    """Parse DigitalOcean size to InstanceSpec."""
    gpu_info = size.get("gpu_info") or {}

    accelerator = None
    accelerator_count = 0
    accelerator_memory_gb = 0.0

    if gpu_info:
        accelerator = _normalize_gpu_model(gpu_info.get("model"))
        accelerator_count = gpu_info.get("count", 0)
        vram = gpu_info.get("vram") or {}
        accelerator_memory_gb = vram.get("amount", 0)

    return InstanceSpec(
        name=size["slug"],
        vcpu=size.get("vcpus", 0),
        memory_gb=size.get("memory", 0) / 1024,
        accelerator=accelerator,
        accelerator_count=accelerator_count,
        accelerator_memory_gb=accelerator_memory_gb,
        price_on_demand=size.get("price_hourly"),
        price_spot=None,  # DigitalOcean does not have spot pricing
        billing_increment_minutes=None,  # DigitalOcean: per-second billing (60s min) since Jan 2026
        metadata={
            "regions": size.get("regions", []),
            "price_monthly": size.get("price_monthly"),
        },
    )


# =============================================================================
# Lifecycle Helpers
# =============================================================================


class _DropletPendingError(Exception):
    """Droplet not yet active - retry."""


def _wait_for_active(client: Client, droplets: list[_Droplet], timeout: float) -> None:
    """Wait for all droplets to become active and get IPs."""

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
            raise TimeoutError(
                f"Droplet {droplet.id} did not become active within {timeout}s"
            ) from e

    for droplet in droplets:
        poll_droplet(droplet)


def _get_gpu_image(accelerator: Accelerator, accelerator_count: int = 1) -> str:
    """Get the appropriate GPU image for the accelerator type."""
    acc_str = accelerator.accelerator if hasattr(accelerator, "accelerator") else str(accelerator)

    if acc_str and "MI3" in acc_str.upper():
        return "gpu-amd-base"

    if accelerator_count == 8:
        return "gpu-h100x8-base"

    return "gpu-h100x1-base"


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
    _droplets: list = field(default_factory=list, repr=False, compare=False, hash=False)

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

        key_info = _get_or_create_ssh_key(self._token)
        object.__setattr__(self, "_resolved_fingerprint", key_info.fingerprint)
        return key_info.fingerprint

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
                os_image = _get_gpu_image(cast(Accelerator, compute.accelerator), spec.accelerator_count)
                username = "ubuntu"
            else:
                os_image = "ubuntu-24-04-x64"
                username = "root"

            object.__setattr__(self, "_username", username)

            # Generate bootstrap script using Image.bootstrap()
            user_data = compute.image.bootstrap(ttl=self.instance_timeout)

            emit(InstanceLaunching(count=compute.nodes, instance_type=spec.name, provider=ProviderName.DigitalOcean))

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
                    metadata=frozenset([
                        ("cluster_id", cluster_id),
                        ("droplet_id", droplet.id),
                        ("droplet_ip", droplet.ip),
                        ("username", username),
                        ("accelerator_count", str(spec.accelerator_count)),
                    ]),
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
    def shutdown(self, instances: tuple[Instance, ...], compute: ComputeSpec) -> tuple[ExitedInstance, ...]:
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
    def create_tunnel(self, instance: Instance, remote_port: int = 18861) -> tuple[int, subprocess.Popen[bytes]]:
        """Create SSH tunnel to Droplet."""
        ip = instance.get_meta("droplet_ip")
        local_port = find_available_port()
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-N",
            "-L", f"{local_port}:127.0.0.1:{remote_port}",
            f"{self._username}@{ip}",
        ]
        return create_tunnel(cmd, local_port)

    @override
    def run_command(self, instance: Instance, command: str, timeout: int = 30) -> str:
        """Run shell command on Droplet via SSH."""
        ip = instance.get_meta("droplet_ip") or instance.public_ip or instance.private_ip
        if not ip:
            raise RuntimeError(f"No IP address for instance {instance.id}")

        username = instance.get_meta("username", self._username)
        result = ssh_run(ip, username, command, timeout)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed on {instance.id}: {result.stderr}")
        return result.stdout

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
                    metadata=frozenset([
                        ("cluster_id", cluster_id),
                        ("droplet_id", droplet["id"]),
                        ("droplet_ip", public_ip or private_ip),
                        ("region", self.region),
                    ]),
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
    @cached(namespace="digitalocean")
    def available_instances(self) -> tuple[InstanceSpec, ...]:
        """List all available droplet sizes."""
        client = self._get_client()
        sizes: list[dict[str, Any]] = []

        page = 1
        while True:
            resp = client.sizes.list(page=page, per_page=100)
            sizes.extend(resp.get("sizes", []))

            links = resp.get("links", {})
            pages = links.get("pages", {})
            if not pages.get("next"):
                break
            page += 1

        specs = [
            _parse_droplet_size(s)
            for s in sizes
            if s.get("available", True)
        ]

        return tuple(sorted(
            specs,
            key=lambda s: (
                s.accelerator or "",
                s.accelerator_count,
                s.vcpu,
                s.memory_gb,
            ),
        ))
