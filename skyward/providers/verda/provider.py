"""Verda Cloud provider for Skyward GPU instances."""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

from verda import VerdaClient
from verda.constants import Actions

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
    RegionAutoSelected,
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
from skyward.spec import Spot, _SpotNever, normalize_spot
from skyward.types import (
    ComputeSpec,
    ExitedInstance,
    Instance,
    InstanceSpec,
    Provider,
    select_instance,
)

from .discovery import (
    NoAvailableRegionError,
    fetch_available_instances,
    find_available_region,
)

if TYPE_CHECKING:
    pass


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
    from tenacity import (
        RetryError,
        retry,
        retry_if_exception_type,
        stop_after_delay,
        wait_exponential,
    )

    class _InstancePendingError(Exception):
        """Instance not yet running - retry."""

    def poll_instance(vinst: _VerdaInstance) -> None:
        @retry(
            stop=stop_after_delay(timeout),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(_InstancePendingError),
            reraise=True,
        )
        def check() -> None:
            all_instances = client.instances.get()
            data = None
            for inst in all_instances:
                if inst.id == vinst.id:
                    data = inst
                    break

            if data is None:
                raise _InstancePendingError(f"Instance {vinst.id} not found")

            vinst.status = data.status

            if data.status != "running":
                raise _InstancePendingError(f"Instance {vinst.id} status: {data.status}")

            if hasattr(data, "ip") and data.ip:
                vinst.ip = data.ip
            else:
                raise _InstancePendingError(f"Instance {vinst.id} has no IP yet")

        try:
            check()
        except RetryError as e:
            raise TimeoutError(f"Instance {vinst.id} did not become running within {timeout}s") from e

    for vinst in instances:
        poll_instance(vinst)


# =============================================================================
# Provider
# =============================================================================


@dataclass(frozen=True)
class Verda(Provider):
    """Provider for Verda Cloud (formerly DataCrunch) GPU instances.

    Executes functions on Verda GPU instances via SSH.
    Supports NVIDIA GPUs: V100, A100, H100, H200, L40S, GB200.

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

    _resolved_ssh_key_id: str | None = field(default=None, repr=False, compare=False, hash=False)
    _username: str = field(default="root", repr=False, compare=False, hash=False)
    _instances: list[_VerdaInstance] = field(default_factory=list, repr=False, compare=False, hash=False)
    _cluster_id: str | None = field(default=None, repr=False, compare=False, hash=False)

    @property
    @override
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

    def _get_transport(self, instance: Instance) -> SSHTransport:
        """Get SSHTransport for an instance."""
        ip = instance.get_meta("instance_ip") or instance.public_ip or instance.private_ip
        username = instance.get_meta("username", self._username)
        key_path = get_private_key_path()
        return SSHTransport(host=ip, username=username, key_path=key_path)

    @override
    def provision(self, compute: ComputeSpec) -> tuple[Instance, ...]:
        """Provision Verda GPU instances."""
        try:
            cluster_id = str(uuid.uuid4())[:8]
            emit(InfraCreating())

            client = self._get_client()

            ssh_key_info = _get_or_create_ssh_key(client)
            object.__setattr__(self, "_resolved_ssh_key_id", ssh_key_info.id)

            emit(InfraCreated(region=self.region))

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

            supported_os = spec.metadata.get("supported_os", ())
            if compute.accelerator:
                os_image = next(filter(lambda os: "cuda" in os, supported_os))
            else:
                os_image = supported_os[0]
            username = "root"
            object.__setattr__(self, "_username", username)

            # Generate bootstrap script using Image.bootstrap()
            user_data = compute.image.bootstrap(ttl=compute.timeout)

            script_name = f"skyward-bootstrap-{cluster_id}"
            startup_script = client.startup_scripts.create(name=script_name, script=user_data)

            spot_strategy = normalize_spot(compute.spot) if hasattr(compute, "spot") else Spot.Never
            use_spot = not isinstance(spot_strategy, _SpotNever)

            # Auto-region discovery
            actual_region = find_available_region(
                client,
                spec.name,
                is_spot=use_spot,
                preferred_region=self.region,
            )

            if actual_region != self.region:
                emit(
                    RegionAutoSelected(
                        requested_region=self.region,
                        selected_region=actual_region,
                        instance_type=spec.name,
                        provider=ProviderName.Verda,
                    )
                )

            emit(
                InstanceLaunching(count=compute.nodes, instance_type=spec.name, provider=ProviderName.Verda)
            )

            verda_instances: list[_VerdaInstance] = []
            for i in range(compute.nodes):
                instance_name = f"skyward-{cluster_id}-{i}"

                created = client.instances.create(
                    instance_type=spec.name,
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

            _wait_for_running(client, verda_instances, timeout=300)

            object.__setattr__(self, "_instances", verda_instances)
            object.__setattr__(self, "_cluster_id", cluster_id)

            instances: list[Instance] = []
            for i, vinst in enumerate(verda_instances):
                instance = Instance(
                    id=str(vinst.id),
                    provider=self,
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
                )
                instances.append(instance)

                emit(
                    InstanceProvisioned(
                        instance_id=str(vinst.id),
                        node=i,
                        spot=use_spot,
                        ip=vinst.ip,
                        instance_type=spec.name,
                        provider=ProviderName.Verda,
                        price_on_demand=spec.price_on_demand,
                        price_spot=spec.price_spot,
                        billing_increment_minutes=spec.billing_increment_minutes,
                    )
                )

            emit(
                ProvisioningCompleted(
                    spot=len([i for i in instances if i.spot]),
                    on_demand=len([i for i in instances if not i.spot]),
                    provider=ProviderName.Verda,
                    region=actual_region,
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

            key_path = get_private_key_path()

            def get_ip(inst: Instance) -> str:
                return inst.get_meta("instance_ip") or inst.public_ip or ""

            wait_for_ssh_bootstrap(instances, get_ip, timeout=300, key_path=key_path)
            install_skyward_wheel(instances, get_ip, key_path=key_path)

            for inst in instances:
                emit(BootstrapCompleted(instance_id=inst.id))

        except Exception as e:
            emit(Error(message=f"Setup failed: {e}"))
            raise

    @override
    def shutdown(
        self, instances: tuple[Instance, ...], compute: ComputeSpec
    ) -> tuple[ExitedInstance, ...]:
        """Shutdown instances (delete them)."""
        client = self._get_client()

        exited: list[ExitedInstance] = []
        for inst in instances:
            instance_id = inst.get_meta("instance_id") or inst.id
            if instance_id:
                emit(InstanceStopping(instance_id=inst.id))
                try:
                    client.instances.action(instance_id, Actions.DELETE)
                except Exception:
                    pass

            exited.append(
                ExitedInstance(
                    instance=inst,
                    exit_code=0,
                    exit_reason="normal",
                )
            )

        object.__setattr__(self, "_instances", [])
        return tuple(exited)

    @override
    def create_tunnel(
        self, instance: Instance, remote_port: int = 18861
    ) -> tuple[int, subprocess.Popen[bytes]]:
        """Create SSH tunnel to instance using SSHTransport."""
        transport = self._get_transport(instance)
        return transport.create_tunnel(remote_port)

    @override
    def run_command(self, instance: Instance, command: str, timeout: int = 30) -> str:
        """Run shell command on instance via SSH using SSHTransport."""
        transport = self._get_transport(instance)
        return transport.run_command(command, timeout)

    @override
    def discover_peers(self, cluster_id: str) -> tuple[Instance, ...]:
        """Discover peer instances in a cluster via Verda API."""
        client = self._get_client()

        hostname_pattern = f"skyward-{cluster_id}-"
        all_instances = client.instances.get()

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
                    spot=getattr(vinst, "is_spot", False),
                    private_ip=ip,
                    public_ip=ip,
                    node=0,
                    metadata=frozenset(
                        [
                            ("cluster_id", cluster_id),
                            ("instance_id", vinst.id),
                            ("instance_ip", ip),
                            ("username", self._username),
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
        """List all available instance types."""
        client = self._get_client()
        return fetch_available_instances(client)
