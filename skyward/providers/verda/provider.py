"""Verda Cloud provider for Skyward GPU instances."""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from verda import VerdaClient
from verda.constants import Actions

from skyward.accelerator import Accelerator
from skyward.callback import emit
from skyward.events import (
    BootstrapCompleted,
    BootstrapStarting,
    Error,
    InstanceLaunching,
    InstanceProvisioned,
    InstanceStopping,
    NetworkReady,
    ProviderName,
    ProvisionedInstance,
    ProvisioningCompleted,
    ProvisioningStarted,
    RegionAutoSelected,
)
from skyward.providers.base import (
    SSHKeyInfo,
    SSHKeyManager,
    SSHTransport,
    get_private_key_path,
)
from skyward.providers.common import (
    install_skyward_wheel_via_transport,
    wait_for_ssh_bootstrap,
)
from skyward.spec import _AllocationOnDemand, normalize_allocation
from skyward.types import (
    ComputeSpec,
    ExitedInstance,
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
        wait_fixed,
    )

    class _InstancePendingError(Exception):
        """Instance not yet running - retry."""

    def poll_instance(vinst: _VerdaInstance) -> None:
        @retry(
            stop=stop_after_delay(timeout),
            wait=wait_fixed(1),
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

    def _get_transport(self, instance: Instance) -> SSHTransport:
        """Get SSHTransport for an instance."""
        ip = instance.get_meta("instance_ip") or instance.public_ip or instance.private_ip
        username = instance.get_meta("username", self._username)
        key_path = get_private_key_path()
        return SSHTransport(host=ip, username=username, key_path=key_path)

    def _make_provisioned(
        self, inst: Instance, spec: InstanceSpec | None = None
    ) -> ProvisionedInstance:
        """Create ProvisionedInstance from Instance for events."""
        return ProvisionedInstance(
            instance_id=inst.id,
            node=inst.node,
            provider=ProviderName.Verda,
            spot=inst.spot,
            spec=spec,
            ip=inst.public_ip or inst.private_ip,
        )

    def provision(self, compute: ComputeSpec) -> tuple[Instance, ...]:
        """Provision Verda GPU instances."""
        try:
            cluster_id = str(uuid.uuid4())[:8]
            logger.info(f"Provisioning {compute.nodes} Verda instances in {self.region}")
            logger.debug(f"Cluster ID: {cluster_id}")
            emit(ProvisioningStarted())

            client = self._get_client()

            ssh_key_info = _get_or_create_ssh_key(client)
            self._resolved_ssh_key_id = ssh_key_info.id

            emit(NetworkReady(region=self.region))

            acc = Accelerator.from_value(compute.accelerator)
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

            logger.info(f"Provisioned {len(instances)} Verda instances: {[v.id for v in verda_instances]}")
            return tuple(instances)

        except Exception as e:
            logger.error(f"Verda provisioning failed: {e}")
            emit(Error(message=f"Provision failed: {e}"))
            raise

    def setup(self, instances: tuple[Instance, ...], compute: ComputeSpec) -> None:
        """Setup instances (wait for bootstrap to complete)."""
        from contextlib import contextmanager
        from typing import Generator

        logger.info(f"Setting up {len(instances)} Verda instances...")
        try:
            # Build provisioned lookup for events
            provisioned_map = {inst.id: self._make_provisioned(inst) for inst in instances}

            for inst in instances:
                emit(BootstrapStarting(instance=provisioned_map[inst.id]))

            key_path = get_private_key_path()

            def get_ip(inst: Instance) -> str:
                return inst.get_meta("instance_ip") or inst.public_ip or ""

            def make_provisioned(inst: Instance) -> ProvisionedInstance:
                return provisioned_map[inst.id]

            @contextmanager
            def ssh_transport(inst: Instance) -> Generator[SSHTransport, None, None]:
                ip = get_ip(inst)
                username = inst.get_meta("username", self._username)
                yield SSHTransport(host=ip, username=username, key_path=key_path)

            wait_for_ssh_bootstrap(
                instances, get_ip, make_provisioned, timeout=300, key_path=key_path
            )
            install_skyward_wheel_via_transport(instances, ssh_transport, compute=compute)

            for inst in instances:
                emit(BootstrapCompleted(instance=provisioned_map[inst.id]))

            logger.info(f"Setup completed for {len(instances)} instances")

        except Exception as e:
            logger.error(f"Verda setup failed: {e}")
            emit(Error(message=f"Setup failed: {e}"))
            raise

    def shutdown(
        self, instances: tuple[Instance, ...], compute: ComputeSpec
    ) -> tuple[ExitedInstance, ...]:
        """Shutdown instances (delete them)."""
        logger.info(f"Shutting down {len(instances)} Verda instances...")
        client = self._get_client()

        exited: list[ExitedInstance] = []
        for inst in instances:
            instance_id = inst.get_meta("instance_id") or inst.id
            if instance_id:
                provisioned = self._make_provisioned(inst)
                emit(InstanceStopping(instance=provisioned))
                logger.debug(f"Deleting instance {instance_id}")
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

        self._instances = []
        logger.info(f"Shutdown complete: {len(exited)} instances deleted")
        return tuple(exited)

    def create_tunnel(
        self, instance: Instance, remote_port: int = 18861
    ) -> tuple[int, subprocess.Popen[bytes]]:
        """Create SSH tunnel to instance using SSHTransport."""
        logger.debug(f"Creating tunnel to instance {instance.id} port {remote_port}")
        transport = self._get_transport(instance)
        return transport.create_tunnel(remote_port)

    def run_command(self, instance: Instance, command: str, timeout: int = 30) -> str:
        """Run shell command on instance via SSH using SSHTransport."""
        transport = self._get_transport(instance)
        return transport.run_command(command, timeout)

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

    def available_instances(self) -> tuple[InstanceSpec, ...]:
        """List all available instance types."""
        client = self._get_client()
        return fetch_available_instances(client)
