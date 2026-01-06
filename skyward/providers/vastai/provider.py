"""Vast.ai provider for Skyward GPU instances.

Vast.ai is a GPU marketplace with dynamic offers from various hosts.
Instances are Docker containers, not VMs, so bootstrap works differently
from traditional cloud providers.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

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
)
from skyward.providers.base import SSHTransport, get_private_key_path
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
    select_instance,
)

from .discovery import (
    build_search_query,
    extract_cuda_version,
    fetch_available_offers,
    offer_to_instance_spec,
    search_offers,
)
from .ssh import (
    attach_ssh_key_to_instance,
    get_or_create_ssh_key,
)

if TYPE_CHECKING:
    from vastai import VastAI as VastAIClient


# =============================================================================
# Internal Types
# =============================================================================


@dataclass
class _VastInstance:
    """Internal representation of a Vast.ai instance."""

    id: int
    offer_id: int
    machine_id: int
    ssh_host: str = ""
    ssh_port: int = 22
    actual_status: str = ""


# =============================================================================
# Provider
# =============================================================================


@dataclass(frozen=True, slots=True)
class VastAI:
    """Vast.ai provider configuration.

    Vast.ai is a GPU marketplace with dynamic offers from various hosts.
    Unlike traditional cloud providers, instances are Docker containers
    running on marketplace hosts with varying reliability.

    SSH keys are automatically detected from ~/.ssh/id_ed25519.pub or
    ~/.ssh/id_rsa.pub and registered on Vast.ai if needed.

    Example:
        from skyward import ComputePool, compute
        from skyward.providers import VastAI

        @compute
        def train(data):
            return model.fit(data)

        pool = ComputePool(
            provider=VastAI(
                min_reliability=0.95,
                geolocation="US",
            ),
            accelerator="RTX_4090",
            nodes=2,
            allocation="spot-if-available",  # Uses interruptible pricing
            pip=["torch", "transformers"],
        )

        with pool:
            result = train(data) >> pool

    Environment Variables:
        VAST_API_KEY: API key (required if not passed directly)

    Attributes:
        api_key: Vast.ai API key. Falls back to VAST_API_KEY env var.
        min_reliability: Minimum host reliability score (0.0-1.0).
                        Higher values = more stable but fewer options.
        geolocation: Preferred region/country code (e.g., "US", "EU").
                    None = any location.
        bid_multiplier: For interruptible (spot) pricing, multiply min bid
                       by this. 1.0 = minimum bid, 1.5 = 50% above minimum.
        instance_timeout: Auto-shutdown in seconds (safety timeout).
        docker_image: Base Docker image for containers.
                     If None, derived from Accelerator's CUDA requirements.
    """

    api_key: str | None = None
    min_reliability: float = 0.95
    geolocation: str | None = None
    bid_multiplier: float = 1.2
    instance_timeout: int = 300
    docker_image: str | None = None
    disk_gb: int = 100

    def build(self) -> VastAIProvider:
        """Build a stateful VastAIProvider from this configuration."""
        return VastAIProvider(
            api_key=self.api_key,
            min_reliability=self.min_reliability,
            geolocation=self.geolocation,
            bid_multiplier=self.bid_multiplier,
            instance_timeout=self.instance_timeout,
            docker_image=self.docker_image,
            disk_gb=self.disk_gb,
        )


class VastAIProvider(Provider):
    """Stateful Vast.ai provider service.

    This class manages Docker container instances on Vast.ai marketplace.
    Created by VastAI.build() or automatically by ComputePool.

    Implements the Provider protocol with provision/setup/shutdown lifecycle.
    """

    def __init__(
        self,
        api_key: str | None,
        min_reliability: float,
        geolocation: str | None,
        bid_multiplier: float,
        instance_timeout: int,
        docker_image: str | None,
        disk_gb: int,
    ) -> None:
        self.api_key = api_key
        self.min_reliability = min_reliability
        self.geolocation = geolocation
        self.bid_multiplier = bid_multiplier
        self.instance_timeout = instance_timeout
        self.docker_image = docker_image
        self.disk_gb = disk_gb

        # Mutable runtime state
        self._ssh_key_id: int | None = None
        self._ssh_public_key: str | None = None
        self._instances: list[_VastInstance] = []

    @property
    def name(self) -> str:
        return "vastai"

    @property
    def _api_key(self) -> str:
        """Get API key from config, environment, or config file.

        Resolution order:
        1. Direct api_key parameter
        2. VAST_API_KEY environment variable
        3. ~/.config/vastai/vast_api_key file (CLI config)
        """
        if self.api_key:
            return self.api_key

        if env_key := os.environ.get("VAST_API_KEY"):
            return env_key

        # Check CLI config file
        config_path = os.path.expanduser("~/.config/vastai/vast_api_key")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    if file_key := f.read().strip():
                        return file_key
            except OSError:
                pass

        raise ValueError(
            "Vast.ai API key not provided. Options:\n"
            "  1. Pass api_key to VastAI()\n"
            "  2. Set VAST_API_KEY environment variable\n"
            "  3. Run: vastai set api-key YOUR_KEY"
        )

    def _get_client(self) -> VastAIClient:
        """Create authenticated Vast.ai client."""
        try:
            from vastai import VastAI as VastAISDK
        except ImportError as e:
            raise ImportError(
                "vastai-sdk package not installed. Install with: pip install skyward[vastai]"
            ) from e

        return VastAISDK(api_key=self._api_key)

    def _get_transport(self, instance: Instance) -> SSHTransport:
        """Get SSHTransport for an instance."""
        ssh_host = instance.get_meta("ssh_host", "")
        ssh_port = int(instance.get_meta("ssh_port", "22"))
        key_path = get_private_key_path()
        return SSHTransport(
            host=ssh_host,
            username="root",  # Vast.ai containers always use root
            key_path=key_path,
            port=ssh_port,
        )

    def _make_provisioned(
        self, inst: Instance, spec: InstanceSpec | None = None
    ) -> ProvisionedInstance:
        """Create ProvisionedInstance from Instance for events."""
        return ProvisionedInstance(
            instance_id=inst.id,
            node=inst.node,
            provider=ProviderName.VastAI,
            spot=inst.spot,
            spec=spec,
            ip=inst.get_meta("ssh_host"),
        )

    def _generate_onstart_script(self, compute: ComputeSpec) -> str:
        """Generate onstart script for Vast.ai container.

        Unlike cloud-init, this runs inside the Docker container.
        We bootstrap Python environment and dependencies.
        """
        script = f"""#!/bin/bash
set -e

# Create skyward directory
mkdir -p /opt/skyward
cd /opt/skyward

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv with specified Python version
uv venv --python {compute.image.python}
source .venv/bin/activate

# Mark uv step complete
touch /opt/skyward/.step_uv

# Install base dependencies
uv pip install cloudpickle rpyc nvidia-ml-py

# Mark base deps complete
touch /opt/skyward/.step_pip
"""

        # Install user dependencies
        if compute.image.pip:
            pip_pkgs = " ".join(f'"{pkg}"' for pkg in compute.image.pip)
            if compute.image.pip_extra_index_url:
                extra_idx = compute.image.pip_extra_index_url
                script += f'uv pip install --extra-index-url "{extra_idx}" {pip_pkgs}\n'
            else:
                script += f"uv pip install {pip_pkgs}\n"

        # Export environment variables
        if compute.image.env:
            for key, value in compute.image.env.items():
                script += f'export {key}="{value}"\n'

        script += """
# Mark bootstrap complete
touch /opt/skyward/.ready

# Keep container running for SSH access
tail -f /dev/null
"""
        return script

    def _wait_for_running(
        self,
        client: VastAIClient,
        instances: list[_VastInstance],
        timeout: float = 300,
    ) -> None:
        """Wait for all instances to be running and populate SSH info."""
        from tenacity import (
            RetryError,
            retry,
            retry_if_exception_type,
            stop_after_delay,
            wait_fixed,
        )

        class _NotReadyError(Exception):
            pass

        def poll_instance(inst: _VastInstance) -> None:
            @retry(
                stop=stop_after_delay(timeout),
                wait=wait_fixed(1),
                retry=retry_if_exception_type(_NotReadyError),
                reraise=True,
            )
            def check() -> None:
                result = client.show_instance(id=inst.id)

                if not result:
                    logger.debug(f"Instance {inst.id}: no response from show_instance")
                    raise _NotReadyError()

                # Handle dict or direct result
                if isinstance(result, dict):
                    status = result.get("actual_status", "")
                else:
                    status = getattr(result, "actual_status", "")

                inst.actual_status = status
                logger.debug(f"Instance {inst.id}: actual_status={status}")

                if status != "running":
                    raise _NotReadyError()

                # Get SSH connection info from instance data
                if isinstance(result, dict):
                    ssh_host = result.get("ssh_host", "")
                    ssh_port = result.get("ssh_port", 0)
                else:
                    ssh_host = getattr(result, "ssh_host", "")
                    ssh_port = getattr(result, "ssh_port", 0)

                if ssh_host and ssh_port:
                    inst.ssh_host = ssh_host
                    inst.ssh_port = int(ssh_port)
                    logger.debug(f"Instance {inst.id}: SSH ready at {ssh_host}:{ssh_port}")
                else:
                    logger.debug(f"Instance {inst.id}: SSH not available yet")
                    raise _NotReadyError()

            try:
                check()
            except RetryError as e:
                raise TimeoutError(
                    f"Instance {inst.id} did not become ready within {timeout}s. "
                    f"Last status: {inst.actual_status}"
                ) from e

        for inst in instances:
            poll_instance(inst)

    def provision(self, compute: ComputeSpec) -> tuple[Instance, ...]:
        """Provision Vast.ai container instances."""
        try:
            cluster_id = str(uuid.uuid4())[:8]
            logger.info(f"Provisioning {compute.nodes} Vast.ai instances")
            logger.debug(f"Cluster ID: {cluster_id}")
            emit(ProvisioningStarted())

            client = self._get_client()

            # Ensure SSH key is registered
            self._ssh_key_id, self._ssh_public_key = get_or_create_ssh_key(client)
            emit(NetworkReady(region=self.geolocation or "global"))

            # Parse compute requirements
            acc = Accelerator.from_value(compute.accelerator)
            allocation = normalize_allocation(compute.allocation)

            # Determine if using interruptible (spot) pricing
            use_interruptible = not isinstance(allocation, _AllocationOnDemand)
            logger.debug(f"Allocation: {allocation}, use_interruptible={use_interruptible}")

            # Determine CUDA version and Docker image
            if self.docker_image:
                # User specified image - extract CUDA version
                docker_image = self.docker_image
                min_cuda = extract_cuda_version(docker_image)
                cuda_versions = [min_cuda] if min_cuda else [None]
            elif acc and acc.metadata.get("cuda"):
                # Derive from Accelerator metadata - try max down to min
                cuda_meta = acc.metadata["cuda"]
                cuda_max = float(cuda_meta.get("max", "12.4"))
                cuda_min = float(cuda_meta.get("min", "11.0"))
                # Generate versions to try: max, max-0.1, max-0.2, ... down to min
                cuda_versions = []
                v = cuda_max
                while v >= cuda_min:
                    cuda_versions.append(v)
                    v = round(v - 0.1, 1)
                docker_image = None  # Will be set when we find offers
            else:
                # Fallback default
                docker_image = "nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04"
                cuda_versions = [12.4]

            # Build search query
            if compute.machine is not None:
                # Direct machine ID override
                query = f"machine_id={compute.machine} rentable=true"
                offers = search_offers(client, query)
            else:
                gpu_name = None
                num_gpus = None

                if acc:
                    # Convert accelerator to Vast.ai GPU name format
                    # Replace spaces/hyphens with underscores, preserve case
                    gpu_name = acc.accelerator.replace(" ", "_").replace("-", "_")
                    if not callable(acc.count):
                        num_gpus = acc.count

                min_mem_gb = parse_memory_mb(compute.memory) / 1024 if compute.memory else None

                # Try CUDA versions from max to min until we find offers
                offers = []
                selected_cuda = None
                for cuda_ver in cuda_versions:
                    query = build_search_query(
                        gpu_name=gpu_name,
                        num_gpus=num_gpus,
                        min_cpu=compute.cpu,
                        min_memory_gb=min_mem_gb,
                        min_cuda_version=cuda_ver,
                        min_disk_gb=self.disk_gb,
                        min_reliability=self.min_reliability,
                        geolocation=self.geolocation,
                    )
                    logger.debug(f"Searching offers with query: {query}")
                    offers = search_offers(client, query)
                    if offers:
                        selected_cuda = cuda_ver
                        logger.debug(f"Found {len(offers)} offers with CUDA >= {cuda_ver}")
                        break
                    logger.debug(f"No offers with CUDA >= {cuda_ver}, trying lower...")

                # Build docker image if not user-specified
                if docker_image is None and selected_cuda:
                    docker_image = f"nvcr.io/nvidia/cuda:{selected_cuda:.1f}.0-runtime-ubuntu22.04"
                    logger.debug(f"Selected Docker image: {docker_image}")

            if not offers:
                raise RuntimeError(
                    f"No Vast.ai offers found matching: {query}. "
                    "Try lowering min_reliability or broadening accelerator requirements."
                )

            # Sort by price (interruptible uses min_bid, on-demand uses dph_total)
            if use_interruptible:
                offers.sort(key=lambda o: o.get("min_bid", float("inf")))
            else:
                offers.sort(key=lambda o: o.get("dph_total", float("inf")))

            # Convert to InstanceSpec for selection
            specs = tuple(offer_to_instance_spec(o) for o in offers)

            # Get initial spec for event (actual offer chosen during creation loop)
            if compute.machine is None and acc:
                # Use select_instance for initial filtering
                spec = select_instance(
                    specs,
                    cpu=compute.cpu or 1,
                    memory_mb=parse_memory_mb(compute.memory),
                    accelerator=acc.accelerator if acc else None,
                    accelerator_count=acc.count if acc and not callable(acc.count) else 1,
                    prefer_spot=use_interruptible,
                )
            else:
                spec = specs[0] if specs else None

            emit(
                InstanceLaunching(
                    count=compute.nodes,
                    candidates=(spec,) if spec else (),
                    provider=ProviderName.VastAI,
                )
            )

            # Generate onstart script
            onstart_script = self._generate_onstart_script(compute)

            # Create instances - try multiple offers since marketplace is dynamic
            instances: list[Instance] = []
            provisioned_instances: list[ProvisionedInstance] = []
            vast_instances: list[_VastInstance] = []

            # Keep track of which offers to try (sorted by price)
            available_offers = list(offers)
            offer_index = 0

            for i in range(compute.nodes):
                instance_created = False
                last_error = None

                # Try offers until one succeeds
                while not instance_created and offer_index < len(available_offers):
                    current_offer = available_offers[offer_index]
                    price_info = (
                        f"bid=${current_offer.get('min_bid', 0) * self.bid_multiplier:.3f}/hr"
                        if use_interruptible
                        else f"on-demand=${current_offer.get('dph_total', 0):.3f}/hr"
                    )
                    gpu = current_offer.get("gpu_name", "?")
                    logger.debug(
                        f"Creating instance {i + 1}/{compute.nodes} "
                        f"from offer {current_offer['id']} ({gpu}) {price_info}"
                    )

                    # Calculate bid price for this offer
                    if use_interruptible:
                        offer_min_bid = current_offer.get("min_bid") or current_offer.get(
                            "dph_total", 0
                        )
                        offer_bid_price = offer_min_bid * self.bid_multiplier
                    else:
                        offer_bid_price = None

                    create_params: dict[str, Any] = {
                        "id": current_offer["id"],
                        "image": docker_image,  # Dynamically selected based on CUDA
                        "onstart_cmd": onstart_script,  # onstart_cmd for script contents
                        "disk": self.disk_gb,
                        "label": f"skyward-{cluster_id}-{i}",
                    }

                    if offer_bid_price is not None:
                        create_params["price"] = offer_bid_price

                    result = client.create_instance(**create_params)

                    # Empty response means offer is no longer available
                    if not result:
                        logger.debug(
                            f"Offer {current_offer['id']} unavailable, trying next..."
                        )
                        offer_index += 1
                        continue

                    # Handle various response formats
                    if isinstance(result, dict):
                        if not result.get("success", True):
                            last_error = f"API error: {result}"
                            logger.debug(f"Offer {current_offer['id']} failed: {result}")
                            offer_index += 1
                            continue
                        instance_id = result.get("new_contract") or result.get("id")
                    else:
                        instance_id = result

                    if not instance_id:
                        last_error = f"No instance ID in response: {result}"
                        offer_index += 1
                        continue

                    # Success!
                    vast_inst = _VastInstance(
                        id=int(instance_id),
                        offer_id=current_offer["id"],
                        machine_id=current_offer.get("machine_id", 0),
                    )
                    vast_instances.append(vast_inst)
                    instance_created = True

                    # Update spec for events
                    spec = offer_to_instance_spec(current_offer)

                    logger.debug(f"Created instance {instance_id}")

                if not instance_created:
                    raise RuntimeError(
                        f"Failed to create instance: no available offers. "
                        f"Tried {offer_index} offers. Last error: {last_error}"
                    )

            # Attach SSH key to all instances
            assert self._ssh_public_key is not None
            for vast_inst in vast_instances:
                attach_ssh_key_to_instance(client, vast_inst.id, self._ssh_public_key)

            # Wait for instances to be running and get SSH info
            logger.debug("Waiting for instances to be running...")
            self._wait_for_running(client, vast_instances, timeout=300)

            self._instances = vast_instances

            # Build Instance objects
            for i, vast_inst in enumerate(vast_instances):
                instance = Instance(
                    id=str(vast_inst.id),
                    provider=self,
                    spot=use_interruptible,
                    private_ip=vast_inst.ssh_host,  # Vast.ai only has public
                    public_ip=vast_inst.ssh_host,
                    node=i,
                    metadata=frozenset(
                        [
                            ("cluster_id", cluster_id),
                            ("offer_id", str(vast_inst.offer_id)),
                            ("machine_id", str(vast_inst.machine_id)),
                            ("ssh_host", vast_inst.ssh_host),
                            ("ssh_port", str(vast_inst.ssh_port)),
                            ("username", "root"),
                            ("accelerator_count", str(spec.accelerator_count if spec else 0)),
                        ]
                    ),
                )
                instances.append(instance)

                provisioned = ProvisionedInstance(
                    instance_id=str(vast_inst.id),
                    node=i,
                    provider=ProviderName.VastAI,
                    spot=use_interruptible,
                    spec=spec,
                    ip=vast_inst.ssh_host,
                )
                provisioned_instances.append(provisioned)
                emit(InstanceProvisioned(instance=provisioned))

            emit(
                ProvisioningCompleted(
                    instances=tuple(provisioned_instances),
                    provider=ProviderName.VastAI,
                    region=self.geolocation or "global",
                )
            )

            logger.info(f"Provisioned {len(instances)} Vast.ai instances")
            return tuple(instances)

        except Exception as e:
            logger.error(f"Vast.ai provisioning failed: {e}")
            emit(Error(message=f"Provision failed: {e}"))
            raise

    def setup(self, instances: tuple[Instance, ...], compute: ComputeSpec) -> None:
        """Setup instances (wait for bootstrap, install wheel)."""
        from collections.abc import Generator
        from contextlib import contextmanager

        logger.info(f"Setting up {len(instances)} Vast.ai instances...")

        try:
            provisioned_map = {inst.id: self._make_provisioned(inst) for inst in instances}

            for inst in instances:
                emit(BootstrapStarting(instance=provisioned_map[inst.id]))

            key_path = get_private_key_path()

            def get_ip(inst: Instance) -> str:
                return inst.get_meta("ssh_host", "")

            def get_port(inst: Instance) -> int:
                return int(inst.get_meta("ssh_port", "22"))

            def make_provisioned(inst: Instance) -> ProvisionedInstance:
                return provisioned_map[inst.id]

            @contextmanager
            def ssh_transport(inst: Instance) -> Generator[SSHTransport]:
                ssh_host = inst.get_meta("ssh_host", "")
                ssh_port = int(inst.get_meta("ssh_port", "22"))
                yield SSHTransport(
                    host=ssh_host,
                    username="root",
                    key_path=key_path,
                    port=ssh_port,
                )

            # Wait for bootstrap using SSH
            wait_for_ssh_bootstrap(
                instances, get_ip, make_provisioned, timeout=300, get_port=get_port
            )

            # Install skyward wheel (Docker containers don't have systemd)
            install_skyward_wheel_via_transport(
                instances, ssh_transport, compute=compute, use_systemd=False
            )

            for inst in instances:
                emit(BootstrapCompleted(instance=provisioned_map[inst.id]))

            logger.info(f"Setup completed for {len(instances)} instances")

        except Exception as e:
            logger.error(f"Vast.ai setup failed: {e}")
            emit(Error(message=f"Setup failed: {e}"))
            raise

    def shutdown(
        self, instances: tuple[Instance, ...], compute: ComputeSpec
    ) -> tuple[ExitedInstance, ...]:
        """Shutdown (destroy) Vast.ai instances."""
        from contextlib import suppress

        logger.info(f"Shutting down {len(instances)} Vast.ai instances...")
        client = self._get_client()

        exited: list[ExitedInstance] = []
        for inst in instances:
            provisioned = self._make_provisioned(inst)
            emit(InstanceStopping(instance=provisioned))

            instance_id = int(inst.id)
            logger.debug(f"Destroying instance {instance_id}")

            with suppress(Exception):
                client.destroy_instance(id=instance_id)

            exited.append(
                ExitedInstance(
                    instance=inst,
                    exit_code=0,
                    exit_reason="normal",
                )
            )

        self._instances = []
        logger.info(f"Shutdown complete: {len(exited)} instances destroyed")
        return tuple(exited)

    def create_tunnel(
        self, instance: Instance, remote_port: int = 18861
    ) -> tuple[int, subprocess.Popen[bytes]]:
        """Create SSH tunnel to instance."""
        transport = self._get_transport(instance)
        return transport.create_tunnel(remote_port)

    def run_command(self, instance: Instance, command: str, timeout: int = 30) -> str:
        """Run shell command on instance via SSH."""
        transport = self._get_transport(instance)
        return transport.run_command(command, timeout)

    def discover_peers(self, cluster_id: str) -> tuple[Instance, ...]:
        """Discover peer instances by cluster label.

        Note: Vast.ai doesn't have native tagging, so we use labels.
        """
        client = self._get_client()

        # List all instances and filter by label
        all_instances = client.show_instances()

        if not all_instances:
            return ()

        # Handle both dict and list responses
        if isinstance(all_instances, dict):
            all_instances = all_instances.get("instances", [])

        instances: list[Instance] = []
        for inst_data in all_instances:
            label = inst_data.get("label", "")
            if not label.startswith(f"skyward-{cluster_id}-"):
                continue

            if inst_data.get("actual_status") != "running":
                continue

            # Get SSH info from instance data
            ssh_host = inst_data.get("ssh_host", "")
            ssh_port = inst_data.get("ssh_port", 22)
            if not ssh_host:
                continue

            instances.append(
                Instance(
                    id=str(inst_data["id"]),
                    provider=self,
                    spot=inst_data.get("is_bid", False),
                    private_ip=ssh_host,
                    public_ip=ssh_host,
                    node=0,
                    metadata=frozenset(
                        [
                            ("cluster_id", cluster_id),
                            ("ssh_host", ssh_host),
                            ("ssh_port", str(ssh_port)),
                        ]
                    ),
                )
            )

        # Sort by IP and assign node indices
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
        """List available offers as InstanceSpecs.

        Note: Vast.ai is a dynamic marketplace, so this searches
        current offers rather than returning a static list.
        """
        client = self._get_client()
        return fetch_available_offers(
            client,
            min_reliability=self.min_reliability,
            geolocation=self.geolocation,
        )
