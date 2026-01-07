"""Vast.ai provider for Skyward GPU instances.

Vast.ai is a GPU marketplace with dynamic offers from various hosts.
Instances are Docker containers, not VMs, so bootstrap works differently
from traditional cloud providers.
"""

from __future__ import annotations

import os
import random
import string
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from skyward.callback import emit
from skyward.events import (
    BootstrapCompleted,
    BootstrapStarting,
    InstanceLaunching,
    InstanceProvisioned,
    InstanceStopping,
    NetworkReady,
    ProviderName,
    ProvisionedInstance,
    ProvisioningCompleted,
    ProvisioningStarted,
)
from skyward.internal.decorators import audit
from skyward.providers.base import get_private_key_path, poll_instances
from skyward.providers.common import (
    install_skyward_wheel_via_transport,
    make_provisioned,
    wait_for_ssh_bootstrap,
)
from skyward.providers.ssh import SSHConfig
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

from ...accelerators import AcceleratorSpec
from .discovery import (
    build_search_query,
    extract_cuda_version,
    fetch_available_offers,
    offer_to_instance_spec,
    search_offers,
)
from .overlay import OverlayError
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

    # Overlay networking (multi-node)
    use_overlay: bool = True
    """Enable overlay networking for multi-node clusters.

    When True (default) and nodes > 1, the provider will:
    1. Search for offers in physical clusters (cluster_id != null)
    2. Create an overlay network on the selected cluster
    3. Join all instances to the overlay
    4. Use overlay IPs for inter-node communication (NCCL, etc.)

    Set to False to disable overlay networking (not recommended for
    distributed training as NCCL won't work without direct connectivity).
    """

    overlay_timeout: int = 120
    """Timeout in seconds for overlay network operations (create, join)."""

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
            use_overlay=self.use_overlay,
            overlay_timeout=self.overlay_timeout,
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
        use_overlay: bool = True,
        overlay_timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.min_reliability = min_reliability
        self.geolocation = geolocation
        self.bid_multiplier = bid_multiplier
        self.instance_timeout = instance_timeout
        self.docker_image = docker_image
        self.disk_gb = disk_gb
        self.use_overlay = use_overlay
        self.overlay_timeout = overlay_timeout

        # Mutable runtime state
        self._ssh_key_id: int | None = None
        self._ssh_public_key: str | None = None
        self._instances: list[_VastInstance] = []

        # Overlay network state
        self._overlay_name: str | None = None
        self._overlay_cluster_id: int | None = None
        self._overlay_ips: dict[int, str] = {}  # instance_id -> overlay IP
        self._overlay_ifaces: dict[int, str] = {}  # instance_id -> interface name

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

    def _generate_onstart_script(self, compute: ComputeSpec) -> str:
        """Generate onstart script for Vast.ai container.

        Unlike cloud-init, this runs inside the Docker container.
        We bootstrap Python environment and dependencies.

        When skyward_source is "github" or "pypi", the script also
        installs skyward and starts the RPyC service via nohup.
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

# Install base dependencies (git needed for github install)
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

        # Install skyward if not using local wheel
        if compute.image.skyward_source == "github":
            script += """
# Install skyward from GitHub
uv pip install "git+https://github.com/gabfssilva/skyward.git"
touch /opt/skyward/.step_wheel
"""
        elif compute.image.skyward_source == "pypi":
            script += """
# Install skyward from PyPI
uv pip install skyward
touch /opt/skyward/.step_wheel
"""
        # "local" mode: skyward installed via SCP after bootstrap

        # Start RPyC service if skyward is installed via user-data
        if compute.image.skyward_source != "local":
            script += """
# Start RPyC server via nohup (Docker doesn't have systemd)
cd /opt/skyward
nohup .venv/bin/python -m skyward.rpc > /var/log/skyward-rpyc.log 2>&1 &
echo $! > /var/run/skyward-rpyc.pid

# Wait for RPyC port (18861)
for i in $(seq 1 30); do
    (echo > /dev/tcp/127.0.0.1/18861) 2>/dev/null && break || sleep 1
done
touch /opt/skyward/.step_server
"""

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

        def fetch_status(inst: _VastInstance) -> tuple[str, dict[str, Any]]:
            result = client.show_instance(id=inst.id)
            if not result:
                return "pending", {}

            # Handle dict or direct result
            if isinstance(result, dict):
                status = result.get("actual_status", "")
                ssh_host = result.get("ssh_host", "")
                ssh_port = result.get("ssh_port", 0)
            else:
                status = getattr(result, "actual_status", "")
                ssh_host = getattr(result, "ssh_host", "")
                ssh_port = getattr(result, "ssh_port", 0)

            # Treat missing SSH as not ready yet
            if status == "running" and not (ssh_host and ssh_port):
                status = "pending"

            return status, {
                "actual_status": status,
                "ssh_host": ssh_host,
                "ssh_port": int(ssh_port) if ssh_port else 0,
            }

        def update_instance(inst: _VastInstance, info: dict[str, Any]) -> None:
            inst.actual_status = info.get("actual_status", "")
            inst.ssh_host = info.get("ssh_host", "")
            inst.ssh_port = info.get("ssh_port", 22)

        poll_instances(
            instances=instances,
            fetch_status=fetch_status,
            target_status="running",
            update_instance=update_instance,
            timeout=timeout,
        )

    def _create_overlay_network(self) -> None:
        """Create overlay network before instances.

        Per VastAI docs, overlay must be created BEFORE instances.
        Instances attach to the overlay via env="-n OVERLAY_NAME" parameter.

        Raises:
            RuntimeError: If overlay creation fails.
        """
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        from skyward.providers.vastai.overlay import (
            OverlayError,
            create_overlay,
        )

        # VastAI overlay names only accept lowercase letters and hyphens (no numbers!)
        suffix = "".join(random.choices(string.ascii_lowercase, k=8))
        self._overlay_name = f"skyward-{suffix}"
        logger.info(f"Creating overlay network: {self._overlay_name}")

        @retry(
            retry=retry_if_exception_type(OverlayError),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
        def _create_with_retry() -> None:
            create_overlay(
                cluster_id=self._overlay_cluster_id,  # type: ignore[arg-type]
                overlay_name=self._overlay_name,
                api_key=self._api_key,
                timeout=self.overlay_timeout,
            )

        try:
            _create_with_retry()
            logger.info(f"Overlay network '{self._overlay_name}' created successfully")
        except OverlayError as e:
            logger.error(f"Overlay network creation failed: {e}")
            self._cleanup_overlay()
            raise RuntimeError(
                f"Failed to create overlay network: {e}. "
                "Multi-node VastAI requires overlay networking for NCCL. "
                "Use VastAI(use_overlay=False) to disable (not recommended)."
            ) from e

    def _join_instances_to_overlay(self, vast_instances: list[_VastInstance]) -> None:
        """Join all instances to the overlay network concurrently."""
        from skyward.conc import for_each_async

        logger.info(f"Joining {len(vast_instances)} instances to overlay '{self._overlay_name}'")

        try:
            for_each_async(
                lambda inst: self._join_single_instance_to_overlay(inst.id),
                vast_instances,
            )
            logger.info(f"All instances joined overlay '{self._overlay_name}'")
        except OverlayError as e:
            logger.error(f"Failed to join instances to overlay: {e}")
            self._cleanup_overlay()
            raise RuntimeError(f"Failed to join instances to overlay: {e}") from e

    @retry(
        retry=retry_if_exception_type(OverlayError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _join_single_instance_to_overlay(self, instance_id: int) -> None:
        """Join a single instance to overlay with retry."""
        from skyward.providers.vastai.overlay import join_overlay

        join_overlay(
            overlay_name=self._overlay_name,  # type: ignore[arg-type]
            instance_id=instance_id,
            api_key=self._api_key,
            timeout=self.overlay_timeout,
        )

    def _cleanup_overlay(self) -> None:
        """Delete overlay network if it exists."""
        if self._overlay_name:
            from skyward.providers.vastai.overlay import delete_overlay

            logger.info(f"Deleting overlay network: {self._overlay_name}")
            delete_overlay(
                overlay_name=self._overlay_name,
                api_key=self._api_key,
                timeout=self.overlay_timeout,
            )
            self._overlay_name = None
            self._overlay_cluster_id = None
            self._overlay_ips = {}
            self._overlay_ifaces = {}

    def _detect_overlay_ips(
        self,
        instances: list[Instance],
    ) -> None:
        """Detect overlay network IPs and interfaces for all instances via SSH.

        After joining an overlay, each instance gets a network interface
        with the overlay IP. This method detects those IPs and interfaces via SSH,
        then updates each Instance's private_ip and metadata.

        Args:
            instances: List of Instance objects to probe.

        Raises:
            RuntimeError: If overlay IP cannot be detected for any instance.
        """
        import time

        from skyward.providers.vastai.overlay import get_instance_overlay_info

        logger.info("Detecting overlay network IPs...")

        for inst in instances:
            # Retry loop - overlay interface may take time to appear
            overlay_info: tuple[str, str] | None = None
            for _attempt in range(10):
                overlay_info = get_instance_overlay_info(inst)
                if overlay_info:
                    ip, iface = overlay_info
                    inst_id = int(inst.id)
                    self._overlay_ips[inst_id] = ip
                    self._overlay_ifaces[inst_id] = iface
                    # Update Instance with overlay info
                    inst.private_ip = ip
                    inst.metadata = inst.metadata | frozenset([("network_interface", iface)])
                    logger.debug(f"Instance {inst.id} overlay: {ip} on {iface}")
                    break
                time.sleep(1)

            if not overlay_info:
                raise RuntimeError(
                    f"Failed to detect overlay IP for instance {inst.id}. "
                    "The instance may not have joined the overlay network correctly."
                )

        logger.info(f"Detected overlay IPs for {len(self._overlay_ips)} instances")

    @audit("Provisioning")
    def provision(self, compute: ComputeSpec) -> tuple[Instance, ...]:
        """Provision Vast.ai container instances."""
        cluster_id = str(uuid.uuid4())[:8]
        logger.debug(f"Cluster ID: {cluster_id}, nodes: {compute.nodes}")
        emit(ProvisioningStarted())

        # Determine if overlay networking is needed
        needs_overlay = compute.nodes > 1 and self.use_overlay
        if needs_overlay:
            logger.debug("Multi-node cluster: overlay networking enabled")

        client = self._get_client()

        # Ensure SSH key is registered
        self._ssh_key_id, self._ssh_public_key = get_or_create_ssh_key(client)
        emit(NetworkReady(region=self.geolocation or "global"))

        # Parse compute requirements
        acc = AcceleratorSpec.from_value(compute.accelerator)
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

        # Track preselected cluster from CUDA version filtering (for multi-node overlay)
        preselected_cluster: tuple[int, list[dict[str, Any]]] | None = None

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

            # Search offers (GPU and CUDA filtering done client-side due to VastAI bugs)
            # VastAI's query parser has issues when combining gpu_name with other filters
            query = build_search_query(
                gpu_name=gpu_name,  # Don't filter by GPU in query (buggy)
                num_gpus=num_gpus,
                min_cpu=compute.cpu,
                min_memory_gb=min_mem_gb,
                # min_cuda_version=min_cuda,  # Don't filter by CUDA in query (buggy)
                min_disk_gb=self.disk_gb,
                min_reliability=self.min_reliability,
                geolocation=self.geolocation,
                require_cluster=needs_overlay,
            )
            logger.debug(f"Searching offers with query: {query}")
            all_offers = search_offers(client, query)

            logger.debug(f"Found {len(all_offers)} offers for {gpu_name}")

            # Filter by GPU name client-side
            if gpu_name:
                all_offers = [
                    o for o in all_offers if o.get("gpu_name", "").replace(" ", "_") == gpu_name
                ]
                logger.debug(f"Filtered to {len(all_offers)} offers for {gpu_name}")

            # Filter by exact GPU count client-side (query uses >= for broader search)
            if num_gpus:
                all_offers = [o for o in all_offers if o.get("num_gpus") == num_gpus]
                logger.debug(f"Filtered to {len(all_offers)} offers with {num_gpus} GPU(s)")

                # Filter by CUDA version client-side (try from max to min)
                # For multi-node with overlay, we need enough offers in a single cluster
                from skyward.providers.vastai.discovery import select_best_cluster

                offers = []
                selected_cuda = None

                for cuda_ver in cuda_versions:
                    if cuda_ver is None:
                        offers = all_offers
                        break
                    cuda_offers = [
                        o for o in all_offers if (o.get("cuda_max_good") or 0) >= cuda_ver
                    ]
                    if not cuda_offers:
                        logger.debug(f"No offers with CUDA >= {cuda_ver}, trying lower...")
                        continue

                    # For multi-node overlay, verify cluster has enough capacity
                    if needs_overlay:
                        cluster_result = select_best_cluster(
                            cuda_offers,
                            nodes_needed=compute.nodes,
                            use_interruptible=use_interruptible,
                        )
                        if cluster_result is None:
                            logger.debug(
                                f"Found {len(cuda_offers)} offers with CUDA >= {cuda_ver}, "
                                f"but no cluster with {compute.nodes} nodes. Trying lower..."
                            )
                            continue
                        # Found a valid cluster - store for later use
                        preselected_cluster = cluster_result
                        offers = cuda_offers
                        selected_cuda = cuda_ver
                        logger.debug(
                            f"Found {len(cuda_offers)} offers with CUDA >= {cuda_ver}, "
                            f"cluster {cluster_result[0]} available with {compute.nodes}+ nodes"
                        )
                        break
                    else:
                        # Single node or no overlay - any offers will do
                        offers = cuda_offers
                        selected_cuda = cuda_ver
                        logger.debug(f"Found {len(cuda_offers)} offers with CUDA >= {cuda_ver}")
                        break

                # Build docker image if not user-specified
                if docker_image is None and selected_cuda:
                    docker_image = f"nvcr.io/nvidia/cuda:{selected_cuda:.1f}.0-runtime-ubuntu22.04"
                    logger.debug(f"Selected Docker image: {docker_image}")

            if not offers:
                if needs_overlay:
                    raise RuntimeError(
                        f"No Vast.ai offers found in physical clusters matching: {query}. "
                        "Multi-node requires offers in physical clusters "
                        "(cluster_id!=None) for overlay networking. "
                        "Options: 1) Try a different GPU type with cluster availability, "
                        "2) Use VastAI(use_overlay=False) to disable overlay "
                        "(not recommended for NCCL), "
                        "3) Lower min_reliability or broadening accelerator requirements."
                    )
                raise RuntimeError(
                    f"No Vast.ai offers found matching: {query}. "
                    "Try lowering min_reliability or broadening accelerator requirements."
                )

            # Sort by price (interruptible uses min_bid, on-demand uses dph_total)
            if use_interruptible:
                offers.sort(key=lambda o: o.get("min_bid", float("inf")))
            else:
                offers.sort(key=lambda o: o.get("dph_total", float("inf")))

            # For multi-node with overlay, select offers from the same physical cluster
            if needs_overlay:
                # Reuse preselected cluster from CUDA version loop if available
                if preselected_cluster is not None:
                    cluster_result = preselected_cluster
                else:
                    cluster_result = select_best_cluster(
                        offers,
                        nodes_needed=compute.nodes,
                        use_interruptible=use_interruptible,
                    )

                if cluster_result is None:
                    raise RuntimeError(
                        f"No physical cluster found with {compute.nodes} available "
                        f"offers for {compute.accelerator}. Multi-node VastAI requires "
                        "instances in the same physical cluster for overlay networking. "
                        "Try reducing node count or use VastAI(use_overlay=False) to "
                        "disable (not recommended for NCCL)."
                    )

                self._overlay_cluster_id, offers = cluster_result
                logger.info(
                    f"Selected physical cluster {self._overlay_cluster_id} "
                    f"with {len(offers)} available offers"
                )

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

            # Create overlay network BEFORE instances (per VastAI docs)
            # Instances will be attached to the overlay via env parameter
            if needs_overlay:
                self._create_overlay_network()

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

                    # Attach to overlay network if enabled (must exist before instance creation)
                    if self._overlay_name:
                        create_params["env"] = f"-n {self._overlay_name}"

                    if offer_bid_price is not None:
                        create_params["price"] = offer_bid_price

                    result = client.create_instance(**create_params)

                    # Empty response means offer is no longer available
                    if not result:
                        logger.debug(f"Offer {current_offer['id']} unavailable, trying next...")
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
                    offer_index += 1  # Move to next offer for next instance

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

            # Build Instance objects first (needed for overlay detection)
            key_path = get_private_key_path()
            for i, vast_inst in enumerate(vast_instances):
                meta_items: list[tuple[str, str]] = [
                    ("cluster_id", cluster_id),
                    ("offer_id", str(vast_inst.offer_id)),
                    ("machine_id", str(vast_inst.machine_id)),
                    ("ssh_host", vast_inst.ssh_host),
                    ("ssh_port", str(vast_inst.ssh_port)),
                    ("username", "root"),
                    ("accelerator_count", str(spec.accelerator_count if spec else 0)),
                ]

                instance = Instance(
                    id=str(vast_inst.id),
                    provider=self,
                    ssh=SSHConfig(
                        host=vast_inst.ssh_host,
                        username="root",
                        port=vast_inst.ssh_port,
                        key_path=key_path,
                    ),
                    spot=use_interruptible,
                    private_ip=vast_inst.ssh_host,
                    public_ip=vast_inst.ssh_host,
                    node=i,
                    metadata=frozenset(meta_items),
                )
                instances.append(instance)

            # Join instances to overlay network if enabled
            # Note: env="-n NAME" in SDK doesn't auto-attach, must use join_overlay
            if needs_overlay and self._overlay_name is not None:
                self._join_instances_to_overlay(vast_instances)
                # Detect overlay IPs and update Instance objects
                self._detect_overlay_ips(instances)

            # Emit provisioned events for all instances
            for inst in instances:
                provisioned = ProvisionedInstance(
                    instance_id=inst.id,
                    node=inst.node,
                    provider=ProviderName.VastAI,
                    spot=inst.spot,
                    spec=spec,
                    ip=inst.public_ip or inst.private_ip,
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

        return tuple(instances)

    @audit("Setup")
    def setup(self, instances: tuple[Instance, ...], compute: ComputeSpec) -> None:
        """Setup instances (wait for bootstrap, install wheel)."""
        provisioned_map = {
            inst.id: make_provisioned(inst, ProviderName.VastAI, ip=inst.get_meta("ssh_host"))
            for inst in instances
        }

        for inst in instances:
            emit(BootstrapStarting(instance=provisioned_map[inst.id]))

        def get_ip(inst: Instance) -> str:
            return inst.get_meta("ssh_host", "")

        def get_port(inst: Instance) -> int:
            return int(inst.get_meta("ssh_port", "22"))

        def get_provisioned(inst: Instance) -> ProvisionedInstance:
            return provisioned_map[inst.id]

        # Wait for bootstrap using SSH
        wait_for_ssh_bootstrap(instances, get_ip, get_provisioned, timeout=300, get_port=get_port)

        # Install skyward wheel (Docker containers don't have systemd)
        install_skyward_wheel_via_transport(
            instances,
            get_transport=lambda inst: inst.connect(),
            compute=compute,
            use_systemd=False,
        )

        for inst in instances:
            emit(BootstrapCompleted(instance=provisioned_map[inst.id]))

    def shutdown(
        self, instances: tuple[Instance, ...], compute: ComputeSpec
    ) -> tuple[ExitedInstance, ...]:
        """Shutdown (destroy) Vast.ai instances."""
        from contextlib import suppress

        logger.info(f"Shutting down {len(instances)} Vast.ai instances...")
        client = self._get_client()

        exited: list[ExitedInstance] = []
        for inst in instances:
            provisioned = make_provisioned(inst, ProviderName.VastAI, ip=inst.get_meta("ssh_host"))
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

        # Cleanup overlay network if it exists
        self._cleanup_overlay()

        self._instances = []
        logger.info(f"Shutdown complete: {len(exited)} instances destroyed")
        return tuple(exited)

    def discover_peers(self, cluster_id: str) -> tuple[Instance, ...]:
        """Discover peer instances by cluster label.

        Note: Vast.ai doesn't have native tagging, so we use labels.
        """
        from skyward.providers.base import assign_node_indices

        client = self._get_client()
        all_instances = client.show_instances()

        if not all_instances:
            return ()

        # Handle both dict and list responses
        if isinstance(all_instances, dict):
            all_instances = all_instances.get("instances", [])

        key_path = get_private_key_path()
        instances: list[Instance] = []

        for inst_data in all_instances:
            label = inst_data.get("label", "")
            if not label.startswith(f"skyward-{cluster_id}-"):
                continue

            if inst_data.get("actual_status") != "running":
                continue

            ssh_host = inst_data.get("ssh_host", "")
            ssh_port = inst_data.get("ssh_port", 22)
            if not ssh_host:
                continue

            instances.append(
                Instance(
                    id=str(inst_data["id"]),
                    provider=self,
                    ssh=SSHConfig(host=ssh_host, username="root", port=ssh_port, key_path=key_path),
                    spot=inst_data.get("is_bid", False),
                    private_ip=ssh_host,
                    public_ip=ssh_host,
                    node=0,
                    metadata=frozenset([
                        ("cluster_id", cluster_id),
                        ("ssh_host", ssh_host),
                        ("ssh_port", str(ssh_port)),
                    ]),
                )
            )

        return assign_node_indices(instances)

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
