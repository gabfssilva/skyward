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
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skyward.types.spec import InstanceStatus

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from skyward.core.callback import emit
from skyward.core.events import (
    InstanceLaunching,
    InstanceProvisioned,
    NetworkReady,
    ProviderName,
    ProvisionedInstance,
    ProvisioningCompleted,
)
from skyward.internal.decorators import audit
from skyward.providers.base import get_private_key_path, poll_instances
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

from ...accelerators import AcceleratorSpec
from ...utils import map_async, for_each_async
from .client import (
    Offer,
    VastAIClient,
    VastAIError,
    extract_cuda_version,
    select_all_valid_clusters,
    select_best_cluster,
)

# =============================================================================
# Minimal Onstart Script
# =============================================================================


def _generate_minimal_onstart() -> str:
    """Generate minimal onstart script for VastAI containers.

    VastAI has a 4048-character limit on onstart_cmd, but the full bootstrap
    script is much larger. This minimal script just prepares the container
    for SSH access - the full bootstrap is executed later via SSH.
    """
    return """#!/bin/bash
set -e
mkdir -p /opt/skyward
tail -f /dev/null
"""


# =============================================================================
# Internal Types
# =============================================================================


@dataclass
class _VastInstance:
    """Internal representation of a Vast.ai instance during provisioning."""

    id: int
    offer: Offer
    ssh_host: str = ""
    ssh_port: int = 22
    actual_status: str = ""
    dph_total: float = 0.0  # Actual $/hr being charged
    public_ipaddr: str = ""  # Direct connection IP
    direct_port: int | None = None  # Direct SSH port


# =============================================================================
# Provider Configuration
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
            allocation="spot-if-available",
            pip=["torch", "transformers"],
        )

        with pool:
            result = train(data) >> pool

    Environment Variables:
        VAST_API_KEY: API key (required if not passed directly)

    Attributes:
        api_key: Vast.ai API key. Falls back to VAST_API_KEY env var.
        min_reliability: Minimum host reliability score (0.0-1.0).
        geolocation: Preferred region/country code (e.g., "US", "EU").
        bid_multiplier: For spot pricing, multiply min bid by this.
        instance_timeout: Auto-shutdown in seconds (safety timeout).
        docker_image: Base Docker image for containers.
        disk_gb: Disk space in GB.
        use_overlay: Enable overlay networking for multi-node clusters.
        overlay_timeout: Timeout for overlay operations in seconds.
    """

    api_key: str | None = None
    min_reliability: float = 0.95
    geolocation: str | None = None
    bid_multiplier: float = 1.2
    instance_timeout: int = 300
    docker_image: str | None = None
    disk_gb: int = 100
    use_overlay: bool = True
    overlay_timeout: int = 120

    @classmethod
    def ubuntu(
        cls,
        version: Literal["22.04", "24.04", "26.04"] | str = "24.04",
        cuda: Literal["12.9.1", "13.1.0", "13.0.1"] | str = "12.9.1",
        cuda_dist: Literal['devel', 'runtime'] = "runtime",
    ) -> str:
        return f"nvcr.io/nvidia/cuda:{cuda}-{cuda_dist}-ubuntu{version}"

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


# =============================================================================
# Provider Implementation
# =============================================================================


class VastAIProvider(Provider):
    """Stateful Vast.ai provider service."""

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
        self._client: VastAIClient | None = None

        # Overlay network state
        self._overlay_name: str | None = None
        self._overlay_cluster_id: int | None = None
        self._overlay_ips: dict[int, str] = {}
        self._overlay_ifaces: dict[int, str] = {}

    @property
    def name(self) -> str:
        return "vastai"

    @property
    def _api_key(self) -> str:
        """Get API key from config, environment, or config file."""
        if self.api_key:
            return self.api_key

        if env_key := os.environ.get("VAST_API_KEY"):
            return env_key

        config_path = os.path.expanduser("~/.config/vastai/vast_api_key")
        if os.path.exists(config_path):
            with suppress(OSError), open(config_path) as f:
                if file_key := f.read().strip():
                    return file_key

        raise ValueError(
            "Vast.ai API key not provided. Options:\n"
            "  1. Pass api_key to VastAI()\n"
            "  2. Set VAST_API_KEY environment variable\n"
            "  3. Run: vastai set api-key YOUR_KEY"
        )

    def _get_client(self) -> VastAIClient:
        """Get or create authenticated Vast.ai client (singleton per provider)."""
        if self._client is None:
            self._client = VastAIClient(api_key=self._api_key)
        return self._client

    @audit("Destroy instance")
    def _destroy_instance(self, instance_id: int) -> None:
        """Destroy Vast.ai instance."""
        with suppress(Exception):
            self._get_client().destroy_instance(instance_id)

    def _wait_for_running(
        self,
        client: VastAIClient,
        instances: list[_VastInstance],
        timeout: float = 300,
    ) -> None:
        """Wait for all instances to be running and populate SSH info."""

        def fetch_status(inst: _VastInstance) -> tuple[str, dict[str, str | int | float | None]]:
            info = client.get_instance(inst.id)
            if not info:
                return "pending", {}

            status = info.actual_status
            if status == "running" and not (info.ssh_host and info.ssh_port):
                status = "pending"

            return status, {
                "actual_status": status,
                "ssh_host": info.ssh_host,
                "ssh_port": info.ssh_port,
                "dph_total": info.dph_total,
                "public_ipaddr": info.public_ipaddr,
                "direct_port": info.direct_port,
            }

        def update_instance(inst: _VastInstance, info: dict[str, str | int | float | None]) -> None:
            inst.actual_status = str(info.get("actual_status", ""))
            inst.ssh_host = str(info.get("ssh_host", ""))
            inst.ssh_port = int(info.get("ssh_port") or 22)
            inst.dph_total = float(info.get("dph_total") or 0)
            inst.public_ipaddr = str(info.get("public_ipaddr") or "")
            inst.direct_port = int(info["direct_port"]) if info.get("direct_port") else None

        poll_instances(
            instances=instances,
            fetch_status=fetch_status,
            target_status="running",
            update_instance=update_instance,
            timeout=timeout,
        )

    def _create_overlay_network(self, client: VastAIClient) -> None:
        """Create overlay network before instances."""
        suffix = "".join(random.choices(string.ascii_lowercase, k=8))
        self._overlay_name = f"skyward-{suffix}"
        logger.info(f"Creating overlay network: {self._overlay_name}")

        @retry(
            retry=retry_if_exception_type(VastAIError),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
        def _create_with_retry() -> None:
            client.create_overlay(
                cluster_id=self._overlay_cluster_id,  # type: ignore[arg-type]
                name=self._overlay_name,  # type: ignore[arg-type]
            )

        try:
            _create_with_retry()
            logger.info(f"Overlay network '{self._overlay_name}' created")
        except VastAIError as e:
            logger.error(f"Overlay creation failed: {e}")
            self._cleanup_overlay()
            raise RuntimeError(f"Failed to create overlay: {e}") from e

    def _join_instances_to_overlay(
        self,
        client: VastAIClient,
        vast_instances: list[_VastInstance],
    ) -> None:
        """Join all instances to the overlay network."""
        logger.info(f"Joining {len(vast_instances)} instances to overlay '{self._overlay_name}'")

        @retry(
            retry=retry_if_exception_type(VastAIError),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
        def _join_single(instance_id: int) -> None:
            client.join_overlay(name=self._overlay_name, instance_id=instance_id)  # type: ignore[arg-type]

        try:
            for_each_async(lambda inst: _join_single(inst.id), vast_instances)
            logger.info(f"All instances joined overlay '{self._overlay_name}'")
        except VastAIError as e:
            logger.error(f"Failed to join overlay: {e}")
            self._cleanup_overlay()
            raise RuntimeError(f"Failed to join overlay: {e}") from e

    def _cleanup_overlay(self) -> None:
        """Delete overlay network if it exists."""
        if self._overlay_name:
            logger.info(f"Deleting overlay: {self._overlay_name}")
            self._get_client().delete_overlay(self._overlay_name)
            self._overlay_name = None
            self._overlay_cluster_id = None
            self._overlay_ips = {}
            self._overlay_ifaces = {}

    def _detect_overlay_ips(self, instances: list[Instance]) -> None:
        """Detect overlay network IPs via SSH."""
        from tenacity import retry, stop_after_delay, wait_fixed

        def _get_overlay_info(instance: Instance) -> tuple[str, str] | None:
            try:
                cmd = """
IFACE=$(awk 'NR>1 && substr($2,7,2)=="0A" {print $1; exit}' /proc/net/route)
IP=$(hostname -I | tr ' ' '\\n' | grep '^10\\.')
echo "$IFACE $IP"
"""
                output = instance.run_command(cmd.strip(), timeout=10)
                parts = output.strip().split()
                if len(parts) >= 2:
                    return parts[1], parts[0]
            except Exception as e:
                logger.debug(f"Failed to get overlay info: {e}")
            return None

        @retry(
            stop=stop_after_delay(300),
            wait=wait_fixed(1),
            reraise=True,
            retry=retry_if_exception_type(RuntimeError),
        )
        def get_overlay_info(i: Instance) -> tuple[str, str]:
            info = _get_overlay_info(i)
            if not info:
                raise RuntimeError(f"Failed to get overlay info for {i}")
            return info

        logger.info("Detecting overlay network IPs...")
        fetched = map_async(lambda i: (i, get_overlay_info(i)), instances)

        for instance, overlay_info in fetched:
            ip, iface = overlay_info
            inst_id = int(instance.id)
            self._overlay_ips[inst_id] = ip
            self._overlay_ifaces[inst_id] = iface
            instance.private_ip = ip
            instance.metadata |= frozenset([("network_interface", iface)])

        logger.info(f"Detected overlay IPs for {len(self._overlay_ips)} instances")

    @audit("Provisioning")
    def provision(self, compute: ComputeSpec) -> tuple[Instance, ...]:
        """Provision Vast.ai container instances."""
        cluster_id = str(uuid.uuid4())[:8]
        logger.debug(f"Cluster ID: {cluster_id}, nodes: {compute.nodes}")

        needs_overlay = compute.nodes > 1 and self.use_overlay
        if needs_overlay:
            logger.debug("Multi-node cluster: overlay networking enabled")

        client = self._get_client()

        # Ensure SSH key is registered
        self._ssh_key_id, self._ssh_public_key = client.ensure_ssh_key()
        emit(NetworkReady(region=self.geolocation or "global"))

        # Parse compute requirements
        acc = AcceleratorSpec.from_value(compute.accelerator)
        allocation = normalize_allocation(compute.allocation)
        use_interruptible = not isinstance(allocation, _AllocationOnDemand)
        logger.debug(f"Allocation: {allocation}, use_interruptible={use_interruptible}")

        # Determine CUDA version and Docker image
        docker_image: str | None = None
        if self.docker_image:
            docker_image = self.docker_image
            min_cuda = extract_cuda_version(docker_image)
            cuda_versions: list[float | None] = [min_cuda] if min_cuda else [None]
        elif acc and acc.metadata.get("cuda"):
            cuda_meta = acc.metadata["cuda"]
            cuda_max = float(cuda_meta.get("max", "12.4"))
            cuda_min = float(cuda_meta.get("min", "11.0"))
            cuda_versions = []
            v = cuda_max
            while v >= cuda_min:
                cuda_versions.append(v)
                v = round(v - 0.1, 1)
        else:
            docker_image = VastAI.ubuntu(cuda="12.9.1")
            cuda_versions = [12.9]

        # Search for offers
        gpu_name = None
        num_gpus = None
        if acc:
            gpu_name = acc.accelerator.replace(" ", "_").replace("-", "_")
            if not callable(acc.count):
                num_gpus = acc.count

        min_mem_gb = parse_memory_mb(compute.memory) / 1024 if compute.memory else None

        if compute.machine is not None:
            all_offers = client.search_by_machine_id(compute.machine, use_interruptible)
        else:
            all_offers = client.search_offers(
                gpu_name=gpu_name,
                num_gpus=num_gpus,
                min_cpu=compute.cpu,
                min_memory_gb=min_mem_gb,
                min_disk_gb=self.disk_gb,
                min_reliability=self.min_reliability,
                geolocation=self.geolocation,
                use_interruptible=use_interruptible,
                with_cluster_id=compute.nodes > 1,
            )

        logger.debug(f"Found {len(all_offers)} offers")

        # Filter by GPU name (with variant matching: H100 matches H100_NVL, H100_SXM, etc.)
        if gpu_name:
            req_norm = gpu_name.upper()
            all_offers = [
                o for o in all_offers
                if (offer_norm := o.gpu_name.replace(" ", "_").upper()) == req_norm
                or offer_norm.startswith(req_norm + "_")
            ]
            logger.debug(f"Filtered to {len(all_offers)} offers for {gpu_name}")

        # Filter by exact GPU count
        if num_gpus:
            all_offers = [o for o in all_offers if o.num_gpus == num_gpus]
            logger.debug(f"Filtered to {len(all_offers)} offers with {num_gpus} GPU(s)")

        # Filter by CUDA version and find cluster
        offers: list[Offer] = []
        preselected_cluster: tuple[int, list[Offer]] | None = None
        selected_cuda: float | None = None

        for cuda_ver in cuda_versions:
            if cuda_ver is None:
                offers = all_offers
                break

            cuda_offers = [o for o in all_offers if o.cuda_max_good >= cuda_ver]
            if not cuda_offers:
                logger.debug(f"No offers with CUDA >= {cuda_ver}, trying lower...")
                continue

            if needs_overlay:
                cluster_result = select_best_cluster(cuda_offers, compute.nodes, use_interruptible)
                if cluster_result is None:
                    logger.debug(f"No cluster with {compute.nodes} nodes for CUDA >= {cuda_ver}")
                    continue
                preselected_cluster = cluster_result
                offers = cuda_offers
                selected_cuda = cuda_ver
                logger.debug(f"Found cluster {cluster_result[0]} with CUDA >= {cuda_ver}")
                break
            else:
                offers = cuda_offers
                selected_cuda = cuda_ver
                logger.debug(f"Found {len(cuda_offers)} offers with CUDA >= {cuda_ver}")
                break

        if docker_image is None and selected_cuda:
            docker_image = f"nvcr.io/nvidia/cuda:{selected_cuda:.1f}.0-runtime-ubuntu22.04"
            logger.debug(f"Selected Docker image: {docker_image}")

        if not offers:
            # Find alternatives: search without GPU filter
            if compute.nodes > 1:
                alt_offers = client.search_offers(
                    num_gpus=num_gpus,
                    min_cpu=compute.cpu,
                    min_memory_gb=min_mem_gb,
                    min_disk_gb=self.disk_gb,
                    min_reliability=self.min_reliability,
                    geolocation=self.geolocation,
                    use_interruptible=use_interruptible,
                    with_cluster_id=compute.nodes > 1,
                )
                alternatives = client.find_available_accelerators(
                    alt_offers, compute.nodes, num_gpus
                )
                if alternatives:
                    alt_list = ", ".join(
                        f"{name} (max {sz} nodes)" for name, sz, _ in alternatives[:5]
                    )
                    logger.info(f"Available accelerators for {compute.nodes}+ nodes: {alt_list}")

            raise RuntimeError(
                f"No Vast.ai offers found matching requirements. "
                f"GPU: {gpu_name}, nodes: {compute.nodes}. "
                "Try lowering min_reliability or broadening requirements."
            )

        # Sort by price
        offers.sort(key=lambda o: o.min_bid if use_interruptible else o.dph_total)

        # Filter by budget
        if compute.max_hourly_cost:
            max_per_instance = compute.max_hourly_cost / compute.nodes

            def offer_price(o: Offer) -> float:
                return o.min_bid * self.bid_multiplier if use_interruptible else o.dph_total

            before_filter = len(offers)
            offers = [o for o in offers if offer_price(o) <= max_per_instance]

            if not offers:
                from skyward.core.exceptions import BudgetExceededError

                raise BudgetExceededError(
                    f"No Vast.ai offers found within budget ${max_per_instance:.2f}/hr per instance. "
                    f"Filtered out {before_filter} offers."
                )
            logger.debug(f"Budget filter: {len(offers)}/{before_filter} offers within ${max_per_instance:.2f}/hr")

        # Select clusters for multi-node (may need to try multiple if overlay creation fails)
        valid_clusters: list[tuple[int, list[Offer]]] = []
        if needs_overlay:
            if preselected_cluster:
                # Start with preselected, but also get alternatives
                valid_clusters = select_all_valid_clusters(offers, compute.nodes, use_interruptible)
                # Ensure preselected is first
                if preselected_cluster[0] != valid_clusters[0][0]:
                    valid_clusters = [preselected_cluster] + [
                        c for c in valid_clusters if c[0] != preselected_cluster[0]
                    ]
            else:
                valid_clusters = select_all_valid_clusters(offers, compute.nodes, use_interruptible)

            if not valid_clusters:
                raise RuntimeError(
                    f"No cluster with {compute.nodes} nodes for {compute.accelerator}"
                )

            logger.info(f"Found {len(valid_clusters)} valid cluster(s) for multi-node provisioning")

        # Convert to InstanceSpec for events
        specs = tuple(o.to_instance_spec() for o in offers)
        spec = specs[0] if specs else None

        if compute.machine is None and acc and specs:
            spec = select_instance(
                specs,
                cpu=compute.cpu or 1,
                memory_mb=parse_memory_mb(compute.memory),
                accelerator=acc.accelerator,
                accelerator_count=acc.count if not callable(acc.count) else 1,
                prefer_spot=use_interruptible,
            )

        emit(
            InstanceLaunching(
                count=compute.nodes,
                candidates=(spec,) if spec else (),
                provider=ProviderName.VastAI,
            )
        )

        # VastAI has 4048-char limit on onstart_cmd - use minimal script
        # Full bootstrap will be executed via SSH after container starts
        onstart_script = _generate_minimal_onstart()

        # Generate full bootstrap script (with nohup since Docker has no systemd)
        full_bootstrap = compute.image.bootstrap(ttl=compute.timeout, use_systemd=False)

        # Try clusters until overlay creation succeeds
        if needs_overlay:
            overlay_created = False
            last_overlay_error: str | None = None

            for cluster_idx, (cluster_id, cluster_offers) in enumerate(valid_clusters):
                self._overlay_cluster_id = cluster_id
                offers = cluster_offers
                logger.info(f"Trying cluster {cluster_id} ({cluster_idx + 1}/{len(valid_clusters)})")

                try:
                    self._create_overlay_network(client)
                    overlay_created = True
                    logger.info(f"Overlay created on cluster {cluster_id}")
                    break
                except RuntimeError as e:
                    last_overlay_error = str(e)
                    logger.warning(
                        f"Overlay creation failed on cluster {cluster_id}: {e}. "
                        f"Trying next cluster..."
                    )
                    # Reset overlay state for next attempt
                    self._overlay_name = None
                    self._overlay_cluster_id = None

            if not overlay_created:
                raise RuntimeError(
                    f"Failed to create overlay on any of {len(valid_clusters)} clusters. "
                    f"Last error: {last_overlay_error}"
                )

        # Create instances
        instances: list[Instance] = []
        provisioned_instances: list[ProvisionedInstance] = []
        vast_instances: list[_VastInstance] = []
        offer_index = 0

        for i in range(compute.nodes):
            instance_created = False
            last_error: str | None = None

            while not instance_created and offer_index < len(offers):
                offer = offers[offer_index]
                price = offer.min_bid * self.bid_multiplier if use_interruptible else None

                logger.debug(
                    f"Creating instance {i + 1}/{compute.nodes} "
                    f"from offer {offer.id} ({offer.gpu_name})"
                )

                try:
                    instance_id = client.create_instance(
                        offer_id=offer.id,
                        image=docker_image or "pytorch/pytorch",
                        disk=self.disk_gb,
                        label=f"skyward-{cluster_id}-{i}",
                        onstart_cmd=onstart_script,
                        # overlay_name=self._overlay_name,
                        price=price,
                    )

                    vast_inst = _VastInstance(id=instance_id, offer=offer)
                    vast_instances.append(vast_inst)
                    instance_created = True
                    spec = offer.to_instance_spec()
                    logger.debug(f"Created instance {instance_id}")

                except VastAIError as e:
                    last_error = str(e)
                    logger.debug(f"Offer {offer.id} failed: {e}")

                offer_index += 1

            if not instance_created:
                raise RuntimeError(
                    f"Failed to create instance: tried {offer_index} offers. "
                    f"Last error: {last_error}"
                )

        # Attach SSH keys
        for vast_inst in vast_instances:
            client.attach_ssh(vast_inst.id, self._ssh_public_key)  # type: ignore[arg-type]

        # Wait for instances to be running
        logger.debug("Waiting for instances to be running...")
        self._wait_for_running(client, vast_instances, timeout=300)
        self._instances = vast_instances

        # Build Instance objects
        key_path = get_private_key_path()
        for i, vast_inst in enumerate(vast_instances):
            # Prefer direct connection if available
            if vast_inst.public_ipaddr and vast_inst.direct_port:
                ssh_host = vast_inst.public_ipaddr
                ssh_port = vast_inst.direct_port
                logger.debug(f"Using direct SSH: {ssh_host}:{ssh_port}")
            else:
                ssh_host = vast_inst.ssh_host
                ssh_port = vast_inst.ssh_port
                logger.debug(f"Using proxy SSH: {ssh_host}:{ssh_port}")

            meta_items: list[tuple[str, str]] = [
                ("cluster_id", str(cluster_id)),
                ("offer_id", str(vast_inst.offer.id)),
                ("machine_id", str(vast_inst.offer.machine_id)),
                ("ssh_host", ssh_host),
                ("ssh_port", str(ssh_port)),
                ("username", "root"),
                ("accelerator_count", str(vast_inst.offer.num_gpus)),
            ]

            instance = Instance(
                id=str(vast_inst.id),
                provider=self,
                ssh=SSHConfig(
                    host=ssh_host,
                    username="root",
                    port=ssh_port,
                    key_path=key_path,
                ),
                spot=use_interruptible,
                private_ip=vast_inst.ssh_host,
                public_ip=vast_inst.public_ipaddr or vast_inst.ssh_host,
                node=i,
                metadata=frozenset(meta_items),
                _destroy_fn=lambda inst_id=vast_inst.id: self._destroy_instance(inst_id),
                ssh_pool_size=compute.concurrency * 2,
            )
            instances.append(instance)

        # Join overlay network
        if needs_overlay and self._overlay_name:
            self._join_instances_to_overlay(client, vast_instances)
            self._detect_overlay_ips(instances)

        # Emit events with actual prices from instance info
        for inst, vast_inst in zip(instances, vast_instances, strict=True):
            # Create spec with actual dph_total if available
            inst_spec = vast_inst.offer.to_instance_spec()
            if vast_inst.dph_total > 0:
                p = vast_inst.dph_total
                inst_spec = replace(
                    inst_spec,
                    price_spot=p if use_interruptible else inst_spec.price_spot,
                    price_on_demand=p if not use_interruptible else inst_spec.price_on_demand,
                )

            provisioned = ProvisionedInstance(
                instance_id=inst.id,
                node=inst.node,
                provider=ProviderName.VastAI,
                spot=inst.spot,
                spec=inst_spec,
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

        # Execute full bootstrap via SSH (two-stage bootstrap)
        # The minimal onstart just started the container - now run the real bootstrap
        from skyward.providers.common import wait_for_ssh_ready

        logger.info("Executing bootstrap via SSH...")
        for inst in instances:
            # Wait for SSH to be ready
            wait_for_ssh_ready(inst.ssh.host, inst.ssh.port, timeout=300)
            # Upload and execute bootstrap script
            inst.bootstrap(full_bootstrap)

        return tuple(instances)

    def discover_peers(self, cluster_id: str) -> tuple[Instance, ...]:
        """Discover peer instances by cluster label."""
        from skyward.providers.base import assign_node_indices

        client = self._get_client()
        all_instances = client.list_instances()

        key_path = get_private_key_path()
        instances: list[Instance] = []

        for inst in all_instances:
            if not inst.label or not inst.label.startswith(f"skyward-{cluster_id}-"):
                continue
            if inst.actual_status != "running":
                continue
            if not inst.ssh_host:
                continue

            # Prefer direct connection if available
            if inst.public_ipaddr and inst.direct_port:
                ssh_host = inst.public_ipaddr
                ssh_port = inst.direct_port
            else:
                ssh_host = inst.ssh_host
                ssh_port = inst.ssh_port

            instances.append(
                Instance(
                    id=str(inst.id),
                    provider=self,
                    ssh=SSHConfig(
                        host=ssh_host,
                        username="root",
                        port=ssh_port,
                        key_path=key_path,
                    ),
                    spot=inst.is_bid,
                    private_ip=inst.ssh_host,
                    public_ip=inst.public_ipaddr or inst.ssh_host,
                    node=0,
                    metadata=frozenset([
                        ("cluster_id", cluster_id),
                        ("ssh_host", ssh_host),
                        ("ssh_port", str(ssh_port)),
                    ]),
                    _destroy_fn=lambda inst_id=inst.id: self._destroy_instance(inst_id),
                )
            )

        return assign_node_indices(instances)

    def available_instances(self) -> tuple[InstanceSpec, ...]:
        """List available offers as InstanceSpecs."""
        client = self._get_client()



        offers = client.search_offers(
            min_reliability=self.min_reliability,
            geolocation=self.geolocation,
        )

        specs = [o.to_instance_spec() for o in offers]
        return tuple(
            sorted(
                specs,
                key=lambda s: (
                    s.accelerator or "",
                    s.accelerator_count,
                    s.price_on_demand or float("inf"),
                ),
            )
        )

    def cleanup(self) -> None:
        """Clean up overlay network after all instances are destroyed."""
        self._cleanup_overlay()

    def get_instance_status(self, instance_id: str) -> InstanceStatus | None:
        """Get current status of an instance.

        Args:
            instance_id: Vast.ai instance ID.

        Returns:
            InstanceStatus with current state, or None if not found.
        """
        from skyward.types.spec import InstanceStatus

        info = self._get_client().get_instance(int(instance_id))
        if not info:
            return None

        return InstanceStatus(
            instance_id=instance_id,
            status=info.actual_status,
            ssh_available=bool(info.ssh_host and info.ssh_port),
        )

    def classify_preemption(self, status: str) -> str | None:
        """Classify if a status indicates preemption.

        Maps Vast.ai status strings to preemption reasons.

        Args:
            status: Vast.ai actual_status value.

        Returns:
            Preemption reason or None if not preempted.
        """
        return {
            "outbid": "outbid",
            "exited": "terminated",
            "offline": "maintenance",
        }.get(status)
