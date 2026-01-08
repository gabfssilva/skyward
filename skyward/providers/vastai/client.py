"""HTTP client for Vast.ai API.

Typesafe client implementation with automatic rate limiting.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from loguru import logger

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

from skyward.internal.rethrow import rethrow
from skyward.utils.throttle import Throttle

if TYPE_CHECKING:
    from skyward.types import InstanceSpec

VAST_API_BASE = "https://console.vast.ai"


class VastAIError(Exception):
    """Error from Vast.ai API."""


# =============================================================================
# Data Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class Offer:
    """A rentable GPU offer from the VastAI marketplace."""

    id: int
    machine_id: int
    gpu_name: str
    num_gpus: int
    cpu_cores: int
    cpu_ram: float  # GB
    gpu_ram: float  # MB
    disk_space: float  # GB
    reliability: float
    dph_total: float  # on-demand $/hr
    min_bid: float  # spot $/hr
    cuda_max_good: float
    geolocation: str
    cluster_id: int | None
    inet_up: float  # Mbps
    inet_down: float  # Mbps
    dlperf: float | None
    verified: bool

    @classmethod
    def from_api(cls, data: dict[str, object]) -> Offer:
        """Parse API response to Offer."""
        # VastAI returns cpu_ram in MB, convert to GB
        cpu_ram_mb = float(data.get("cpu_ram") or 0)
        return cls(
            id=int(data.get("id") or 0),
            machine_id=int(data.get("machine_id") or 0),
            gpu_name=str(data.get("gpu_name") or ""),
            num_gpus=int(data.get("num_gpus") or 0),
            cpu_cores=int(data.get("cpu_cores") or 0),
            cpu_ram=cpu_ram_mb / 1024,  # MB -> GB
            gpu_ram=float(data.get("gpu_ram") or 0),
            disk_space=float(data.get("disk_space") or 0),
            reliability=float(data.get("reliability") or 0),
            dph_total=float(data.get("dph_total") or 0),
            min_bid=float(data.get("min_bid") or 0),
            cuda_max_good=float(data.get("cuda_max_good") or 0),
            geolocation=str(data.get("geolocation") or ""),
            cluster_id=int(data["cluster_id"]) if data.get("cluster_id") else None,
            inet_up=float(data.get("inet_up") or 0),
            inet_down=float(data.get("inet_down") or 0),
            dlperf=float(data["dlperf"]) if data.get("dlperf") else None,
            verified=bool(data.get("verified", False)),
        )

    @property
    def gpu_ram_gb(self) -> float:
        """GPU RAM in GB."""
        return self.gpu_ram / 1024 if self.gpu_ram else 0

    @property
    def normalized_gpu_name(self) -> str:
        """Normalized GPU name (e.g., 'RTX 5090' -> 'RTX_5090')."""
        name = self.gpu_name.upper()
        for suffix in ("_PCIE", "_SXM", "_80GB", "_40GB"):
            name = name.replace(suffix, "")
        return name.replace(" ", "_")

    def to_instance_spec(self) -> InstanceSpec:
        """Convert to InstanceSpec for provider compatibility."""
        from skyward.types import InstanceSpec

        return InstanceSpec(
            name=str(self.id),
            vcpu=self.cpu_cores,
            memory_gb=self.cpu_ram,
            accelerator=self.gpu_name,  # Keep original name for display
            accelerator_count=self.num_gpus,
            accelerator_memory_gb=self.gpu_ram_gb,
            price_on_demand=self.dph_total,
            price_spot=self.min_bid,
            billing_increment_minutes=None,
            metadata={
                "offer_id": self.id,
                "machine_id": self.machine_id,
                "reliability": self.reliability,
                "geolocation": self.geolocation,
                "cuda_version": self.cuda_max_good,
                "dlperf": self.dlperf,
                "inet_up": self.inet_up,
                "inet_down": self.inet_down,
                "disk_space": self.disk_space,
                "verified": self.verified,
            },
        )


@dataclass(frozen=True, slots=True)
class InstanceInfo:
    """Information about a VastAI instance."""

    id: int
    actual_status: str
    ssh_host: str
    ssh_port: int
    gpu_name: str
    num_gpus: int
    machine_id: int
    label: str | None
    is_bid: bool
    dph_total: float  # Actual $/hr being charged
    public_ipaddr: str  # Direct connection IP
    direct_port: int | None  # Direct SSH port (from ports mapping)

    @classmethod
    def from_api(cls, data: dict[str, object]) -> InstanceInfo:
        """Parse API response to InstanceInfo."""
        # Extract direct SSH port from ports mapping
        direct_port = None
        ports = data.get("ports")
        if isinstance(ports, dict):
            ssh_mapping = ports.get("22/tcp")
            if isinstance(ssh_mapping, list) and ssh_mapping:
                direct_port = int(ssh_mapping[0].get("HostPort", 0)) or None

        return cls(
            id=int(data.get("id") or 0),
            actual_status=str(data.get("actual_status") or ""),
            ssh_host=str(data.get("ssh_host") or ""),
            ssh_port=int(data.get("ssh_port") or 0),
            gpu_name=str(data.get("gpu_name") or ""),
            num_gpus=int(data.get("num_gpus") or 0),
            machine_id=int(data.get("machine_id") or 0),
            label=str(data["label"]) if data.get("label") else None,
            is_bid=bool(data.get("is_bid", False)),
            dph_total=float(data.get("dph_total") or 0),
            public_ipaddr=str(data.get("public_ipaddr") or ""),
            direct_port=direct_port,
        )


@dataclass(frozen=True, slots=True)
class SSHKey:
    """SSH key registered on VastAI."""

    id: int
    public_key: str

    @classmethod
    def from_api(cls, data: dict[str, object]) -> SSHKey:
        """Parse API response to SSHKey."""
        return cls(
            id=int(data.get("id", 0)),
            public_key=str(data.get("public_key", "")),
        )


@dataclass(frozen=True, slots=True)
class OverlayNetwork:
    """VastAI overlay network for multi-node communication."""

    id: int
    name: str
    cluster_id: int
    instance_ids: frozenset[int]

    @classmethod
    def from_api(cls, data: dict[str, object]) -> OverlayNetwork:
        """Parse API response to OverlayNetwork."""
        return cls(
            id=int(data.get("overlay_id") or data.get("id", 0)),
            name=str(data.get("name", "")),
            cluster_id=int(data.get("cluster_id", 0)),
            instance_ids=frozenset(data.get("instances", [])),  # type: ignore[arg-type]
        )


# =============================================================================
# Query Builder
# =============================================================================

# Known GPU variants for base model names (Vast.ai uses spaces, not hyphens)
_GPU_VARIANTS: dict[str, list[str]] = {
    "H100": ["H100", "H100 NVL", "H100 SXM", "H100 PCIe"],
    "H200": ["H200", "H200 NVL", "H200 SXM"],
    "A100": ["A100", "A100 SXM4", "A100 PCIe", "A100 SXM"],
    "A800": ["A800", "A800 PCIe", "A800 SXM"],
    "L40": ["L40", "L40S"],
    "RTX 4090": ["RTX 4090", "RTX 4090D"],
    "RTX 5090": ["RTX 5090", "RTX 5090D"],
}


def _get_gpu_variants(gpu_name: str) -> list[str]:
    """Get all known variants for a GPU name.

    Args:
        gpu_name: GPU name with spaces (e.g., "H100", "RTX 4090").

    Returns:
        List of variant names to search for.
    """
    # Check if it's a known base name with variants
    upper = gpu_name.upper()
    for base, variants in _GPU_VARIANTS.items():
        if upper == base.upper():
            return variants

    # Return as-is if no variants known
    return [gpu_name]


def build_search_query(
    *,
    gpu_name: str | None = None,
    num_gpus: int | None = None,
    min_cpu: int | None = None,
    min_memory_gb: float | None = None,
    min_reliability: float = 0.95,
    min_disk_gb: float | None = None,
    geolocation: str | None = None,
    verified_only: bool = False,
    use_interruptible: bool = False,
    with_cluster_id: bool = False,
) -> dict[str, object]:
    """Build VastAI search query dict."""
    query: dict[str, object] = {
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "order": [["score", "desc"]],
        "type": "bid" if use_interruptible else "on-demand",
        "allocated_storage": 5.0,
    }

    if with_cluster_id:
        query["cluster_id"] = {"neq": None}

    if gpu_name:
        normalized = gpu_name.replace("_", " ")
        variants = _get_gpu_variants(normalized)
        if len(variants) == 1:
            query["gpu_name"] = {"eq": variants[0]}
        else:
            query["gpu_name"] = {"in": variants}
    if num_gpus:
        query["num_gpus"] = {"gte": num_gpus}
    if min_cpu:
        query["cpu_cores"] = {"gte": min_cpu}
    if min_memory_gb:
        query["cpu_ram"] = {"gte": min_memory_gb}
    if min_disk_gb:
        query["disk_space"] = {"gte": min_disk_gb}
    if min_reliability > 0:
        query["reliability"] = {"gt": min_reliability}
    if geolocation:
        query["geolocation"] = {"eq": geolocation}
    if verified_only:
        query["verified"] = {"eq": True}

    return query


# =============================================================================
# Client
# =============================================================================


class VastAIClient:
    """Typesafe HTTP client for Vast.ai API.

    Includes rate limiting (2 requests/second) to avoid 429 errors.

    Example:
        client = VastAIClient(api_key="...")

        # Search offers
        offers = client.search_offers(gpu_name="RTX 4090", num_gpus=1)
        print(offers[0].dph_total)

        # Create instance
        instance_id = client.create_instance(
            offer_id=offers[0].id,
            image="pytorch/pytorch",
        )

        # SSH key management
        key_id, public_key = client.ensure_ssh_key()
        client.attach_ssh(instance_id, public_key)
    """

    __slots__ = ("_api_key", "_base_url", "_timeout")

    _throttle = Throttle(calls=2, period=1)

    def __init__(
        self,
        api_key: str,
        base_url: str = VAST_API_BASE,
        timeout: int = 60,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}

    @_throttle
    @rethrow(httpx.RequestError, lambda e: VastAIError(f"Request failed: {e}"))
    @rethrow(
        httpx.HTTPStatusError,
        lambda e: VastAIError(f"API error {e.response.status_code}: {e.response.text}"),
    )
    @retry(retry=retry_if_exception(lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429), wait=wait_fixed(1), stop=stop_after_attempt(5))
    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, object] | None = None,
        params: dict[str, object] | None = None,
        timeout: int | None = None,
    ) -> dict[str, object] | list[object] | None:
        """Execute HTTP request and return JSON response."""
        url = f"{self._base_url}{path}"
        resp = httpx.request(
            method,
            url,
            headers=self._headers(),
            json=json,
            params=params,
            timeout=timeout or self._timeout,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else None  # type: ignore[return-value]

    # =========================================================================
    # SSH Key Management
    # =========================================================================

    def list_ssh_keys(self) -> list[SSHKey]:
        """List all SSH keys registered on this account."""
        result = self._request("GET", "/api/v0/ssh/")
        if not isinstance(result, dict):
            return []

        raw_keys = result.get("ssh_keys", [])
        if not isinstance(raw_keys, list):
            return []

        return [SSHKey.from_api(k) for k in raw_keys if isinstance(k, dict)]

    def create_ssh_key(self, public_key: str) -> int | None:
        """Register a new SSH public key.

        Returns:
            Key ID if created, None if already exists.
        """
        try:
            result = self._request("POST", "/api/v0/ssh/", {"ssh_key": public_key})
        except VastAIError as e:
            # API returns 400 when key already exists - not an error
            if "duplicate" in str(e).lower() or "already exists" in str(e).lower():
                logger.debug("SSH key already exists on VastAI")
                return None
            raise

        if not isinstance(result, dict):
            raise VastAIError("Failed to create SSH key: empty response")
        if not result.get("success", True):
            # Also handle duplicate in response body
            if result.get("error") == "duplicate":
                return None
            raise VastAIError(f"Failed to create SSH key: {result}")

        key_id = result.get("ssh_key_id") or result.get("id")
        if not key_id:
            raise VastAIError(f"No key ID in response: {result}")
        return int(key_id)

    def attach_ssh(self, instance_id: int, public_key: str) -> None:
        """Attach SSH key to an instance."""
        result = self._request(
            "POST", f"/api/v0/instances/{instance_id}/ssh/", {"ssh_key": public_key}
        )
        if isinstance(result, dict) and not result.get("success", True):
            logger.warning(f"Failed to attach SSH key to {instance_id}: {result}")

    def ensure_ssh_key(self) -> tuple[int, str]:
        """Get or create SSH key on VastAI.

        Finds local SSH key, checks if registered on VastAI, creates if not.

        Returns:
            Tuple of (key_id, public_key_content).

        Raises:
            RuntimeError: If no local SSH key found.
        """
        # Find local SSH key
        key_paths = [
            Path.home() / ".ssh" / "id_ed25519.pub",
            Path.home() / ".ssh" / "id_rsa.pub",
            Path.home() / ".ssh" / "id_ecdsa.pub",
        ]

        public_key = None
        key_path = None
        for path in key_paths:
            if path.exists():
                public_key = path.read_text().strip()
                key_path = path
                break

        if not public_key or not key_path:
            raise RuntimeError(
                "No SSH key found. Create one with: ssh-keygen -t ed25519\n"
                "Searched: ~/.ssh/id_ed25519.pub, ~/.ssh/id_rsa.pub, ~/.ssh/id_ecdsa.pub"
            )

        # Extract key data for comparison
        local_parts = public_key.split()
        local_key_data = local_parts[1] if len(local_parts) >= 2 else public_key

        # Check existing keys
        for key in self.list_ssh_keys():
            stored_parts = key.public_key.strip().split()
            stored_data = stored_parts[1] if len(stored_parts) >= 2 else key.public_key
            if stored_data == local_key_data:
                logger.debug(f"Found existing SSH key on VastAI: {key.id}")
                return key.id, public_key

        # Create new key (or detect duplicate)
        key_name = f"skyward-{os.environ.get('USER', 'user')}-{key_path.stem}"
        logger.info(f"Creating SSH key on VastAI: {key_name}")
        key_id = self.create_ssh_key(public_key)

        if key_id is not None:
            logger.debug(f"Created SSH key on VastAI: {key_id}")
            return key_id, public_key

        # Key already exists but wasn't found in list - refetch to get ID
        logger.debug("SSH key duplicate detected, refetching key list")
        for key in self.list_ssh_keys():
            stored_parts = key.public_key.strip().split()
            stored_data = stored_parts[1] if len(stored_parts) >= 2 else key.public_key
            if stored_data == local_key_data:
                logger.debug(f"Found existing SSH key on VastAI: {key.id}")
                return key.id, public_key

        # Fallback: use key ID 0 and rely on attach_ssh
        logger.warning("Could not determine SSH key ID, using public key directly")
        return 0, public_key

    # =========================================================================
    # Offer Search
    # =========================================================================

    def search_offers(
        self,
        *,
        gpu_name: str | None = None,
        num_gpus: int | None = None,
        min_cpu: int | None = None,
        min_memory_gb: float | None = None,
        min_reliability: float = 0.95,
        min_disk_gb: float | None = None,
        geolocation: str | None = None,
        verified_only: bool = False,
        use_interruptible: bool = False,
        with_cluster_id: bool = False,
        limit: int = 1000,
    ) -> list[Offer]:
        """Search available GPU offers.

        Args:
            gpu_name: GPU model (e.g., "RTX 4090", "H100").
            num_gpus: Minimum number of GPUs.
            min_cpu: Minimum CPU cores.
            min_memory_gb: Minimum system RAM in GB.
            min_reliability: Minimum host reliability (0-1).
            min_disk_gb: Minimum disk space in GB.
            geolocation: Region filter (e.g., "US", "EU").
            verified_only: Only verified hosts.
            use_interruptible: Use spot/bid pricing.
            with_cluster_id: If true, filter for offers with cluster.
            limit: Max results.

        Returns:
            List of Offer objects sorted by score.
        """
        query = build_search_query(
            gpu_name=gpu_name,
            num_gpus=num_gpus,
            min_cpu=min_cpu,
            min_memory_gb=min_memory_gb,
            min_reliability=min_reliability,
            min_disk_gb=min_disk_gb,
            geolocation=geolocation,
            verified_only=verified_only,
            use_interruptible=use_interruptible,
            with_cluster_id=with_cluster_id
        )
        query["limit"] = limit

        result = self._request("POST", "/api/v0/bundles/", json=query)
        if not isinstance(result, dict):
            return []

        raw_offers = result.get("offers", [])
        if not isinstance(raw_offers, list):
            return []

        return [Offer.from_api(o) for o in raw_offers if isinstance(o, dict)]

    def search_by_machine_id(
        self, machine_id: int, use_interruptible: bool = False
    ) -> list[Offer]:
        """Search offers by machine ID."""
        query: dict[str, object] = {
            "machine_id": {"eq": machine_id},
            "rentable": {"eq": True},
            "order": [["score", "desc"]],
            "type": "bid" if use_interruptible else "on-demand",
        }

        result = self._request("POST", "/api/v0/bundles/", json=query)
        if not isinstance(result, dict):
            return []

        raw_offers = result.get("offers", [])
        if not isinstance(raw_offers, list):
            return []

        return [Offer.from_api(o) for o in raw_offers if isinstance(o, dict)]

    # =========================================================================
    # Instance Management
    # =========================================================================

    def create_instance(
        self,
        offer_id: int,
        image: str,
        disk: float = 10.0,
        label: str | None = None,
        onstart_cmd: str | None = None,
        overlay_name: str | None = None,
        price: float | None = None,
    ) -> int:
        """Create a new instance from an offer.

        Args:
            offer_id: Offer ID to rent.
            image: Docker image name.
            disk: Disk space in GB.
            label: Instance label.
            onstart_cmd: Startup script content.
            overlay_name: Overlay network to join.
            price: Bid price for interruptible.

        Returns:
            New instance ID.

        Raises:
            VastAIError: If creation fails.
        """
        body: dict[str, object] = {
            "client_id": "me",
            "image": image,
            "disk": disk,
        }

        if label:
            body["label"] = label
        if onstart_cmd:
            body["onstart"] = onstart_cmd
        if overlay_name:
            body["env"] = {f"-n {overlay_name}": "1"}
        if price is not None:
            body["price"] = price

        result = self._request("PUT", f"/api/v0/asks/{offer_id}/", body)

        if not result:
            raise VastAIError(f"Offer {offer_id} unavailable")
        if not isinstance(result, dict):
            raise VastAIError(f"Unexpected response: {result}")

        # Check for instance ID first - API may return success=False with valid new_contract
        instance_id = result.get("new_contract") or result.get("id")
        if instance_id:
            return int(instance_id)

        # Only fail if no instance ID and success is explicitly False
        if not result.get("success", True):
            raise VastAIError(f"Creation failed: {result}")

        raise VastAIError(f"No instance ID in response: {result}")

    def get_instance(self, instance_id: int) -> InstanceInfo | None:
        """Get instance details. Returns None if not found."""
        result = self._request("GET", f"/api/v0/instances/{instance_id}/", params={"owner": "me"})

        if not isinstance(result, dict):
            return None

        data = result.get("instances")
        if not isinstance(data, dict):
            return None

        return InstanceInfo.from_api(data)

    def list_instances(self) -> list[InstanceInfo]:
        """List all instances owned by this account."""
        result = self._request("GET", "/api/v0/instances/", params={"owner": "me"})

        if not isinstance(result, dict):
            return []

        raw = result.get("instances", [])
        if not isinstance(raw, list):
            return []

        return [InstanceInfo.from_api(i) for i in raw if isinstance(i, dict)]

    def destroy_instance(self, instance_id: int) -> None:
        """Destroy an instance."""
        self._request("DELETE", f"/api/v0/instances/{instance_id}/", {})

    # =========================================================================
    # Overlay Network Management
    # =========================================================================

    def create_overlay(self, cluster_id: int, name: str) -> None:
        """Create an overlay network on a physical cluster.

        Args:
            cluster_id: Physical cluster ID.
            name: Overlay name (lowercase letters and hyphens only).

        Raises:
            VastAIError: If overlay creation fails.
        """
        logger.debug(f"Creating overlay '{name}' on cluster {cluster_id}")
        result = self._request("POST", "/api/v0/overlay/", {"cluster_id": cluster_id, "name": name})

        if isinstance(result, dict):
            if result.get("success"):
                return
            error_msg = result.get("msg") or result.get("error") or "unknown error"
            raise VastAIError(f"Failed to create overlay on cluster {cluster_id}: {error_msg}")

        raise VastAIError(f"Failed to create overlay on cluster {cluster_id}: unexpected response")

    def join_overlay(self, name: str, instance_id: int) -> None:
        """Join an instance to an overlay network."""
        logger.debug(f"Joining instance {instance_id} to overlay '{name}'")
        response = self._request("PUT", "/api/v0/overlay/", {"name": name, "instance_id": instance_id})
        logger.debug(f"Join overlay response: {response}")

    def list_overlays(self) -> list[OverlayNetwork]:
        """List all overlay networks for this account."""
        try:
            result = self._request("GET", "/api/v0/overlay/")
        except VastAIError as e:
            logger.debug(f"list_overlays failed: {e}")
            return []

        if not isinstance(result, list):
            return []

        return [OverlayNetwork.from_api(item) for item in result if isinstance(item, dict)]

    def delete_overlay(self, name: str) -> None:
        """Delete an overlay network by name."""
        logger.debug(f"Deleting overlay '{name}'")
        try:
            self._request("DELETE", "/api/v0/overlay/", {"overlay_name": name})
        except VastAIError as e:
            logger.warning(f"Failed to delete overlay '{name}': {e}")

    def find_available_accelerators(
        self,
        offers: list[Offer],
        nodes_needed: int,
        num_gpus: int | None = None,
    ) -> list[tuple[str, int, int]]:
        """Find accelerators with clusters that have enough capacity.

        Groups offers by GPU name and checks which have clusters
        that can satisfy the node requirement.

        Args:
            offers: List of offers to analyze.
            nodes_needed: Required number of nodes in a cluster.
            num_gpus: Filter by exact GPU count per node.

        Returns:
            List of (gpu_name, max_cluster_size, total_offers) tuples,
            sorted by max_cluster_size descending.
        """
        filtered = [o for o in offers if o.num_gpus == num_gpus] if num_gpus else offers

        by_gpu: dict[str, list[Offer]] = {}
        for offer in filtered:
            by_gpu.setdefault(offer.gpu_name, []).append(offer)

        results: list[tuple[str, int, int]] = []
        for gpu_name, gpu_offers in by_gpu.items():
            clusters = group_offers_by_cluster(gpu_offers)
            if not clusters:
                continue

            max_cluster_size = max(len(c) for c in clusters.values())
            if max_cluster_size >= nodes_needed:
                results.append((gpu_name, max_cluster_size, len(gpu_offers)))

        results.sort(key=lambda x: (-x[1], -x[2]))
        return results


# =============================================================================
# Utility Functions
# =============================================================================


def extract_cuda_version(docker_image: str) -> float | None:
    """Extract CUDA version from Docker image name.

    Example:
        >>> extract_cuda_version("nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu22.04")
        12.8
    """
    match = re.search(r"cuda:(\d+\.\d+)", docker_image)
    return float(match.group(1)) if match else None


def group_offers_by_cluster(offers: list[Offer]) -> dict[int, list[Offer]]:
    """Group offers by physical cluster_id.

    Offers without cluster_id are excluded.
    """
    clusters: dict[int, list[Offer]] = {}
    for offer in offers:
        if offer.cluster_id is not None:
            clusters.setdefault(offer.cluster_id, []).append(offer)
    return clusters


def select_best_cluster(
    offers: list[Offer],
    nodes_needed: int,
    use_interruptible: bool = True,
) -> tuple[int, list[Offer]] | None:
    """Select best cluster with enough capacity.

    Returns:
        Tuple of (cluster_id, sorted offers) or None if no cluster has capacity.
    """
    all_clusters = select_all_valid_clusters(offers, nodes_needed, use_interruptible)
    return all_clusters[0] if all_clusters else None


def select_all_valid_clusters(
    offers: list[Offer],
    nodes_needed: int,
    use_interruptible: bool = True,
) -> list[tuple[int, list[Offer]]]:
    """Select all valid clusters with enough capacity, sorted by price.

    Returns:
        List of (cluster_id, sorted offers) tuples, sorted by total cluster cost.
        Empty list if no clusters have sufficient capacity.
    """
    clusters = group_offers_by_cluster(offers)

    valid = [
        (cid, cluster_offers)
        for cid, cluster_offers in clusters.items()
        if len(cluster_offers) >= nodes_needed
    ]

    if not valid:
        return []

    # Sort offers within each cluster by price
    for _, cluster_offers in valid:
        cluster_offers.sort(
            key=lambda o: o.min_bid if use_interruptible else o.dph_total
        )

    # Sort clusters by total cost
    def cluster_cost(item: tuple[int, list[Offer]]) -> float:
        _, cluster_offers = item
        price_fn = (lambda o: o.min_bid) if use_interruptible else (lambda o: o.dph_total)
        return sum(price_fn(o) for o in cluster_offers[:nodes_needed])

    valid.sort(key=cluster_cost)
    return valid


__all__ = [
    "Offer",
    "InstanceInfo",
    "SSHKey",
    "OverlayNetwork",
    "VastAIClient",
    "VastAIError",
    "extract_cuda_version",
    "group_offers_by_cluster",
    "select_best_cluster",
    "select_all_valid_clusters",
]
