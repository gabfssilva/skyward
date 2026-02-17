"""Async HTTP client for Vast.ai API."""

from __future__ import annotations

import os
import re
from contextlib import suppress
from pathlib import Path
from typing import Any

from skyward.infra.http import BearerAuth, HttpClient, HttpError
from skyward.infra.retry import on_status_code, retry
from skyward.infra.throttle import throttle
from skyward.observability.logger import logger

from .config import VastAI
from .types import (
    BundlesResponse,
    CreateInstanceResponse,
    CreateSSHKeyResponse,
    InstanceGetResponse,
    InstanceResponse,
    InstancesListResponse,
    OfferResponse,
    OverlayCreateResponse,
    OverlayResponse,
    SSHKeyResponse,
)

VAST_API_BASE = "https://console.vast.ai"


class VastAIError(Exception):
    """Error from Vast.ai API."""


# =============================================================================
# Query Builder
# =============================================================================

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
    """Get all known variants for a GPU name."""
    upper = gpu_name.upper()
    for base, variants in _GPU_VARIANTS.items():
        if upper == base.upper():
            return variants
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
    require_direct_port: bool = False,
) -> dict[str, Any]:
    """Build VastAI search query dict."""
    query: dict[str, Any] = {
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
    # Note: cuda_vers filter doesn't work in API, filter locally using cuda_max_good
    if geolocation:
        query["geolocation"] = {"eq": geolocation}
    if verified_only:
        query["verified"] = {"eq": True}
    if require_direct_port:
        query["direct_port_count"] = {"gte": 1}

    return query


# =============================================================================
# Async Client
# =============================================================================


class VastAIClient:
    """Async HTTP client for Vast.ai API."""

    def __init__(self, api_key: str, config: VastAI | None = None) -> None:
        self._api_key = api_key
        self.config = config or VastAI()
        self._http = HttpClient(
            VAST_API_BASE, BearerAuth(api_key), timeout=self.config.request_timeout,
        )
        self._log = logger.bind(provider="vastai", component="client")

    async def __aenter__(self) -> VastAIClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    async def close(self) -> None:
        await self._http.close()

    @retry(on=on_status_code(429, 503), max_attempts=10, base_delay=0.5)
    @throttle(max_concurrent=2, interval=1.5)
    async def _do_request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        return await self._http.request(method, path, json=json, params=params)

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        self._log.debug("{method} {path}", method=method, path=path)
        try:
            return await self._do_request(method, path, json, params)
        except HttpError as e:
            self._log.warning(
                "API error {method} {path}: {status}",
                method=method, path=path, status=e.status,
            )
            raise VastAIError(f"API error {e.status}: {e.body}") from e

    # =========================================================================
    # SSH Key Management
    # =========================================================================

    async def list_ssh_keys(self) -> list[SSHKeyResponse]:
        """List all SSH keys registered on this account."""
        result = await self._request("GET", "/api/v0/ssh/")
        return result if result else []

    async def create_ssh_key(self, public_key: str) -> int | None:
        """Register a new SSH public key. Returns key ID or None if exists."""
        try:
            result: CreateSSHKeyResponse = await self._request(
                "POST", "/api/v0/ssh/", {"ssh_key": public_key}
            )
        except VastAIError as e:
            if "duplicate" in str(e).lower() or "already exists" in str(e).lower():
                self._log.debug("SSH key already exists")
                return None
            raise

        if result.get("error") == "duplicate":
            return None

        key_id = result.get("ssh_key_id") or result.get("id")
        return int(key_id) if key_id else None

    async def attach_ssh(self, instance_id: int, public_key: str) -> None:
        """Attach SSH key to an instance."""

        @retry(on=VastAIError, max_attempts=5, base_delay=2.0)
        async def do_attach() -> None:
            result = await self._request(
                "POST",
                f"/api/v0/instances/{instance_id}/ssh/",
                {"ssh_key": public_key},
            )
            if result and not result.get("success", True):
                error = result.get("msg") or result.get("error") or "unknown"
                # Ignore "already associated" - key was auto-attached on instance creation
                if "already associated" in error.lower():
                    self._log.debug(
                        "SSH key already associated with instance {iid}",
                        iid=instance_id,
                    )
                    return
                raise VastAIError(f"Failed to attach SSH key to instance {instance_id}: {error}")

        await do_attach()

    async def ensure_ssh_key(self) -> tuple[int, str]:
        """Get or create SSH key on VastAI. Returns (key_id, public_key)."""
        key_paths = [
            Path.home() / ".ssh" / "id_ed25519.pub",
            Path.home() / ".ssh" / "id_rsa.pub",
            Path.home() / ".ssh" / "id_ecdsa.pub",
        ]

        public_key = None
        for path in key_paths:
            if path.exists():
                public_key = path.read_text().strip()
                break

        if not public_key:
            raise RuntimeError("No SSH key found. Create with: ssh-keygen -t ed25519")

        local_parts = public_key.split()
        local_key_data = local_parts[1] if len(local_parts) >= 2 else public_key

        for key in await self.list_ssh_keys():
            stored_parts = key["public_key"].strip().split()
            stored_data = stored_parts[1] if len(stored_parts) >= 2 else key["public_key"]
            if stored_data == local_key_data:
                return key["id"], public_key

        key_id = await self.create_ssh_key(public_key)
        if key_id:
            return key_id, public_key

        # Refetch if duplicate
        for key in await self.list_ssh_keys():
            stored_parts = key["public_key"].strip().split()
            stored_data = stored_parts[1] if len(stored_parts) >= 2 else key["public_key"]
            if stored_data == local_key_data:
                return key["id"], public_key

        return 0, public_key

    # =========================================================================
    # Offer Search
    # =========================================================================

    async def search_offers(
        self,
        *,
        gpu_name: str | None = None,
        num_gpus: int | None = None,
        min_cpu: int | None = None,
        min_memory_gb: float | None = None,
        min_reliability: float | None = None,
        min_cuda: float | None = None,
        min_disk_gb: float | None = None,
        geolocation: str | None = None,
        verified_only: bool = False,
        use_interruptible: bool = False,
        with_cluster_id: bool = False,
        require_direct_port: bool | None = None,
        limit: int = 5000,
    ) -> list[OfferResponse]:
        """Search available GPU offers."""
        reliability = (
            min_reliability
            if min_reliability is not None
            else self.config.min_reliability
        )
        cuda = (
            min_cuda if min_cuda is not None else self.config.min_cuda
        )
        geo = geolocation or self.config.geolocation
        direct_port = (
            require_direct_port
            if require_direct_port is not None
            else self.config.require_direct_port
        )

        query = build_search_query(
            gpu_name=gpu_name,
            num_gpus=num_gpus,
            min_cpu=min_cpu,
            min_memory_gb=min_memory_gb,
            min_reliability=reliability,
            min_disk_gb=min_disk_gb,
            geolocation=geo,
            verified_only=verified_only,
            use_interruptible=use_interruptible,
            with_cluster_id=with_cluster_id,
            require_direct_port=direct_port,
        )
        query["limit"] = limit

        self._log.debug("Search query: {query}", query=query)
        result: BundlesResponse | None = await self._request("POST", "/api/v0/bundles/", json=query)
        offers = result.get("offers", []) if result else []

        # Filter by CUDA version locally (API filter doesn't work)
        if cuda > 0:
            offers = [o for o in offers if o.get("cuda_max_good", 0) >= cuda]

        return offers

    async def search_by_machine_id(
        self, machine_id: int, use_interruptible: bool = False
    ) -> list[OfferResponse]:
        """Search offers by machine ID."""
        query: dict[str, Any] = {
            "machine_id": {"eq": machine_id},
            "rentable": {"eq": True},
            "order": [["score", "desc"]],
            "type": "bid" if use_interruptible else "on-demand",
        }

        result: BundlesResponse | None = await self._request("POST", "/api/v0/bundles/", json=query)
        return result.get("offers", []) if result else []

    # =========================================================================
    # Instance Management
    # =========================================================================

    async def create_instance(
        self,
        offer_id: int,
        image: str,
        disk: float | None = None,
        label: str | None = None,
        onstart_cmd: str | None = None,
        overlay_name: str | None = None,
        price: float | None = None,
        use_direct_port: bool | None = None,
    ) -> int:
        """Create a new instance. Returns instance ID."""
        direct = use_direct_port if use_direct_port is not None else self.config.require_direct_port
        body: dict[str, Any] = {
            "client_id": "me",
            "image": image,
            "disk": disk or self.config.disk_gb,
            "runtype": "ssh_direc" if direct else "ssh_proxy",
        }

        if label:
            body["label"] = label
        if onstart_cmd:
            body["onstart"] = onstart_cmd
        if overlay_name:
            body["env"] = {f"-n {overlay_name}": "1"}
        if price is not None:
            body["price"] = price

        self._log.debug("Creating instance from offer {oid}, image={img}", oid=offer_id, img=image)
        result: CreateInstanceResponse = await self._request(
            "PUT", f"/api/v0/asks/{offer_id}/", json=body,
        )

        instance_id = result.get("new_contract") or result.get("id")
        if instance_id:
            self._log.debug("Created instance {iid}", iid=instance_id)
            return int(instance_id)

        raise VastAIError(f"No instance ID in response: {result}")

    async def get_instance(self, instance_id: int) -> InstanceResponse | None:
        """Get instance details. Returns None if not found."""
        result: InstanceGetResponse | None = await self._request(
            "GET", f"/api/v0/instances/{instance_id}/", params={"owner": "me"}
        )
        if not result:
            return None
        data = result.get("instances")
        match data:
            case dict():
                return data
            case _:
                return None

    async def list_instances(self) -> list[InstanceResponse]:
        """List all instances owned by this account."""
        result: InstancesListResponse | None = await self._request(
            "GET", "/api/v0/instances/", params={"owner": "me"}
        )
        if not result:
            return []
        raw = result.get("instances", [])
        match raw:
            case list():
                return raw
            case _:
                return []

    async def destroy_instance(self, instance_id: int) -> None:
        """Destroy an instance."""
        self._log.debug("Destroying instance {iid}", iid=instance_id)
        await self._request("DELETE", f"/api/v0/instances/{instance_id}/", json={})

    # =========================================================================
    # Overlay Network Management
    # =========================================================================

    async def create_overlay(self, cluster_id: int, name: str) -> None:
        """Create an overlay network on a physical cluster."""
        self._log.debug("Creating overlay '{name}' on cluster {cid}", name=name, cid=cluster_id)
        result: OverlayCreateResponse | None = await self._request(
            "POST", "/api/v0/overlay/", json={"cluster_id": cluster_id, "name": name}
        )

        if result and not result.get("success"):
            error = result.get("msg") or result.get("error") or "unknown"
            raise VastAIError(f"Overlay creation failed: {error}")

    async def join_overlay(self, name: str, instance_id: int) -> None:
        """Join an instance to an overlay network."""
        self._log.debug("Joining instance {iid} to overlay '{name}'", iid=instance_id, name=name)
        await self._request(
            "PUT", "/api/v0/overlay/",
            json={"name": name, "instance_id": instance_id},
        )

    async def list_overlays(self) -> list[OverlayResponse]:
        """List all overlay networks for this account."""
        try:
            result: list[OverlayResponse] | None = await self._request(
                "GET", "/api/v0/overlay/",
            )
            return result or []
        except VastAIError:
            return []

    async def delete_overlay(self, name: str) -> None:
        """Delete an overlay network by name."""
        self._log.debug("Deleting overlay '{name}'", name=name)
        with suppress(VastAIError):
            await self._request(
                "DELETE", "/api/v0/overlay/",
                json={"overlay_name": name},
            )


# =============================================================================
# Utility Functions
# =============================================================================


def get_api_key() -> str:
    """Get Vast.ai API key from environment or config file."""
    if env_key := os.environ.get("VAST_API_KEY"):
        return env_key

    config_path = os.path.expanduser("~/.config/vastai/vast_api_key")
    if os.path.exists(config_path):
        with suppress(OSError), open(config_path) as f:
            if file_key := f.read().strip():
                return file_key

    raise ValueError("Vast.ai API key not found. Set VAST_API_KEY or run: vastai set api-key")


def extract_cuda_version(docker_image: str) -> float | None:
    """Extract CUDA version from Docker image name."""
    match = re.search(r"cuda:(\d+\.\d+)", docker_image)
    return float(match.group(1)) if match else None


def group_offers_by_cluster(offers: list[OfferResponse]) -> dict[int, list[OfferResponse]]:
    """Group offers by physical cluster_id."""
    clusters: dict[int, list[OfferResponse]] = {}
    for offer in offers:
        cid = offer.get("cluster_id")
        if cid is not None:
            clusters.setdefault(cid, []).append(offer)
    return clusters


def select_best_cluster(
    offers: list[OfferResponse],
    nodes_needed: int,
    use_interruptible: bool = True,
) -> tuple[int, list[OfferResponse]] | None:
    """Select best cluster with enough capacity."""
    all_clusters = select_all_valid_clusters(offers, nodes_needed, use_interruptible)
    return all_clusters[0] if all_clusters else None


def select_all_valid_clusters(
    offers: list[OfferResponse],
    nodes_needed: int,
    use_interruptible: bool = True,
) -> list[tuple[int, list[OfferResponse]]]:
    """Select all valid clusters with enough capacity, sorted by CUDA version (desc) then price."""
    clusters = group_offers_by_cluster(offers)

    valid = [
        (cid, cluster_offers)
        for cid, cluster_offers in clusters.items()
        if len(cluster_offers) >= nodes_needed
    ]

    if not valid:
        return []

    price_key = "min_bid" if use_interruptible else "dph_total"

    # Sort offers within each cluster by CUDA (desc), then price (asc)
    for _, cluster_offers in valid:
        cluster_offers.sort(key=lambda o: (
            -o.get("cuda_max_good", 0),
            o.get(price_key, float("inf")),
        ))

    def cluster_score(item: tuple[int, list[OfferResponse]]) -> tuple[float, float]:
        """Sort by min CUDA in cluster (desc), then total cost (asc)."""
        _, cluster_offers = item
        selected = cluster_offers[:nodes_needed]
        # Use min CUDA of selected offers (we want all nodes to have good CUDA)
        min_cuda = min(o.get("cuda_max_good", 0) for o in selected)
        total_cost = sum(o.get(price_key, float("inf")) for o in selected)
        return (-min_cuda, total_cost)

    valid.sort(key=cluster_score)
    return valid
