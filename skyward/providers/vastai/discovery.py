"""Vast.ai offer discovery and InstanceSpec conversion.

Vast.ai is a GPU marketplace with dynamic offers from various hosts.
Unlike traditional cloud providers with fixed instance types, offers
change frequently based on availability and host pricing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from skyward.types import InstanceSpec

if TYPE_CHECKING:
    from vastai import VastAI


def normalize_gpu_name(gpu_name: str | None) -> str | None:
    """Normalize Vast.ai GPU name to standard format.

    Vast.ai uses names like "RTX_4090", "A100_PCIE", "H100_SXM".
    Normalize to "RTX4090", "A100", "H100".

    Args:
        gpu_name: Raw GPU name from Vast.ai API.

    Returns:
        Normalized GPU name or None.

    Examples:
        >>> normalize_gpu_name("RTX_4090")
        'RTX4090'
        >>> normalize_gpu_name("A100_PCIE")
        'A100'
        >>> normalize_gpu_name("H100_SXM")
        'H100'
        >>> normalize_gpu_name("H100_NVL")
        'H100NVL'
    """
    if not gpu_name:
        return None

    name = gpu_name.upper()

    # Remove interface suffixes but keep variant suffixes like NVL
    for suffix in ("_PCIE", "_SXM", "_80GB", "_40GB"):
        name = name.replace(suffix, "")

    # Keep NVL as it indicates a different variant (96GB vs 80GB)
    # Just remove underscores
    name = name.replace("_", "")

    return name


def offer_to_instance_spec(offer: dict[str, Any]) -> InstanceSpec:
    """Convert Vast.ai offer to InstanceSpec.

    Args:
        offer: Offer dict from Vast.ai API.

    Returns:
        InstanceSpec with normalized fields.
    """
    gpu_name = normalize_gpu_name(offer.get("gpu_name"))

    # gpu_ram is in MB, convert to GB
    gpu_ram_mb = offer.get("gpu_ram", 0)
    gpu_ram_gb = gpu_ram_mb / 1024 if gpu_ram_mb else 0

    return InstanceSpec(
        name=str(offer.get("id", "")),  # Use offer ID as "instance type"
        vcpu=offer.get("cpu_cores", 0),
        memory_gb=offer.get("cpu_ram", 0),
        accelerator=gpu_name,
        accelerator_count=offer.get("num_gpus", 0),
        accelerator_memory_gb=gpu_ram_gb,
        price_on_demand=offer.get("dph_total"),  # dollars per hour
        price_spot=offer.get("min_bid"),  # minimum bid for interruptible
        billing_increment_minutes=None,  # per-second billing
        metadata={
            "offer_id": offer.get("id"),
            "machine_id": offer.get("machine_id"),
            "reliability": offer.get("reliability"),
            "geolocation": offer.get("geolocation"),
            "cuda_version": offer.get("cuda_max_good"),
            "driver_version": offer.get("driver_version"),
            "dlperf": offer.get("dlperf"),  # DL performance score
            "dlperf_per_dphtotal": offer.get("dlperf_per_dphtotal"),  # perf/cost
            "inet_up": offer.get("inet_up"),  # upload bandwidth Mbps
            "inet_down": offer.get("inet_down"),  # download bandwidth Mbps
            "disk_space": offer.get("disk_space"),  # available disk GB
            "pcie_bw": offer.get("pcie_bw"),  # PCIe bandwidth
            "verified": offer.get("verified", False),
        },
    )


def extract_cuda_version(docker_image: str) -> float | None:
    """Extract CUDA version from Docker image name.

    Args:
        docker_image: Docker image name (e.g., "nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu22.04").

    Returns:
        CUDA version as float (e.g., 12.8), or None if not found.
    """
    import re

    # Match CUDA version in image tag (e.g., "12.8.0" or "12.8")
    match = re.search(r"cuda:(\d+\.\d+)", docker_image)
    if match:
        return float(match.group(1))
    return None


def build_search_query(
    *,
    gpu_name: str | None = None,
    num_gpus: int | None = None,
    min_cpu: int | None = None,
    min_memory_gb: float | None = None,
    min_reliability: float = 0.95,
    min_cuda_version: float | None = None,
    min_disk_gb: float | None = None,
    geolocation: str | None = None,
    verified_only: bool = False,
    require_cluster: bool = False,
) -> str:
    """Build Vast.ai search query string.

    Args:
        gpu_name: GPU model name (e.g., "RTX_4090", "H100_SXM").
        num_gpus: Minimum number of GPUs.
        min_cpu: Minimum CPU cores.
        min_memory_gb: Minimum system RAM in GB.
        min_reliability: Minimum host reliability score (0-1).
        min_cuda_version: Minimum CUDA version supported by host (e.g., 12.8).
        min_disk_gb: Minimum available disk space in GB.
        geolocation: Country/region code (e.g., "US", "EU").
        verified_only: Only include verified hosts.
        require_cluster: If True, only return offers in physical clusters
                        (required for overlay networking in multi-node setups).

    Returns:
        Query string for Vast.ai search API.

    Examples:
        >>> build_search_query(gpu_name="RTX_4090", num_gpus=1, min_cuda_version=12.8)
        'gpu_name=RTX_4090 num_gpus>=1 cuda_vers>=12.8 reliability>0.95 rentable=true'
    """
    parts = []

    # NOTE: gpu_name filter is buggy in VastAI when combined with other filters.
    # The combination often returns empty results even when valid offers exist.
    # GPU filtering should be done client-side instead.
    if gpu_name:
        gpu_query = gpu_name.replace(" ", "_")
        parts.append(f"gpu_name={gpu_query}")

    if num_gpus:
        parts.append(f"num_gpus>={num_gpus}")

    if min_cpu:
        parts.append(f"cpu_cores>={min_cpu}")

    if min_memory_gb:
        parts.append(f"cpu_ram>={min_memory_gb}")

    # NOTE: cuda_vers filter is buggy in VastAI when combined with gpu_name.
    # The combination returns empty results even when valid offers exist.
    # CUDA filtering should be done client-side instead.
    # if min_cuda_version:
    #     parts.append(f"cuda_vers>={min_cuda_version}")

    if min_disk_gb:
        parts.append(f"disk_space>={min_disk_gb}")

    if min_reliability > 0:
        parts.append(f"reliability>{min_reliability}")

    if geolocation:
        parts.append(f"geolocation={geolocation}")

    if verified_only:
        parts.append("verified=true")

    # NOTE: cluster_id is NOT a searchable field in VastAI CLI.
    # Filtering by cluster must be done client-side after fetching offers.
    # The require_cluster parameter is kept for API compatibility but does nothing.

    # Always filter to rentable offers
    parts.append("rentable=true")

    return " ".join(parts)


def search_offers(client: VastAI, query: str) -> list[dict[str, Any]]:
    """Search Vast.ai offers with caching.

    Args:
        client: Authenticated VastAI client.
        query: Vast.ai query string.

    Returns:
        List of offer dicts from Vast.ai API.
    """
    result = client.search_offers(query=query, limit=1000)

    if not result:
        return []

    # Handle both dict with 'offers' key and direct list
    if isinstance(result, dict):
        return result.get("offers", [])

    return list(result)


def fetch_available_offers(
    client: VastAI,
    min_reliability: float = 0.95,
    geolocation: str | None = None,
) -> tuple[InstanceSpec, ...]:
    """Fetch all available offers as InstanceSpecs.

    Unlike traditional providers with fixed instance types, this
    searches current marketplace offers.

    Args:
        client: Authenticated VastAI client.
        min_reliability: Minimum host reliability score.
        geolocation: Optional region filter.

    Returns:
        Tuple of InstanceSpec sorted by accelerator, count, price.
    """
    query = build_search_query(
        min_reliability=min_reliability,
        geolocation=geolocation,
    )

    offers = search_offers(client, query)
    specs = [offer_to_instance_spec(o) for o in offers]

    return tuple(
        sorted(
            specs,
            key=lambda s: (
                s.Accelerator or "",
                s.accelerator_count,
                s.price_on_demand or float("inf"),
            ),
        )
    )


def group_offers_by_cluster(
    offers: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Group offers by their physical cluster_id.

    Physical clusters are groups of machines with fast local networking,
    required for overlay network creation.

    Args:
        offers: List of VastAI offer dicts.

    Returns:
        Dict mapping cluster_id -> list of offers in that cluster.
        Offers without cluster_id (None) are excluded.
    """
    clusters: dict[int, list[dict[str, Any]]] = {}

    for offer in offers:
        cluster_id = offer.get("cluster_id")
        if cluster_id is not None:
            clusters.setdefault(cluster_id, []).append(offer)

    return clusters


def select_best_cluster(
    offers: list[dict[str, Any]],
    nodes_needed: int,
    use_interruptible: bool = True,
) -> tuple[int, list[dict[str, Any]]] | None:
    """Select the best cluster that can accommodate required nodes.

    Finds the physical cluster with the lowest total cost for the
    required number of nodes.

    Args:
        offers: List of VastAI offers (should include cluster_id field).
        nodes_needed: Number of instances to provision.
        use_interruptible: If True, sort by min_bid; otherwise by dph_total.

    Returns:
        Tuple of (cluster_id, sorted list of offers) or None if no cluster
        has enough capacity.
    """
    clusters = group_offers_by_cluster(offers)

    # Filter clusters with enough capacity
    valid_clusters = [
        (cid, cluster_offers)
        for cid, cluster_offers in clusters.items()
        if len(cluster_offers) >= nodes_needed
    ]

    if not valid_clusters:
        return None

    # Sort offers within each cluster by price
    price_key = "min_bid" if use_interruptible else "dph_total"

    for _, cluster_offers in valid_clusters:
        cluster_offers.sort(key=lambda o: o.get(price_key) or float("inf"))

    # Select cluster with lowest total cost for required nodes
    def cluster_cost(item: tuple[int, list[dict[str, Any]]]) -> float:
        _, cluster_offers = item
        return sum(
            o.get(price_key) or float("inf")
            for o in cluster_offers[:nodes_needed]
        )

    valid_clusters.sort(key=cluster_cost)

    cluster_id, cluster_offers = valid_clusters[0]
    return cluster_id, cluster_offers
