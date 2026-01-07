"""AWS instance type discovery, parsing, and region availability checking."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final, Literal

from botocore.exceptions import ClientError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from skyward.cache import cached
from skyward.types import InstanceSpec

if TYPE_CHECKING:
    from mypy_boto3_ec2 import EC2Client


# =============================================================================
# Constants
# =============================================================================

type Architecture = Literal["x86_64", "arm64"]

DISCOVERY_CACHE_TTL: Final[timedelta] = timedelta(hours=24)
VANTAGE_AWS_URL: Final[str] = "https://instances.vantage.sh/instances.json"

# AWS API GPU name -> Skyward internal name
AWS_GPU_NAME_MAP: Final[dict[str, str]] = {
    "M60": "M60",
    "T4": "T4",
    "T4g": "T4",
    "V100": "V100",
    "A10G": "A10G",
    "L4": "L4",
    "L40S": "L40S",
    "H100": "H100",
    "H200": "H200",
    "B200": "B200",
}

# Trainium/Inferentia device name -> Skyward internal name
AWS_NEURON_NAME_MAP: Final[dict[str, str]] = {
    "Trainium": "Trainium1",
    "Inferentia": "Inferentia1",
    "Inferentia2": "Inferentia2",
}

DLAMI_GPU_SSM: dict[Architecture, str] = {
    "x86_64": "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id",
    "arm64": "/aws/service/deeplearning/ami/arm64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id",
}

# Ubuntu base for fractional GPU (requires GRID driver, not DLAMI)
UBUNTU_BASE_SSM: dict[Architecture, str] = {
    "x86_64": "/aws/service/canonical/ubuntu/server/jammy/stable/current/amd64/hvm/ebs-gp2/ami-id",
    "arm64": "/aws/service/canonical/ubuntu/server/jammy/stable/current/arm64/hvm/ebs-gp2/ami-id",
}


# =============================================================================
# Exceptions
# =============================================================================


class NoAvailableRegionError(Exception):
    """No region has the requested instance type available."""

    def __init__(self, instance_type: str, regions_checked: list[str]):
        regions_str = ", ".join(regions_checked) if regions_checked else "none"
        super().__init__(
            f"No region has instance type '{instance_type}' available. "
            f"Regions checked: {regions_str}. "
            "Try a different instance type or check AWS capacity."
        )
        self.instance_type = instance_type
        self.regions_checked = regions_checked


# =============================================================================
# GPU/Neuron Name Normalization
# =============================================================================


def normalize_gpu_name(aws_name: str, memory_mib: int) -> str:
    """Normalize AWS GPU name to Skyward internal name."""
    memory_gb = memory_mib / 1024

    if aws_name == "A100":
        return "A100-40" if memory_gb < 50 else "A100-80"

    return AWS_GPU_NAME_MAP.get(aws_name, aws_name)


def normalize_neuron_name(aws_name: str, instance_type: str = "") -> str:
    """Normalize AWS Neuron device name to Skyward internal name."""
    if aws_name == "Trainium" and instance_type.startswith("trn2"):
        return "Trainium2"

    return AWS_NEURON_NAME_MAP.get(aws_name, aws_name)


# =============================================================================
# Instance Type Parsing
# =============================================================================


def parse_instance_info(info: dict[str, Any]) -> dict[str, Any]:
    """Parse instance type info from EC2 API."""
    instance_type = info.get("InstanceType", "")
    vcpu = info.get("VCpuInfo", {}).get("DefaultVCpus", 0)
    memory_mib = info.get("MemoryInfo", {}).get("SizeInMiB", 0)
    memory_gb = memory_mib / 1024

    placement_info = info.get("PlacementGroupInfo", {})
    supported_strategies = placement_info.get("SupportedStrategies", [])
    supports_cluster = "cluster" in supported_strategies

    accelerator: str | None = None
    accelerator_count = 0
    accelerator_memory_gb = 0.0

    gpu_info = info.get("GpuInfo", {})
    gpus = gpu_info.get("Gpus", [])
    if gpus:
        gpu = gpus[0]
        aws_name = gpu.get("Name", "")
        count = gpu.get("Count", 0)
        gpu_memory_mib = gpu.get("MemoryInfo", {}).get("SizeInMiB", 0)

        accelerator = normalize_gpu_name(aws_name, gpu_memory_mib)
        accelerator_memory_gb = gpu_memory_mib / 1024

        # Fractional GPU: count=0 but has memory (e.g., G6f instances)
        if count == 0 and gpu_memory_mib > 0:
            from skyward.accelerators import AcceleratorSpec as Acc

            base = Acc.from_name(aws_name)
            if base and base.memory:
                full_memory_gb = int(base.memory.replace("GB", ""))
                count = accelerator_memory_gb / full_memory_gb

        accelerator_count = count

    neuron_info = info.get("NeuronInfo", {})
    neuron_devices = neuron_info.get("NeuronDevices", [])
    if neuron_devices:
        device = neuron_devices[0]
        aws_name = device.get("Name", "")
        device_count = device.get("Count", 0)
        core_info = device.get("CoreInfo", {})
        cores_per_device = core_info.get("Count", 2)
        device_memory_mib = device.get("MemoryInfo", {}).get("SizeInMiB", 0)

        accelerator = normalize_neuron_name(aws_name, instance_type)
        accelerator_count = device_count * cores_per_device
        accelerator_memory_gb = (device_memory_mib / 1024) / cores_per_device

    # Get architecture from API (prefer arm64 if both supported)
    processor_info = info.get("ProcessorInfo", {})
    supported_archs = processor_info.get("SupportedArchitectures", ["x86_64"])
    architecture: Architecture = "arm64" if "arm64" in supported_archs else "x86_64"

    return {
        "type_name": instance_type,
        "vcpu": vcpu,
        "memory_gb": memory_gb,
        "architecture": architecture,
        "accelerator": accelerator,
        "accelerator_count": accelerator_count,
        "supports_cluster_placement": supports_cluster,
        "accelerator_memory_gb": accelerator_memory_gb,
    }


# =============================================================================
# Vantage Pricing Integration
# =============================================================================


@cached(namespace="aws.vantage_pricing", ttl=DISCOVERY_CACHE_TTL)
def fetch_vantage_pricing() -> dict[str, dict[str, Any]]:
    """Fetch AWS pricing from Vantage, indexed by instance_type."""
    import httpx

    data = httpx.get(VANTAGE_AWS_URL, timeout=30).json()
    return {inst.get("instance_type", "").lower(): inst for inst in data}


def extract_aws_pricing(
    instance_type: str,
    region: str,
    vantage: dict[str, dict[str, Any]],
) -> tuple[float | None, float | None]:
    """Extract on-demand and spot pricing for an instance type in a region.

    Returns:
        Tuple of (on_demand_price, spot_price) in USD/hour.
    """
    inst = vantage.get(instance_type.lower())
    if not inst:
        return None, None

    pricing = inst.get("pricing", {}).get(region, {}).get("linux", {})

    on_demand: float | None = None
    spot: float | None = None

    if od := pricing.get("ondemand"):
        try:
            on_demand = float(od)
        except (ValueError, TypeError):
            pass

    if sp := pricing.get("spot_avg"):
        try:
            spot = float(sp)
        except (ValueError, TypeError):
            pass

    return on_demand, spot


# =============================================================================
# Instance Type Discovery
# =============================================================================


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(ClientError),
    reraise=True,
)
def fetch_all_instances_from_api(ec2: EC2Client) -> list[dict[str, Any]]:
    """Fetch all instance types from EC2 API with retry."""
    specs: list[dict[str, Any]] = []
    paginator = ec2.get_paginator("describe_instance_types")

    for page in paginator.paginate():
        for instance_info in page.get("InstanceTypes", []):
            specs.append(parse_instance_info(instance_info))

    return sorted(
        specs,
        key=lambda s: (s.get("accelerator") or "", s.get("accelerator_count", 0), s.get("vcpu", 0)),
    )


def discover_all_instances(ec2: EC2Client, region: str) -> list[dict[str, Any]]:
    """Discover all instance types in a region (cached)."""

    @cached(
        namespace="aws.instance_types",
        ttl=DISCOVERY_CACHE_TTL,
        key_func=lambda *_: f"all:{region}",
    )
    def _cached_discover() -> list[dict[str, Any]]:
        return fetch_all_instances_from_api(ec2)

    return _cached_discover()


def build_instance_specs(
    instance_data: list[dict[str, Any]],
    region: str,
) -> tuple[InstanceSpec, ...]:
    """Build InstanceSpec tuple from raw instance data with pricing."""
    vantage = fetch_vantage_pricing()

    def make_spec(s: dict[str, Any]) -> InstanceSpec:
        od, sp = extract_aws_pricing(s["type_name"], region, vantage)
        return InstanceSpec(
            name=s["type_name"],
            vcpu=s["vcpu"],
            memory_gb=s["memory_gb"],
            accelerator=s.get("accelerator"),
            accelerator_count=s.get("accelerator_count", 0),
            accelerator_memory_gb=s.get("accelerator_memory_gb", 0),
            price_on_demand=od,
            price_spot=sp,
            billing_increment_minutes=1,  # AWS charges per minute
            metadata={
                "architecture": s.get("architecture", "x86_64"),
                "supports_cluster_placement": s.get("supports_cluster_placement", False),
            },
        )

    return tuple(make_spec(s) for s in instance_data)


# =============================================================================
# Region Discovery
# =============================================================================


@cached(
    namespace="aws.regions",
    ttl=DISCOVERY_CACHE_TTL,
    key_func=lambda region: "enabled_regions",
)
def get_enabled_regions(region: str) -> tuple[str, ...]:
    """Get all enabled regions for this AWS account."""
    import boto3

    ec2 = boto3.client("ec2", region_name=region)
    response = ec2.describe_regions(AllRegions=False)
    regions = [r["RegionName"] for r in response.get("Regions", [])]
    return tuple(sorted(regions))


def check_instance_type_in_region(instance_type: str, region: str) -> bool:
    """Check if an instance type is available in a specific region."""
    import boto3

    try:
        ec2 = boto3.client("ec2", region_name=region)
        response = ec2.describe_instance_type_offerings(
            LocationType="region",
            Filters=[{"Name": "instance-type", "Values": [instance_type]}],
            MaxResults=5,
        )
        return len(response.get("InstanceTypeOfferings", [])) > 0
    except ClientError:
        return False


def find_available_region(
    instance_type: str,
    preferred_region: str,
) -> str:
    """Find a region where the instance type is available.

    Args:
        instance_type: The EC2 instance type (e.g., 'p4d.24xlarge')
        preferred_region: Try this region first

    Returns:
        Region name where the instance type is available

    Raises:
        NoAvailableRegionError: If no region has the instance type
    """
    regions_to_check = list(get_enabled_regions(preferred_region))

    # Preferred region first
    if preferred_region in regions_to_check:
        regions_to_check.remove(preferred_region)
        regions_to_check.insert(0, preferred_region)

    checked_regions: list[str] = []
    for region in regions_to_check:
        checked_regions.append(region)
        if check_instance_type_in_region(instance_type, region):
            return region

    raise NoAvailableRegionError(instance_type, checked_regions)
