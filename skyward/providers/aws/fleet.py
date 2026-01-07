"""EC2 Fleet management for launching instances."""

from __future__ import annotations

import base64
import contextlib
import uuid
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError
from loguru import logger

from skyward.core.constants import DEFAULT_INSTANCE_NAME, SkywardTag
from skyward.spec.allocation import Allocation, AllocationStrategy, NormalizedAllocation, _AllocationOnDemand

from .infra import AWSResources

if TYPE_CHECKING:
    from mypy_boto3_ec2 import EC2Client


# =============================================================================
# Instance Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class InstanceConfig:
    """Configuration for an instance type with its AMI and pricing."""

    instance_type: str
    ami_id: str
    price_spot: float | None = None
    price_on_demand: float | None = None


# =============================================================================
# Fleet Launch
# =============================================================================


def launch_fleet(
    ec2: EC2Client,
    resources: AWSResources,
    n: int,
    instances: tuple[InstanceConfig, ...],
    user_data: str,
    requirements_hash: str,
    spot: bool = False,
    subnet_ids: tuple[str, ...] | None = None,
    allocation_strategy: AllocationStrategy = "price-capacity-optimized",
    root_device: str = "/dev/sda1",
    min_volume_size: int = 30,
) -> list[str]:
    """Low-level Fleet launch with intelligent AZ and instance type selection.

    Uses EC2 Fleet with type='instant' and SingleAvailabilityZone=True to:
    1. Evaluate capacity across all availability zones and instance types
    2. Select the best AZ + instance type based on allocation_strategy
    3. Launch ALL instances in that single AZ with same type (cluster locality)

    Args:
        ec2: EC2 client
        resources: AWS resources (security group, instance profile, etc.)
        n: Number of instances to launch
        instances: Instance configurations (type + AMI). Fleet picks one with capacity.
        user_data: User data script
        requirements_hash: Content hash for tagging
        spot: Whether to use spot instances
        subnet_ids: Specific subnets to use. If None, uses resources.subnet_ids.
        allocation_strategy: Spot allocation strategy
        root_device: Root device name
        min_volume_size: Minimum root volume size in GB
    """
    # Filter to subnets in AZs that support at least one of the instance types
    instance_types = tuple(inst.instance_type for inst in instances)
    market_type = "spot" if spot else "on-demand"
    logger.info(f"Launching EC2 fleet: {n} {market_type} instances")
    logger.debug(f"Candidate instance types: {instance_types[:5]}{'...' if len(instance_types) > 5 else ''}")

    if subnet_ids:
        target_subnets = subnet_ids
    else:
        target_subnets = get_valid_subnets_for_instance_types(
            ec2, instance_types, resources.subnet_ids
        )
    logger.debug(f"Target subnets: {target_subnets}")

    # Create temporary launch template (without InstanceType/ImageId - goes in overrides)
    template_name = f"skyward-{uuid.uuid4().hex[:8]}"
    template_data: dict[str, Any] = {
        "IamInstanceProfile": {"Arn": resources.instance_profile_arn},
        "UserData": base64.b64encode(user_data.encode()).decode(),
        # Use NetworkInterfaces to ensure public IP is assigned (required for SSH)
        # SubnetId comes from Fleet overrides, not here
        "NetworkInterfaces": [
            {
                "DeviceIndex": 0,
                "AssociatePublicIpAddress": True,
                "Groups": [resources.security_group_id],
            }
        ],
        "BlockDeviceMappings": [
            {
                "DeviceName": root_device,
                "Ebs": {
                    "VolumeSize": max(min_volume_size, 30),
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": DEFAULT_INSTANCE_NAME},
                    {"Key": SkywardTag.MANAGED, "Value": "true"},
                    {"Key": SkywardTag.REQUIREMENTS_HASH, "Value": requirements_hash},
                ],
            }
        ],
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
        "InstanceInitiatedShutdownBehavior": "terminate",
    }

    launch_template = ec2.create_launch_template(
        LaunchTemplateName=template_name,
        LaunchTemplateData=template_data,
    )
    template_id = launch_template["LaunchTemplate"]["LaunchTemplateId"]

    try:
        # Overrides: cartesian product of subnets × instances
        # Fleet will choose the best combination based on capacity and price
        overrides = [
            {"SubnetId": sid, "InstanceType": inst.instance_type, "ImageId": inst.ami_id}
            for sid in target_subnets
            for inst in instances
        ]

        fleet_response = ec2.create_fleet(
            Type="instant",
            LaunchTemplateConfigs=[
                {
                    "LaunchTemplateSpecification": {
                        "LaunchTemplateId": template_id,
                        "Version": "$Latest",
                    },
                    "Overrides": overrides,
                }
            ],
            TargetCapacitySpecification={
                "TotalTargetCapacity": n,
                "DefaultTargetCapacityType": "spot" if spot else "on-demand",
                "SpotTargetCapacity": n if spot else 0,
                "OnDemandTargetCapacity": 0 if spot else n,
            },
            # SingleAvailabilityZone + SingleInstanceType ensures ALL instances
            # land in the SAME AZ with the SAME instance type (cluster locality)
            SpotOptions={
                "AllocationStrategy": allocation_strategy,
                "SingleAvailabilityZone": True,
                "SingleInstanceType": True,
            },
            OnDemandOptions={
                "AllocationStrategy": "lowest-price",
                "SingleAvailabilityZone": True,
                "SingleInstanceType": True,
            },
        )

        # Extract instance IDs from response
        instance_ids: list[str] = []
        for instance_set in fleet_response.get("Instances", []):
            instance_ids.extend(instance_set.get("InstanceIds", []))

        # Check for errors
        errors = fleet_response.get("Errors", [])
        if errors and not instance_ids:
            error_msgs = [f"{e.get('ErrorCode', 'Unknown')}: {e.get('ErrorMessage', '')}" for e in errors]
            logger.error(f"Fleet launch failed: {'; '.join(error_msgs)}")
            raise RuntimeError(f"Fleet launch failed: {'; '.join(error_msgs)}")

        if len(instance_ids) < n:
            logger.error(f"Fleet launched {len(instance_ids)}/{n} instances. Errors: {errors}")
            raise RuntimeError(f"Fleet launched {len(instance_ids)}/{n} instances. " f"Errors: {errors}")

        logger.info(f"Fleet launched {len(instance_ids)} instances: {instance_ids}")
        return instance_ids

    finally:
        # Always clean up the temporary launch template
        with contextlib.suppress(ClientError):
            ec2.delete_launch_template(LaunchTemplateId=template_id)

def get_valid_subnets_for_instance_types(
    ec2: EC2Client,
    instance_types: tuple[str, ...],
    subnet_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Filter subnets to only those in AZs that support at least one instance type."""
    # Get AZs that support any of these instance types
    response = ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": list(instance_types)}],
    )
    valid_azs = {o["Location"] for o in response.get("InstanceTypeOfferings", [])}

    # Filter subnets to only those in valid AZs
    subnets = ec2.describe_subnets(SubnetIds=list(subnet_ids))
    valid_subnets = [s["SubnetId"] for s in subnets["Subnets"] if s["AvailabilityZone"] in valid_azs]

    if not valid_subnets:
        region = subnets["Subnets"][0]["AvailabilityZone"][:-1] if subnets["Subnets"] else "unknown"
        raise RuntimeError(f"No AZs in {region} support any of {instance_types}")

    return tuple(valid_subnets)


def get_instance_az(ec2: EC2Client, instance_id: str) -> str:
    """Get the availability zone of an instance."""
    response = ec2.describe_instances(InstanceIds=[instance_id])
    return response["Reservations"][0]["Instances"][0]["Placement"]["AvailabilityZone"]


def find_subnet_for_az(ec2: EC2Client, az: str, subnet_ids: tuple[str, ...]) -> str:
    """Find a subnet in the specified availability zone."""
    subnets = ec2.describe_subnets(SubnetIds=list(subnet_ids))
    for subnet in subnets["Subnets"]:
        if subnet["AvailabilityZone"] == az:
            return subnet["SubnetId"]
    raise RuntimeError(f"No subnet found for AZ {az}")


# =============================================================================
# Instance Launch with Allocation Strategy
# =============================================================================


def launch_instances(
    ec2: EC2Client,
    resources: AWSResources,
    n: int,
    instances: tuple[InstanceConfig, ...],
    user_data: str,
    requirements_hash: str,
    allocation: NormalizedAllocation,
    fleet_strategy: AllocationStrategy = "price-capacity-optimized",
    root_device: str = "/dev/sda1",
    min_volume_size: int = 30,
) -> list[str]:
    """Launch instances with allocation strategy support.

    Handles all allocation strategies:
    - Allocation.AlwaysSpot(): 100% spot, fail if unavailable
    - Allocation.OnDemand: 100% on-demand
    - Allocation.SpotIfAvailable(): try spot, fallback to on-demand
    - Allocation.Percent(spot=x): minimum x% spot, rest on-demand
    - Allocation.Cheapest(): compare spot vs on-demand prices, pick cheapest

    Args:
        instances: Instance configurations (type + AMI + prices).
        allocation: Normalized allocation strategy.
        fleet_strategy: EC2 Fleet allocation strategy.
    """
    common_args = {
        "ec2": ec2,
        "resources": resources,
        "instances": instances,
        "user_data": user_data,
        "requirements_hash": requirements_hash,
        "allocation_strategy": fleet_strategy,
        "root_device": root_device,
        "min_volume_size": min_volume_size,
    }

    logger.debug(f"Launch strategy: {type(allocation).__name__}")

    match allocation:
        case _AllocationOnDemand():
            # 100% on-demand
            logger.debug("Using on-demand allocation")
            return launch_fleet(n=n, spot=False, **common_args)

        case Allocation.AlwaysSpot():
            # 100% spot, fail if unavailable
            logger.debug("Using always-spot allocation")
            return launch_fleet(n=n, spot=True, **common_args)

        case Allocation.SpotIfAvailable():
            # Try spot first, fallback to on-demand
            logger.debug("Using spot-if-available allocation")
            try:
                return launch_fleet(n=n, spot=True, **common_args)
            except (RuntimeError, ClientError) as e:
                logger.warning(f"Spot not available: {e}, falling back to on-demand")
                return launch_fleet(n=n, spot=False, **common_args)

        case Allocation.Cheapest():
            # Compare spot vs on-demand prices and try cheapest first
            return _launch_cheapest(n=n, **common_args)

        case Allocation.Percent(spot=min_pct):
            # Minimum min_pct% spot, rest on-demand in same AZ
            min_spot = ceil(n * min_pct)

            # Try 100% spot first
            try:
                return launch_fleet(n=n, spot=True, **common_args)
            except (RuntimeError, ClientError):
                pass

            # Fallback: minimum spot + on-demand in same AZ
            spot_ids = launch_fleet(n=min_spot, spot=True, **common_args)
            spot_az = get_instance_az(ec2, spot_ids[0])
            subnet_in_az = find_subnet_for_az(ec2, spot_az, resources.subnet_ids)

            remaining = n - len(spot_ids)
            if remaining > 0:
                ondemand_ids = launch_fleet(
                    n=remaining,
                    spot=False,
                    subnet_ids=(subnet_in_az,),
                    **common_args,
                )
                return spot_ids + ondemand_ids

            return spot_ids

        case _:
            raise ValueError(f"Unknown allocation strategy: {allocation}")


def _launch_cheapest(
    ec2: EC2Client,
    resources: AWSResources,
    n: int,
    instances: tuple[InstanceConfig, ...],
    user_data: str,
    requirements_hash: str,
    allocation_strategy: AllocationStrategy,
    root_device: str,
    min_volume_size: int,
) -> list[str]:
    """Launch instances using cheapest option (spot or on-demand).

    Compares the cheapest spot price vs cheapest on-demand price and tries
    the cheaper market first. Falls back to the other if capacity unavailable.
    """
    common_args = {
        "ec2": ec2,
        "resources": resources,
        "instances": instances,
        "user_data": user_data,
        "requirements_hash": requirements_hash,
        "allocation_strategy": allocation_strategy,
        "root_device": root_device,
        "min_volume_size": min_volume_size,
    }

    # Find cheapest spot and on-demand prices
    spot_prices = [i.price_spot for i in instances if i.price_spot is not None]
    ondemand_prices = [i.price_on_demand for i in instances if i.price_on_demand is not None]

    min_spot = min(spot_prices) if spot_prices else float("inf")
    min_ondemand = min(ondemand_prices) if ondemand_prices else float("inf")

    # Try cheapest market first, fallback to other
    if min_spot <= min_ondemand:
        # Spot is cheaper or equal, try spot first
        try:
            return launch_fleet(n=n, spot=True, **common_args)
        except (RuntimeError, ClientError):
            return launch_fleet(n=n, spot=False, **common_args)
    else:
        # On-demand is cheaper, try on-demand first
        try:
            return launch_fleet(n=n, spot=False, **common_args)
        except (RuntimeError, ClientError):
            return launch_fleet(n=n, spot=True, **common_args)


# =============================================================================
# Instance Waiting
# =============================================================================


def wait_running(ec2: EC2Client, instance_ids: list[str]) -> None:
    """Wait for instances to be in running state."""
    from skyward.core.constants import INSTANCE_RUNNING_MAX_ATTEMPTS, INSTANCE_RUNNING_WAIT_DELAY

    if not instance_ids:
        return

    logger.debug(f"Waiting for {len(instance_ids)} instances to reach running state...")
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(
        InstanceIds=instance_ids,
        WaiterConfig={
            "Delay": INSTANCE_RUNNING_WAIT_DELAY,
            "MaxAttempts": INSTANCE_RUNNING_MAX_ATTEMPTS,
        },
    )
    logger.debug(f"All {len(instance_ids)} instances are running")


def get_instance_details(ec2: EC2Client, instance_ids: list[str]) -> list[dict[str, Any]]:
    """Get instance details (ID, private IP, spot status)."""
    if not instance_ids:
        return []

    response = ec2.describe_instances(InstanceIds=instance_ids)

    instances = []
    for r in response["Reservations"]:
        for i in r["Instances"]:
            instances.append(
                {
                    "id": i["InstanceId"],
                    "private_ip": i.get("PrivateIpAddress", ""),
                    "public_ip": i.get("PublicIpAddress"),
                    "spot": i.get("InstanceLifecycle") == "spot",
                    "instance_type": i.get("InstanceType", ""),
                }
            )

    instances.sort(key=lambda x: x["id"])
    return instances
