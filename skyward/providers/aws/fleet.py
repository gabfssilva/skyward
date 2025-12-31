"""EC2 Fleet management for launching instances."""

from __future__ import annotations

import base64
import contextlib
import uuid
from math import ceil
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError

from skyward.constants import DEFAULT_INSTANCE_NAME, SkywardTag
from skyward.spec import AllocationStrategy, NormalizedSpot, Spot, _SpotNever

from .infra import AWSResources

if TYPE_CHECKING:
    from mypy_boto3_ec2 import EC2Client


# =============================================================================
# Fleet Launch
# =============================================================================


def launch_fleet(
    ec2: EC2Client,
    resources: AWSResources,
    n: int,
    instance_type: str,
    ami_id: str,
    user_data: str,
    requirements_hash: str,
    spot: bool = False,
    subnet_ids: tuple[str, ...] | None = None,
    allocation_strategy: AllocationStrategy = "price-capacity-optimized",
    root_device: str = "/dev/sda1",
    min_volume_size: int = 30,
) -> list[str]:
    """Low-level Fleet launch with intelligent AZ selection.

    Uses EC2 Fleet with type='instant' and SingleAvailabilityZone=True to:
    1. Evaluate capacity across all availability zones
    2. Select the best AZ based on allocation_strategy
    3. Launch ALL instances in that single AZ (required for cluster locality)

    Args:
        ec2: EC2 client
        resources: AWS resources (security group, instance profile, etc.)
        n: Number of instances to launch
        instance_type: EC2 instance type
        ami_id: AMI ID
        user_data: User data script
        requirements_hash: Content hash for tagging
        spot: Whether to use spot instances
        subnet_ids: Specific subnets to use. If None, uses resources.subnet_ids.
        allocation_strategy: Spot allocation strategy
        root_device: Root device name
        min_volume_size: Minimum root volume size in GB
    """
    # Filter to subnets in AZs that support this instance type
    if subnet_ids:
        target_subnets = subnet_ids
    else:
        target_subnets = get_valid_subnets_for_instance_type(ec2, instance_type, resources.subnet_ids)

    # Create temporary launch template
    template_name = f"skyward-{uuid.uuid4().hex[:8]}"
    template_data: dict[str, Any] = {
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "IamInstanceProfile": {"Arn": resources.instance_profile_arn},
        "UserData": base64.b64encode(user_data.encode()).decode(),
        "SecurityGroupIds": [resources.security_group_id],
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
        # Overrides with target subnets - Fleet will choose the best AZ
        overrides = [{"SubnetId": sid} for sid in target_subnets]

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
            raise RuntimeError(f"Fleet launch failed: {'; '.join(error_msgs)}")

        if len(instance_ids) < n:
            raise RuntimeError(f"Fleet launched {len(instance_ids)}/{n} instances. " f"Errors: {errors}")

        return instance_ids

    finally:
        # Always clean up the temporary launch template
        with contextlib.suppress(ClientError):
            ec2.delete_launch_template(LaunchTemplateId=template_id)


def get_valid_subnets_for_instance_type(
    ec2: EC2Client,
    instance_type: str,
    subnet_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Filter subnets to only those in AZs that support the instance type."""
    # Get AZs that support this instance type
    response = ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": [instance_type]}],
    )
    valid_azs = {o["Location"] for o in response.get("InstanceTypeOfferings", [])}

    # Filter subnets to only those in valid AZs
    subnets = ec2.describe_subnets(SubnetIds=list(subnet_ids))
    valid_subnets = [s["SubnetId"] for s in subnets["Subnets"] if s["AvailabilityZone"] in valid_azs]

    if not valid_subnets:
        region = subnets["Subnets"][0]["AvailabilityZone"][:-1] if subnets["Subnets"] else "unknown"
        raise RuntimeError(f"No AZs in {region} support {instance_type}")

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
# Instance Launch with Spot Strategy
# =============================================================================


def launch_instances(
    ec2: EC2Client,
    resources: AWSResources,
    n: int,
    instance_type: str,
    ami_id: str,
    user_data: str,
    requirements_hash: str,
    spot_strategy: NormalizedSpot,
    allocation_strategy: AllocationStrategy = "price-capacity-optimized",
    root_device: str = "/dev/sda1",
    min_volume_size: int = 30,
) -> list[str]:
    """Launch instances with full SpotStrategy support.

    Handles all spot strategies:
    - Spot.Always(): 100% spot, fail if unavailable
    - Spot.Never: 100% on-demand
    - Spot.IfAvailable(): try spot, fallback to on-demand
    - Spot.Percent(x): minimum x% spot, rest on-demand in same AZ
    """
    common_args = {
        "ec2": ec2,
        "resources": resources,
        "instance_type": instance_type,
        "ami_id": ami_id,
        "user_data": user_data,
        "requirements_hash": requirements_hash,
        "allocation_strategy": allocation_strategy,
        "root_device": root_device,
        "min_volume_size": min_volume_size,
    }

    match spot_strategy:
        case _SpotNever():
            # 100% on-demand
            return launch_fleet(n=n, spot=False, **common_args)

        case Spot.Always():
            # 100% spot, fail if unavailable
            return launch_fleet(n=n, spot=True, **common_args)

        case Spot.IfAvailable():
            # Try spot first, fallback to on-demand
            try:
                return launch_fleet(n=n, spot=True, **common_args)
            except (RuntimeError, ClientError):
                return launch_fleet(n=n, spot=False, **common_args)

        case Spot.Percent(percentage=min_pct):
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
            raise ValueError(f"Unknown spot strategy: {spot_strategy}")


# =============================================================================
# Instance Waiting
# =============================================================================


def wait_running(ec2: EC2Client, instance_ids: list[str]) -> None:
    """Wait for instances to be in running state."""
    from skyward.constants import INSTANCE_RUNNING_MAX_ATTEMPTS, INSTANCE_RUNNING_WAIT_DELAY

    if not instance_ids:
        return

    waiter = ec2.get_waiter("instance_running")
    waiter.wait(
        InstanceIds=instance_ids,
        WaiterConfig={
            "Delay": INSTANCE_RUNNING_WAIT_DELAY,
            "MaxAttempts": INSTANCE_RUNNING_MAX_ATTEMPTS,
        },
    )


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
                    "spot": i.get("InstanceLifecycle") == "spot",
                    "instance_type": i.get("InstanceType", ""),
                }
            )

    instances.sort(key=lambda x: x["id"])
    return instances
