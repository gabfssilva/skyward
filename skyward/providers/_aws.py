"""AWS EC2 provider for Skyward GPU instances using SSM connectivity.

This module is self-contained - no legacy imports required.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from functools import cached_property, lru_cache
from math import ceil
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, cast, override

from botocore.exceptions import ClientError
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_fixed,
)

from skyward.cache import cached
from skyward.callback import emit
from skyward.constants import (
    DEFAULT_INSTANCE_NAME,
    INSTANCE_RUNNING_MAX_ATTEMPTS,
    INSTANCE_RUNNING_WAIT_DELAY,
    SkywardTag,
)
from skyward.events import (
    BootstrapCompleted,
    BootstrapStarting,
    Error,
    InfraCreated,
    InfraCreating,
    InstanceLaunching,
    InstanceProvisioned,
    InstanceStopping,
    ProviderName,
    ProvisioningCompleted,
    RegionAutoSelected,
)
from skyward.exceptions import InstanceTerminatedError
from skyward.providers.common import (
    Checkpoint,
    create_tunnel,
    find_available_port,
    wait_for_bootstrap as _wait_for_bootstrap_common,
)
from skyward.spec import (
    AllocationStrategy,
    NormalizedSpot,
    Spot,
    _SpotNever,
)
from skyward.types import ComputeSpec, ExitedInstance, Instance, InstanceSpec, Provider, select_instance

if TYPE_CHECKING:
    from subprocess import Popen

    from mypy_boto3_ec2 import EC2Client
    from mypy_boto3_iam import IAMClient
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_s3.literals import BucketLocationConstraintType
    from mypy_boto3_ssm import SSMClient
    from mypy_boto3_sts import STSClient

    from skyward.volume import Volume


# =============================================================================
# Constants
# =============================================================================

type Architecture = Literal["x86_64", "arm64"]

DISCOVERY_CACHE_TTL: timedelta = timedelta(hours=24)
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

# SSM parameter paths for public AMIs
AL2023_CPU_SSM: dict[Architecture, str] = {
    "x86_64": "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64",
    "arm64": "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-arm64",
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

# AWS-specific bootstrap checkpoints (in addition to common ones)
AWS_EXTRA_CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(".step_download", "downloading deps"),
    Checkpoint(".step_volumes", "volumes"),
)


# =============================================================================
# AWS Infrastructure (AWSResources, AWSInfraManager)
# =============================================================================


@dataclass
class AWSResources:
    """Container for AWS resource identifiers."""

    bucket: str
    iam_role_arn: str
    instance_profile_arn: str
    security_group_id: str
    region: str
    subnet_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, str | list[str]]:
        """Serialize to dictionary."""
        return {
            "bucket": self.bucket,
            "iam_role_arn": self.iam_role_arn,
            "instance_profile_arn": self.instance_profile_arn,
            "security_group_id": self.security_group_id,
            "region": self.region,
            "subnet_ids": list(self.subnet_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | list[str]]) -> AWSResources:
        """Deserialize from dictionary."""
        subnet_ids_raw = data.get("subnet_ids")
        subnet_ids: tuple[str, ...]
        if subnet_ids_raw is None:
            old_subnet = data.get("subnet_id", "")
            subnet_ids = (str(old_subnet),) if old_subnet else ()
        elif isinstance(subnet_ids_raw, list):
            subnet_ids = tuple(str(s) for s in subnet_ids_raw)
        else:
            subnet_ids = (str(subnet_ids_raw),)
        return cls(
            bucket=str(data["bucket"]),
            iam_role_arn=str(data["iam_role_arn"]),
            instance_profile_arn=str(data["instance_profile_arn"]),
            security_group_id=str(data["security_group_id"]),
            region=str(data["region"]),
            subnet_ids=subnet_ids,
        )


class AWSInfraManager:
    """Creates and destroys AWS infrastructure for Skyward."""

    EC2_POLICY_DOCUMENT = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "S3ObjectAccess",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                "Resource": "arn:aws:s3:::{bucket}/*",
            },
            {
                "Sid": "S3BucketAccess",
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": "arn:aws:s3:::{bucket}",
            },
            {
                "Sid": "S3VolumeAccess",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                    "s3:GetBucketLocation",
                ],
                "Resource": ["arn:aws:s3:::*", "arn:aws:s3:::*/*"],
            },
            {
                "Sid": "SSMCore",
                "Effect": "Allow",
                "Action": [
                    "ssm:UpdateInstanceInformation",
                    "ssmmessages:CreateControlChannel",
                    "ssmmessages:CreateDataChannel",
                    "ssmmessages:OpenControlChannel",
                    "ssmmessages:OpenDataChannel",
                ],
                "Resource": "*",
            },
            {
                "Sid": "EC2Messages",
                "Effect": "Allow",
                "Action": [
                    "ec2messages:AcknowledgeMessage",
                    "ec2messages:DeleteMessage",
                    "ec2messages:FailMessage",
                    "ec2messages:GetEndpoint",
                    "ec2messages:GetMessages",
                    "ec2messages:SendReply",
                ],
                "Resource": "*",
            },
        ],
    }

    ASSUME_ROLE_POLICY = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    def __init__(self, region: str, prefix: str = "skyward") -> None:
        self.region = region
        self.prefix = prefix

    @cached_property
    def _s3(self) -> S3Client:
        import boto3
        return boto3.client("s3", region_name=self.region)

    @cached_property
    def _ec2(self) -> EC2Client:
        import boto3
        return boto3.client("ec2", region_name=self.region)

    @cached_property
    def _iam(self) -> IAMClient:
        import boto3
        return boto3.client("iam", region_name=self.region)

    @cached_property
    def _sts(self) -> STSClient:
        import boto3
        return boto3.client("sts", region_name=self.region)

    @cached_property
    def _ssm(self) -> SSMClient:
        import boto3
        return boto3.client("ssm", region_name=self.region)

    @cached_property
    def _account_id(self) -> str:
        return self._sts.get_caller_identity()["Account"]

    def _bucket_name(self) -> str:
        return f"{self.prefix}-{self._account_id}-{self.region}"

    def _role_name(self) -> str:
        return f"{self.prefix}-role"

    def _instance_profile_name(self) -> str:
        return f"{self.prefix}-instance-profile"

    def _security_group_name(self) -> str:
        return f"{self.prefix}-sg"

    def _policy_name(self) -> str:
        return f"{self.prefix}-ec2-policy"

    def _get_default_subnets(self) -> tuple[str, ...]:
        vpcs = self._ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
        if not vpcs["Vpcs"]:
            raise RuntimeError("No default VPC found. Please create one or specify a VPC.")
        vpc_id = vpcs["Vpcs"][0]["VpcId"]

        subnets = self._ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
        if not subnets["Subnets"]:
            raise RuntimeError("No subnets found in default VPC.")

        return tuple(s["SubnetId"] for s in subnets["Subnets"])

    def ensure_infrastructure(self) -> AWSResources:
        data = self._ensure_infrastructure_cached()
        resources = AWSResources.from_dict(data)
        self._ensure_iam_role(resources.bucket)
        return resources

    @cached(
        namespace="aws.infrastructure",
        key_func=lambda self: f"{self.region}:{self.prefix}",
    )
    def _ensure_infrastructure_cached(self) -> dict[str, str | list[str]]:
        return self._create_infrastructure().to_dict()

    def _create_infrastructure(self) -> AWSResources:
        bucket = self._ensure_bucket()
        role_arn, profile_arn = self._ensure_iam_role(bucket)
        security_group_id = self._ensure_security_group()
        subnet_ids = self._get_default_subnets()

        return AWSResources(
            bucket=bucket,
            iam_role_arn=role_arn,
            instance_profile_arn=profile_arn,
            security_group_id=security_group_id,
            region=self.region,
            subnet_ids=subnet_ids,
        )

    def _ensure_bucket(self) -> str:
        bucket_name = self._bucket_name()

        try:
            self._s3.head_bucket(Bucket=bucket_name)
        except Exception:
            if self.region == "us-east-1":
                self._s3.create_bucket(Bucket=bucket_name)
            else:
                location = cast("BucketLocationConstraintType", self.region)
                self._s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": location},
                )

        return bucket_name

    def _ensure_iam_role(self, bucket: str) -> tuple[str, str]:
        role_name = self._role_name()
        profile_name = self._instance_profile_name()
        policy_name = self._policy_name()

        try:
            get_role_resp = self._iam.get_role(RoleName=role_name)
            role_arn = get_role_resp["Role"]["Arn"]
        except self._iam.exceptions.NoSuchEntityException:
            create_role_resp = self._iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(self.ASSUME_ROLE_POLICY),
                Description="Skyward EC2 instance role",
                Tags=[{"Key": "skyward:managed", "Value": "true"}],
            )
            role_arn = create_role_resp["Role"]["Arn"]

        policy_doc = json.loads(json.dumps(self.EC2_POLICY_DOCUMENT))
        policy_doc["Statement"][0]["Resource"] = f"arn:aws:s3:::{bucket}/*"
        policy_doc["Statement"][1]["Resource"] = f"arn:aws:s3:::{bucket}"

        self._iam.put_role_policy(
            RoleName=role_name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_doc),
        )

        try:
            self._iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
            )
        except self._iam.exceptions.EntityAlreadyExistsException:
            pass

        try:
            get_profile_resp = self._iam.get_instance_profile(InstanceProfileName=profile_name)
            profile_arn = get_profile_resp["InstanceProfile"]["Arn"]

            roles = get_profile_resp["InstanceProfile"].get("Roles", [])
            if not any(r["RoleName"] == role_name for r in roles):
                self._iam.add_role_to_instance_profile(
                    InstanceProfileName=profile_name,
                    RoleName=role_name,
                )
        except self._iam.exceptions.NoSuchEntityException:
            create_profile_resp = self._iam.create_instance_profile(
                InstanceProfileName=profile_name,
                Tags=[{"Key": "skyward:managed", "Value": "true"}],
            )
            profile_arn = create_profile_resp["InstanceProfile"]["Arn"]

            self._iam.add_role_to_instance_profile(
                InstanceProfileName=profile_name,
                RoleName=role_name,
            )

            waiter = self._iam.get_waiter("instance_profile_exists")
            waiter.wait(
                InstanceProfileName=profile_name,
                WaiterConfig={"Delay": 1, "MaxAttempts": 10},
            )

        return role_arn, profile_arn

    def _ensure_self_ref_rule(self, sg_id: str) -> None:
        sg_resp = self._ec2.describe_security_groups(GroupIds=[sg_id])
        rules = sg_resp["SecurityGroups"][0].get("IpPermissions", [])

        has_self_ref = False
        for rule in rules:
            if rule.get("IpProtocol") == "-1":
                for pair in rule.get("UserIdGroupPairs", []):
                    if pair.get("GroupId") == sg_id:
                        has_self_ref = True

        if not has_self_ref:
            self._ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "-1",
                        "UserIdGroupPairs": [
                            {
                                "GroupId": sg_id,
                                "Description": "All traffic from same security group (DDP/NCCL)",
                            }
                        ],
                    }
                ],
            )

    def _ensure_security_group(self) -> str:
        sg_name = self._security_group_name()

        try:
            describe_sg_resp = self._ec2.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": [sg_name]}]
            )
            if describe_sg_resp["SecurityGroups"]:
                sg_id = describe_sg_resp["SecurityGroups"][0]["GroupId"]
                self._ensure_self_ref_rule(sg_id)
                return sg_id
        except Exception:
            pass

        vpcs = self._ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
        if not vpcs["Vpcs"]:
            raise RuntimeError("No default VPC found. Please create one or specify a VPC.")
        vpc_id = vpcs["Vpcs"][0]["VpcId"]

        create_sg_resp = self._ec2.create_security_group(
            GroupName=sg_name,
            Description="Skyward EC2 worker security group",
            VpcId=vpc_id,
            TagSpecifications=[
                {
                    "ResourceType": "security-group",
                    "Tags": [
                        {"Key": "Name", "Value": sg_name},
                        {"Key": "skyward:managed", "Value": "true"},
                    ],
                }
            ],
        )

        sg_id = create_sg_resp["GroupId"]

        self._ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    "IpProtocol": "-1",
                    "UserIdGroupPairs": [
                        {
                            "GroupId": sg_id,
                            "Description": "All traffic from same security group (DDP/NCCL)",
                        }
                    ],
                }
            ],
        )

        return sg_id

    def cleanup(self) -> None:
        cleanups = [
            ("instances", self._cleanup_instances),
            ("eice", self._cleanup_eice),
            ("security_group", self._cleanup_security_group),
            ("iam", self._cleanup_iam),
            ("bucket", self._cleanup_bucket),
        ]

        for _, cleanup_fn in cleanups:
            with contextlib.suppress(Exception):
                cleanup_fn()

    def _cleanup_eice(self) -> None:
        with contextlib.suppress(Exception):
            response = self._ec2.describe_instance_connect_endpoints(
                Filters=[{"Name": "tag:skyward:managed", "Values": ["true"]}]
            )

            for endpoint in response.get("InstanceConnectEndpoints", []):
                endpoint_id = endpoint["InstanceConnectEndpointId"]
                state = endpoint["State"]

                if state in ("create-complete", "create-failed"):
                    with contextlib.suppress(Exception):
                        self._ec2.delete_instance_connect_endpoint(
                            InstanceConnectEndpointId=endpoint_id
                        )

    def _cleanup_instances(self) -> None:
        try:
            response = self._ec2.describe_instances(
                Filters=[
                    {"Name": "tag:skyward:managed", "Values": ["true"]},
                    {"Name": "instance-state-name", "Values": ["running", "pending", "stopped"]},
                ]
            )
            instance_ids = []
            for reservation in response.get("Reservations", []):
                for instance in reservation.get("Instances", []):
                    instance_ids.append(instance["InstanceId"])

            if instance_ids:
                self._ec2.terminate_instances(InstanceIds=instance_ids)
                waiter = self._ec2.get_waiter("instance_terminated")
                waiter.wait(InstanceIds=instance_ids)
        except Exception:
            pass

    def _cleanup_security_group(self) -> None:
        sg_name = self._security_group_name()
        try:
            response = self._ec2.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": [sg_name]}]
            )
            for sg in response.get("SecurityGroups", []):
                self._ec2.delete_security_group(GroupId=sg["GroupId"])
        except Exception:
            pass

    def _cleanup_iam(self) -> None:
        role_name = self._role_name()
        profile_name = self._instance_profile_name()
        policy_name = self._policy_name()

        with contextlib.suppress(Exception):
            self._iam.remove_role_from_instance_profile(
                InstanceProfileName=profile_name,
                RoleName=role_name,
            )

        with contextlib.suppress(Exception):
            self._iam.delete_instance_profile(InstanceProfileName=profile_name)

        with contextlib.suppress(Exception):
            self._iam.delete_role_policy(
                RoleName=role_name,
                PolicyName=policy_name,
            )

        with contextlib.suppress(Exception):
            self._iam.delete_role(RoleName=role_name)

    def _cleanup_bucket(self) -> None:
        bucket_name = self._bucket_name()

        try:
            paginator = self._s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket_name):
                objects = page.get("Contents", [])
                if objects:
                    self._s3.delete_objects(
                        Bucket=bucket_name,
                        Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                    )

            self._s3.delete_bucket(Bucket=bucket_name)
        except Exception:
            pass


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
# SSM Session (Command Execution)
# =============================================================================


class _SSMPendingError(Exception):
    """Command still pending - retry."""


class _SSMNotReadyError(Exception):
    """SSM agent not ready - retry."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Immutable result of SSM command execution."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def raise_on_failure(self, context: str = "") -> None:
        if not self.success:
            msg = f"{context}: {self.stderr}" if context else self.stderr
            raise RuntimeError(msg)


class SSMSession:
    """SSM session for command execution on EC2 instances."""

    def __init__(self, region: str) -> None:
        self.region = region

    @cached_property
    def _ssm(self) -> SSMClient:
        import boto3
        return boto3.client("ssm", region_name=self.region)

    def run_command(
        self,
        instance_id: str,
        command: str,
        timeout: int = 300,
    ) -> CommandResult:
        response = self._ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [command]},
            TimeoutSeconds=min(timeout, 3600),
        )

        command_id = response["Command"]["CommandId"]

        @retry(
            stop=stop_after_delay(timeout + 30),
            wait=wait_fixed(2),
            retry=retry_if_exception_type(_SSMPendingError),
            reraise=True,
        )
        def _poll() -> CommandResult:
            try:
                result = self._ssm.get_command_invocation(
                    CommandId=command_id,
                    InstanceId=instance_id,
                )
            except self._ssm.exceptions.InvocationDoesNotExist:
                raise _SSMPendingError()

            status = result["Status"]

            if status == "Success":
                return CommandResult(
                    exit_code=0,
                    stdout=result.get("StandardOutputContent", ""),
                    stderr=result.get("StandardErrorContent", ""),
                )
            elif status in ("Failed", "TimedOut", "Cancelled"):
                stderr = result.get("StandardErrorContent", "") or f"Command {status}"
                return CommandResult(
                    exit_code=1,
                    stdout=result.get("StandardOutputContent", ""),
                    stderr=stderr,
                )
            else:
                raise _SSMPendingError()

        try:
            return _poll()
        except RetryError as e:
            raise TimeoutError(f"Command timed out after {timeout}s") from e

    def wait_for_ssm_agent(
        self,
        instance_id: str,
        timeout: int = 600,
    ) -> None:
        @retry(
            stop=stop_after_delay(timeout),
            wait=wait_fixed(5),
            retry=retry_if_exception_type(_SSMNotReadyError),
            reraise=True,
        )
        def _check() -> None:
            try:
                result = self.run_command(instance_id, "echo ok", timeout=30)
                if result.success:
                    return
            except Exception:
                pass
            raise _SSMNotReadyError()

        try:
            _check()
        except RetryError as e:
            raise TimeoutError(
                f"SSM agent not available on {instance_id} after {timeout}s"
            ) from e

    def cleanup(self) -> None:
        pass


# =============================================================================
# S3 Object Store
# =============================================================================


class ObjectStore(Protocol):
    """Protocol for object store."""

    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...


class S3ObjectStore:
    """Object store using S3 for data persistence."""

    def __init__(self, bucket: str, prefix: str = "objects/") -> None:
        self.bucket = bucket
        self.prefix = prefix

    @cached_property
    def _s3(self) -> S3Client:
        import boto3
        return boto3.client("s3")

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def put(self, key: str, data: bytes) -> None:
        self._s3.put_object(
            Bucket=self.bucket,
            Key=self._full_key(key),
            Body=data,
        )

    def get(self, key: str) -> bytes:
        try:
            response = self._s3.get_object(
                Bucket=self.bucket,
                Key=self._full_key(key),
            )
            return response["Body"].read()
        except self._s3.exceptions.NoSuchKey as e:
            raise KeyError(f"Object not found: {key}") from e

    def delete(self, key: str) -> None:
        self._s3.delete_object(
            Bucket=self.bucket,
            Key=self._full_key(key),
        )

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(
                Bucket=self.bucket,
                Key=self._full_key(key),
            )
            return True
        except Exception:
            return False

# =============================================================================
# Volume Script Generation
# =============================================================================


def _generate_volume_script(
    volumes: tuple[Volume, ...],
    username: str = "ec2-user",
) -> str:
    """Generate shell script to install and mount volumes."""
    if not volumes:
        return ""

    if username == "root":
        home_dir = "/root"
    else:
        home_dir = f"/home/{username}"

    lines = ["# Install and mount volumes"]

    install_cmds: set[str] = set()
    for vol in volumes:
        install_cmds.update(vol.install_commands())

    for cmd in install_cmds:
        lines.append(cmd)

    lines.append(f'MOUNT_UID=$(id -u {username})')
    lines.append(f'MOUNT_GID=$(id -g {username})')

    for vol in volumes:
        for cmd in vol.mount_commands():
            expanded_cmd = cmd.replace("~/", f"{home_dir}/")
            if expanded_cmd == "~":
                expanded_cmd = home_dir
            elif expanded_cmd.startswith("~ "):
                expanded_cmd = home_dir + expanded_cmd[1:]
            expanded_cmd = expanded_cmd.replace("UID_PLACEHOLDER", "$MOUNT_UID")
            expanded_cmd = expanded_cmd.replace("GID_PLACEHOLDER", "$MOUNT_GID")
            lines.append(expanded_cmd)

    return "\n".join(lines)


# =============================================================================
# AWS Bootstrap (SSM-based)
# =============================================================================


def _is_instance_terminated_error(exc: BaseException) -> bool:
    """Check if exception indicates the instance is terminated."""
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in ("InvalidInstanceId", "InvalidInstanceId.NotFound")
    msg = str(exc).lower()
    return "invalidinstanceid" in msg or "not in a valid state" in msg


def _create_ssm_runner(
    ssm_session: SSMSession,
    instance_id: str,
) -> Callable[[str], str]:
    """Create a command runner using SSM."""

    def run_command(cmd: str) -> str:
        try:
            result = ssm_session.run_command(instance_id, cmd, timeout=30)
            if result.success:
                return result.stdout
            return ""
        except Exception as e:
            if _is_instance_terminated_error(e):
                raise InstanceTerminatedError(
                    instance_id, "instance not in valid state"
                ) from e
            raise

    return run_command


def wait_for_bootstrap(
    ssm_session: SSMSession,
    instance_id: str,
    verified_instances: set[str] | None = None,
    timeout: int = 600,
) -> None:
    """Wait for instance bootstrap with progress tracking."""
    if verified_instances is None:
        verified_instances = set()

    runner = _create_ssm_runner(ssm_session, instance_id)

    try:
        _wait_for_bootstrap_common(
            run_command=runner,
            instance_id=instance_id,
            timeout=timeout,
            extra_checkpoints=AWS_EXTRA_CHECKPOINTS,
        )
        verified_instances.add(instance_id)
    except RuntimeError:
        error_file = ""
        bootstrap_log = ""
        cloud_init_logs = ""

        try:
            result = ssm_session.run_command(
                instance_id,
                "cat /opt/skyward/.error 2>/dev/null || echo ''",
                timeout=30,
            )
            error_file = result.stdout.strip() if result.success else ""

            result = ssm_session.run_command(
                instance_id,
                "tail -100 /opt/skyward/bootstrap.log 2>/dev/null || echo ''",
                timeout=30,
            )
            bootstrap_log = result.stdout.strip() if result.success else ""

            result = ssm_session.run_command(
                instance_id,
                "cat /var/log/cloud-init-output.log 2>/dev/null || echo ''",
                timeout=30,
            )
            cloud_init_logs = result.stdout.strip() if result.success else ""
        except Exception:
            pass

        msg_parts = [f"Bootstrap failed on {instance_id}."]

        if error_file:
            msg_parts.append(f"\n--- Error file ---\n{error_file}")

        if bootstrap_log:
            msg_parts.append(f"\n--- Bootstrap log (last 100 lines) ---\n{bootstrap_log}")

        if cloud_init_logs and not error_file and not bootstrap_log:
            msg_parts.append(f"\n--- Cloud-init logs ---\n{cloud_init_logs}")

        raise RuntimeError("\n".join(msg_parts)) from None


# =============================================================================
# AWS Provider
# =============================================================================

# Cache for AWS resources per region
_resources_cache: dict[str, AWSResources] = {}
_resources_lock = threading.Lock()


# =============================================================================
# Vantage Pricing Integration
# =============================================================================


@cached(namespace="aws.vantage_pricing", ttl=DISCOVERY_CACHE_TTL)
def _fetch_vantage_pricing() -> dict[str, dict[str, Any]]:
    """Fetch AWS pricing from Vantage, indexed by instance_type."""
    import httpx

    data = httpx.get(VANTAGE_AWS_URL, timeout=30).json()
    return {inst.get("instance_type", "").lower(): inst for inst in data}


def _extract_aws_pricing(
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


@dataclass(frozen=True, slots=True)
class AWS(Provider):
    """Provider for AWS EC2 with SSM connectivity.

    Executes functions on EC2 instances using UV for fast dependency installation.
    Supports NVIDIA GPUs, Trainium, distributed clusters, and spot instances.

    Uses SSM (Systems Manager) for all connectivity - no SSH required.
    Infrastructure (S3 bucket, IAM roles, security groups) is auto-created.

    Example:
        from skyward import ComputePool, AWS, compute

        @compute
        def train(data):
            return model.fit(data)

        pool = ComputePool(
            provider=AWS(region="us-east-1"),
            accelerator="A100",
            pip=["torch"],
        )

        with pool:
            result = train(data) >> pool

    Environment Variables:
        AWS_REGION: Default region (can be overridden by region parameter)
        AWS_ACCESS_KEY_ID: AWS credentials
        AWS_SECRET_ACCESS_KEY: AWS credentials
    """

    region: str = "us-east-1"
    ami: str | None = None
    subnet_id: str | None = None
    security_group_id: str | None = None
    instance_profile_arn: str | None = None
    username: str | None = None
    instance_timeout: int = 300
    allocation_strategy: AllocationStrategy = "price-capacity-optimized"

    # Mutable state (not part of equality/hash)
    _resolved_region: str | None = field(
        default=None, compare=False, repr=False, hash=False
    )
    _resources: AWSResources | None = field(
        default=None, compare=False, repr=False, hash=False
    )
    _store: S3ObjectStore | None = field(
        default=None, compare=False, repr=False, hash=False
    )
    _ssm_session: SSMSession | None = field(
        default=None, compare=False, repr=False, hash=False
    )
    _verified_instances: set[str] = field(
        default_factory=set, compare=False, repr=False, hash=False
    )

    @property
    @override
    def name(self) -> str:
        return "aws"

    @property
    def _active_region(self) -> str:
        """Return resolved region (from auto-discovery) or configured region."""
        return self._resolved_region or self.region

    # =========================================================================
    # Resource Management
    # =========================================================================

    def _get_resources(self) -> AWSResources:
        """Get or create AWS resources."""
        if self._resources is not None:
            return self._resources

        region = self._active_region
        cache_key = region

        with _resources_lock:
            if cache_key in _resources_cache:
                resources = _resources_cache[cache_key]
                object.__setattr__(self, "_resources", resources)
                return resources

            infra = AWSInfraManager(region=region)
            resources = infra.ensure_infrastructure()
            _resources_cache[cache_key] = resources
            object.__setattr__(self, "_resources", resources)
            return resources

    @cached_property
    def _store_instance(self) -> S3ObjectStore:
        """Get S3 object store."""
        resources = self._get_resources()
        return S3ObjectStore(resources.bucket, prefix="skyward/")

    def _get_ssm_session(self) -> SSMSession:
        """Get or create SSM session."""
        if self._ssm_session is not None:
            return self._ssm_session

        resources = self._get_resources()
        session = SSMSession(region=resources.region)
        object.__setattr__(self, "_ssm_session", session)
        return session

    @property
    def _ec2(self) -> EC2Client:
        """EC2 client for active region."""
        import boto3
        return boto3.client("ec2", region_name=self._active_region)

    # =========================================================================
    # Accelerator Name Normalization
    # =========================================================================

    @staticmethod
    def _normalize_gpu_name(aws_name: str, memory_mib: int) -> str:
        """Normalize AWS GPU name to Skyward internal name."""
        memory_gb = memory_mib / 1024

        if aws_name == "A100":
            return "A100-40" if memory_gb < 50 else "A100-80"

        return AWS_GPU_NAME_MAP.get(aws_name, aws_name)

    @staticmethod
    def _normalize_neuron_name(aws_name: str, instance_type: str = "") -> str:
        """Normalize AWS Neuron device name to Skyward internal name."""
        if aws_name == "Trainium" and instance_type.startswith("trn2"):
            return "Trainium2"

        return AWS_NEURON_NAME_MAP.get(aws_name, aws_name)

    # =========================================================================
    # AMI Resolution
    # =========================================================================

    def _resolve_ami_id(
        self,
        compute: ComputeSpec,
        architecture: Architecture = "x86_64",
        acc: "Accelerator | None" = None,
    ) -> str:
        """Resolve AMI ID based on configuration and architecture."""
        if self.ami:
            return self.ami

        # Fractional GPU needs Ubuntu base + GRID driver (not DLAMI)
        if acc and acc.fractional:
            return self._get_ubuntu_base_ami(architecture=architecture)

        return self._get_public_ami(gpu=compute.accelerator is not None, architecture=architecture)

    @lru_cache(maxsize=8)
    def _get_ubuntu_base_ami(self, architecture: Architecture = "x86_64") -> str:
        """Get Ubuntu base AMI for fractional GPU (requires GRID driver)."""
        import boto3

        param_name = UBUNTU_BASE_SSM[architecture]
        region = self._active_region
        try:
            ssm = boto3.client("ssm", region_name=region)
            response = ssm.get_parameter(Name=param_name)
            return response["Parameter"]["Value"]
        except ClientError as e:
            raise RuntimeError(
                f"Could not find Ubuntu 22.04 AMI in region {region}. "
                f"SSM parameter {param_name} not found."
            ) from e

    @lru_cache(maxsize=32)
    def _get_public_ami(self, gpu: bool = False, architecture: Architecture = "x86_64") -> str:
        """Get the best public AMI for the given configuration."""
        import boto3

        if gpu:
            param_name = DLAMI_GPU_SSM[architecture]
            ami_type = f"Deep Learning AMI (Ubuntu 22.04, {architecture})"
        else:
            param_name = UBUNTU_BASE_SSM[architecture]
            ami_type = f"Ubuntu 22.04 ({architecture})"

        region = self._active_region
        try:
            ssm = boto3.client("ssm", region_name=region)
            response = ssm.get_parameter(Name=param_name)
            return response["Parameter"]["Value"]
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "ParameterNotFound":
                raise RuntimeError(
                    f"Could not find {ami_type} AMI in region {region}. "
                    f"SSM parameter {param_name} not found. "
                    "Please specify an AMI explicitly using AWS(ami='ami-xxx')"
                ) from e
            raise RuntimeError(f"Failed to resolve AMI in region {region}: {e}") from e

    def _resolve_username(self, ami_id: str) -> str:
        """Resolve system username for the given AMI."""
        if self.username:
            return self.username

        return self._get_ami_username(ami_id)

    @lru_cache(maxsize=32)
    def _get_ami_block_device_info(self, ami_id: str) -> tuple[str, int]:
        """Get root device name and minimum volume size from AMI.

        Returns:
            Tuple of (device_name, min_volume_size_gb).
        """
        try:
            response = self._ec2.describe_images(ImageIds=[ami_id])
            if response.get("Images"):
                image = response["Images"][0]
                root_device = image.get("RootDeviceName", "/dev/sda1")
                for bdm in image.get("BlockDeviceMappings", []):
                    if bdm.get("DeviceName") == root_device:
                        min_size = bdm.get("Ebs", {}).get("VolumeSize", 30)
                        return root_device, min_size
                return root_device, 30
        except ClientError:
            pass
        return "/dev/sda1", 30

    @lru_cache(maxsize=32)
    def _get_ami_username(self, ami_id: str) -> str:
        """Detect the default system username for an AMI."""
        try:
            response = self._ec2.describe_images(ImageIds=[ami_id])

            if not response.get("Images"):
                return "ec2-user"

            image = response["Images"][0]
            name = (image.get("Name") or "").lower()
            description = (image.get("Description") or "").lower()

            if "ubuntu" in name or "ubuntu" in description:
                return "ubuntu"

            if "debian" in name or "debian" in description:
                return "admin"

            return "ec2-user"

        except ClientError:
            return "ec2-user"

    # =========================================================================
    # Instance Type Discovery
    # =========================================================================

    def _parse_instance_info(self, info: dict[str, Any]) -> dict[str, Any]:
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

            accelerator = self._normalize_gpu_name(aws_name, gpu_memory_mib)
            accelerator_memory_gb = gpu_memory_mib / 1024

            # Fractional GPU: count=0 but has memory (e.g., G6f instances)
            if count == 0 and gpu_memory_mib > 0:
                from skyward.accelerator import Accelerator as Acc
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

            accelerator = self._normalize_neuron_name(aws_name, instance_type)
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

    @cached(
        namespace="aws.instance_types",
        ttl=DISCOVERY_CACHE_TTL,
        key_func=lambda self: f"all:{self.region}",
    )
    def _discover_all_instances(self) -> list[dict[str, Any]]:
        """Discover all instance types in this region."""
        return self._fetch_all_instances_from_api()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ClientError),
        reraise=True,
    )
    def _fetch_all_instances_from_api(self) -> list[dict[str, Any]]:
        """Fetch all instance types from EC2 API with retry."""
        specs: list[dict[str, Any]] = []
        paginator = self._ec2.get_paginator("describe_instance_types")

        for page in paginator.paginate():
            for instance_info in page.get("InstanceTypes", []):
                specs.append(self._parse_instance_info(instance_info))

        return sorted(
            specs, key=lambda s: (s.get("accelerator") or "", s.get("accelerator_count", 0), s.get("vcpu", 0))
        )

    # =========================================================================
    # Region Discovery
    # =========================================================================

    @cached(
        namespace="aws.regions",
        ttl=DISCOVERY_CACHE_TTL,
        key_func=lambda self: "enabled_regions",
    )
    def _get_enabled_regions(self) -> tuple[str, ...]:
        """Get all enabled regions for this AWS account."""
        import boto3

        ec2 = boto3.client("ec2", region_name=self.region)
        response = ec2.describe_regions(AllRegions=False)
        regions = [r["RegionName"] for r in response.get("Regions", [])]
        return tuple(sorted(regions))

    def _check_instance_type_in_region(self, instance_type: str, region: str) -> bool:
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

    def _find_available_region(
        self,
        instance_type: str,
        preferred_region: str | None = None,
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
        regions_to_check = list(self._get_enabled_regions())

        # Preferred region first
        if preferred_region and preferred_region in regions_to_check:
            regions_to_check.remove(preferred_region)
            regions_to_check.insert(0, preferred_region)

        checked_regions: list[str] = []
        for region in regions_to_check:
            checked_regions.append(region)
            if self._check_instance_type_in_region(instance_type, region):
                return region

        raise NoAvailableRegionError(instance_type, checked_regions)

    # =========================================================================
    # Instance Lifecycle
    # =========================================================================

    def _launch_fleet(
        self,
        n: int,
        instance_type: str,
        ami_id: str,
        user_data: str,
        requirements_hash: str,
        spot: bool = False,
        subnet_ids: tuple[str, ...] | None = None,
    ) -> list[str]:
        """Low-level Fleet launch with intelligent AZ selection.

        Uses EC2 Fleet with type='instant' and SingleAvailabilityZone=True to:
        1. Evaluate capacity across all availability zones
        2. Select the best AZ based on allocation_strategy
        3. Launch ALL instances in that single AZ (required for cluster locality)

        Args:
            subnet_ids: Specific subnets to use. If None, filters to AZs that support the instance type.
        """
        resources = self._get_resources()

        # Filter to subnets in AZs that support this instance type
        if subnet_ids:
            target_subnets = subnet_ids
        else:
            target_subnets = self._get_valid_subnets_for_instance_type(instance_type)

        # Create temporary launch template
        template_name = f"skyward-{uuid.uuid4().hex[:8]}"
        root_device, min_volume_size = self._get_ami_block_device_info(ami_id)
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
        }

        if self.instance_timeout:
            template_data["InstanceInitiatedShutdownBehavior"] = "terminate"

        launch_template = self._ec2.create_launch_template(
            LaunchTemplateName=template_name,
            LaunchTemplateData=template_data,
        )
        template_id = launch_template["LaunchTemplate"]["LaunchTemplateId"]

        try:
            # Overrides with target subnets - Fleet will choose the best AZ
            overrides = [{"SubnetId": sid} for sid in target_subnets]

            fleet_response = self._ec2.create_fleet(
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
                    "AllocationStrategy": self.allocation_strategy,
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
                error_msgs = [
                    f"{e.get('ErrorCode', 'Unknown')}: {e.get('ErrorMessage', '')}"
                    for e in errors
                ]
                raise RuntimeError(f"Fleet launch failed: {'; '.join(error_msgs)}")

            if len(instance_ids) < n:
                raise RuntimeError(
                    f"Fleet launched {len(instance_ids)}/{n} instances. "
                    f"Errors: {errors}"
                )

            return instance_ids

        finally:
            # Always clean up the temporary launch template
            with contextlib.suppress(ClientError):
                self._ec2.delete_launch_template(LaunchTemplateId=template_id)

    def _get_valid_subnets_for_instance_type(self, instance_type: str) -> tuple[str, ...]:
        """Filter subnets to only those in AZs that support the instance type."""
        resources = self._get_resources()

        # Get AZs that support this instance type
        response = self._ec2.describe_instance_type_offerings(
            LocationType="availability-zone",
            Filters=[{"Name": "instance-type", "Values": [instance_type]}],
        )
        valid_azs = {o["Location"] for o in response.get("InstanceTypeOfferings", [])}

        # Filter subnets to only those in valid AZs
        subnets = self._ec2.describe_subnets(SubnetIds=list(resources.subnet_ids))
        valid_subnets = [
            s["SubnetId"]
            for s in subnets["Subnets"]
            if s["AvailabilityZone"] in valid_azs
        ]

        if not valid_subnets:
            raise RuntimeError(
                f"No AZs in {self._active_region} support {instance_type}"
            )

        return tuple(valid_subnets)

    def _get_instance_az(self, instance_id: str) -> str:
        """Get the availability zone of an instance."""
        response = self._ec2.describe_instances(InstanceIds=[instance_id])
        return response["Reservations"][0]["Instances"][0]["Placement"]["AvailabilityZone"]

    def _find_subnet_for_az(self, az: str) -> str:
        """Find a subnet in the specified availability zone."""
        resources = self._get_resources()
        subnets = self._ec2.describe_subnets(SubnetIds=list(resources.subnet_ids))
        for subnet in subnets["Subnets"]:
            if subnet["AvailabilityZone"] == az:
                return subnet["SubnetId"]
        raise RuntimeError(f"No subnet found for AZ {az}")

    def _launch_instances(
        self,
        n: int,
        instance_type: str,
        ami_id: str,
        user_data: str,
        requirements_hash: str,
        spot_strategy: NormalizedSpot,
    ) -> list[str]:
        """Launch instances with full SpotStrategy support.

        Handles all spot strategies:
        - Spot.Always(): 100% spot, fail if unavailable
        - Spot.Never: 100% on-demand
        - Spot.IfAvailable(): try spot, fallback to on-demand
        - Spot.Percent(x): minimum x% spot, rest on-demand in same AZ
        """
        common_args = {
            "instance_type": instance_type,
            "ami_id": ami_id,
            "user_data": user_data,
            "requirements_hash": requirements_hash,
        }

        match spot_strategy:
            case _SpotNever():
                # 100% on-demand
                return self._launch_fleet(n=n, spot=False, **common_args)

            case Spot.Always():
                # 100% spot, fail if unavailable
                return self._launch_fleet(n=n, spot=True, **common_args)

            case Spot.IfAvailable():
                # Try spot first, fallback to on-demand
                try:
                    return self._launch_fleet(n=n, spot=True, **common_args)
                except (RuntimeError, ClientError):
                    return self._launch_fleet(n=n, spot=False, **common_args)

            case Spot.Percent(percentage=min_pct):
                # Minimum min_pct% spot, rest on-demand in same AZ
                min_spot = ceil(n * min_pct)

                # Try 100% spot first
                try:
                    return self._launch_fleet(n=n, spot=True, **common_args)
                except (RuntimeError, ClientError):
                    pass

                # Fallback: minimum spot + on-demand in same AZ
                spot_ids = self._launch_fleet(n=min_spot, spot=True, **common_args)
                spot_az = self._get_instance_az(spot_ids[0])
                subnet_in_az = self._find_subnet_for_az(spot_az)

                remaining = n - len(spot_ids)
                if remaining > 0:
                    ondemand_ids = self._launch_fleet(
                        n=remaining,
                        spot=False,
                        subnet_ids=(subnet_in_az,),
                        **common_args,
                    )
                    return spot_ids + ondemand_ids

                return spot_ids

            case _:
                raise ValueError(f"Unknown spot strategy: {spot_strategy}")

    def _wait_running(self, instance_ids: list[str]) -> None:
        """Wait for instances to be in running state."""
        if not instance_ids:
            return

        waiter = self._ec2.get_waiter("instance_running")
        waiter.wait(
            InstanceIds=instance_ids,
            WaiterConfig={
                "Delay": INSTANCE_RUNNING_WAIT_DELAY,
                "MaxAttempts": INSTANCE_RUNNING_MAX_ATTEMPTS,
            },
        )

    def _wait_ssm_parallel(self, instance_ids: list[str]) -> None:
        """Wait for SSM agent to be ready on all instances."""
        from skyward.conc import for_each_async

        if not instance_ids:
            return

        ssm = self._get_ssm_session()

        def wait_ssm(iid: str) -> None: ssm.wait_for_ssm_agent(iid, timeout=600)

        for_each_async(wait_ssm, instance_ids)

    def _get_instance_details(self, instance_ids: list[str]) -> list[dict[str, Any]]:
        """Get instance details (ID, private IP, spot status)."""
        if not instance_ids:
            return []

        response = self._ec2.describe_instances(InstanceIds=instance_ids)

        instances = []
        for r in response["Reservations"]:
            for i in r["Instances"]:
                instances.append({
                    "id": i["InstanceId"],
                    "private_ip": i.get("PrivateIpAddress", ""),
                    "spot": i.get("InstanceLifecycle") == "spot",
                    "instance_type": i.get("InstanceType", ""),
                })

        instances.sort(key=lambda x: x["id"])
        return instances

    # =========================================================================
    # Skyward Wheel Installation
    # =========================================================================

    def _install_skyward_wheel(self, instances: tuple[Instance, ...]) -> None:
        """Install skyward wheel on all instances via SSM SSH tunnel.

        Creates an SSM tunnel to port 22 and uses SSH/SCP to transfer
        and install the wheel, same approach as Verda/DigitalOcean.
        """
        import json

        from skyward.conc import for_each_async
        from skyward.constants import SKYWARD_DIR
        from skyward.providers.common import build_wheel, get_private_key_path, scp_upload, ssh_run

        # Build wheel locally
        wheel_path = build_wheel()
        key_path = get_private_key_path()
        ssm = self._get_ssm_session()

        def install_on_instance(inst: Instance) -> None:
            # Ensure SSM agent is ready for sessions (may have just started with cached AMI)
            ssm.wait_for_ssm_agent(inst.id, timeout=120)

            # Wait for SSH daemon to be ready (critical for cached AMIs where sshd may not be up yet)
            ssm.run_command(
                inst.id,
                "for i in $(seq 1 30); do nc -z 127.0.0.1 22 && break || sleep 1; done",
                timeout=60,
            )

            # Create SSM tunnel to SSH port (22)
            local_port = find_available_port()
            tunnel_cmd = [
                "aws", "ssm", "start-session",
                "--target", inst.id,
                "--document-name", "AWS-StartPortForwardingSession",
                "--parameters", json.dumps({
                    "portNumber": ["22"],
                    "localPortNumber": [str(local_port)],
                }),
                "--region", self._active_region,
            ]
            # Give more time for tunnel (Session Manager can be slower than RunCommand)
            local_port, tunnel_proc = create_tunnel(tunnel_cmd, local_port, timeout=60)

            try:
                username = inst.get_meta("username", "ubuntu")

                # Upload wheel to /tmp first (user has write access), then move to SKYWARD_DIR
                tmp_wheel = f"/tmp/{wheel_path.name}"
                remote_wheel = f"{SKYWARD_DIR}/{wheel_path.name}"
                scp_upload("localhost", username, wheel_path, tmp_wheel, port=local_port, key_path=key_path)

                # Move to SKYWARD_DIR with sudo
                ssh_run("localhost", username, f"sudo mv {tmp_wheel} {remote_wheel}", timeout=30, port=local_port, key_path=key_path)

                # Find uv path
                find_uv_result = ssh_run(
                    "localhost", username,
                    "which uv || find /root /home -name uv -type f 2>/dev/null | head -1",
                    timeout=30, port=local_port, key_path=key_path,
                )
                uv_path = find_uv_result.stdout.strip()
                if not uv_path:
                    uv_path = "/root/.local/bin/uv"

                # Install wheel with all dependencies
                install_result = ssh_run(
                    "localhost", username,
                    f"sudo bash -c 'cd {SKYWARD_DIR} && {uv_path} pip install {remote_wheel}'",
                    timeout=120, port=local_port, key_path=key_path,
                )
                if install_result.returncode != 0:
                    raise RuntimeError(f"Failed to install wheel: {install_result.stderr}")

                # Create and start systemd service (only after wheel is installed)
                # The service runs python -m skyward.rpc which requires the skyward package
                from skyward.bootstrap.worker import rpyc_service_unit
                from skyward.constants import RPYC_PORT

                unit_content = rpyc_service_unit()
                unit_path = "/etc/systemd/system/skyward-rpyc.service"

                # Write unit file, reload daemon, enable and start
                setup_service_cmd = f"""sudo bash -c '
cat > {unit_path} << EOF
{unit_content}
EOF
systemctl daemon-reload
systemctl enable skyward-rpyc
systemctl restart skyward-rpyc
'"""
                ssh_run("localhost", username, setup_service_cmd, timeout=60, port=local_port, key_path=key_path)

                # Wait for port to be ready
                wait_port_cmd = f"""
for i in $(seq 1 30); do
    if nc -z 127.0.0.1 {RPYC_PORT} 2>/dev/null; then
        echo "Port {RPYC_PORT} ready"
        exit 0
    fi
    sleep 1
done
echo "Timeout waiting for port {RPYC_PORT}"
exit 1
"""
                port_result = ssh_run("localhost", username, f"sudo bash -c '{wait_port_cmd}'", timeout=60, port=local_port, key_path=key_path)
                if port_result.returncode != 0:
                    raise RuntimeError(f"RPyC service failed to start: {port_result.stderr}")

            finally:
                tunnel_proc.terminate()

        for_each_async(install_on_instance, instances)

    # =========================================================================
    # Provider Protocol Implementation
    # =========================================================================

    @override
    def provision(self, compute: ComputeSpec) -> tuple[Instance, ...]:
        """Provision EC2 instances."""
        from skyward.accelerator import Accelerator
        from skyward.bootstrap import grid_driver, group, ssm_restart
        from skyward.spec import normalize_spot

        try:
            cluster_id = str(uuid.uuid4())[:8]

            emit(InfraCreating())

            # Select instance type first (before region discovery)
            acc = Accelerator.from_value(compute.accelerator)
            accelerator_type = acc.accelerator if acc else None
            requested_gpu_count = acc.count if acc else 1

            spec = select_instance(
                self.available_instances(),
                cpu=1,
                memory_mb=512,
                accelerator=accelerator_type,
                accelerator_count=requested_gpu_count,
            )
            instance_type = spec.name
            accelerator_count = spec.accelerator_count
            architecture = spec.metadata.get("architecture", "x86_64")

            # Find available region for this instance type
            actual_region = self._find_available_region(
                instance_type=instance_type,
                preferred_region=self.region,
            )

            # Set resolved region if different from configured
            if actual_region != self.region:
                object.__setattr__(self, "_resolved_region", actual_region)
                emit(RegionAutoSelected(
                    requested_region=self.region,
                    selected_region=actual_region,
                    instance_type=instance_type,
                    provider=ProviderName.AWS,
                ))

            # Now create resources in the actual region
            resources = self._get_resources()
            emit(InfraCreated(region=self._active_region))

            # Get image from compute spec
            image = compute.image
            content_hash = image.content_hash()

            ami_id = self._resolve_ami_id(compute, architecture=architecture, acc=acc)
            username = self._resolve_username(ami_id)

            # Get local SSH public key to inject into instance
            from skyward.bootstrap import inject_ssh_key
            from skyward.providers.common import find_local_ssh_key

            ssh_key_op = None
            key_info = find_local_ssh_key()
            if key_info:
                _, public_key = key_info
                ssh_key_op = inject_ssh_key(public_key)

            # Generate bootstrap script
            base_preamble = group(ssm_restart(), grid_driver()) if acc and acc.fractional else ssm_restart()
            preamble = group(base_preamble, ssh_key_op) if ssh_key_op else base_preamble

            # Build postamble (volume mounting)
            volume_script = _generate_volume_script(compute.volumes, username)
            postamble = volume_script if volume_script else None

            user_data = image.bootstrap(
                ttl=compute.timeout,
                preamble=preamble,
                postamble=postamble,
            )

            # Normalize spot strategy
            spot_strategy = normalize_spot(compute.spot)

            emit(InstanceLaunching(count=compute.nodes, instance_type=instance_type, provider=ProviderName.AWS))

            # Launch instances with full spot strategy support
            instance_ids = self._launch_instances(
                n=compute.nodes,
                instance_type=instance_type,
                ami_id=ami_id,
                user_data=user_data,
                requirements_hash=content_hash,
                spot_strategy=spot_strategy,
            )

            # Wait for instances to be running and SSM ready
            self._wait_running(instance_ids)
            self._wait_ssm_parallel(instance_ids)

            # Get instance details
            instance_details = self._get_instance_details(instance_ids)

            # Build Instance objects
            instances: list[Instance] = []
            for i, details in enumerate(instance_details):
                instance = Instance(
                    id=details["id"],
                    provider=self,
                    spot=details["spot"],
                    private_ip=details["private_ip"],
                    node=i,
                    metadata=frozenset([
                        ("cluster_id", cluster_id),
                        ("region", self._active_region),
                        ("instance_type", instance_type),
                        ("content_hash", content_hash),
                        ("accelerator_count", str(accelerator_count)),
                    ]),
                )
                instances.append(instance)

                emit(InstanceProvisioned(
                    instance_id=instance.id,
                    node=i,
                    spot=instance.spot,
                    instance_type=instance_type,
                    provider=ProviderName.AWS,
                    price_on_demand=spec.price_on_demand,
                    price_spot=spec.price_spot,
                    billing_increment_minutes=spec.billing_increment_minutes,
                ))

            spot_count = sum(1 for inst in instances if inst.spot)
            emit(ProvisioningCompleted(
                spot=spot_count,
                on_demand=len(instances) - spot_count,
                provider=ProviderName.AWS,
                region=self._active_region,
                instances=[inst.id for inst in instances],
            ))

            return tuple(instances)

        except Exception as e:
            emit(Error(message=f"Provision failed: {e}"))
            raise

    @override
    def setup(self, instances: tuple[Instance, ...], compute: ComputeSpec) -> None:
        """Setup instances (bootstrap, install dependencies)."""
        from skyward.conc import for_each_async

        try:
            ssm_session = self._get_ssm_session()

            def bootstrap_instance(inst: Instance) -> None:
                try:
                    emit(BootstrapStarting(instance_id=inst.id))

                    wait_for_bootstrap(
                        ssm_session=ssm_session,
                        instance_id=inst.id,
                        verified_instances=self._verified_instances,
                    )

                    emit(BootstrapCompleted(instance_id=inst.id))
                except Exception as e:
                    emit(Error(message=f"Bootstrap failed on {inst.id}: {e}", instance_id=inst.id))
                    raise

            for_each_async(bootstrap_instance, instances)

            # Install skyward wheel on all instances
            self._install_skyward_wheel(instances)

        except Exception as e:
            emit(Error(message=f"Setup failed: {e}"))
            raise

    @override
    def shutdown(self, instances: tuple[Instance, ...], compute: ComputeSpec) -> tuple[ExitedInstance, ...]:
        """Terminate instances."""
        if not instances:
            return ()

        instance_ids = [inst.id for inst in instances]

        # Terminate all instances
        self._ec2.terminate_instances(InstanceIds=instance_ids)

        # Build ExitedInstance objects
        exited: list[ExitedInstance] = []
        for inst in instances:
            emit(InstanceStopping(instance_id=inst.id))
            exited.append(ExitedInstance(instance=inst, exit_code=0, exit_reason="terminated"))

        # Cleanup SSM session
        if self._ssm_session is not None:
            self._ssm_session.cleanup()

        self._verified_instances.clear()

        return tuple(exited)

    @override
    def create_tunnel(self, instance: Instance, remote_port: int = 18861) -> tuple[int, Popen[bytes]]:
        """Create SSM tunnel to instance."""
        import json
        import subprocess

        try:
            result = subprocess.run(
                ["session-manager-plugin", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError("session-manager-plugin returned non-zero")
        except FileNotFoundError:
            raise RuntimeError(
                "session-manager-plugin not found. Please install it:\n"
                "  macOS: brew install session-manager-plugin\n"
                "  Linux: See https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html"
            ) from None

        local_port = find_available_port()
        cmd = [
            "aws", "ssm", "start-session",
            "--target", instance.id,
            "--document-name", "AWS-StartPortForwardingSession",
            "--parameters", json.dumps({
                "portNumber": [str(remote_port)],
                "localPortNumber": [str(local_port)],
            }),
            "--region", self._active_region,
        ]
        return create_tunnel(cmd, local_port)

    @override
    def run_command(self, instance: Instance, command: str, timeout: int = 30) -> str:
        """Run shell command on instance via SSM."""
        ssm = self._get_ssm_session()
        result = ssm.run_command(instance.id, command, timeout)
        if not result.success:
            raise RuntimeError(f"Command failed on {instance.id}: {result.stderr}")
        return result.stdout

    @override
    def discover_peers(self, cluster_id: str) -> tuple[Instance, ...]:
        """Discover peer instances in a cluster via EC2 API."""
        response = self._ec2.describe_instances(
            Filters=[
                {"Name": f"tag:{SkywardTag.MANAGED}", "Values": ["true"]},
                {"Name": f"tag:{SkywardTag.CLUSTER_ID}", "Values": [cluster_id]},
                {"Name": "instance-state-name", "Values": ["running"]},
            ]
        )

        instances = []
        for reservation in response.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                instances.append(
                    Instance(
                        id=inst["InstanceId"],
                        provider=self,
                        spot=inst.get("InstanceLifecycle") == "spot",
                        private_ip=inst.get("PrivateIpAddress", ""),
                        public_ip=inst.get("PublicIpAddress"),
                        node=0,
                        metadata=frozenset([
                            ("cluster_id", cluster_id),
                            ("instance_type", inst.get("InstanceType", "")),
                            ("region", self._active_region),
                        ]),
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

    @override
    def available_instances(self) -> tuple[InstanceSpec, ...]:
        """List all available instance types in this region with pricing."""
        instance_data = self._discover_all_instances()
        vantage = _fetch_vantage_pricing()
        region = self._active_region

        def make_spec(s: dict[str, Any]) -> InstanceSpec:
            od, sp = _extract_aws_pricing(s["type_name"], region, vantage)
            return InstanceSpec(
                name=s["type_name"],
                vcpu=s["vcpu"],
                memory_gb=s["memory_gb"],
                accelerator=s.get("accelerator"),
                accelerator_count=s.get("accelerator_count", 0),
                accelerator_memory_gb=s.get("accelerator_memory_gb", 0),
                price_on_demand=od,
                price_spot=sp,
                metadata={
                    "architecture": s.get("architecture", "x86_64"),
                    "supports_cluster_placement": s.get("supports_cluster_placement", False),
                },
            )

        return tuple(make_spec(s) for s in instance_data)
