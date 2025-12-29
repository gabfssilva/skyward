"""AWS infrastructure management for Skyward."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, cast

from skyward.cache import cached

if TYPE_CHECKING:
    from mypy_boto3_ec2 import EC2Client
    from mypy_boto3_iam import IAMClient
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_s3.literals import BucketLocationConstraintType
    from mypy_boto3_ssm import SSMClient
    from mypy_boto3_sts import STSClient

@dataclass
class AWSResources:
    """Container for AWS resource identifiers."""

    bucket: str
    iam_role_arn: str
    instance_profile_arn: str
    security_group_id: str
    region: str
    subnet_ids: tuple[str, ...]  # All subnets in default VPC for multi-AZ Fleet

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
        # Handle both old format (subnet_id: str) and new format (subnet_ids: list)
        subnet_ids_raw = data.get("subnet_ids")
        subnet_ids: tuple[str, ...]
        if subnet_ids_raw is None:
            # Legacy format: single subnet_id
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

    # IAM policy for EC2 instances
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
                "Resource": [
                    "arn:aws:s3:::*",
                    "arn:aws:s3:::*/*"
                ],
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

    # Trust policy for EC2 to assume the role
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
        """Initialize infrastructure manager.

        Args:
            region: AWS region.
            prefix: Prefix for all resource names.
        """
        self.region = region
        self.prefix = prefix

    @cached_property
    def _s3(self) -> S3Client:
        """Cliente S3 com inicialização lazy."""
        import boto3

        return boto3.client("s3", region_name=self.region)

    @cached_property
    def _ec2(self) -> EC2Client:
        """Cliente EC2 com inicialização lazy."""
        import boto3

        return boto3.client("ec2", region_name=self.region)

    @cached_property
    def _iam(self) -> IAMClient:
        """Cliente IAM com inicialização lazy."""
        import boto3

        return boto3.client("iam", region_name=self.region)

    @cached_property
    def _sts(self) -> STSClient:
        """Cliente STS com inicialização lazy."""
        import boto3

        return boto3.client("sts", region_name=self.region)

    @cached_property
    def _ssm(self) -> SSMClient:
        """Cliente SSM com inicialização lazy."""
        import boto3

        return boto3.client("ssm", region_name=self.region)

    @cached_property
    def _account_id(self) -> str:
        """AWS account ID."""
        return self._sts.get_caller_identity()["Account"]

    def _bucket_name(self) -> str:
        """Generate unique bucket name."""
        return f"{self.prefix}-{self._account_id}-{self.region}"

    def _role_name(self) -> str:
        """Generate IAM role name."""
        return f"{self.prefix}-role"

    def _instance_profile_name(self) -> str:
        """Generate instance profile name."""
        return f"{self.prefix}-instance-profile"

    def _security_group_name(self) -> str:
        """Generate security group name."""
        return f"{self.prefix}-sg"

    def _policy_name(self) -> str:
        """Generate IAM policy name."""
        return f"{self.prefix}-ec2-policy"

    def _get_default_subnets(self) -> tuple[str, ...]:
        """Get all subnets from the default VPC.

        Returns:
            Tuple of subnet IDs (one per availability zone).

        Raises:
            RuntimeError: If no default VPC or subnets found.
        """
        # Get default VPC
        vpcs = self._ec2.describe_vpcs(
            Filters=[{"Name": "is-default", "Values": ["true"]}]
        )
        if not vpcs["Vpcs"]:
            raise RuntimeError("No default VPC found. Please create one or specify a VPC.")
        vpc_id = vpcs["Vpcs"][0]["VpcId"]

        # Get subnets in default VPC
        subnets = self._ec2.describe_subnets(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
        )
        if not subnets["Subnets"]:
            raise RuntimeError("No subnets found in default VPC.")

        # Return all subnet IDs for multi-AZ Fleet support
        return tuple(s["SubnetId"] for s in subnets["Subnets"])

    def ensure_infrastructure(self) -> AWSResources:
        """Create all required AWS resources if they don't exist.

        Returns:
            AWSResources with all resource identifiers.
        """
        data = self._ensure_infrastructure_cached()
        resources = AWSResources.from_dict(data)

        # Always refresh IAM policy to ensure it's up to date
        # (cache doesn't track policy document changes)
        self._ensure_iam_role(resources.bucket)

        return resources

    @cached(
        namespace="aws.infrastructure",
        key_func=lambda self: f"{self.region}:{self.prefix}",
    )
    def _ensure_infrastructure_cached(self) -> dict[str, str | list[str]]:
        """Create all resources and return as dict for caching."""
        return self._create_infrastructure().to_dict()

    def _create_infrastructure(self) -> AWSResources:
        """Actually create all AWS resources."""
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
        """Ensure S3 bucket exists.

        Returns:
            Bucket name.
        """
        bucket_name = self._bucket_name()

        try:
            self._s3.head_bucket(Bucket=bucket_name)
        except Exception:
            # Bucket doesn't exist, create it
            # LocationConstraint is required for non-us-east-1 regions
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
        """Ensure IAM role and instance profile exist.

        Args:
            bucket: S3 bucket name for policy.

        Returns:
            Tuple of (role_arn, instance_profile_arn).
        """
        role_name = self._role_name()
        profile_name = self._instance_profile_name()
        policy_name = self._policy_name()

        # Create or get role
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

        # Create or update inline policy
        policy_doc = json.loads(json.dumps(self.EC2_POLICY_DOCUMENT))
        policy_doc["Statement"][0]["Resource"] = f"arn:aws:s3:::{bucket}/*"
        policy_doc["Statement"][1]["Resource"] = f"arn:aws:s3:::{bucket}"

        self._iam.put_role_policy(
            RoleName=role_name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_doc),
        )

        # Attach AWS managed policy for SSM (more reliable than inline)
        try:
            self._iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
            )
        except self._iam.exceptions.EntityAlreadyExistsException:
            pass  # Already attached

        # Create or get instance profile
        try:
            get_profile_resp = self._iam.get_instance_profile(
                InstanceProfileName=profile_name
            )
            profile_arn = get_profile_resp["InstanceProfile"]["Arn"]

            # Check if role is attached
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

            # Wait for new profile to propagate (only when newly created)
            waiter = self._iam.get_waiter("instance_profile_exists")
            waiter.wait(
                InstanceProfileName=profile_name,
                WaiterConfig={"Delay": 1, "MaxAttempts": 10},
            )

        return role_arn, profile_arn

    def _ensure_self_ref_rule(self, sg_id: str) -> None:
        """Add self-referencing rule for DDP/NCCL if not present.

        Args:
            sg_id: Security group ID.
        """
        # Check existing rules
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
        """Ensure security group exists.

        With SSM, we no longer need SSH port 22 open.
        Only need self-referencing rule for inter-instance communication (DDP/NCCL).

        Returns:
            Security group ID.
        """
        sg_name = self._security_group_name()

        # Check if security group exists
        try:
            describe_sg_resp = self._ec2.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": [sg_name]}]
            )
            if describe_sg_resp["SecurityGroups"]:
                sg_id = describe_sg_resp["SecurityGroups"][0]["GroupId"]
                # Ensure self-referencing rule exists for DDP/NCCL
                self._ensure_self_ref_rule(sg_id)
                return sg_id
        except Exception:
            pass

        # Get default VPC
        vpcs = self._ec2.describe_vpcs(
            Filters=[{"Name": "is-default", "Values": ["true"]}]
        )
        if not vpcs["Vpcs"]:
            raise RuntimeError("No default VPC found. Please create one or specify a VPC.")
        vpc_id = vpcs["Vpcs"][0]["VpcId"]

        # Create security group
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

        # Add self-referencing rule for inter-instance communication (DDP/NCCL)
        # No SSH rules needed with SSM
        self._ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    "IpProtocol": "-1",  # All traffic
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
        """Destroy all AWS resources created by Skyward.

        Executa cleanup de todos os recursos em ordem, coletando erros
        sem interromper o processo.
        """
        # Ordem é importante: dependências primeiro
        cleanups = [
            ("instances", self._cleanup_instances),
            ("eice", self._cleanup_eice),  # Keep for legacy cleanup
            ("security_group", self._cleanup_security_group),
            ("iam", self._cleanup_iam),
            ("bucket", self._cleanup_bucket),
            ("ami", self._cleanup_ami),
        ]

        for _, cleanup_fn in cleanups:
            with contextlib.suppress(Exception):
                cleanup_fn()

    def _cleanup_eice(self) -> None:
        """Delete EC2 Instance Connect Endpoints."""
        with contextlib.suppress(Exception):
            response = self._ec2.describe_instance_connect_endpoints(
                Filters=[{"Name": "tag:skyward:managed", "Values": ["true"]}]
            )

            deleted = 0
            for endpoint in response.get("InstanceConnectEndpoints", []):
                endpoint_id = endpoint["InstanceConnectEndpointId"]
                state = endpoint["State"]

                if state in ("create-complete", "create-failed"):
                    with contextlib.suppress(Exception):
                        self._ec2.delete_instance_connect_endpoint(
                            InstanceConnectEndpointId=endpoint_id
                        )

    def _cleanup_ami(self) -> None:
        """Delete skyward-managed AMIs (legacy cleanup).

        With Instance Pool approach, we no longer create custom AMIs.
        This is kept for cleaning up any legacy AMIs.
        """
        with contextlib.suppress(Exception):
            # Find any skyward-managed AMIs
            response = self._ec2.describe_images(
                Owners=["self"],
                Filters=[{"Name": "tag:skyward:managed", "Values": ["true"]}],
            )

            for image in response.get("Images", []):
                ami_id = image["ImageId"]

                # Get snapshot IDs before deregistering
                snapshot_ids = [
                    m["Ebs"]["SnapshotId"]
                    for m in image.get("BlockDeviceMappings", [])
                    if "Ebs" in m and "SnapshotId" in m["Ebs"]
                ]

                with contextlib.suppress(Exception):
                    self._ec2.deregister_image(ImageId=ami_id)

                # Delete associated snapshots
                for snapshot_id in snapshot_ids:
                    with contextlib.suppress(Exception):
                        self._ec2.delete_snapshot(SnapshotId=snapshot_id)

    def _cleanup_instances(self) -> None:
        """Terminate all Skyward-managed instances (including stopped ones)."""
        try:
            # Include stopped instances (from Instance Pool)
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
                # Wait for termination
                waiter = self._ec2.get_waiter("instance_terminated")
                waiter.wait(InstanceIds=instance_ids)
        except Exception:
            pass

    def _cleanup_security_group(self) -> None:
        """Delete the Skyward security group."""
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
        """Delete IAM role and instance profile."""
        role_name = self._role_name()
        profile_name = self._instance_profile_name()
        policy_name = self._policy_name()

        # Remove role from instance profile
        with contextlib.suppress(Exception):
            self._iam.remove_role_from_instance_profile(
                InstanceProfileName=profile_name,
                RoleName=role_name,
            )

        # Delete instance profile
        with contextlib.suppress(Exception):
            self._iam.delete_instance_profile(InstanceProfileName=profile_name)

        # Delete inline policy
        with contextlib.suppress(Exception):
            self._iam.delete_role_policy(
                RoleName=role_name,
                PolicyName=policy_name,
            )

        # Delete role
        with contextlib.suppress(Exception):
            self._iam.delete_role(RoleName=role_name)

    def _cleanup_bucket(self) -> None:
        """Empty and delete S3 bucket."""
        bucket_name = self._bucket_name()

        try:
            # Delete all objects
            paginator = self._s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket_name):
                objects = page.get("Contents", [])
                if objects:
                    self._s3.delete_objects(
                        Bucket=bucket_name,
                        Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                    )

            # Delete bucket
            self._s3.delete_bucket(Bucket=bucket_name)
        except Exception:
            pass
