"""AWS infrastructure management (IAM, S3, security groups)."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Protocol, cast

from skyward.utils.cache import cached

if TYPE_CHECKING:
    from mypy_boto3_ec2 import EC2Client
    from mypy_boto3_iam import IAMClient
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_s3.literals import BucketLocationConstraintType
    from mypy_boto3_sts import STSClient


# =============================================================================
# AWS Resources
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
# AWS Infrastructure Manager
# =============================================================================


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

    def _ensure_security_group_rules(self, sg_id: str) -> None:
        """Ensure security group has required rules (self-ref for NCCL, SSH for access)."""
        sg_resp = self._ec2.describe_security_groups(GroupIds=[sg_id])
        rules = sg_resp["SecurityGroups"][0].get("IpPermissions", [])

        has_self_ref = False
        has_ssh = False

        for rule in rules:
            if rule.get("IpProtocol") == "-1":
                for pair in rule.get("UserIdGroupPairs", []):
                    if pair.get("GroupId") == sg_id:
                        has_self_ref = True
            if rule.get("IpProtocol") == "tcp" and rule.get("FromPort") == 22:
                has_ssh = True

        permissions_to_add = []

        if not has_self_ref:
            permissions_to_add.append({
                "IpProtocol": "-1",
                "UserIdGroupPairs": [
                    {
                        "GroupId": sg_id,
                        "Description": "All traffic from same security group (DDP/NCCL)",
                    }
                ],
            })

        if not has_ssh:
            permissions_to_add.append({
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH access"}],
            })

        if permissions_to_add:
            self._ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=permissions_to_add,
            )

    def _ensure_security_group(self) -> str:
        sg_name = self._security_group_name()

        try:
            describe_sg_resp = self._ec2.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": [sg_name]}]
            )
            if describe_sg_resp["SecurityGroups"]:
                sg_id = describe_sg_resp["SecurityGroups"][0]["GroupId"]
                self._ensure_security_group_rules(sg_id)
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
                },
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH access"}],
                },
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
                        self._ec2.delete_instance_connect_endpoint(InstanceConnectEndpointId=endpoint_id)

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
            response = self._ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [sg_name]}])
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
