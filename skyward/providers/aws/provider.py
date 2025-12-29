"""AWS Provider using EC2 with SSM connectivity."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, override

from skyward.spec import AllocationStrategy
from skyward.types import ComputeSpec, ExitedInstance, Instance, Provider

if TYPE_CHECKING:
    import subprocess

    from skyward.providers.aws.infra import AWSResources
    from skyward.providers.aws.pool import InstancePool
    from skyward.providers.aws.s3 import S3ObjectStore
    from skyward.providers.aws.ssm import SSMSession

# Cache for AWS resources per region
_resources_cache: dict[str, AWSResources] = {}
_resources_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class AWS(Provider):
    """Provider for AWS EC2.

    Executes functions on EC2 instances using UV for fast dependency installation.
    Supports NVIDIA GPUs, Trainium, distributed clusters, and spot instances.

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
    """

    region: str = "us-east-1"
    ami: str | None = None
    subnet_id: str | None = None
    security_group_id: str | None = None
    instance_profile_arn: str | None = None
    username: str | None = None
    instance_timeout: int = 300  # Auto-terminate instance after N seconds. Defaults to 5 min
    allocation_strategy: AllocationStrategy = "price-capacity-optimized"

    # Mutable state (not part of equality/hash)
    _resources: AWSResources | None = field(
        default=None, compare=False, repr=False, hash=False
    )
    _store: S3ObjectStore | None = field(
        default=None, compare=False, repr=False, hash=False
    )
    _ssm_session: SSMSession | None = field(
        default=None, compare=False, repr=False, hash=False
    )
    _pool: InstancePool | None = field(
        default=None, compare=False, repr=False, hash=False
    )
    _verified_instances: set[str] = field(
        default_factory=set, compare=False, repr=False, hash=False
    )

    @property
    def name(self) -> str:
        """Provider name."""
        return "aws"

    def _get_resources(self) -> AWSResources:
        """Get or create AWS resources."""
        if self._resources is not None:
            return self._resources

        cache_key = self.region

        with _resources_lock:
            if cache_key in _resources_cache:
                resources = _resources_cache[cache_key]
                object.__setattr__(self, "_resources", resources)
                return resources

            from skyward.providers.aws.infra import AWSInfraManager

            infra = AWSInfraManager(region=self.region)
            resources = infra.ensure_infrastructure()
            _resources_cache[cache_key] = resources
            object.__setattr__(self, "_resources", resources)
            return resources

    @cached_property
    def _store_instance(self) -> S3ObjectStore:
        """Get S3 object store."""
        from skyward.providers.aws.s3 import S3ObjectStore

        resources = self._get_resources()
        return S3ObjectStore(resources.bucket, prefix="skyward/")

    def _get_ssm_session(self) -> SSMSession:
        """Get or create SSM session."""
        if self._ssm_session is not None:
            return self._ssm_session

        from skyward.providers.aws.ssm import SSMSession

        resources = self._get_resources()
        session = SSMSession(region=resources.region)
        object.__setattr__(self, "_ssm_session", session)
        return session

    def _get_pool(self) -> InstancePool:
        """Get or create instance pool."""
        if self._pool is not None:
            return self._pool

        from skyward.providers.aws.pool import InstancePool

        resources = self._get_resources()
        pool = InstancePool(resources, self._get_ssm_session())
        object.__setattr__(self, "_pool", pool)
        return pool

    def _resolve_ami(self, compute: ComputeSpec) -> str:
        """Resolve AMI ID based on configuration."""
        if self.ami:
            return self.ami

        from skyward.providers.aws.ami import resolve_ami

        needs_accelerator = compute.accelerator is not None
        return resolve_ami(self.region, gpu=needs_accelerator)

    def _resolve_username(self, ami_id: str) -> str:
        """Resolve system username for the given AMI."""
        if self.username:
            return self.username

        from skyward.providers.aws.ami import get_ami_username

        return get_ami_username(self.region, ami_id)

    @override
    def create_tunnel(self, instance: Instance) -> tuple[int, subprocess.Popen[bytes]]:
        """Create SSM tunnel to instance."""
        import json
        import subprocess

        from skyward.providers.common import RPYC_PORT, create_tunnel, find_available_port

        # Check SSM plugin is installed
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
                "portNumber": [str(RPYC_PORT)],
                "localPortNumber": [str(local_port)],
            }),
            "--region", self.region,
        ]
        return create_tunnel(cmd, local_port)

    @override
    def provision(
        self,
        compute: ComputeSpec,
    ) -> tuple[Instance, ...]:
        """Provision EC2 instances."""
        from skyward.providers.aws.lifecycle import provision
        return provision(self, compute)

    @override
    def setup(
        self,
        instances: tuple[Instance, ...],
        compute: ComputeSpec,
    ) -> None:
        """Setup instances (bootstrap, install dependencies)."""
        from skyward.providers.aws.lifecycle import setup
        setup(self, instances, compute)

    @override
    def shutdown(
        self,
        instances: tuple[Instance, ...],
        compute: ComputeSpec,
    ) -> tuple[ExitedInstance, ...]:
        """Shutdown instances (stop on-demand, terminate spot)."""
        from skyward.providers.aws.lifecycle import shutdown
        return shutdown(self, instances, compute)

    @override
    def run_command(self, instance: Instance, command: str, timeout: int = 30) -> str:
        """Run shell command on instance via SSM.

        Args:
            instance: Target EC2 instance.
            command: Shell command string to execute.
            timeout: Maximum time to wait in seconds.

        Returns:
            stdout from the command.

        Raises:
            RuntimeError: If command fails (non-zero exit code).
        """
        ssm = self._get_ssm_session()
        result = ssm.run_command(instance.id, command, timeout)
        if not result.success:
            raise RuntimeError(f"Command failed on {instance.id}: {result.stderr}")
        return result.stdout

    @override
    def discover_peers(self, cluster_id: str) -> tuple[Instance, ...]:
        """Discover peer instances in a cluster via EC2 API.

        Queries EC2 for all running instances with the given cluster_id tag.
        Instances are sorted by private IP, with node index assigned by order.

        Args:
            cluster_id: Unique identifier for the cluster (job_id).

        Returns:
            Tuple of Instance objects for all running peers, sorted by private_ip.
        """
        import boto3

        from skyward.constants import SkywardTag

        ec2 = boto3.client("ec2", region_name=self.region)

        response = ec2.describe_instances(
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
                        node=0,  # Will be assigned by sort order
                        metadata=frozenset([
                            ("cluster_id", cluster_id),
                            ("instance_type", inst.get("InstanceType", "")),
                            ("region", self.region),
                        ]),
                    )
                )

        # Sort by private_ip for deterministic ordering
        instances.sort(key=lambda i: i.private_ip)

        # Reassign node index based on sort order
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

