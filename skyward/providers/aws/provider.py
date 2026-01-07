"""AWS EC2 provider for Skyward GPU instances using SSH connectivity."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError
from loguru import logger

from skyward.callback import emit
from skyward.constants import SkywardTag
from skyward.events import (
    BootstrapCompleted,
    BootstrapStarting,
    InstanceLaunching,
    InstanceProvisioned,
    InstanceStopping,
    NetworkReady,
    ProviderName,
    ProvisionedInstance,
    ProvisioningCompleted,
    ProvisioningStarted,
    RegionAutoSelected,
)
from skyward.exceptions import InstanceTerminatedError
from skyward.internal.decorators import audit
from skyward.providers.base.ssh_keys import find_local_ssh_key, get_private_key_path
from skyward.providers.common import (
    Checkpoint,
    install_skyward_wheel_via_transport,
)
from skyward.providers.common import (
    wait_for_bootstrap as _wait_for_bootstrap_common,
)
from skyward.providers.ssh import SSHConfig
from skyward.spec import AllocationStrategy, _AllocationOnDemand, normalize_allocation
from skyward.types import (
    Auto,
    ComputeSpec,
    ExitedInstance,
    Instance,
    InstanceSpec,
    Provider,
    parse_memory_mb,
    select_instances,
)

from .discovery import (
    DLAMI_GPU_SSM,
    UBUNTU_BASE_SSM,
    Architecture,
    build_instance_specs,
    discover_all_instances,
    find_available_region,
)
from .fleet import InstanceConfig, get_instance_details, launch_instances, wait_running
from .infra import AWSInfraManager, AWSResources, S3ObjectStore

if TYPE_CHECKING:
    from mypy_boto3_ec2 import EC2Client

    from skyward.accelerators import AcceleratorSpec
    from skyward.volume import Volume

# AWS-specific bootstrap checkpoints (in addition to common ones)
AWS_EXTRA_CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(".step_download", "downloading deps"),
    Checkpoint(".step_volumes", "volumes"),
)


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

    lines.append(f"MOUNT_UID=$(id -u {username})")
    lines.append(f"MOUNT_GID=$(id -g {username})")

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


def wait_for_bootstrap_ssh(
    inst: Instance,
    provisioned: ProvisionedInstance,
    timeout: int = 600,
) -> None:
    """Wait for instance bootstrap with progress tracking via SSH."""

    def runner(cmd: str) -> str:
        try:
            return inst.run_command(cmd, timeout=30)
        except Exception as e:
            if "Connection refused" in str(e) or "timed out" in str(e).lower():
                raise InstanceTerminatedError(inst.id, "instance not reachable") from e
            raise

    try:
        _wait_for_bootstrap_common(
            run_command=runner,
            instance=provisioned,
            timeout=timeout,
            extra_checkpoints=AWS_EXTRA_CHECKPOINTS,
        )
    except RuntimeError:
        error_file = ""
        bootstrap_log = ""
        cloud_init_logs = ""

        try:
            error_file = inst.run_command(
                "cat /opt/skyward/.error 2>/dev/null || echo ''",
                timeout=30,
            ).strip()

            bootstrap_log = inst.run_command(
                "tail -100 /opt/skyward/bootstrap.log 2>/dev/null || echo ''",
                timeout=30,
            ).strip()

            cloud_init_logs = inst.run_command(
                "cat /var/log/cloud-init-output.log 2>/dev/null || echo ''",
                timeout=30,
            ).strip()
        except Exception:
            pass

        msg_parts = [f"Bootstrap failed on {provisioned.instance_id}."]

        if error_file:
            msg_parts.append(f"\n--- Error file ---\n{error_file}")

        if bootstrap_log:
            msg_parts.append(f"\n--- Bootstrap log (last 100 lines) ---\n{bootstrap_log}")

        if cloud_init_logs and not error_file and not bootstrap_log:
            msg_parts.append(f"\n--- Cloud-init logs ---\n{cloud_init_logs}")

        raise RuntimeError("\n".join(msg_parts)) from None


_resources_cache: dict[str, AWSResources] = {}
_resources_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class AWS:
    """AWS provider configuration.

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

    def build(self) -> AWSProvider:
        """Build a stateful AWSProvider from this configuration."""
        return AWSProvider(
            region=self.region,
            ami=self.ami,
            subnet_id=self.subnet_id,
            security_group_id=self.security_group_id,
            instance_profile_arn=self.instance_profile_arn,
            username=self.username,
            instance_timeout=self.instance_timeout,
            allocation_strategy=self.allocation_strategy,
        )


class AWSProvider(Provider):
    """Stateful AWS provider service.

    This class manages EC2 instances and maintains runtime state.
    Created by AWS.build() or automatically by ComputePool.

    Implements the Provider protocol with provision/setup/shutdown lifecycle.
    """

    def __init__(
        self,
        region: str,
        ami: str | None,
        subnet_id: str | None,
        security_group_id: str | None,
        instance_profile_arn: str | None,
        username: str | None,
        instance_timeout: int,
        allocation_strategy: AllocationStrategy,
    ) -> None:
        self.region = region
        self.ami = ami
        self.subnet_id = subnet_id
        self.security_group_id = security_group_id
        self.instance_profile_arn = instance_profile_arn
        self.username = username
        self.instance_timeout = instance_timeout
        self.allocation_strategy = allocation_strategy

        # Mutable runtime state
        self._resolved_region: str | None = None
        self._resources: AWSResources | None = None
        self._store: S3ObjectStore | None = None

    @property
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
                self._resources = resources
                return resources

            infra = AWSInfraManager(region=region)
            resources = infra.ensure_infrastructure()
            _resources_cache[cache_key] = resources
            self._resources = resources
            return resources

    @property
    def _ec2(self) -> EC2Client:
        """EC2 client for active region."""
        import boto3

        return boto3.client("ec2", region_name=self._active_region)

    # =========================================================================
    # AMI Resolution
    # =========================================================================

    def _resolve_ami_id(
        self,
        compute: ComputeSpec,
        architecture: Architecture = "x86_64",
        acc: AcceleratorSpec | None = None,
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
    # Provider Protocol Implementation
    # =========================================================================

    @audit("Provisioning")
    def provision(self, compute: ComputeSpec) -> tuple[Instance, ...]:
        """Provision EC2 instances."""
        from skyward.accelerators import AcceleratorSpec
        from skyward.bootstrap import grid_driver, group, inject_ssh_key

        cluster_id = str(uuid.uuid4())[:8]
        logger.debug(f"Cluster ID: {cluster_id}, nodes: {compute.nodes}, region: {self.region}")

        emit(ProvisioningStarted())

        # Select instance type (direct override or inferred from resources)
        acc = AcceleratorSpec.from_value(compute.accelerator)
        allocation = normalize_allocation(compute.allocation)
        prefer_spot = not isinstance(allocation, _AllocationOnDemand)

        if compute.machine is not None:
            # Direct instance type override
            # Look up spec for metadata (architecture, accelerator_count)
            specs = self.available_instances()
            spec = next((s for s in specs if s.name == compute.machine), None)
            if spec:
                accelerator_count = spec.accelerator_count
                architecture = spec.metadata.get("architecture", "x86_64")
            else:
                # Defaults if instance type not found in catalog
                if acc and callable(acc.count):
                    accelerator_count = 1
                else:
                    accelerator_count = acc.count if acc else 0
                architecture = "x86_64"
            # Will construct instance_configs after region/resources are resolved
            filtered_candidates = None  # Signal to use direct machine override
        else:
            # Get all matching candidates (Fleet will pick one with capacity)
            accelerator_type = acc.accelerator if acc else None
            requested_gpu_count = acc.count if acc else 1

            candidates = select_instances(
                self.available_instances(),
                cpu=compute.cpu or 1,
                memory_mb=parse_memory_mb(compute.memory),
                accelerator=accelerator_type,
                accelerator_count=requested_gpu_count,
                prefer_spot=prefer_spot,
            )

            # Filter by architecture based on compute.architecture
            match compute.architecture:
                case Auto():
                    # Auto: include ALL architectures, sorted by price
                    filtered_candidates = list(candidates)
                case arch:
                    # Explicit architecture requested - filter to it
                    filtered_candidates = [
                        c for c in candidates if c.metadata.get("architecture") == arch
                    ]

            # AWS Fleet API has payload size limits - keep candidate list reasonable
            _MAX_INSTANCE_TYPES = 50
            if len(filtered_candidates) > _MAX_INSTANCE_TYPES:
                filtered_candidates = filtered_candidates[:_MAX_INSTANCE_TYPES]

            accelerator_count = filtered_candidates[0].accelerator_count

        # Find available region for any of these instance types
        first_candidate_type = (
            compute.machine if filtered_candidates is None else filtered_candidates[0].name
        )
        actual_region = find_available_region(
            instance_type=first_candidate_type,
            preferred_region=self.region,
        )

        # Set resolved region if different from configured
        if actual_region != self.region:
            self._resolved_region = actual_region
            # Get the spec for the first candidate for the event
            first_spec = (
                filtered_candidates[0]
                if filtered_candidates
                else next(
                    (s for s in self.available_instances() if s.name == compute.machine), None
                )
            )
            if first_spec:
                emit(
                    RegionAutoSelected(
                        requested_region=self.region,
                        selected_region=actual_region,
                        spec=first_spec,
                        provider=ProviderName.AWS,
                    )
                )

        # Now create resources in the actual region
        logger.debug(f"Ensuring infrastructure in {self._active_region}")
        resources = self._get_resources()
        logger.debug(f"Infrastructure ready: subnets={resources.subnet_ids}")
        emit(NetworkReady(region=self._active_region))

        # Get image from compute spec
        image = compute.image
        content_hash = image.content_hash()

        # Resolve AMI(s) and build InstanceConfigs with pricing
        if filtered_candidates is None:
            # Direct machine override - single AMI
            ami_id = self._resolve_ami_id(compute, architecture=architecture, acc=acc)
            # Look up spec from catalog for pricing and details
            specs = self.available_instances()
            spec = next((s for s in specs if s.name == compute.machine), None)
            instance_configs = (
                InstanceConfig(
                    instance_type=compute.machine,
                    ami_id=ami_id,
                    price_spot=spec.price_spot if spec else None,
                    price_on_demand=spec.price_on_demand if spec else None,
                ),
            )
            # Use spec for candidates if found, otherwise create minimal one
            candidates_for_log: tuple[InstanceSpec, ...] = (spec,) if spec else ()
            username = self._resolve_username(ami_id)
        else:
            # Candidates from select_instances - may have mixed architectures
            architectures = {c.metadata.get("architecture", "x86_64") for c in filtered_candidates}
            ami_by_arch = {
                arch: self._resolve_ami_id(compute, architecture=arch, acc=acc)
                for arch in architectures
            }
            instance_configs = tuple(
                InstanceConfig(
                    instance_type=c.name,
                    ami_id=ami_by_arch[c.metadata.get("architecture", "x86_64")],
                    price_spot=c.price_spot,
                    price_on_demand=c.price_on_demand,
                )
                for c in filtered_candidates
            )
            candidates_for_log = tuple(filtered_candidates)
            # Use first AMI for username (all same type, so same username)
            first_ami = next(iter(ami_by_arch.values()))
            username = self._resolve_username(first_ami)

        # Get local SSH public key to inject into instance
        ssh_key_op = None
        key_info = find_local_ssh_key()
        if key_info:
            _, public_key = key_info
            ssh_key_op = inject_ssh_key(public_key)

        # Generate bootstrap script
        # For fractional GPU, include GRID driver installation
        base_preamble = grid_driver() if acc and acc.fractional else None
        preamble = (
            group(base_preamble, ssh_key_op)
            if base_preamble and ssh_key_op
            else (ssh_key_op or base_preamble)
        )

        # Build postamble (volume mounting)
        volume_script = _generate_volume_script(compute.volumes, username)
        postamble = volume_script if volume_script else None

        user_data = image.bootstrap(
            ttl=compute.timeout,
            preamble=preamble,
            postamble=postamble,
        )

        emit(
            InstanceLaunching(
                count=compute.nodes,
                candidates=candidates_for_log,
                provider=ProviderName.AWS,
            )
        )

        # Get AMI block device info (use first AMI - all same type)
        first_ami = instance_configs[0].ami_id
        root_device, min_volume_size = self._get_ami_block_device_info(first_ami)

        # Launch instances with allocation strategy
        # Fleet will pick an instance type with available capacity
        logger.debug(
            f"Launching {compute.nodes} instances with {len(instance_configs)} candidate types"
        )
        instance_ids = launch_instances(
            ec2=self._ec2,
            resources=resources,
            n=compute.nodes,
            instances=instance_configs,
            user_data=user_data,
            requirements_hash=content_hash,
            allocation=allocation,
            fleet_strategy=self.allocation_strategy,
            root_device=root_device,
            min_volume_size=min_volume_size,
        )
        logger.debug(f"Fleet launched: {instance_ids}")

        # Wait for instances to be running
        logger.debug("Waiting for instances to reach running state...")
        wait_running(self._ec2, instance_ids)
        logger.debug("All instances running")

        # Get instance details (includes public_ip)
        instance_details = get_instance_details(self._ec2, instance_ids)

        # Build spec lookup for pricing info
        all_specs = {s.name: s for s in self.available_instances()}

        # Build Instance objects and ProvisionedInstance for events
        instances: list[Instance] = []
        provisioned_instances: list[ProvisionedInstance] = []
        key_path = get_private_key_path()

        for i, details in enumerate(instance_details):
            # Look up spec for this instance type (for pricing info)
            actual_type = details["instance_type"]
            spec = all_specs.get(actual_type)
            ssh_host = details.get("public_ip") or details["private_ip"]

            instance = Instance(
                id=details["id"],
                provider=self,
                ssh=SSHConfig(
                    host=ssh_host,
                    username=username,
                    key_path=key_path,
                ),
                spot=details["spot"],
                private_ip=details["private_ip"],
                public_ip=details.get("public_ip"),
                node=i,
                metadata=frozenset(
                    [
                        ("cluster_id", cluster_id),
                        ("region", self._active_region),
                        ("instance_type", actual_type),
                        ("content_hash", content_hash),
                        ("accelerator_count", str(accelerator_count)),
                        ("username", username),
                    ]
                ),
            )
            instances.append(instance)

            # Create ProvisionedInstance for events
            provisioned = ProvisionedInstance(
                instance_id=instance.id,
                node=i,
                provider=ProviderName.AWS,
                spot=instance.spot,
                spec=spec,
                ip=instance.private_ip,
            )
            provisioned_instances.append(provisioned)

            emit(InstanceProvisioned(instance=provisioned))

        spot_count = sum(1 for inst in instances if inst.spot)
        logger.info(
            f"Provisioned {len(instances)} instances: {spot_count} spot, "
            f"{len(instances) - spot_count} on-demand"
        )
        emit(
            ProvisioningCompleted(
                instances=tuple(provisioned_instances),
                provider=ProviderName.AWS,
                region=self._active_region,
            )
        )

        return tuple(instances)

    def _make_provisioned(self, inst: Instance) -> ProvisionedInstance:
        """Create ProvisionedInstance from Instance for events."""
        instance_type = inst.get_meta("instance_type", "")
        all_specs = {s.name: s for s in self.available_instances()}
        spec = all_specs.get(instance_type)

        return ProvisionedInstance(
            instance_id=inst.id,
            node=inst.node,
            provider=ProviderName.AWS,
            spot=inst.spot,
            spec=spec,
            ip=inst.private_ip,
        )

    @audit("Setup")
    def setup(self, instances: tuple[Instance, ...], compute: ComputeSpec) -> None:
        """Setup instances (bootstrap, install dependencies) via SSH."""
        from skyward.conc import for_each_async

        def bootstrap_instance(inst: Instance) -> None:
            provisioned = self._make_provisioned(inst)
            emit(BootstrapStarting(instance=provisioned))
            wait_for_bootstrap_ssh(inst, provisioned)
            emit(BootstrapCompleted(instance=provisioned))

        for_each_async(bootstrap_instance, instances)

        install_skyward_wheel_via_transport(
            instances,
            get_transport=lambda inst: inst.connect(),
            compute=compute,
        )

    def shutdown(
        self, instances: tuple[Instance, ...], compute: ComputeSpec
    ) -> tuple[ExitedInstance, ...]:
        """Terminate instances."""
        if not instances:
            return ()

        instance_ids = [inst.id for inst in instances]
        logger.info(f"Terminating {len(instances)} instances...")
        logger.debug(f"Instance IDs: {instance_ids}")

        # Terminate all instances
        self._ec2.terminate_instances(InstanceIds=instance_ids)

        # Build ExitedInstance objects
        exited: list[ExitedInstance] = []
        for inst in instances:
            provisioned = self._make_provisioned(inst)
            emit(InstanceStopping(instance=provisioned))
            exited.append(ExitedInstance(instance=inst, exit_code=0, exit_reason="terminated"))

        logger.info(f"Terminated {len(instances)} instances")
        return tuple(exited)

    def discover_peers(self, cluster_id: str) -> tuple[Instance, ...]:
        """Discover peer instances in a cluster via EC2 API."""
        from skyward.providers.base import assign_node_indices

        response = self._ec2.describe_instances(
            Filters=[
                {"Name": f"tag:{SkywardTag.MANAGED}", "Values": ["true"]},
                {"Name": f"tag:{SkywardTag.CLUSTER_ID}", "Values": [cluster_id]},
                {"Name": "instance-state-name", "Values": ["running"]},
            ]
        )

        key_path = get_private_key_path()
        instances = []

        for reservation in response.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                public_ip = inst.get("PublicIpAddress")
                private_ip = inst.get("PrivateIpAddress", "")
                ssh_host = public_ip or private_ip

                instances.append(
                    Instance(
                        id=inst["InstanceId"],
                        provider=self,
                        ssh=SSHConfig(host=ssh_host, username="ubuntu", key_path=key_path),
                        spot=inst.get("InstanceLifecycle") == "spot",
                        private_ip=private_ip,
                        public_ip=public_ip,
                        node=0,
                        metadata=frozenset([
                            ("cluster_id", cluster_id),
                            ("instance_type", inst.get("InstanceType", "")),
                            ("region", self._active_region),
                        ]),
                    )
                )

        return assign_node_indices(instances)

    def available_instances(self) -> tuple[InstanceSpec, ...]:
        """List all available instance types in this region with pricing."""
        instance_data = discover_all_instances(self._ec2, self._active_region)
        return build_instance_specs(instance_data, self._active_region)
