"""AWS EC2 provider for Skyward GPU instances using SSH connectivity."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, override

from botocore.exceptions import ClientError

from skyward.callback import emit
from skyward.constants import SkywardTag
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
from skyward.providers.base import SSHTransport
from skyward.providers.base.ssh_keys import find_local_ssh_key, get_private_key_path
from skyward.providers.common import (
    Checkpoint,
    install_skyward_wheel_via_transport,
)
from skyward.providers.common import (
    wait_for_bootstrap as _wait_for_bootstrap_common,
)
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
    from subprocess import Popen

    from mypy_boto3_ec2 import EC2Client

    from skyward.accelerator import Accelerator
    from skyward.volume import Volume


# =============================================================================
# Constants
# =============================================================================

# AWS-specific bootstrap checkpoints (in addition to common ones)
AWS_EXTRA_CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(".step_download", "downloading deps"),
    Checkpoint(".step_volumes", "volumes"),
)


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


# =============================================================================
# AWS Bootstrap (SSH-based)
# =============================================================================


def _create_ssh_runner(transport: SSHTransport, instance_id: str) -> Callable[[str], str]:
    """Create a command runner using SSH."""

    def run_command(cmd: str) -> str:
        try:
            return transport.run_command(cmd, timeout=30)
        except Exception as e:
            if "Connection refused" in str(e) or "timed out" in str(e).lower():
                raise InstanceTerminatedError(instance_id, "instance not reachable") from e
            raise

    return run_command


def wait_for_bootstrap_ssh(
    transport: SSHTransport,
    instance_id: str,
    timeout: int = 600,
) -> None:
    """Wait for instance bootstrap with progress tracking via SSH."""
    runner = _create_ssh_runner(transport, instance_id)

    try:
        _wait_for_bootstrap_common(
            run_command=runner,
            instance_id=instance_id,
            timeout=timeout,
            extra_checkpoints=AWS_EXTRA_CHECKPOINTS,
        )
    except RuntimeError:
        error_file = ""
        bootstrap_log = ""
        cloud_init_logs = ""

        try:
            error_file = transport.run_command(
                "cat /opt/skyward/.error 2>/dev/null || echo ''",
                timeout=30,
            ).strip()

            bootstrap_log = transport.run_command(
                "tail -100 /opt/skyward/bootstrap.log 2>/dev/null || echo ''",
                timeout=30,
            ).strip()

            cloud_init_logs = transport.run_command(
                "cat /var/log/cloud-init-output.log 2>/dev/null || echo ''",
                timeout=30,
            ).strip()
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


@dataclass(frozen=True, slots=True)
class AWS(Provider):
    """Provider for AWS EC2 with SSH connectivity.

    Executes functions on EC2 instances using UV for fast dependency installation.
    Supports NVIDIA GPUs, Trainium, distributed clusters, and spot instances.

    Uses SSH for connectivity. Instances must have public IPs.
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
    _resolved_region: str | None = field(default=None, compare=False, repr=False, hash=False)
    _resources: AWSResources | None = field(default=None, compare=False, repr=False, hash=False)
    _store: S3ObjectStore | None = field(default=None, compare=False, repr=False, hash=False)

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

    def _get_transport(self, instance: Instance) -> SSHTransport:
        """Get SSHTransport for an instance."""
        key_path = get_private_key_path()
        username = instance.get_meta("username", "ubuntu")
        ip = instance.public_ip or instance.private_ip
        return SSHTransport(host=ip, username=username, key_path=key_path)

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
        acc: Accelerator | None = None,
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

    @override
    def provision(self, compute: ComputeSpec) -> tuple[Instance, ...]:
        """Provision EC2 instances."""
        from skyward.accelerator import Accelerator
        from skyward.bootstrap import grid_driver, group, inject_ssh_key

        try:
            cluster_id = str(uuid.uuid4())[:8]

            emit(InfraCreating())

            # Select instance type (direct override or inferred from resources)
            acc = Accelerator.from_value(compute.accelerator)
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
                            c for c in candidates
                            if c.metadata.get("architecture") == arch
                        ]

                # AWS Fleet API has payload size limits - keep candidate list reasonable
                _MAX_INSTANCE_TYPES = 50
                if len(filtered_candidates) > _MAX_INSTANCE_TYPES:
                    filtered_candidates = filtered_candidates[:_MAX_INSTANCE_TYPES]

                accelerator_count = filtered_candidates[0].accelerator_count

            # Find available region for any of these instance types
            first_candidate_type = (
                compute.machine if filtered_candidates is None
                else filtered_candidates[0].name
            )
            actual_region = find_available_region(
                instance_type=first_candidate_type,
                preferred_region=self.region,
            )

            # Set resolved region if different from configured
            if actual_region != self.region:
                object.__setattr__(self, "_resolved_region", actual_region)
                emit(
                    RegionAutoSelected(
                        requested_region=self.region,
                        selected_region=actual_region,
                        instance_type=first_candidate_type,
                        provider=ProviderName.AWS,
                    )
                )

            # Now create resources in the actual region
            resources = self._get_resources()
            emit(InfraCreated(region=self._active_region))

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
                architectures = {
                    c.metadata.get("architecture", "x86_64") for c in filtered_candidates
                }
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
            preamble = group(base_preamble, ssh_key_op) if base_preamble and ssh_key_op else (ssh_key_op or base_preamble)

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

            # Wait for instances to be running
            wait_running(self._ec2, instance_ids)

            # Get instance details (includes public_ip)
            instance_details = get_instance_details(self._ec2, instance_ids)

            # Build spec lookup for pricing info
            all_specs = {s.name: s for s in self.available_instances()}

            # Build Instance objects
            instances: list[Instance] = []
            for i, details in enumerate(instance_details):
                # Look up spec for this instance type (for pricing info)
                actual_type = details["instance_type"]
                spec = all_specs.get(actual_type)

                instance = Instance(
                    id=details["id"],
                    provider=self,
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

                emit(
                    InstanceProvisioned(
                        instance_id=instance.id,
                        node=i,
                        spot=instance.spot,
                        provider=ProviderName.AWS,
                        spec=spec,
                        ip=instance.private_ip,
                    )
                )

            spot_count = sum(1 for inst in instances if inst.spot)
            emit(
                ProvisioningCompleted(
                    spot=spot_count,
                    on_demand=len(instances) - spot_count,
                    provider=ProviderName.AWS,
                    region=self._active_region,
                    instances=[inst.id for inst in instances],
                )
            )

            return tuple(instances)

        except Exception as e:
            emit(Error(message=f"Provision failed: {e}"))
            raise

    @override
    def setup(self, instances: tuple[Instance, ...], compute: ComputeSpec) -> None:
        """Setup instances (bootstrap, install dependencies) via SSH."""
        from contextlib import contextmanager

        from skyward.conc import for_each_async

        try:
            def bootstrap_instance(inst: Instance) -> None:
                try:
                    emit(BootstrapStarting(instance_id=inst.id))

                    transport = self._get_transport(inst)
                    wait_for_bootstrap_ssh(
                        transport=transport,
                        instance_id=inst.id,
                    )

                    emit(BootstrapCompleted(instance_id=inst.id))
                except Exception as e:
                    emit(Error(message=f"Bootstrap failed on {inst.id}: {e}", instance_id=inst.id))
                    raise

            for_each_async(bootstrap_instance, instances)

            # Install skyward wheel on all instances using Transport abstraction
            @contextmanager
            def get_transport(inst: Instance):
                yield self._get_transport(inst)

            install_skyward_wheel_via_transport(
                instances,
                get_transport=get_transport,
                compute=compute,
            )

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

        return tuple(exited)

    @override
    def create_tunnel(self, instance: Instance, remote_port: int = 18861) -> tuple[int, Popen[bytes]]:
        """Create SSH tunnel to instance."""
        transport = self._get_transport(instance)
        return transport.create_tunnel(remote_port)

    @override
    def run_command(self, instance: Instance, command: str, timeout: int = 30) -> str:
        """Run shell command on instance via SSH."""
        transport = self._get_transport(instance)
        return transport.run_command(command, timeout)

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
                        metadata=frozenset(
                            [
                                ("cluster_id", cluster_id),
                                ("instance_type", inst.get("InstanceType", "")),
                                ("region", self._active_region),
                            ]
                        ),
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
        instance_data = discover_all_instances(self._ec2, self._active_region)
        return build_instance_specs(instance_data, self._active_region)
