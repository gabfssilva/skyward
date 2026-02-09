"""AWS Provider Handler - event-driven with Event Pipeline.

Uses intermediate events (InstanceLaunched, InstanceRunning) for
decoupled instance lifecycle management.
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from loguru import logger

from skyward.app import component, on
from skyward.bus import AsyncEventBus
from skyward.retry import on_exception_message, retry
from skyward.events import (
    BootstrapFailed,
    BootstrapPhase,
    BootstrapRequested,
    ClusterDestroyed,
    ClusterProvisioned,
    ClusterRequested,
    InstanceBootstrapped,
    InstanceId,
    InstanceLaunched,
    InstanceRunning,
    InstanceRequested,
    ShutdownRequested,
)
from skyward.monitors import SSHCredentialsRegistry

from skyward.utils.pricing import get_instance_pricing

from .clients import EC2ClientFactory
from .config import AWS, AllocationStrategy
from .state import AWSClusterState, AWSResources, InstanceConfig

# AWS instance family to GPU model mapping
_AWS_INSTANCE_GPU = {
    "g4dn": "T4",
    "g5": "A10G",
    "g6": "L4",
    "p4d": "A100",
    "p4de": "A100",
    "p5": "H100",
}

if TYPE_CHECKING:
    from skyward.spec import Architecture, PoolSpec


@component
class AWSHandler:
    """Event-driven AWS provider using Event Pipeline.

    Flow:
        ClusterRequested → setup infra → ClusterProvisioned
        InstanceRequested → launch via Fleet → InstanceLaunched
        InstanceLaunched → wait running → InstanceRunning
        BootstrapRequested → wait for EventStreamer → InstanceBootstrapped
        ShutdownRequested → terminate all → ClusterDestroyed

    The InstanceOrchestrator handles:
        InstanceRunning → InstanceProvisioned + BootstrapRequested

    Note: AWS uses EventStreamer for bootstrap streaming. The handler
    waits for BootstrapPhase events via the event bus.
    """

    bus: AsyncEventBus
    config: AWS
    ec2: EC2ClientFactory
    ssh_credentials: SSHCredentialsRegistry

    def __post_init__(self) -> None:
        """Initialize runtime state."""
        self._clusters: dict[str, AWSClusterState] = {}
        self._bootstrap_waiters: dict[InstanceId, asyncio.Future[bool]] = {}

    # -------------------------------------------------------------------------
    # Cluster Lifecycle
    # -------------------------------------------------------------------------

    @on(ClusterRequested, match=lambda self, e: e.provider == "aws")
    async def handle_cluster_requested(self, _: Any, event: ClusterRequested) -> None:
        """Provision AWS infrastructure and launch all instances atomically."""
        resources, (ssh_key_name, ssh_key_path), instance_configs = await asyncio.gather(
            self._ensure_infrastructure(),
            self._ensure_key_pair(),
            self._resolve_instance_configs(event.spec),
        )

        cluster_id = f"aws-{uuid.uuid4().hex[:8]}"

        state = AWSClusterState(
            cluster_id=cluster_id,
            spec=event.spec,
            resources=resources,
            region=self.config.region,
            ssh_key_name=ssh_key_name,
            ssh_key_path=ssh_key_path,
            username=self.config.username or "ubuntu",
        )
        self._clusters[cluster_id] = state

        self.ssh_credentials.register(cluster_id, state.username, state.ssh_key_path)

        user_data = self._generate_user_data(state.spec)

        instance_ids = await self._launch_fleet(
            cluster=state,
            instances=tuple(instance_configs),
            user_data=user_data,
            n=event.spec.nodes,
        )

        for node_id, instance_id in enumerate(instance_ids):
            state.fleet_instance_ids[node_id] = instance_id

        self.bus.emit(
            ClusterProvisioned(
                request_id=event.request_id,
                cluster_id=cluster_id,
                provider="aws",
            )
        )

    @on(ShutdownRequested)
    async def handle_shutdown_requested(self, _: Any, event: ShutdownRequested) -> None:
        """Terminate all instances in a cluster."""
        cluster = self._clusters.pop(event.cluster_id, None)
        if not cluster:
            return

        if cluster.instance_ids:
            await self._terminate_instances(cluster.instance_ids)

        self.bus.emit(ClusterDestroyed(cluster_id=event.cluster_id))

    # -------------------------------------------------------------------------
    # Instance Lifecycle - Event Pipeline
    # -------------------------------------------------------------------------

    @on(InstanceRequested, match=lambda self, e: e.provider == "aws")
    async def handle_instance_requested(self, _: Any, event: InstanceRequested) -> None:
        """Emit InstanceLaunched for pre-assigned or replacement instances."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        # Initial provision: use pre-launched instance from atomic fleet
        pre_assigned = cluster.fleet_instance_ids.pop(event.node_id, None)
        if event.replacing is None and pre_assigned:
            instance_id = pre_assigned
        else:
            # Replacement: launch a single instance
            instance_configs = await self._resolve_instance_configs(cluster.spec)
            user_data = self._generate_user_data(cluster.spec)

            instance_ids = await self._launch_fleet(
                cluster=cluster,
                instances=tuple(instance_configs),
                user_data=user_data,
                n=1,
            )

            if not instance_ids:
                logger.error(f"AWS: Failed to launch instance for node {event.node_id}")
                return

            instance_id = instance_ids[0]

        cluster.pending_nodes.add(event.node_id)

        self.bus.emit(
            InstanceLaunched(
                request_id=event.request_id,
                cluster_id=event.cluster_id,
                node_id=event.node_id,
                provider="aws",
                instance_id=instance_id,
            )
        )

    @on(InstanceLaunched, match=lambda self, e: e.provider == "aws")
    async def handle_instance_launched(self, _: Any, event: InstanceLaunched) -> None:
        """Wait for instance to be running and emit InstanceRunning."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        # Wait for running state
        await self._wait_running([event.instance_id])

        # Get instance details
        details = await self._get_instance_details([event.instance_id])
        if not details:
            logger.error(f"AWS: Could not get details for {event.instance_id}")
            return

        detail = details[0]
        instance_type = detail.get("instance_type", "")
        is_spot = detail["spot"]

        # Fetch pricing from Vantage API
        hourly_rate = 0.0
        on_demand_rate = 0.0
        gpu_count = 0
        gpu_model = ""
        vcpus = 0
        memory_gb = 0.0

        pricing = get_instance_pricing(instance_type, "aws", self.config.region)
        if pricing:
            on_demand_rate = pricing.ondemand or 0.0
            hourly_rate = (pricing.spot_avg if is_spot and pricing.spot_avg else on_demand_rate)
            gpu_count = pricing.gpu_count
            vcpus = pricing.vcpu
            memory_gb = pricing.memory_gb

        # Determine GPU model from instance type family
        from skyward.accelerators.catalog import get_gpu_vram_gb

        instance_family = instance_type.split(".")[0] if "." in instance_type else ""
        gpu_model = _AWS_INSTANCE_GPU.get(instance_family, "")
        gpu_vram_gb = get_gpu_vram_gb(gpu_model)

        # Emit InstanceRunning - InstanceOrchestrator handles the rest
        self.bus.emit(
            InstanceRunning(
                request_id=event.request_id,
                cluster_id=event.cluster_id,
                node_id=event.node_id,
                provider="aws",
                instance_id=event.instance_id,
                ip=detail["public_ip"] or detail["private_ip"],
                private_ip=detail["private_ip"],
                ssh_port=22,
                spot=is_spot,
                # Pricing info from Vantage
                hourly_rate=hourly_rate,
                on_demand_rate=on_demand_rate,
                billing_increment=1,  # AWS bills per-minute
                instance_type=instance_type,
                gpu_count=gpu_count,
                gpu_model=gpu_model,
                # Hardware specs
                vcpus=vcpus,
                memory_gb=memory_gb,
                gpu_vram_gb=gpu_vram_gb,
                # Location info
                region=self.config.region,
            )
        )

    @on(BootstrapRequested, match=lambda self, e: e.instance.provider == "aws")
    async def handle_bootstrap_requested(self, _: Any, event: BootstrapRequested) -> None:
        """Wait for bootstrap completion via EventStreamer and emit InstanceBootstrapped."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        # Wait for bootstrap completion via BootstrapPhase event
        # EventStreamer handles the actual streaming
        await self._wait_for_bootstrap_event(event.instance)

        # Install local skyward wheel if skyward_source == 'local'
        if cluster.spec.image and cluster.spec.image.skyward_source == "local":
            await self._install_local_skyward(event.instance, cluster)

        # Track instance in cluster state
        cluster.add_instance(event.instance)

        # Emit InstanceBootstrapped - Node will signal NodeReady
        self.bus.emit(InstanceBootstrapped(instance=event.instance))

    @on(BootstrapPhase, audit=False)  # High-frequency event
    async def handle_bootstrap_phase(self, _: Any, event: BootstrapPhase) -> None:
        """Handle bootstrap phase events - resolve waiters when bootstrap completes."""
        if event.phase != "bootstrap" or event.event not in ("completed", "failed"):
            return

        instance_id = event.instance.id
        waiter = self._bootstrap_waiters.get(instance_id)
        if waiter and not waiter.done():
            waiter.set_result(event.event == "completed")

    @on(BootstrapFailed, audit=False)
    async def handle_bootstrap_failed(self, _: Any, event: BootstrapFailed) -> None:
        """Handle bootstrap failure - resolve waiter with failure."""
        instance_id = event.instance.id
        waiter = self._bootstrap_waiters.get(instance_id)
        if waiter and not waiter.done():
            waiter.set_result(False)

    # -------------------------------------------------------------------------
    # Infrastructure Management
    # -------------------------------------------------------------------------

    async def _ensure_infrastructure(self, prefix: str = "skyward") -> AWSResources:
        """Ensure minimal AWS infrastructure exists."""
        subnet_ids = await self._get_default_subnets()

        if self.config.security_group_id:
            security_group_id = self.config.security_group_id
        else:
            security_group_id = await self._ensure_security_group(prefix)

        instance_profile_arn = self.config.instance_profile_arn or ""

        return AWSResources(
            bucket="",
            iam_role_arn="",
            instance_profile_arn=instance_profile_arn,
            security_group_id=security_group_id,
            region=self.config.region,
            subnet_ids=subnet_ids,
        )

    async def _get_default_subnets(self) -> tuple[str, ...]:
        """Get subnet IDs from default VPC."""
        async with self.ec2() as ec2:
            vpcs = await ec2.describe_vpcs(
                Filters=[{"Name": "is-default", "Values": ["true"]}]
            )
            if not vpcs["Vpcs"]:
                raise RuntimeError("No default VPC found")

            vpc_id = vpcs["Vpcs"][0]["VpcId"]
            subnets = await ec2.describe_subnets(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
            )

            if not subnets["Subnets"]:
                raise RuntimeError("No subnets found in default VPC")

            return tuple(s["SubnetId"] for s in subnets["Subnets"])

    async def _ensure_security_group(self, prefix: str) -> str:
        """Ensure security group exists with required rules."""
        sg_name = f"{prefix}-sg"

        async with self.ec2() as ec2:
            try:
                resp = await ec2.describe_security_groups(
                    Filters=[{"Name": "group-name", "Values": [sg_name]}]
                )
                if resp["SecurityGroups"]:
                    sg_id = resp["SecurityGroups"][0]["GroupId"]
                    await self._ensure_sg_rules(ec2, sg_id)
                    return sg_id
            except Exception:
                pass

            vpcs = await ec2.describe_vpcs(
                Filters=[{"Name": "is-default", "Values": ["true"]}]
            )
            if not vpcs["Vpcs"]:
                raise RuntimeError("No default VPC found")
            vpc_id = vpcs["Vpcs"][0]["VpcId"]

            resp = await ec2.create_security_group(
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
            sg_id = resp["GroupId"]

            await ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "-1",
                        "UserIdGroupPairs": [
                            {"GroupId": sg_id, "Description": "All traffic from same SG"}
                        ],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH"}],
                    },
                ],
            )

            return sg_id

    async def _ensure_sg_rules(self, ec2: Any, sg_id: str) -> None:
        """Ensure security group has required rules."""
        resp = await ec2.describe_security_groups(GroupIds=[sg_id])
        rules = resp["SecurityGroups"][0].get("IpPermissions", [])

        has_self = any(
            r.get("IpProtocol") == "-1"
            and any(p.get("GroupId") == sg_id for p in r.get("UserIdGroupPairs", []))
            for r in rules
        )
        has_ssh = any(
            r.get("IpProtocol") == "tcp" and r.get("FromPort") == 22 for r in rules
        )

        to_add = []
        if not has_self:
            to_add.append({
                "IpProtocol": "-1",
                "UserIdGroupPairs": [{"GroupId": sg_id, "Description": "All traffic from same SG"}],
            })
        if not has_ssh:
            to_add.append({
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH"}],
            })

        if to_add:
            await ec2.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=to_add)

    # -------------------------------------------------------------------------
    # Fleet Launch
    # -------------------------------------------------------------------------

    async def _resolve_instance_configs(self, spec: PoolSpec) -> list[InstanceConfig]:
        """Resolve multiple instance configs from spec for Fleet flexibility."""
        spot = spec.allocation in ("spot", "spot-if-available")

        candidates = await self._select_instances(spec)

        configs = []
        for instance_type, arch in candidates:
            if spec.accelerator_name is not None:
                ami = self.config.ami or await self._get_dlami(arch)
            else:
                ami = self.config.ami or await self._get_ubuntu_ami(arch)
            configs.append(InstanceConfig(instance_type=instance_type, ami=ami, spot=spot))

        return configs

    async def _select_instances(self, spec: PoolSpec, max_candidates: int = 5) -> list[tuple[str, Architecture]]:
        """Select cheapest instances that meet spec requirements.

        Queries AWS for instances matching vcpus, memory, architecture, and
        accelerator requirements, then returns the cheapest ones by spot price.

        Returns:
            List of (instance_type, architecture) tuples, sorted by price.
        """
        import aioboto3

        min_vcpus = spec.vcpus or 2
        min_memory_mib = (spec.memory_gb or 8) * 1024
        archs = [spec.architecture] if spec.architecture else ["x86_64", "arm64"]

        # Build instance requirements
        requirements: dict[str, Any] = {
            "VCpuCount": {"Min": min_vcpus},
            "MemoryMiB": {"Min": min_memory_mib},
            "InstanceGenerations": ["current"],
            "BurstablePerformance": "excluded",
            "BareMetal": "excluded",
        }

        session = aioboto3.Session()
        async with session.client("ec2", region_name=self.config.region) as ec2:
            # Map instance type -> architecture (from query)
            candidate_arch: dict[str, Architecture] = {}

            for a in archs:
                arch_requirements = requirements.copy()

                # Add accelerator requirements if specified
                if spec.accelerator_name:
                    arch_requirements["AcceleratorTypes"] = ["gpu"]
                    arch_requirements["AcceleratorNames"] = [spec.accelerator_name.lower()]
                    arch_requirements["AcceleratorCount"] = {"Min": spec.accelerator_count or 1}

                response = await ec2.get_instance_types_from_instance_requirements(
                    ArchitectureTypes=[a],
                    VirtualizationTypes=["hvm"],
                    InstanceRequirements=arch_requirements,
                )
                for r in response.get("InstanceTypes", []):
                    candidate_arch[r["InstanceType"]] = a  # type: ignore[assignment]

            if not candidate_arch:
                return [self._fallback_instance(spec)]

            # Get spot prices - query in batches
            all_types = list(candidate_arch.keys())
            price_map: dict[str, float] = {}
            for i in range(0, len(all_types), 20):
                batch = all_types[i:i + 20]
                prices = await ec2.describe_spot_price_history(
                    InstanceTypes=batch,
                    ProductDescriptions=["Linux/UNIX"],
                )
                for p in prices.get("SpotPriceHistory", []):
                    itype = p["InstanceType"]
                    price = float(p["SpotPrice"])
                    if itype not in price_map or price < price_map[itype]:
                        price_map[itype] = price

            if price_map:
                # Sort by price and take top N
                sorted_types = sorted(price_map.keys(), key=lambda k: price_map[k])[:max_candidates]
                result: list[tuple[str, Architecture]] = [
                    (t, candidate_arch[t]) for t in sorted_types
                ]
                prices_str = ", ".join(f"{t}=${price_map[t]:.4f}" for t in sorted_types)
                logger.info(f"Instance candidates: {prices_str}")
                return result

            # No prices - return first few candidates
            result = [(t, candidate_arch[t]) for t in all_types[:max_candidates]]
            return result

    def _fallback_instance(self, spec: PoolSpec) -> tuple[str, Architecture]:
        """Return fallback instance when API returns no candidates."""
        if spec.accelerator_name:
            # GPU fallbacks
            gpu_fallbacks: dict[str, tuple[str, Architecture]] = {
                "T4": ("g4dn.xlarge", "x86_64"),
                "A10G": ("g5.xlarge", "x86_64"),
                "A100": ("p4d.24xlarge", "x86_64"),
                "H100": ("p5.48xlarge", "x86_64"),
            }
            fallback = gpu_fallbacks.get(spec.accelerator_name, ("g4dn.xlarge", "x86_64"))
        else:
            # CPU fallbacks
            if spec.architecture == "arm64":
                fallback = ("m7g.large", "arm64")
            elif spec.architecture == "x86_64":
                fallback = ("m5.large", "x86_64")
            else:
                fallback = ("m7g.large", "arm64")  # arm64 is usually cheaper

        logger.warning(f"No instances found for spec. Using fallback {fallback[0]}")
        return fallback

    def _arch_from_instance_type(self, instance_type: str) -> Architecture:
        """Infer architecture from instance type name."""
        # Graviton instances: m7g, c7g, r7g, t4g, m6g, c6g, etc.
        # Pattern: letter(s) + number + "g" (optionally followed by d/n/e)
        # NOT GPU instances like g4dn, g5, p4d which start with g/p
        family = instance_type.split(".")[0]
        # Graviton families end with a number followed by 'g' (optionally 'd', 'n', 'e')
        # Examples: m7g, c7gd, r7gn, t4g, m6gd
        import re
        if re.match(r"^[a-z]+\d+g[den]*$", family):
            return "arm64"
        return "x86_64"

    async def _get_ubuntu_ami(self, arch: Architecture) -> str:
        import aioboto3

        arch_path = "arm64" if arch == "arm64" else "amd64"
        version = self.config.ubuntu_version
        ebs_type = "ebs-gp3" if version >= "24.04" else "ebs-gp2"
        param_name = f"/aws/service/canonical/ubuntu/server/{version}/stable/current/{arch_path}/hvm/{ebs_type}/ami-id"

        session = aioboto3.Session()
        async with session.client("ssm", region_name=self.config.region) as ssm:
            try:
                response = await ssm.get_parameter(Name=param_name)
                return response["Parameter"]["Value"]
            except Exception as e:
                raise RuntimeError(f"Failed to get Ubuntu AMI: {e}") from e

    async def _get_dlami(self, arch: Architecture) -> str:
        """Get Deep Learning AMI for region via SSM Parameter Store."""
        import aioboto3

        arch_path = "arm64" if arch == "arm64" else "x86_64"
        version = self.config.ubuntu_version
        param_name = f"/aws/service/deeplearning/ami/{arch_path}/base-oss-nvidia-driver-gpu-ubuntu-{version}/latest/ami-id"

        session = aioboto3.Session()
        async with session.client("ssm", region_name=self.config.region) as ssm:
            try:
                response = await ssm.get_parameter(Name=param_name)
                return response["Parameter"]["Value"]
            except Exception as e:
                raise RuntimeError(
                    f"Could not find Deep Learning AMI in region {self.config.region}. "
                    f"SSM parameter {param_name} not found. Error: {e}"
                ) from e

    def _generate_user_data(self, spec: PoolSpec) -> str:
        """Generate bootstrap user data script."""
        ttl = spec.ttl or self.config.instance_timeout
        return spec.image.generate_bootstrap(ttl=ttl)

    async def _launch_fleet(
        self,
        cluster: AWSClusterState,
        instances: tuple[InstanceConfig, ...],
        user_data: str,
        n: int,
        allocation_strategy: AllocationStrategy | None = None,
    ) -> list[str]:
        """Launch EC2 Fleet."""
        strategy = allocation_strategy or self.config.allocation_strategy
        spot = instances[0].spot if instances else False

        async with self.ec2() as ec2:
            target_subnets = await self._get_valid_subnets(
                ec2,
                tuple(i.instance_type for i in instances),
                cluster.resources.subnet_ids,
            )

            template_name = f"skyward-{uuid.uuid4().hex[:8]}"
            template_data: dict[str, Any] = {
                "UserData": base64.b64encode(user_data.encode()).decode(),
                "NetworkInterfaces": [
                    {
                        "DeviceIndex": 0,
                        "AssociatePublicIpAddress": True,
                        "Groups": [cluster.resources.security_group_id],
                    }
                ],
                "BlockDeviceMappings": [
                    {
                        "DeviceName": "/dev/sda1",
                        "Ebs": {
                            "VolumeSize": 100,
                            "VolumeType": "gp3",
                            "DeleteOnTermination": True,
                        },
                    }
                ],
                "TagSpecifications": [
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": f"skyward-{cluster.cluster_id}"},
                            {"Key": "skyward:managed", "Value": "true"},
                            {"Key": "skyward:cluster", "Value": cluster.cluster_id},
                        ],
                    }
                ],
                "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
                "InstanceInitiatedShutdownBehavior": "terminate",
            }

            if cluster.ssh_key_name:
                template_data["KeyName"] = cluster.ssh_key_name

            if cluster.resources.instance_profile_arn:
                template_data["IamInstanceProfile"] = {"Arn": cluster.resources.instance_profile_arn}

            template = await ec2.create_launch_template(
                LaunchTemplateName=template_name,
                LaunchTemplateData=template_data,
            )
            template_id = template["LaunchTemplate"]["LaunchTemplateId"]

            try:
                overrides = [
                    {"SubnetId": sid, "InstanceType": inst.instance_type, "ImageId": inst.ami}
                    for sid in target_subnets
                    for inst in instances
                ]

                fleet_response = await ec2.create_fleet(
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
                    SpotOptions={
                        "AllocationStrategy": strategy,
                        "SingleAvailabilityZone": True,
                        "SingleInstanceType": True,
                    },
                    OnDemandOptions={
                        "AllocationStrategy": "lowest-price",
                        "SingleAvailabilityZone": True,
                        "SingleInstanceType": True,
                    },
                )

                instance_ids: list[str] = []
                for instance_set in fleet_response.get("Instances", []):
                    instance_ids.extend(instance_set.get("InstanceIds", []))

                errors = fleet_response.get("Errors", [])
                if errors and not instance_ids:
                    msgs = [f"{e.get('ErrorCode')}: {e.get('ErrorMessage')}" for e in errors]
                    raise RuntimeError(f"Fleet failed: {'; '.join(msgs)}")

                if len(instance_ids) < n:
                    raise RuntimeError(f"Fleet launched {len(instance_ids)}/{n}. Errors: {errors}")

                return instance_ids

            finally:
                with suppress(Exception):
                    await ec2.delete_launch_template(LaunchTemplateId=template_id)

    async def _get_valid_subnets(
        self,
        ec2: Any,
        instance_types: tuple[str, ...],
        subnet_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Filter subnets to AZs that support the instance types."""
        response = await ec2.describe_instance_type_offerings(
            LocationType="availability-zone",
            Filters=[{"Name": "instance-type", "Values": list(instance_types)}],
        )
        valid_azs = {o["Location"] for o in response.get("InstanceTypeOfferings", [])}

        subnets = await ec2.describe_subnets(SubnetIds=list(subnet_ids))
        valid = [s["SubnetId"] for s in subnets["Subnets"] if s["AvailabilityZone"] in valid_azs]

        if not valid:
            raise RuntimeError(f"No AZs support {instance_types}")

        return tuple(valid)

    # -------------------------------------------------------------------------
    # Instance Management
    # -------------------------------------------------------------------------

    async def _wait_running(self, instance_ids: list[str], timeout: float = 300) -> None:
        """Wait for instances to reach running state."""
        if not instance_ids:
            return

        from skyward.providers.wait import wait_for_ready

        async def poll_instances() -> list[dict[str, str]] | None:
            from botocore.exceptions import ClientError

            async with self.ec2() as ec2:
                try:
                    response = await ec2.describe_instances(InstanceIds=instance_ids)
                except ClientError as e:
                    # Handle AWS eventual consistency - instance may not be visible yet
                    if e.response.get("Error", {}).get("Code") == "InvalidInstanceID.NotFound":
                        logger.debug(f"Instance not found yet (eventual consistency): {instance_ids}")
                        return None  # Will retry on next poll interval
                    raise
                return [
                    {"id": inst["InstanceId"], "state": inst["State"]["Name"]}
                    for r in response["Reservations"]
                    for inst in r["Instances"]
                ]

        def all_running(instances: list[dict[str, str]]) -> bool:
            return all(i["state"] == "running" for i in instances)

        def any_terminal(instances: list[dict[str, str]]) -> bool:
            terminal_states = {"terminated", "shutting-down"}
            return any(i["state"] in terminal_states for i in instances)

        await wait_for_ready(
            poll_fn=poll_instances,
            ready_check=all_running,
            terminal_check=any_terminal,
            timeout=timeout,
            interval=5.0,
            description=f"EC2 instances {instance_ids}",
        )

    @retry(on=on_exception_message("InvalidInstanceID.NotFound"), max_attempts=5, base_delay=2.0)
    async def _get_instance_details(self, instance_ids: list[str]) -> list[dict[str, Any]]:
        """Get instance details."""
        if not instance_ids:
            return []

        async with self.ec2() as ec2:
            response = await ec2.describe_instances(InstanceIds=instance_ids)

            instances = []
            for r in response["Reservations"]:
                for i in r["Instances"]:
                    instances.append({
                        "id": i["InstanceId"],
                        "private_ip": i.get("PrivateIpAddress", ""),
                        "public_ip": i.get("PublicIpAddress"),
                        "spot": i.get("InstanceLifecycle") == "spot",
                        "instance_type": i.get("InstanceType", ""),
                    })

            return sorted(instances, key=lambda x: x["id"])

    async def _terminate_instances(self, instance_ids: list[InstanceId]) -> None:
        """Terminate EC2 instances."""
        if not instance_ids:
            return

        async with self.ec2() as ec2:
            await ec2.terminate_instances(InstanceIds=list(instance_ids))

    async def _wait_for_bootstrap_event(
        self,
        info: Any,
        timeout: float = 600.0,
    ) -> None:
        """Wait for bootstrap completion via BootstrapPhase event."""
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[bool] = loop.create_future()
        self._bootstrap_waiters[info.id] = waiter

        try:
            success = await asyncio.wait_for(waiter, timeout=timeout)
            if not success:
                raise RuntimeError(f"Bootstrap failed on {info.id}")
        finally:
            self._bootstrap_waiters.pop(info.id, None)

    async def _install_local_skyward(
        self,
        info: Any,
        cluster: AWSClusterState,
    ) -> None:
        """Install local skyward wheel."""
        from skyward.providers.bootstrap import install_local_skyward, wait_for_ssh

        transport = await wait_for_ssh(
            host=info.ip,
            user=cluster.username,
            key_path=cluster.ssh_key_path,
            timeout=60.0,
            log_prefix="AWS: ",
        )

        try:
            await install_local_skyward(
                transport=transport,
                info=info,
                log_prefix="AWS: ",
            )
        finally:
            await transport.close()

    async def _ensure_key_pair(self) -> tuple[str, str]:
        """Import SSH key pair to AWS if not exists."""
        import hashlib
        from pathlib import Path

        pub_key_paths = [
            Path.home() / ".ssh" / "id_ed25519.pub",
            Path.home() / ".ssh" / "id_rsa.pub",
            Path.home() / ".ssh" / "id_ecdsa.pub",
        ]

        public_key = None
        private_key_path = None
        for pub_path in pub_key_paths:
            if pub_path.exists():
                public_key = pub_path.read_text().strip()
                private_key_path = str(pub_path.with_suffix(""))
                break

        if not public_key or not private_key_path:
            raise RuntimeError("No SSH public key found. Create with: ssh-keygen -t ed25519")

        fingerprint = hashlib.md5(public_key.encode()).hexdigest()[:12]
        key_name = f"skyward-{fingerprint}"

        async with self.ec2() as ec2:
            try:
                response = await ec2.describe_key_pairs(KeyNames=[key_name])
                if response.get("KeyPairs"):
                    return key_name, private_key_path
            except Exception as e:
                if "InvalidKeyPair.NotFound" not in str(e):
                    raise
            await ec2.import_key_pair(
                KeyName=key_name,
                PublicKeyMaterial=public_key.encode(),
            )

        return key_name, private_key_path


__all__ = ["AWSHandler"]
