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

from skyward.v2.app import component, on
from skyward.v2.bus import AsyncEventBus
from skyward.v2.events import (
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
from skyward.v2.monitors import SSHCredentialsRegistry

from skyward.utils.pricing import get_instance_pricing

from .clients import (
    EC2ClientFactory,
    IAMClientFactory,
    S3ClientFactory,
    STSClientFactory,
)
from .config import AWS, AllocationStrategy
from .state import AWSClusterState, AWSResources, InstanceConfig

# GPU model mapping for AWS instance types
_GPU_MODELS = {
    "g4dn": "T4",
    "g5": "A10G",
    "g6": "L4",
    "p4d": "A100",
    "p4de": "A100",
    "p5": "H100",
}

# GPU VRAM mapping (GB per GPU)
_GPU_VRAM = {
    "T4": 16,
    "A10G": 24,
    "L4": 24,
    "A100": 40,  # 40GB variant (p4de uses 80GB)
    "H100": 80,
}

if TYPE_CHECKING:
    from skyward.v2.spec import PoolSpec


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
    s3: S3ClientFactory
    iam: IAMClientFactory
    sts: STSClientFactory
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
        """Provision AWS infrastructure for a new cluster."""
        resources = await self._ensure_infrastructure()
        ssh_key_name, ssh_key_path = await self._ensure_key_pair()

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

        # Register SSH credentials for EventStreamer
        self.ssh_credentials.register(cluster_id, state.username, state.ssh_key_path)

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
        """Launch EC2 instance via Fleet and emit InstanceLaunched."""
        cluster = self._clusters.get(event.cluster_id)
        if not cluster:
            return

        instance_config = await self._resolve_instance_config(cluster.spec)
        user_data = self._generate_user_data(cluster.spec)

        instance_ids = await self._launch_fleet(
            cluster=cluster,
            instances=(instance_config,),
            user_data=user_data,
            n=1,
        )

        if not instance_ids:
            logger.error(f"AWS: Failed to launch instance for node {event.node_id}")
            return

        instance_id = instance_ids[0]

        # Track pending
        cluster.pending_nodes.add(event.node_id)

        # Store instance config for later (needed in handle_instance_launched)
        if not hasattr(cluster, "_pending_instance_configs"):
            cluster._pending_instance_configs = {}  # type: ignore
        cluster._pending_instance_configs[instance_id] = instance_config  # type: ignore

        # Emit intermediate event
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
        instance_family = instance_type.split(".")[0] if "." in instance_type else ""
        gpu_model = _GPU_MODELS.get(instance_family, "")
        gpu_vram_gb = _GPU_VRAM.get(gpu_model, 0)

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

    async def _resolve_instance_config(self, spec: PoolSpec) -> InstanceConfig:
        """Resolve instance type and AMI from spec."""
        accelerator_map = {
            "T4": "g4dn.xlarge",
            "A10G": "g5.xlarge",
            "A100": "p4d.24xlarge",
            "H100": "p5.48xlarge",
        }
        instance_type = accelerator_map.get(spec.accelerator_name, "g4dn.xlarge")

        ami = self.config.ami or await self._get_dlami()
        spot = spec.allocation in ("spot", "spot-if-available")

        return InstanceConfig(instance_type=instance_type, ami=ami, spot=spot)

    async def _get_dlami(self) -> str:
        """Get Deep Learning AMI for region via SSM Parameter Store."""
        import aioboto3

        param_name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"

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
        return spec.image.generate_bootstrap(ttl=spec.ttl, use_systemd=True)

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

        async with self.ec2() as ec2:
            start = asyncio.get_event_loop().time()
            while True:
                response = await ec2.describe_instances(InstanceIds=instance_ids)

                all_running = all(
                    inst["State"]["Name"] == "running"
                    for r in response["Reservations"]
                    for inst in r["Instances"]
                )

                if all_running:
                    return

                if asyncio.get_event_loop().time() - start > timeout:
                    raise TimeoutError(f"Instances not running after {timeout}s")

                await asyncio.sleep(5)

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
        """Install local skyward wheel and start RPyC server."""
        from skyward.v2.providers.bootstrap import install_local_skyward, wait_for_ssh

        env = cluster.spec.image.env if cluster.spec.image else None

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
                env=env,
                use_systemd=True,
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
