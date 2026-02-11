"""AWS Provider Actor - Casty behavior for cluster lifecycle.

Story: idle -> active -> stopped.

The actor receives ProviderMsg and communicates lifecycle events
via pool_ref. Observability is provided transparently via Behaviors.spy().
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from contextlib import suppress
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from casty import ActorContext, ActorRef, Behavior, Behaviors
from loguru import logger

from skyward.actors.messages import (
    BootstrapDone,
    BootstrapRequested,
    ClusterProvisioned,
    ClusterRequested,
    InstanceBootstrapped,
    InstanceId,
    InstanceMetadata,
    InstanceRequested,
    InstanceRunning,
    ProviderMsg,
    ShutdownCompleted,
    ShutdownRequested,
    _InstanceNowRunning,
    _InstanceWaitFailed,
    _LocalInstallDone,
    _LocalInstallFailed,
)
from skyward.actors.streaming import instance_monitor
from skyward.infra.pricing import get_instance_pricing
from skyward.infra.retry import on_exception_message, retry

from .clients import EC2ClientFactory
from .config import AWS, AllocationStrategy
from .state import AWSClusterState, AWSResources, InstanceConfig

if TYPE_CHECKING:
    from skyward.api.spec import Architecture, PoolSpec

_AWS_INSTANCE_GPU = {
    "g4dn": "T4",
    "g5": "A10G",
    "g6": "L4",
    "p4d": "A100",
    "p4de": "A100",
    "p5": "H100",
}


# =============================================================================
# Actor Behavior
# =============================================================================


def aws_provider_actor(
    config: AWS,
    ec2: EC2ClientFactory,
    pool_ref: ActorRef,
) -> Behavior[ProviderMsg]:
    """An AWS provider tells this story: idle -> active -> stopped."""
    log = logger.bind(provider="aws")

    def idle() -> Behavior[ProviderMsg]:
        async def receive(
            ctx: ActorContext[ProviderMsg], msg: ProviderMsg,
        ) -> Behavior[ProviderMsg]:
            match msg:
                case ClusterRequested(
                    request_id=request_id, provider="aws", spec=spec,
                ):
                    log.debug("Provisioning infrastructure, keys, and instance configs")
                    (
                        resources,
                        (ssh_key_name, ssh_key_path),
                        instance_configs,
                    ) = await asyncio.gather(
                        _ensure_infrastructure(config, ec2),
                        _ensure_key_pair(config, ec2),
                        _resolve_instance_configs(config, ec2, spec),
                    )

                    cluster_id = f"aws-{uuid.uuid4().hex[:8]}"
                    username = config.username or "ubuntu"
                    user_data = _generate_user_data(config, spec)

                    state = AWSClusterState(
                        cluster_id=cluster_id,
                        spec=spec,
                        resources=resources,
                        region=config.region,
                        ssh_key_name=ssh_key_name,
                        ssh_key_path=ssh_key_path,
                        username=username,
                    )

                    instance_ids = await _launch_fleet(
                        config=config,
                        ec2=ec2,
                        cluster=state,
                        instances=tuple(instance_configs),
                        user_data=user_data,
                        n=spec.nodes,
                    )

                    fleet_ids = MappingProxyType(dict(enumerate(instance_ids)))
                    state = replace(state, fleet_instance_ids=fleet_ids)

                    provisioned = ClusterProvisioned(
                        request_id=request_id,
                        cluster_id=cluster_id,
                        provider="aws",
                    )
                    pool_ref.tell(provisioned)

                    return active(state=state)

                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    def active(state: AWSClusterState) -> Behavior[ProviderMsg]:
        async def receive(
            ctx: ActorContext[ProviderMsg], msg: ProviderMsg,
        ) -> Behavior[ProviderMsg]:
            match msg:
                case InstanceRequested(
                    request_id=request_id,
                    provider="aws",
                    cluster_id=cluster_id,
                    node_id=node_id,
                    replacing=replacing,
                ) if cluster_id == state.cluster_id:
                    pre_assigned = state.fleet_instance_ids.get(node_id)
                    new_fleet = MappingProxyType({
                        k: v
                        for k, v in state.fleet_instance_ids.items()
                        if k != node_id
                    })

                    if replacing is None and pre_assigned:
                        instance_id = pre_assigned
                    else:
                        instance_configs = await _resolve_instance_configs(config, ec2, state.spec)
                        user_data = _generate_user_data(config, state.spec)

                        instance_ids = await _launch_fleet(
                            config=config,
                            ec2=ec2,
                            cluster=state,
                            instances=tuple(instance_configs),
                            user_data=user_data,
                            n=1,
                        )

                        if not instance_ids:
                            log.error(
                                "Failed to launch instance for node {node_id}",
                                node_id=node_id,
                            )
                            return Behaviors.same()

                        instance_id = instance_ids[0]

                    new_state = replace(
                        state,
                        pending_nodes=state.pending_nodes | {node_id},
                        fleet_instance_ids=new_fleet,
                    )

                    ctx.pipe_to_self(
                        coro=_wait_and_build_running(
                            config=config,
                            ec2=ec2,
                            state=new_state,
                            request_id=request_id,
                            cluster_id=cluster_id,
                            node_id=node_id,
                            instance_id=instance_id,
                        ),
                        mapper=lambda event: _InstanceNowRunning(event=event),
                        on_failure=lambda e: _InstanceWaitFailed(
                            instance_id=instance_id,
                            node_id=node_id,
                            error=str(e),
                        ),
                    )

                    return active(state=new_state)

                case BootstrapRequested(
                    instance=info,
                    cluster_id=cluster_id,
                ) if cluster_id == state.cluster_id:
                    ctx.spawn(
                        instance_monitor(
                            info=info,
                            ssh_user=state.username,
                            ssh_key_path=state.ssh_key_path,
                            pool_ref=pool_ref,
                            reply_to=ctx.self,
                        ),
                        f"monitor-{info.id}",
                    )
                    return Behaviors.same()

                case BootstrapDone(instance=info, success=True):
                    if state.spec.image and state.spec.image.skyward_source == "local":
                        ctx.pipe_to_self(
                            coro=_install_local_skyward(info, state),
                            mapper=lambda _, i=info: _LocalInstallDone(instance=i),
                            on_failure=lambda e, i=info: _LocalInstallFailed(
                                instance=i, error=str(e),
                            ),
                        )
                        return Behaviors.same()

                    new_state = replace(state,
                        instances=MappingProxyType({**state.instances, info.id: info}),
                        pending_nodes=state.pending_nodes - {info.node},
                    )
                    pool_ref.tell(InstanceBootstrapped(instance=info))
                    return active(state=new_state)

                case BootstrapDone(instance=info, success=False, error=error):
                    log.error("Bootstrap failed on {iid}: {error}", iid=info.id, error=error)
                    return Behaviors.same()

                case _LocalInstallDone(instance=info):
                    new_state = replace(state,
                        instances=MappingProxyType({**state.instances, info.id: info}),
                        pending_nodes=state.pending_nodes - {info.node},
                    )
                    pool_ref.tell(InstanceBootstrapped(instance=info))
                    return active(state=new_state)

                case _LocalInstallFailed(instance=info, error=error):
                    log.error(
                        "Local skyward install failed on {iid}: {error}",
                        iid=info.id, error=error,
                    )
                    return Behaviors.same()

                case _InstanceNowRunning(event=event):
                    pool_ref.tell(event)
                    return Behaviors.same()

                case _InstanceWaitFailed(
                    instance_id=iid, node_id=nid, error=error,
                ):
                    log.error(
                        "Instance {iid} (node {nid}) failed to reach running state: {error}",
                        iid=iid, nid=nid, error=error,
                    )
                    return Behaviors.same()

                case ShutdownRequested(
                    cluster_id=cluster_id, reply_to=reply_to,
                ) if cluster_id == state.cluster_id:
                    all_ids: set[str] = set(state.instances.keys())
                    all_ids.update(state.fleet_instance_ids.values())
                    if all_ids:
                        await _terminate_instances(ec2, list(all_ids))

                    if reply_to is not None:
                        reply_to.tell(ShutdownCompleted(cluster_id=cluster_id))
                    return Behaviors.stopped()

                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()


# =============================================================================
# Background Tasks
# =============================================================================


async def _wait_and_build_running(
    config: AWS,
    ec2: EC2ClientFactory,
    state: AWSClusterState,
    request_id: str,
    cluster_id: str,
    node_id: int,
    instance_id: str,
) -> InstanceRunning:
    await _wait_running(ec2, [instance_id])

    details = await _get_instance_details(ec2, [instance_id])
    if not details:
        raise RuntimeError(f"Could not get details for {instance_id}")

    detail = details[0]
    instance_type = detail.get("instance_type", "")
    is_spot = detail["spot"]

    hourly_rate = 0.0
    on_demand_rate = 0.0
    gpu_count = 0
    gpu_model = ""
    vcpus = 0
    memory_gb = 0.0

    pricing = get_instance_pricing(instance_type, "aws", config.region)
    if pricing:
        on_demand_rate = pricing.ondemand or 0.0
        hourly_rate = (pricing.spot_avg if is_spot and pricing.spot_avg else on_demand_rate)
        gpu_count = pricing.gpu_count
        vcpus = pricing.vcpu
        memory_gb = pricing.memory_gb

    from skyward.accelerators.catalog import get_gpu_vram_gb

    instance_family = instance_type.split(".")[0] if "." in instance_type else ""
    gpu_model = _AWS_INSTANCE_GPU.get(instance_family, "")
    gpu_vram_gb = get_gpu_vram_gb(gpu_model)

    return InstanceRunning(
        request_id=request_id,
        cluster_id=cluster_id,
        node_id=node_id,
        provider="aws",
        instance_id=instance_id,
        ip=detail["public_ip"] or detail["private_ip"],
        private_ip=detail["private_ip"],
        ssh_port=22,
        spot=is_spot,
        hourly_rate=hourly_rate,
        on_demand_rate=on_demand_rate,
        billing_increment=1,
        instance_type=instance_type,
        gpu_count=gpu_count,
        gpu_model=gpu_model,
        vcpus=vcpus,
        memory_gb=memory_gb,
        gpu_vram_gb=gpu_vram_gb,
        region=config.region,
    )


# =============================================================================
# Infrastructure Management
# =============================================================================


async def _ensure_infrastructure(
    config: AWS,
    ec2: EC2ClientFactory,
    prefix: str = "skyward",
) -> AWSResources:
    subnet_ids = await _get_default_subnets(ec2)

    if config.security_group_id:
        security_group_id = config.security_group_id
    else:
        security_group_id = await _ensure_security_group(ec2, prefix)

    instance_profile_arn = config.instance_profile_arn or ""

    return AWSResources(
        bucket="",
        iam_role_arn="",
        instance_profile_arn=instance_profile_arn,
        security_group_id=security_group_id,
        region=config.region,
        subnet_ids=subnet_ids,
    )


async def _get_default_subnets(ec2: EC2ClientFactory) -> tuple[str, ...]:
    async with ec2() as client:
        vpcs = await client.describe_vpcs(
            Filters=[{"Name": "is-default", "Values": ["true"]}]
        )
        if not vpcs["Vpcs"]:
            raise RuntimeError("No default VPC found")

        vpc_id = vpcs["Vpcs"][0]["VpcId"]
        subnets = await client.describe_subnets(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
        )

        if not subnets["Subnets"]:
            raise RuntimeError("No subnets found in default VPC")

        return tuple(s["SubnetId"] for s in subnets["Subnets"])


async def _ensure_security_group(ec2: EC2ClientFactory, prefix: str) -> str:
    sg_name = f"{prefix}-sg"

    async with ec2() as client:
        try:
            resp = await client.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": [sg_name]}]
            )
            if resp["SecurityGroups"]:
                sg_id = resp["SecurityGroups"][0]["GroupId"]
                await _ensure_sg_rules(client, sg_id)
                return sg_id
        except Exception:
            pass

        vpcs = await client.describe_vpcs(
            Filters=[{"Name": "is-default", "Values": ["true"]}]
        )
        if not vpcs["Vpcs"]:
            raise RuntimeError("No default VPC found")
        vpc_id = vpcs["Vpcs"][0]["VpcId"]

        resp = await client.create_security_group(
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

        await client.authorize_security_group_ingress(
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


async def _ensure_sg_rules(client: Any, sg_id: str) -> None:
    resp = await client.describe_security_groups(GroupIds=[sg_id])
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
        await client.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=to_add)


# =============================================================================
# Fleet Launch
# =============================================================================


async def _resolve_instance_configs(
    config: AWS,
    ec2: EC2ClientFactory,
    spec: PoolSpec,
) -> list[InstanceConfig]:
    spot = spec.allocation in ("spot", "spot-if-available")

    candidates = await _select_instances(config, spec)

    configs = []
    for instance_type, arch in candidates:
        if spec.accelerator_name is not None:
            ami = config.ami or await _get_dlami(config, arch)
        else:
            ami = config.ami or await _get_ubuntu_ami(config, arch)
        configs.append(InstanceConfig(instance_type=instance_type, ami=ami, spot=spot))

    return configs


async def _select_instances(
    config: AWS,
    spec: PoolSpec,
    max_candidates: int = 5,
) -> list[tuple[str, Architecture]]:
    import aioboto3  # type: ignore[reportMissingImports]

    min_vcpus = spec.vcpus or 2
    min_memory_mib = (spec.memory_gb or 8) * 1024
    archs = [spec.architecture] if spec.architecture else ["x86_64", "arm64"]

    requirements: dict[str, Any] = {
        "VCpuCount": {"Min": min_vcpus},
        "MemoryMiB": {"Min": min_memory_mib},
        "InstanceGenerations": ["current"],
        "BurstablePerformance": "excluded",
        "BareMetal": "excluded",
    }

    session = aioboto3.Session()
    async with session.client("ec2", region_name=config.region) as client:  # type: ignore[reportGeneralTypeIssues]
        candidate_arch: dict[str, Architecture] = {}

        for a in archs:
            arch_requirements = requirements.copy()

            if spec.accelerator_name:
                arch_requirements["AcceleratorTypes"] = ["gpu"]
                arch_requirements["AcceleratorNames"] = [spec.accelerator_name.lower()]
                arch_requirements["AcceleratorCount"] = {"Min": spec.accelerator_count or 1}

            response = await client.get_instance_types_from_instance_requirements(
                ArchitectureTypes=[a],
                VirtualizationTypes=["hvm"],
                InstanceRequirements=arch_requirements,
            )
            for r in response.get("InstanceTypes", []):
                candidate_arch[r["InstanceType"]] = a  # type: ignore[assignment]

        if not candidate_arch:
            return [_fallback_instance(spec)]

        all_types = list(candidate_arch.keys())
        price_map: dict[str, float] = {}
        for i in range(0, len(all_types), 20):
            batch = all_types[i:i + 20]
            prices = await client.describe_spot_price_history(
                InstanceTypes=batch,
                ProductDescriptions=["Linux/UNIX"],
            )
            for p in prices.get("SpotPriceHistory", []):
                itype = p["InstanceType"]
                price = float(p["SpotPrice"])
                if itype not in price_map or price < price_map[itype]:
                    price_map[itype] = price

        if price_map:
            sorted_types = sorted(price_map.keys(), key=lambda k: price_map[k])[:max_candidates]
            result: list[tuple[str, Architecture]] = [
                (t, candidate_arch[t]) for t in sorted_types
            ]
            prices_str = ", ".join(f"{t}=${price_map[t]:.4f}" for t in sorted_types)
            logger.info("Instance candidates: {prices}", prices=prices_str)
            return result

        result = [(t, candidate_arch[t]) for t in all_types[:max_candidates]]
        return result


def _fallback_instance(spec: PoolSpec) -> tuple[str, Architecture]:
    if spec.accelerator_name:
        gpu_fallbacks: dict[str, tuple[str, Architecture]] = {
            "T4": ("g4dn.xlarge", "x86_64"),
            "A10G": ("g5.xlarge", "x86_64"),
            "A100": ("p4d.24xlarge", "x86_64"),
            "H100": ("p5.48xlarge", "x86_64"),
        }
        fallback = gpu_fallbacks.get(spec.accelerator_name, ("g4dn.xlarge", "x86_64"))
    else:
        match spec.architecture:
            case "arm64":
                fallback = ("m7g.large", "arm64")
            case "x86_64":
                fallback = ("m5.large", "x86_64")
            case _:
                fallback = ("m7g.large", "arm64")

    logger.warning("No instances found for spec, using fallback {fallback}", fallback=fallback[0])
    return fallback


def _arch_from_instance_type(instance_type: str) -> Architecture:
    import re

    family = instance_type.split(".")[0]
    if re.match(r"^[a-z]+\d+g[den]*$", family):
        return "arm64"
    return "x86_64"


async def _get_ubuntu_ami(config: AWS, arch: Architecture) -> str:
    import aioboto3  # type: ignore[reportMissingImports]

    arch_path = "arm64" if arch == "arm64" else "amd64"
    version = config.ubuntu_version
    ebs_type = "ebs-gp3" if version >= "24.04" else "ebs-gp2"
    param_name = (
        f"/aws/service/canonical/ubuntu/server/"
        f"{version}/stable/current/{arch_path}/hvm/"
        f"{ebs_type}/ami-id"
    )

    session = aioboto3.Session()
    async with session.client("ssm", region_name=config.region) as ssm:  # type: ignore[reportGeneralTypeIssues]
        try:
            response = await ssm.get_parameter(Name=param_name)
            return response["Parameter"]["Value"]
        except Exception as e:
            raise RuntimeError(f"Failed to get Ubuntu AMI: {e}") from e


async def _get_dlami(config: AWS, arch: Architecture) -> str:
    import aioboto3  # type: ignore[reportMissingImports]

    arch_path = "arm64" if arch == "arm64" else "x86_64"
    version = config.ubuntu_version
    param_name = (
        f"/aws/service/deeplearning/ami/{arch_path}/"
        f"base-oss-nvidia-driver-gpu-ubuntu-{version}"
        f"/latest/ami-id"
    )

    session = aioboto3.Session()
    async with session.client("ssm", region_name=config.region) as ssm:  # type: ignore[reportGeneralTypeIssues]
        try:
            response = await ssm.get_parameter(Name=param_name)
            return response["Parameter"]["Value"]
        except Exception as e:
            raise RuntimeError(
                f"Could not find Deep Learning AMI in region {config.region}. "
                f"SSM parameter {param_name} not found. Error: {e}"
            ) from e


def _generate_user_data(config: AWS, spec: PoolSpec) -> str:
    ttl = spec.ttl or config.instance_timeout
    return spec.image.generate_bootstrap(ttl=ttl)


async def _launch_fleet(
    config: AWS,
    ec2: EC2ClientFactory,
    cluster: AWSClusterState,
    instances: tuple[InstanceConfig, ...],
    user_data: str,
    n: int,
    allocation_strategy: AllocationStrategy | None = None,
) -> list[str]:
    strategy = allocation_strategy or config.allocation_strategy
    spot = instances[0].spot if instances else False

    async with ec2() as client:
        target_subnets = await _get_valid_subnets(
            client,
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

        template = await client.create_launch_template(
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

            fleet_response = await client.create_fleet(
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
                    "MinTargetCapacity": n,
                },
                OnDemandOptions={
                    "AllocationStrategy": "lowest-price",
                    "SingleAvailabilityZone": True,
                    "SingleInstanceType": True,
                    "MinTargetCapacity": n,
                },
            )

            instance_ids = [
                iid
                for instance_set in fleet_response.get("Instances", [])
                for iid in instance_set.get("InstanceIds", [])
            ]

            errors = fleet_response.get("Errors", [])
            if errors and not instance_ids:
                msgs = [f"{e.get('ErrorCode')}: {e.get('ErrorMessage')}" for e in errors]
                raise RuntimeError(f"Fleet failed: {'; '.join(msgs)}")

            if len(instance_ids) < n:
                raise RuntimeError(f"Fleet launched {len(instance_ids)}/{n}. Errors: {errors}")

            return instance_ids

        finally:
            with suppress(Exception):
                await client.delete_launch_template(LaunchTemplateId=template_id)


async def _get_valid_subnets(
    client: Any,
    instance_types: tuple[str, ...],
    subnet_ids: tuple[str, ...],
) -> tuple[str, ...]:
    response = await client.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": list(instance_types)}],
    )
    valid_azs = {o["Location"] for o in response.get("InstanceTypeOfferings", [])}

    subnets = await client.describe_subnets(SubnetIds=list(subnet_ids))
    valid = [s["SubnetId"] for s in subnets["Subnets"] if s["AvailabilityZone"] in valid_azs]

    if not valid:
        raise RuntimeError(f"No AZs support {instance_types}")

    return tuple(valid)


# =============================================================================
# Instance Management
# =============================================================================


async def _wait_running(
    ec2: EC2ClientFactory,
    instance_ids: list[str],
    timeout: float = 300,
) -> None:
    if not instance_ids:
        return

    from skyward.providers.wait import wait_for_ready

    async def poll_instances() -> list[dict[str, str]] | None:
        from botocore.exceptions import ClientError  # type: ignore[reportMissingImports]

        async with ec2() as client:
            try:
                response = await client.describe_instances(InstanceIds=instance_ids)
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") == "InvalidInstanceID.NotFound":
                    logger.debug(
                        "Instance not found yet (eventual consistency): {ids}",
                        ids=instance_ids,
                    )
                    return None
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
async def _get_instance_details(
    ec2: EC2ClientFactory,
    instance_ids: list[str],
) -> list[dict[str, Any]]:
    if not instance_ids:
        return []

    async with ec2() as client:
        response = await client.describe_instances(InstanceIds=instance_ids)

        instances = [
            {
                "id": i["InstanceId"],
                "private_ip": i.get("PrivateIpAddress", ""),
                "public_ip": i.get("PublicIpAddress"),
                "spot": i.get("InstanceLifecycle") == "spot",
                "instance_type": i.get("InstanceType", ""),
            }
            for r in response["Reservations"]
            for i in r["Instances"]
        ]

        return sorted(instances, key=lambda x: x["id"])


async def _terminate_instances(ec2: EC2ClientFactory, instance_ids: list[InstanceId]) -> None:
    if not instance_ids:
        return

    async with ec2() as client:
        await client.terminate_instances(InstanceIds=list(instance_ids))



async def _install_local_skyward(
    info: InstanceMetadata,
    cluster: AWSClusterState,
) -> None:
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


async def _ensure_key_pair(config: AWS, ec2: EC2ClientFactory) -> tuple[str, str]:
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

    async with ec2() as client:
        try:
            response = await client.describe_key_pairs(KeyNames=[key_name])
            if response.get("KeyPairs"):
                return key_name, private_key_path
        except Exception as e:
            if "InvalidKeyPair.NotFound" not in str(e):
                raise
        await client.import_key_pair(
            KeyName=key_name,
            PublicKeyMaterial=public_key.encode(),
        )

    return key_name, private_key_path
