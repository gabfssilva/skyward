from __future__ import annotations

import asyncio
import base64
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from typing import cast

from skyward.api import PoolSpec
from skyward.api.model import Cluster, Instance, InstanceStatus, InstanceType, Offer
from skyward.api.spec import Architecture as SpecArchitecture
from skyward.infra.cache import cached
from skyward.infra.retry import on_exception_message, retry
from skyward.observability.logger import logger
from skyward.providers.provider import Provider

from .clients import EC2ClientFactory
from .config import AWS, AllocationStrategy
from .instances import get_instance_spec

type Architecture = str

log = logger.bind(provider="aws")


@dataclass(frozen=True, slots=True)
class AWSOfferSpecific:
    ami: str


@dataclass(frozen=True, slots=True)
class AWSSpecific:
    resources: AWSResources
    ssh_key_name: str


@dataclass(frozen=True, slots=True)
class AWSResources:
    instance_profile_arn: str
    security_group_id: str
    subnet_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _InstanceConfig:
    instance_type: str
    ami: str
    spot: bool = True


@dataclass(frozen=True, slots=True)
class _InstanceDetail:
    id: str
    private_ip: str
    public_ip: str | None
    spot: bool
    instance_type: str
    state: str


class AWSProvider(Provider[AWS, AWSSpecific]):
    """Stateless AWS provider. Holds only immutable config + client factory."""

    def __init__(self, config: AWS, ec2: EC2ClientFactory) -> None:
        self._config = config
        self._ec2 = ec2

    @classmethod
    async def create(cls, config: AWS) -> AWSProvider:
        from contextlib import asynccontextmanager

        import aioboto3  # type: ignore[reportMissingImports]

        @asynccontextmanager
        async def ec2_factory() -> AsyncIterator[object]:
            session = aioboto3.Session(region_name=config.region)
            async with session.client("ec2", region_name=config.region) as ec2:  # type: ignore[reportGeneralTypeIssues]
                yield ec2

        return cls(config, EC2ClientFactory(ec2_factory))

    async def offers(self, spec: PoolSpec) -> AsyncIterator[Offer]:
        from skyward.accelerators import Accelerator

        from .instances import get_instance_resources

        candidates = await _select_instances(self._config, spec)

        if spec.allocation in ("spot", "spot-if-available"):
            scores = await _get_spot_capacity_scores(
                self._ec2,
                [itype for itype, _ in candidates],
                self._config.region,
                spec.nodes,
            )
            if scores is not None:
                viable = {t for t, score in scores.items() if score >= 3}
                excluded = [t for t, _ in candidates if t not in viable]
                if excluded:
                    log.info(
                        "Excluded {n} types with low spot capacity: {types}",
                        n=len(excluded), types=excluded,
                    )
                candidates = [(t, a) for t, a in candidates if t in viable]

        if not candidates:
            return

        ami_lookup = _get_dlami if spec.accelerator_name is not None else _get_ubuntu_ami
        if self._config.ami:
            ami_by_arch: dict[str, str] = {arch: self._config.ami for _, arch in candidates}
        else:
            unique_archs: list[str] = list({arch for _, arch in candidates})
            ami_results = await asyncio.gather(*(ami_lookup(self._config, a) for a in unique_archs))
            ami_by_arch = dict(zip(unique_archs, ami_results, strict=True))

        resource_specs = await asyncio.gather(*(
            get_instance_resources(itype, self._config.region) for itype, _ in candidates
        ))
        price_specs = await asyncio.gather(*(
            get_instance_spec(itype, self._config.region) for itype, _ in candidates
        ))

        for (itype, arch), resources, prices in zip(
            candidates, resource_specs, price_specs, strict=True,
        ):
            ami = ami_by_arch[arch]

            accelerator: Accelerator | None = None
            if resources and resources.gpu_count > 0 and resources.gpu_model:
                accelerator = Accelerator(
                    name=resources.gpu_model,
                    count=resources.gpu_count,
                )

            it = InstanceType(
                name=itype,
                accelerator=accelerator,
                vcpus=float(resources.vcpus) if resources else 0.0,
                memory_gb=resources.memory_gb if resources else 0.0,
                architecture=cast(SpecArchitecture, arch),
                specific=None,
            )

            yield Offer(
                id=f"aws-{self._config.region}-{itype}",
                instance_type=it,
                spot_price=prices.spot_price if prices else None,
                on_demand_price=prices.ondemand_price if prices else None,
                billing_unit="second",
                specific=AWSOfferSpecific(ami=ami),
            )

    async def prepare(self, spec: PoolSpec, offer: Offer) -> Cluster[AWSSpecific]:
        log.info("Setting up AWS infrastructure")
        resources, (ssh_key_name, ssh_key_path) = await asyncio.gather(
            _ensure_infrastructure(self._config, self._ec2),
            _ensure_key_pair(self._config, self._ec2),
        )
        log.info(
            "Infrastructure ready: sg={sg}, subnets={n}",
            sg=resources.security_group_id, n=len(resources.subnet_ids),
        )

        return Cluster(
            id=f"aws-{uuid.uuid4().hex[:8]}",
            status="setting_up",
            spec=spec,
            offer=offer,
            ssh_key_path=ssh_key_path,
            ssh_user=self._config.username or "ubuntu",
            use_sudo=True,
            shutdown_command="shutdown -h now",
            specific=AWSSpecific(
                resources=resources,
                ssh_key_name=ssh_key_name,
            ),
        )

    async def provision(
        self, cluster: Cluster[AWSSpecific], count: int,
    ) -> tuple[Cluster[AWSSpecific], Sequence[Instance]]:
        offer = cluster.offer
        spot = cluster.spec.allocation in ("spot", "spot-if-available")

        instance_configs = (_InstanceConfig(
            instance_type=offer.instance_type.name,
            ami=offer.specific.ami,
            spot=spot,
        ),)

        ttl = cluster.spec.ttl or self._config.instance_timeout
        user_data = _self_destruction_script(ttl, cluster.shutdown_command)

        instance_ids = await _launch_fleet(
            config=self._config,
            ec2=self._ec2,
            resources=cluster.specific.resources,
            ssh_key_name=cluster.specific.ssh_key_name,
            cluster_id=cluster.id,
            instances=instance_configs,
            user_data=user_data,
            n=count,
        )

        details = await _get_instance_details(self._ec2, instance_ids)
        detail_map = {d.id: d for d in details}

        instances: list[Instance] = []
        for iid in instance_ids:
            detail = detail_map.get(iid)
            spot_actual = detail.spot if detail else spot
            instances.append(Instance(
                id=iid,
                status="provisioning",
                offer=offer,
                spot=spot_actual,
                region=self._config.region,
            ))
        return cluster, instances

    async def get_instance(
        self, cluster: Cluster[AWSSpecific], instance_id: str,
    ) -> tuple[Cluster[AWSSpecific], Instance | None]:
        details = await _get_instance_details(self._ec2, [instance_id])
        if not details:
            return cluster, None

        detail = details[0]
        match detail.state:
            case "terminated" | "shutting-down":
                return cluster, None
            case "running":
                return cluster, _build_instance(
                    detail, status="provisioned",
                    offer=cluster.offer,
                    region=self._config.region,
                )
            case _:
                return cluster, _build_instance(
                    detail, status="provisioning",
                    offer=cluster.offer,
                    region=self._config.region,
                )

    async def terminate(
        self, cluster: Cluster[AWSSpecific], instance_ids: tuple[str, ...],
    ) -> Cluster[AWSSpecific]:
        if not instance_ids:
            return cluster
        await _terminate_instances(self._ec2, list(instance_ids))
        return cluster

    async def teardown(self, cluster: Cluster[AWSSpecific]) -> Cluster[AWSSpecific]:
        return cluster


def _self_destruction_script(ttl: int, shutdown_command: str) -> str:
    from skyward.providers.bootstrap.compose import resolve
    from skyward.providers.bootstrap.ops import instance_timeout

    lines = ["#!/bin/bash", "set -e"]
    if ttl:
        lines.append(resolve(instance_timeout(ttl, shutdown_command=shutdown_command)))
    return "\n".join(lines) + "\n"


async def _ensure_infrastructure(
    config: AWS, ec2: EC2ClientFactory, prefix: str = "skyward",
) -> AWSResources:
    if config.security_group_id:
        subnet_ids = await _get_default_subnets(ec2)
        security_group_id = config.security_group_id
    else:
        subnet_ids, security_group_id = await asyncio.gather(
            _get_default_subnets(ec2),
            _ensure_security_group(ec2, prefix),
        )

    return AWSResources(
        instance_profile_arn=config.instance_profile_arn or "",
        security_group_id=security_group_id,
        subnet_ids=subnet_ids,
    )


async def _get_default_subnets(ec2: EC2ClientFactory) -> tuple[str, ...]:
    async with ec2() as client:
        vpcs = await client.describe_vpcs(
            Filters=[{"Name": "is-default", "Values": ["true"]}],
        )
        if not vpcs["Vpcs"]:
            raise RuntimeError("No default VPC found")

        vpc_id = vpcs["Vpcs"][0]["VpcId"]
        log.debug("Found default VPC {vpc}", vpc=vpc_id)
        subnets = await client.describe_subnets(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}],
        )
        if not subnets["Subnets"]:
            raise RuntimeError("No subnets found in default VPC")

        subnet_ids = tuple(s["SubnetId"] for s in subnets["Subnets"])
        log.debug("Discovered {n} subnets in VPC {vpc}", n=len(subnet_ids), vpc=vpc_id)
        return subnet_ids


async def _ensure_security_group(ec2: EC2ClientFactory, prefix: str) -> str:
    sg_name = f"{prefix}-sg"

    async with ec2() as client:
        try:
            resp = await client.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": [sg_name]}],
            )
            if resp["SecurityGroups"]:
                sg_id = resp["SecurityGroups"][0]["GroupId"]
                log.info("Found existing security group {sg}", sg=sg_id)
                await _ensure_sg_rules(client, sg_id)
                return sg_id
        except Exception:
            pass

        vpcs = await client.describe_vpcs(
            Filters=[{"Name": "is-default", "Values": ["true"]}],
        )
        if not vpcs["Vpcs"]:
            raise RuntimeError("No default VPC found")

        resp = await client.create_security_group(
            GroupName=sg_name,
            Description="Skyward EC2 worker security group",
            VpcId=vpcs["Vpcs"][0]["VpcId"],
            TagSpecifications=[{
                "ResourceType": "security-group",
                "Tags": [
                    {"Key": "Name", "Value": sg_name},
                    {"Key": "skyward:managed", "Value": "true"},
                ],
            }],
        )
        sg_id = resp["GroupId"]
        log.info("Created security group {sg}", sg=sg_id)

        await client.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    "IpProtocol": "-1",
                    "UserIdGroupPairs": [
                        {"GroupId": sg_id, "Description": "All traffic from same SG"},
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


async def _ensure_sg_rules(client: object, sg_id: str) -> None:
    resp = await client.describe_security_groups(GroupIds=[sg_id])  # type: ignore[union-attr]
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
        await client.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=to_add)  # type: ignore[union-attr]


async def _ensure_key_pair(config: AWS, ec2: EC2ClientFactory) -> tuple[str, str]:
    import hashlib
    from pathlib import Path

    pub_key_paths = (
        Path.home() / ".ssh" / "id_ed25519.pub",
        Path.home() / ".ssh" / "id_rsa.pub",
        Path.home() / ".ssh" / "id_ecdsa.pub",
    )

    def _find_ssh_key(paths: tuple[Path, ...]) -> tuple[str, str]:
        pub_path = next((p for p in paths if p.exists()), None)
        if pub_path is None:
            raise RuntimeError("No SSH public key found. Create with: ssh-keygen -t ed25519")
        return pub_path.read_text().strip(), str(pub_path.with_suffix(""))

    public_key, private_key_path = await asyncio.to_thread(_find_ssh_key, pub_key_paths)

    fingerprint = hashlib.md5(public_key.encode()).hexdigest()[:12]
    key_name = f"skyward-{fingerprint}"

    async with ec2() as client:
        try:
            response = await client.describe_key_pairs(KeyNames=[key_name])
            if response.get("KeyPairs"):
                log.info("Found existing SSH key pair {name}", name=key_name)
                return key_name, private_key_path
        except Exception as e:
            if "InvalidKeyPair.NotFound" not in str(e):
                raise
        await client.import_key_pair(
            KeyName=key_name,
            PublicKeyMaterial=public_key.encode(),
        )
        log.info("Imported SSH key pair {name}", name=key_name)

    return key_name, private_key_path


def _select_instances_cache_key(config: AWS, spec: PoolSpec, max_candidates: int = 5) -> str:
    import hashlib

    parts = (
        config.region,
        config.exclude_burstable,
        spec.vcpus,
        spec.memory_gb,
        spec.architecture,
        spec.accelerator_name,
        spec.accelerator_count,
        max_candidates,
    )
    return hashlib.sha256(str(parts).encode()).hexdigest()[:16]


@cached(namespace="aws-instances", ttl=timedelta(hours=6), key_func=_select_instances_cache_key)
async def _select_instances(
    config: AWS, spec: PoolSpec, max_candidates: int = 5,
) -> list[tuple[str, Architecture]]:
    import aioboto3

    min_vcpus = spec.vcpus or 2
    min_memory_mib = int((spec.memory_gb or 1) * 1024)
    archs = [spec.architecture] if spec.architecture else ["x86_64", "arm64"]

    requirements: dict[str, object] = {
        "VCpuCount": {"Min": min_vcpus},
        "MemoryMiB": {"Min": min_memory_mib},
        "InstanceGenerations": ["current"],
        "BurstablePerformance": "excluded" if config.exclude_burstable else "included",
        "BareMetal": "excluded",
    }

    session = aioboto3.Session()
    async with session.client("ec2", region_name=config.region) as client:  # type: ignore[union-attr]

        async def _query_arch(a: str) -> dict[str, Architecture]:
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
            return {r["InstanceType"]: a for r in response.get("InstanceTypes", [])}

        arch_results = await asyncio.gather(*(_query_arch(a) for a in archs))
        candidate_arch: dict[str, Architecture] = {}
        for arch_map in arch_results:
            candidate_arch.update(arch_map)

        if not candidate_arch:
            return [_fallback_instance(spec)]

        offered = await _get_offered_types(client, list(candidate_arch.keys()))
        excluded = set(candidate_arch) - offered
        if excluded:
            log.debug(
                "Excluded {n} types not offered in any AZ: {types}",
                n=len(excluded), types=excluded,
            )
        candidate_arch = {t: a for t, a in candidate_arch.items() if t in offered}

        if not candidate_arch:
            log.warning("No instance types offered in any AZ for {region}", region=config.region)
            return [_fallback_instance(spec)]

        all_types = list(candidate_arch.keys())

        async def _query_prices(batch: list[str]) -> dict[str, float]:
            prices = await client.describe_spot_price_history(
                InstanceTypes=batch,
                ProductDescriptions=["Linux/UNIX"],
            )
            batch_prices: dict[str, float] = {}
            for p in prices.get("SpotPriceHistory", []):
                itype = p["InstanceType"]
                price = float(p["SpotPrice"])
                if itype not in batch_prices or price < batch_prices[itype]:
                    batch_prices[itype] = price
            return batch_prices

        batches = [all_types[i:i + 20] for i in range(0, len(all_types), 20)]
        price_results = await asyncio.gather(*(_query_prices(b) for b in batches))
        price_map: dict[str, float] = {}
        for batch_prices in price_results:
            for itype, price in batch_prices.items():
                if itype not in price_map or price < price_map[itype]:
                    price_map[itype] = price

        if price_map:
            sorted_types = sorted(price_map.keys(), key=lambda k: price_map[k])[:max_candidates]
            prices_str = ", ".join(f"{t}=${price_map[t]:.4f}" for t in sorted_types)
            log.info("Instance candidates: {prices}", prices=prices_str)
            return [(t, candidate_arch[t]) for t in sorted_types]

        return [(t, candidate_arch[t]) for t in all_types[:max_candidates]]


async def _get_offered_types(client: object, instance_types: list[str]) -> set[str]:
    offered: set[str] = set()
    batches = [instance_types[i:i + 100] for i in range(0, len(instance_types), 100)]
    for batch in batches:
        response = await client.describe_instance_type_offerings(  # type: ignore[union-attr]
            LocationType="availability-zone",
            Filters=[{"Name": "instance-type", "Values": batch}],
        )
        offered.update(o["InstanceType"] for o in response.get("InstanceTypeOfferings", []))
    return offered


async def _get_spot_capacity_scores(
    ec2: EC2ClientFactory,
    instance_types: list[str],
    region: str,
    target_capacity: int,
) -> dict[str, int] | None:
    """Query spot placement scores per instance type. Returns None if the API is unavailable."""
    async def _score_type(client: object, itype: str) -> tuple[str, int]:
        response = await client.get_spot_placement_scores(  # type: ignore[union-attr]
            InstanceTypes=[itype],
            TargetCapacity=target_capacity,
            TargetCapacityUnitType="units",
            SingleAvailabilityZone=True,
            RegionNames=[region],
        )
        best = max(
            (s["Score"] for s in response.get("SpotPlacementScores", [])),
            default=0,
        )
        return itype, best

    try:
        async with ec2() as client:
            results = await asyncio.gather(*(_score_type(client, t) for t in instance_types))
            scores = dict(results)
            log.info(
                "Spot capacity scores: {scores}",
                scores=", ".join(f"{t}={s}/10" for t, s in scores.items()),
            )
            return scores
    except Exception as e:
        log.debug("GetSpotPlacementScores unavailable: {err}", err=e)
        return None


def _fallback_instance(spec: PoolSpec) -> tuple[str, Architecture]:
    gpu_fallbacks: dict[str, tuple[str, Architecture]] = {
        "T4": ("g4dn.xlarge", "x86_64"),
        "A10G": ("g5.xlarge", "x86_64"),
        "A100": ("p4d.24xlarge", "x86_64"),
        "H100": ("p5.48xlarge", "x86_64"),
    }

    match spec.accelerator_name:
        case str() as name if name in gpu_fallbacks:
            fallback = gpu_fallbacks[name]
        case str():
            fallback = ("g4dn.xlarge", "x86_64")
        case None:
            match spec.architecture:
                case "arm64":
                    fallback = ("m7g.large", "arm64")
                case "x86_64":
                    fallback = ("m5.large", "x86_64")
                case _:
                    fallback = ("m7g.large", "arm64")

    log.warning("No instances found for spec, using fallback {fallback}", fallback=fallback[0])
    return fallback


@cached(namespace="aws-amis", ttl=timedelta(hours=24))
async def _get_ubuntu_ami(config: AWS, arch: Architecture) -> str:
    import aioboto3

    arch_path = "arm64" if arch == "arm64" else "amd64"
    version = config.ubuntu_version
    ebs_type = "ebs-gp3" if version >= "24.04" else "ebs-gp2"
    param_name = (
        f"/aws/service/canonical/ubuntu/server/"
        f"{version}/stable/current/{arch_path}/hvm/"
        f"{ebs_type}/ami-id"
    )

    session = aioboto3.Session()
    async with session.client("ssm", region_name=config.region) as ssm:  # type: ignore[union-attr]
        response = await ssm.get_parameter(Name=param_name)
        ami = response["Parameter"]["Value"]
        log.debug("Resolved Ubuntu AMI {ami} for {arch}", ami=ami, arch=arch)
        return ami


@cached(namespace="aws-amis", ttl=timedelta(hours=24))
async def _get_dlami(config: AWS, arch: Architecture) -> str:
    import aioboto3

    arch_path = "arm64" if arch == "arm64" else "x86_64"
    version = config.ubuntu_version
    param_name = (
        f"/aws/service/deeplearning/ami/{arch_path}/"
        f"base-oss-nvidia-driver-gpu-ubuntu-{version}"
        f"/latest/ami-id"
    )

    session = aioboto3.Session()
    async with session.client("ssm", region_name=config.region) as ssm:  # type: ignore[union-attr]
        response = await ssm.get_parameter(Name=param_name)
        ami = response["Parameter"]["Value"]
        log.debug("Resolved DLAMI {ami} for {arch}", ami=ami, arch=arch)
        return ami


async def _launch_fleet(
    config: AWS,
    ec2: EC2ClientFactory,
    resources: AWSResources,
    ssh_key_name: str,
    cluster_id: str,
    instances: tuple[_InstanceConfig, ...],
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
            resources.subnet_ids,
        )

        template_name = f"skyward-{uuid.uuid4().hex[:8]}"
        template_data: dict[str, object] = {
            "UserData": base64.b64encode(user_data.encode()).decode(),
            "NetworkInterfaces": [{
                "DeviceIndex": 0,
                "AssociatePublicIpAddress": True,
                "Groups": [resources.security_group_id],
            }],
            "BlockDeviceMappings": [{
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": 100, "VolumeType": "gp3", "DeleteOnTermination": True},
            }],
            "TagSpecifications": [{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"skyward-{cluster_id}"},
                    {"Key": "skyward:managed", "Value": "true"},
                    {"Key": "skyward:cluster", "Value": cluster_id},
                ],
            }],
            "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
            "InstanceInitiatedShutdownBehavior": "terminate",
        }

        if ssh_key_name:
            template_data["KeyName"] = ssh_key_name

        if resources.instance_profile_arn:
            template_data["IamInstanceProfile"] = {"Arn": resources.instance_profile_arn}

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

            log.info("Submitting fleet request for {n} instances (spot={spot})", n=n, spot=spot)

            fleet_response = await client.create_fleet(
                Type="instant",
                LaunchTemplateConfigs=[{
                    "LaunchTemplateSpecification": {
                        "LaunchTemplateId": template_id,
                        "Version": "$Latest",
                    },
                    "Overrides": overrides,
                }],
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

            log.info(
                "Fleet launched {count} instances: {ids}",
                count=len(instance_ids), ids=instance_ids,
            )
            return instance_ids

        finally:
            with suppress(Exception):
                await client.delete_launch_template(LaunchTemplateId=template_id)


async def _get_valid_subnets(
    client: object,
    instance_types: tuple[str, ...],
    subnet_ids: tuple[str, ...],
) -> tuple[str, ...]:
    response = await client.describe_instance_type_offerings(  # type: ignore[union-attr]
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": list(instance_types)}],
    )
    valid_azs = {o["Location"] for o in response.get("InstanceTypeOfferings", [])}

    subnets = await client.describe_subnets(SubnetIds=list(subnet_ids))  # type: ignore[union-attr]
    valid = [s["SubnetId"] for s in subnets["Subnets"] if s["AvailabilityZone"] in valid_azs]

    if not valid:
        raise RuntimeError(f"No AZs support {instance_types}")

    return tuple(valid)


@retry(on=on_exception_message("InvalidInstanceID.NotFound"), max_attempts=5, base_delay=2.0)
async def _get_instance_details(
    ec2: EC2ClientFactory, instance_ids: list[str],
) -> list[_InstanceDetail]:
    if not instance_ids:
        return []

    log.debug("Polling instance details for {ids}", ids=instance_ids)
    async with ec2() as client:
        response = await client.describe_instances(InstanceIds=instance_ids)
        return sorted(
            [
                _InstanceDetail(
                    id=i["InstanceId"],
                    private_ip=i.get("PrivateIpAddress", ""),
                    public_ip=i.get("PublicIpAddress"),
                    spot=i.get("InstanceLifecycle") == "spot",
                    instance_type=i.get("InstanceType", ""),
                    state=i["State"]["Name"],
                )
                for r in response["Reservations"]
                for i in r["Instances"]
            ],
            key=lambda x: x.id,
        )


async def _terminate_instances(ec2: EC2ClientFactory, instance_ids: list[str]) -> None:
    if not instance_ids:
        return
    async with ec2() as client:
        await client.terminate_instances(InstanceIds=instance_ids)


def _build_instance(
    detail: _InstanceDetail, status: str, offer: Offer, region: str,
) -> Instance:
    return Instance(
        id=detail.id,
        status=cast(InstanceStatus, status),
        offer=offer,
        ip=detail.public_ip or detail.private_ip,
        private_ip=detail.private_ip,
        ssh_port=22,
        spot=detail.spot,
        region=region,
    )
