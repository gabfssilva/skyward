import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, NamedTuple, Self

import aioboto3
from aiobotocore.client import AioBaseClient
from botocore.exceptions import ClientError

from skyward.application.errors import CapabilityMismatchError
from skyward.application.provider import Binding, Machine
from skyward.protocol.schemas import ComputeSpec, Market, Offer

DEFAULT_REGIONS = (
    "us-east-1",
    "us-east-2",
    "us-west-2",
    "eu-west-1",
    "eu-central-1",
)

PRICING_REGION = "us-east-1"
MIB = 1024

COMPUTE_TAG = "skyward:compute"
MANAGED_TAG = "skyward:managed"
SSH_USER = "ubuntu"
DEFAULT_UBUNTU = "24.04"
DEFAULT_DISK_GB = 100
FLEET_STRATEGY = "price-capacity-optimized"


class AWSProvider:
    """AWS EC2 — a catalog, an on-demand price list, and a live spot market.

    Three different APIs, because AWS keeps the three in three different places:
    ``ec2:DescribeInstanceTypes`` is the hardware, ``pricing:GetProducts`` is the
    on-demand rate card, and ``ec2:DescribeSpotPriceHistory`` is the market. The
    Pricing API only answers in ``us-east-1`` and is queried with a ``regionCode``
    filter, so it is one client for every region we list.

    The TTL is driven by the volatile third: on-demand rates move about as often as
    AWS issues a press release, but the spot price of a given instance type moves
    within the hour. Thirty minutes keeps the spot column honest without paying for
    a full catalog re-walk on every question.
    """

    kind: ClassVar[str] = "aws"
    credential_fields: ClassVar[tuple[str, ...]] = ("access_key_id", "secret_access_key")
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=30)

    def __init__(
        self,
        provider_id: str,
        name: str,
        access_key_id: str,
        secret_access_key: str,
        session_token: str | None,
        config: Mapping[str, Any],
    ) -> None:
        self._id = provider_id
        self._name = name
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        access_key_id = credentials.get("access_key_id")
        secret_access_key = credentials.get("secret_access_key")
        if not access_key_id or not secret_access_key:
            raise CapabilityMismatchError("aws requires access_key_id and secret_access_key credentials", provider=name)
        return cls(provider_id, name, access_key_id, secret_access_key, credentials.get("session_token") or None, config)

    @property
    def _regions(self) -> tuple[str, ...]:
        return tuple(self._config.get("regions") or DEFAULT_REGIONS)

    def _session(self) -> aioboto3.Session:
        """Always the explicit credentials — never boto3's default chain.

        The default chain would fall back to the caller's environment and
        ``~/.aws``, which is exactly what makes two AWS accounts in one process
        impossible.
        """
        return aioboto3.Session(
            aws_access_key_id=self._access_key_id,
            aws_secret_access_key=self._secret_access_key,
            aws_session_token=self._session_token,
        )

    @asynccontextmanager
    async def _ec2(self, region: str):  # noqa: ANN202 — aioboto3's client type is unexpressible; let it infer
        async with self._session().client("ec2", region_name=region) as client:
            yield client

    async def offers(self) -> AsyncIterator[Offer]:
        regions = self._regions
        results = await asyncio.gather(*(self._fetch_region(region) for region in regions))

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for region, (instance_types, spot, on_demand) in zip(regions, results, strict=True):
            for raw in instance_types:
                instance_type = raw["InstanceType"]
                gpu = _gpu(raw)
                network = raw.get("NetworkInfo") or {}
                storage = raw.get("InstanceStorageInfo") or {}

                yield Offer(
                    id=f"aws-{region}-{instance_type}",
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    billing_unit="second",
                    instance_type=instance_type,
                    accelerator=gpu.name,
                    accelerator_count=gpu.count,
                    vram=gpu.vram,
                    cpus=int((raw.get("VCpuInfo") or {}).get("DefaultVCpus") or 0),
                    memory_gb=float((raw.get("MemoryInfo") or {}).get("SizeInMiB") or 0) / MIB,
                    region=region,
                    disk_gb=float(storage.get("TotalSizeInGB")) if storage.get("TotalSizeInGB") else None,
                    spot_price=spot.get(instance_type),
                    on_demand_price=on_demand.get(instance_type),
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "architectures": (raw.get("ProcessorInfo") or {}).get("SupportedArchitectures"),
                        "hypervisor": raw.get("Hypervisor"),
                        "bare_metal": raw.get("BareMetal"),
                        "current_generation": raw.get("CurrentGeneration"),
                        "burstable": raw.get("BurstablePerformanceSupported"),
                        "ena_support": network.get("EnaSupport"),
                        "efa_supported": network.get("EfaSupported"),
                        "network_performance": network.get("NetworkPerformance"),
                        "instance_storage_supported": raw.get("InstanceStorageSupported"),
                        "instance_storage_nvme": storage.get("NvmeSupport"),
                        "instance_storage_disks": storage.get("Disks"),
                        "gpu_manufacturer": gpu.manufacturer,
                        "gpu_raw_name": gpu.name,
                        "supported_usage_classes": raw.get("SupportedUsageClasses"),
                        "supported_root_devices": raw.get("SupportedRootDeviceTypes"),
                        "supported_virtualization": raw.get("SupportedVirtualizationTypes"),
                    },
                )

    async def _fetch_region(self, region: str) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
        session = self._session()
        instance_types, spot, on_demand = await asyncio.gather(
            self._instance_types(session, region),
            self._spot_prices(session, region),
            self._on_demand_prices(session, region),
        )
        return instance_types, spot, on_demand

    async def _instance_types(self, session: aioboto3.Session, region: str) -> list[dict[str, Any]]:
        types: list[dict[str, Any]] = []
        async with session.client("ec2", region_name=region) as ec2:
            paginator = ec2.get_paginator("describe_instance_types")
            async for page in paginator.paginate():
                types.extend(page.get("InstanceTypes", []))
        return types

    async def _spot_prices(self, session: aioboto3.Session, region: str) -> dict[str, float]:
        """Cheapest current spot price per instance type, across the region's AZs.

        ``DescribeSpotPriceHistory`` with ``StartTime=now`` returns the price in
        force right now in each availability zone. We keep the minimum: the offer
        is a region, and a caller who asks for spot in that region can be placed
        in the AZ that quotes it.
        """
        prices: dict[str, float] = {}
        now = datetime.now(UTC)
        async with session.client("ec2", region_name=region) as ec2:
            paginator = ec2.get_paginator("describe_spot_price_history")
            async for page in paginator.paginate(ProductDescriptions=["Linux/UNIX"], StartTime=now, EndTime=now):
                for entry in page.get("SpotPriceHistory", []):
                    instance_type = entry["InstanceType"]
                    price = float(entry["SpotPrice"])
                    if price < prices.get(instance_type, float("inf")):
                        prices[instance_type] = price
        return prices

    async def _on_demand_prices(self, session: aioboto3.Session, region: str) -> dict[str, float]:
        """Linux/shared-tenancy on-demand rate card for one region, in one walk.

        The alternative — a ``GetProducts`` call per instance type, as v1 does — is
        roughly 900 round trips per region. Filtering by ``regionCode`` and paginating
        the whole rate card is the same data in a couple of dozen.
        """
        prices: dict[str, float] = {}
        filters = [
            {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
            {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            {"Type": "TERM_MATCH", "Field": "licenseModel", "Value": "No License required"},
            {"Type": "TERM_MATCH", "Field": "marketoption", "Value": "OnDemand"},
        ]
        async with session.client("pricing", region_name=PRICING_REGION) as pricing:
            paginator = pricing.get_paginator("get_products")
            async for page in paginator.paginate(ServiceCode="AmazonEC2", Filters=filters):
                for raw in page.get("PriceList", []):
                    product = json.loads(raw) if isinstance(raw, str) else raw
                    instance_type = (product.get("product", {}).get("attributes", {})).get("instanceType")
                    price = _hourly(product)
                    if instance_type and price:
                        prices[instance_type] = price
        return prices

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        region = offer.region
        if not region:
            raise CapabilityMismatchError("an aws offer without a region cannot be launched", provider=self._name)

        name = f"skyward-{compute_id}"
        session = self._session()
        async with session.client("ec2", region_name=region) as ec2:
            vpc = await self._vpc(ec2)

            async with asyncio.TaskGroup() as group:
                key = group.create_task(self._key_pair(ec2, name, public_key))
                security_group = group.create_task(self._security_group(ec2, "skyward-sg", vpc))
                subnets = group.create_task(self._subnets(ec2, vpc, offer.instance_type))
                image = group.create_task(self._image(session, region, offer))

        return {
            "compute_id": compute_id,
            "region": region,
            "instance_type": offer.instance_type,
            "image": image.result(),
            "key_name": key.result(),
            "security_group_id": security_group.result(),
            "subnets": subnets.result(),
            "user": SSH_USER,
            "disk_gb": int(self._config.get("disk_gb") or DEFAULT_DISK_GB),
        }

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        """Ask an EC2 Fleet for the machines, and pin the compute to the zone it chose.

        Fleet places the whole request into a single availability zone, choosing it
        across the candidate subnets by the spot allocation strategy rather than by a
        blind walk — a zone that has capacity beats the first zone in a list. The
        compute is pinned to that zone from the first launch on, because a peer in
        another zone pays for its own traffic and cannot reach the others on a private
        address.

        The launch template lives only for the length of the call: Fleet takes no
        inline instance config, so the shape is written to a template, spent once, and
        deleted whether or not the fleet came up.
        """
        subnets: Mapping[str, str] = binding["subnets"]
        pinned: str | None = binding.get("az")
        candidate = {pinned: subnets[pinned]} if pinned else subnets

        session = self._session()
        async with session.client("ec2", region_name=binding["region"]) as ec2:
            template = await ec2.create_launch_template(
                LaunchTemplateName=f"skyward-{uuid.uuid4().hex[:8]}",
                LaunchTemplateData=_template(binding),
            )
            template_id = str(template["LaunchTemplate"]["LaunchTemplateId"])

            try:
                fleet = await ec2.create_fleet(**_fleet(binding, market, template_id, candidate, count, min_count))
                ids = [iid for spec in fleet.get("Instances", []) for iid in spec.get("InstanceIds", [])]
                if len(ids) < min_count:
                    errors = "; ".join(
                        f"{error.get('ErrorCode')}: {error.get('ErrorMessage')}"
                        for error in fleet.get("Errors", [])
                    )
                    raise CapabilityMismatchError(
                        f"fleet launched {len(ids)}/{count} {binding['instance_type']} instances: {errors}",
                        provider=self._name,
                    )

                described = await self._describe(ec2, ids)
                landed = next(
                    (raw["Placement"]["AvailabilityZone"] for raw in described if raw.get("Placement")),
                    pinned,
                )
                machines = tuple(_machine(raw, binding["user"]) for raw in described)
                return {**binding, "az": landed}, machines
            finally:
                with suppress(ClientError):
                    await ec2.delete_launch_template(LaunchTemplateId=template_id)

    async def _describe(self, ec2: AioBaseClient, ids: list[str]) -> list[dict[str, Any]]:
        """Read the fleet's instances back, past the window where a fresh id is not yet describable.

        A ``create_fleet`` that has just returned an id is ahead of the eventual
        consistency of ``describe_instances``, which answers ``InvalidInstanceID.NotFound``
        for a second or two. The zone the fleet landed in is only knowable from the
        instance, so the read is retried rather than guessed.
        """
        for attempt in range(5):
            try:
                response = await ec2.describe_instances(InstanceIds=ids)
            except ClientError as error:
                if _code(error) != "InvalidInstanceID.NotFound" or attempt == 4:
                    raise
                await asyncio.sleep(2.0)
                continue
            return [raw for reservation in response["Reservations"] for raw in reservation["Instances"]]
        return []

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        found: dict[str, Machine] = {}
        async with self._session().client("ec2", region_name=binding["region"]) as ec2:
            paginator = ec2.get_paginator("describe_instances")
            pages = paginator.paginate(
                Filters=[
                    {"Name": f"tag:{COMPUTE_TAG}", "Values": [binding["compute_id"]]},
                    {"Name": "instance-state-name", "Values": ["pending", "running"]},
                ],
            )
            async for page in pages:
                for reservation in page.get("Reservations", []):
                    for raw in reservation.get("Instances", []):
                        machine = _machine(raw, binding["user"])
                        found[machine.id] = machine
        return found

    async def interruptions(self, binding: Binding, machine_ids: tuple[str, ...]) -> Mapping[str, str]:
        """Which of these spot machines AWS has flagged for reclamation, mapped to the reason.

        Proactive and best-effort. A spot instance carries a two-minute warning
        before it is taken, and that warning lands on its spot request as a
        ``marked-for-*`` status code while the instance is still running. Reading
        it here turns the reclamation into a deficit before SSH drops, rather than
        after the machine has already vanished from :meth:`machines`.

        An on-demand machine has no spot request and is never flagged. An API that
        will not answer is read as no warning at all: a poll that failed is not a
        signal, so the batch is reported clean rather than mistaken for reclaimed.
        """
        if not machine_ids:
            return {}

        try:
            async with self._ec2(binding["region"]) as ec2:
                response = await ec2.describe_spot_instance_requests(
                    Filters=[{"Name": "instance-id", "Values": list(machine_ids)}],
                )
        except ClientError:
            return {}

        flagged: dict[str, str] = {}
        for request in response.get("SpotInstanceRequests", []):
            instance_id = request.get("InstanceId")
            code = str((request.get("Status") or {}).get("Code") or "")
            if instance_id and _reclaimed(code):
                flagged[str(instance_id)] = "spot-interruption"
        return flagged

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        """Terminate the batch; if one of them is already gone, terminate the rest.

        ``TerminateInstances`` rejects the whole call when a single id is unknown to
        it — an id EC2 has finished purging. Swallowing that error would be reading
        "one of these was already dead" as "all of these are now dead", and would
        leave the others running and billed.
        """
        if not machine_ids:
            return

        async with self._session().client("ec2", region_name=binding["region"]) as ec2:
            try:
                await ec2.terminate_instances(InstanceIds=list(machine_ids))
            except ClientError as error:
                if _code(error) != "InvalidInstanceID.NotFound":
                    raise
                alive = set(machine_ids) & set(await self.machines(binding))
                if alive:
                    await ec2.terminate_instances(InstanceIds=sorted(alive))

    async def release(self, binding: Binding) -> None:
        """Take back the key pair. The security group stays.

        The group is shared across computes and never deleted, because deleting one
        means outwaiting AWS: the ENIs of terminated instances hold it for minutes
        — measured between three and seven — and until the last lets go the call
        answers ``DependencyViolation``. A group costs nothing to keep, so teardown
        does not buy that wait.
        """
        async with self._ec2(binding["region"]) as ec2:
            await _ignoring(ec2.delete_key_pair(KeyName=binding["key_name"]), "InvalidKeyPair.NotFound")

    async def _vpc(self, ec2: AioBaseClient) -> str:
        response = await ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
        vpcs = response["Vpcs"]
        if not vpcs:
            raise CapabilityMismatchError("the account has no default vpc to launch into", provider=self._name)
        return str(vpcs[0]["VpcId"])

    async def _key_pair(self, ec2: AioBaseClient, name: str, public_key: str) -> str:
        """Import over a duplicate rather than accept it.

        A second ``initialize`` is a first one whose binding was lost, and with it
        the private key the control plane had generated. The key already registered
        under this compute's name is that lost key: keeping it would leave every
        machine launched from here on unreachable. Deleting it takes nothing down —
        an instance carries the key material it booted with.
        """
        request: dict[str, Any] = {
            "KeyName": name,
            "PublicKeyMaterial": public_key.encode(),
        }

        try:
            await ec2.import_key_pair(**request)
        except ClientError as error:
            if _code(error) != "InvalidKeyPair.Duplicate":
                raise
            await ec2.delete_key_pair(KeyName=name)
            await ec2.import_key_pair(**request)
        return name

    async def _security_group(self, ec2: AioBaseClient, name: str, vpc: str) -> str:
        """Create the group, and authorize it whether or not this call is the one that created it.

        A crash between the create and the authorize leaves a group that exists and
        opens nothing; the retry has to finish the job it finds half done rather
        than take the group's existence as proof that it is usable.
        """
        try:
            created = await ec2.create_security_group(
                GroupName=name,
                Description="Skyward EC2 worker security group",
                VpcId=vpc,
                TagSpecifications=[{"ResourceType": "security-group", "Tags": _tags(name)}],
            )
            group_id = str(created["GroupId"])
        except ClientError as error:
            if _code(error) != "InvalidGroup.Duplicate":
                raise
            existing = await ec2.describe_security_groups(
                Filters=[
                    {"Name": "group-name", "Values": [name]},
                    {"Name": "vpc-id", "Values": [vpc]},
                ],
            )
            group_id = str(existing["SecurityGroups"][0]["GroupId"])

        await _ignoring(
            ec2.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=[
                    {
                        "IpProtocol": "-1",
                        "UserIdGroupPairs": [{"GroupId": group_id, "Description": "peers"}],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "ssh"}],
                    },
                ],
            ),
            "InvalidPermission.Duplicate",
        )
        return group_id

    async def _subnets(self, ec2: AioBaseClient, vpc: str, instance_type: str) -> dict[str, str]:
        """One subnet per availability zone that actually offers the instance type.

        An instance type is not sold in every zone of its region, and a subnet in a
        zone that does not sell it is a launch that fails outright. Resolving the
        pairing once is what lets :meth:`launch` treat the zones as a list to walk.
        """
        async with asyncio.TaskGroup() as group:
            offerings = group.create_task(
                ec2.describe_instance_type_offerings(
                    LocationType="availability-zone",
                    Filters=[{"Name": "instance-type", "Values": [instance_type]}],
                ),
            )
            subnets = group.create_task(
                ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc]}]),
            )

        zones = {offering["Location"] for offering in offerings.result().get("InstanceTypeOfferings", [])}
        usable = {
            str(subnet["AvailabilityZone"]): str(subnet["SubnetId"])
            for subnet in subnets.result()["Subnets"]
            if subnet["AvailabilityZone"] in zones
        }
        if not usable:
            raise CapabilityMismatchError(
                f"the default vpc has no subnet in a zone that offers {instance_type}",
                provider=self._name,
            )
        return usable

    async def _image(self, session: aioboto3.Session, region: str, offer: Offer) -> str:
        """The AMI id: the config's own, or the current one from SSM."""
        return str(self._config.get("ami") or await self._latest(session, region, offer))

    async def _latest(self, session: aioboto3.Session, region: str, offer: Offer) -> str:
        """The current AMI, from the SSM public parameters rather than from a hardcoded id.

        AMI ids are per region and change on every rebuild, so the only stable name
        for "current Ubuntu" or "current NVIDIA driver base" is the parameter that
        points at it. A GPU offer gets the Deep Learning base AMI, which is the one
        that ships the driver.
        """
        architectures = offer.specific.get("architectures") or ()
        arm = "arm64" in architectures
        version = str(self._config.get("ubuntu_version") or DEFAULT_UBUNTU)

        if offer.accelerator_count:
            parameter = (
                f"/aws/service/deeplearning/ami/{'arm64' if arm else 'x86_64'}"
                f"/base-oss-nvidia-driver-gpu-ubuntu-{version}/latest/ami-id"
            )
        else:
            ebs = "ebs-gp3" if version >= "24.04" else "ebs-gp2"
            parameter = (
                f"/aws/service/canonical/ubuntu/server/{version}/stable/current"
                f"/{'arm64' if arm else 'amd64'}/hvm/{ebs}/ami-id"
            )

        async with session.client("ssm", region_name=region) as ssm:
            response = await ssm.get_parameter(Name=parameter)
            return str(response["Parameter"]["Value"])


def _template(binding: Binding) -> dict[str, Any]:
    """The instance shape, minus what Fleet varies per subnet.

    ``ImageId`` and ``InstanceType`` live in the fleet overrides, not here, because
    Fleet is the thing choosing the subnet and needs the pair alongside each one.
    Everything a machine of this compute shares is what remains.
    """
    return {
        "KeyName": binding["key_name"],
        "NetworkInterfaces": [{
            "DeviceIndex": 0,
            "AssociatePublicIpAddress": True,
            "Groups": [binding["security_group_id"]],
        }],
        "BlockDeviceMappings": [{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": binding["disk_gb"],
                "VolumeType": "gp3",
                "DeleteOnTermination": True,
            },
        }],
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": _tags(f"skyward-{binding['compute_id']}", binding["compute_id"]),
        }],
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
        "InstanceInitiatedShutdownBehavior": "terminate",
    }


def _fleet(binding: Binding, market: Market, template_id: str, subnets: Mapping[str, str], count: int, min_count: int) -> dict[str, Any]:
    """An instant fleet: the machines come back with the call, not through a callback.

    ``SingleAvailabilityZone`` is what makes the whole request land together, so that
    the compute can be pinned to one zone and its peers stay on a private network.
    ``MinTargetCapacity`` is the partial-readiness floor — a fleet that can bring up
    ``min_count`` is worth taking, one that cannot is a failure.
    """
    spot = market == "spot"
    overrides = [
        {"SubnetId": subnet, "InstanceType": binding["instance_type"], "ImageId": binding["image"]}
        for subnet in subnets.values()
    ]
    return {
        "Type": "instant",
        "LaunchTemplateConfigs": [{
            "LaunchTemplateSpecification": {"LaunchTemplateId": template_id, "Version": "$Latest"},
            "Overrides": overrides,
        }],
        "TargetCapacitySpecification": {
            "TotalTargetCapacity": count,
            "DefaultTargetCapacityType": "spot" if spot else "on-demand",
            "SpotTargetCapacity": count if spot else 0,
            "OnDemandTargetCapacity": 0 if spot else count,
        },
        "SpotOptions": {
            "AllocationStrategy": FLEET_STRATEGY,
            "SingleAvailabilityZone": True,
            "SingleInstanceType": True,
            "MinTargetCapacity": min_count,
        },
        "OnDemandOptions": {
            "AllocationStrategy": "lowest-price",
            "SingleAvailabilityZone": True,
            "SingleInstanceType": True,
            "MinTargetCapacity": min_count,
        },
    }


def _tags(name: str, compute_id: str | None = None) -> list[dict[str, str]]:
    tags = [{"Key": "Name", "Value": name}, {"Key": MANAGED_TAG, "Value": "true"}]
    if compute_id:
        tags.append({"Key": COMPUTE_TAG, "Value": compute_id})
    return tags


def _machine(raw: dict[str, Any], user: str) -> Machine:
    return Machine(
        id=str(raw["InstanceId"]),
        state="running" if raw["State"]["Name"] == "running" else "pending",
        host=raw.get("PublicIpAddress"),
        user=user,
        private_host=raw.get("PrivateIpAddress"),
    )


async def _ignoring(call: Awaitable[object], *codes: str) -> None:
    """Await an EC2 call, reading the named error codes as the state already being right.

    Every one of them means "already imported", "already authorized" or "already
    gone" — the three answers a daemon that restarts mid-reconcile has to be able
    to receive.
    """
    try:
        await call
    except ClientError as error:
        if _code(error) not in codes:
            raise


def _code(error: ClientError) -> str:
    return str(error.response.get("Error", {}).get("Code", ""))


def _reclaimed(code: str) -> bool:
    """Whether a spot request status code is a reclamation warning rather than a healthy state.

    A fulfilled request reads ``fulfilled``; a request whose instance AWS is taking
    back reads ``marked-for-termination`` or ``marked-for-stop-*`` during the notice
    window, and ``instance-terminated-*`` / ``instance-stopped-*`` once it acts.
    """
    return code.startswith("marked-for-") or "terminated" in code or "stopped" in code


class _Gpu(NamedTuple):
    name: str | None
    count: int
    vram: float | None
    manufacturer: str | None


def _gpu(raw: dict[str, Any]) -> _Gpu:
    """The instance's accelerators, as AWS spells them.

    ``GpuInfo.Gpus`` is a list because an instance could in principle mix models;
    none does today, so the first entry names the card and the counts add up.
    ``MemoryInfo.SizeInMiB`` there is already per card — it is passed through, and
    the shared catalog is left to normalize the name.
    """
    gpus = (raw.get("GpuInfo") or {}).get("Gpus") or []
    if not gpus:
        return _Gpu(None, 0, None, None)

    first = gpus[0]
    count = sum(int(gpu.get("Count") or 0) for gpu in gpus)
    vram = float((first.get("MemoryInfo") or {}).get("SizeInMiB") or 0) / MIB or None
    return _Gpu(first.get("Name") or None, count, vram, first.get("Manufacturer") or None)


def _hourly(product: dict[str, Any]) -> float | None:
    terms = (product.get("terms") or {}).get("OnDemand") or {}
    for term in terms.values():
        for dimension in (term.get("priceDimensions") or {}).values():
            price = (dimension.get("pricePerUnit") or {}).get("USD")
            if price and float(price) > 0:
                return float(price)
    return None
