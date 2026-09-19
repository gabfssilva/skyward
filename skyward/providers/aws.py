from __future__ import annotations

import asyncio
import json
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar, Self

import aioboto3
import msgspec
from aiobotocore.config import AioConfig
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from types_aiobotocore_ec2.client import EC2Client
    from types_aiobotocore_ec2.type_defs import RequestLaunchTemplateDataTypeDef, TagTypeDef
    from types_aiobotocore_pricing.type_defs import FilterTypeDef as PricingFilterTypeDef

from skyward.shared.architectures import architecture
from skyward.shared.errors import CapabilityMismatchError
from skyward.shared.observability import logger
from skyward.shared.provider import Binding, Machine, Mount
from skyward.shared.providers import AWS
from skyward.shared.schemas import ComputeSpec, Endpoint, Market, Offer, Volume
from skyward.worker import bootstrap

PRICING_REGION = "us-east-1"
MIB = 1024

COMPUTE_TAG = "skyward:compute"
MANAGED_TAG = "skyward:managed"
NODE_TAG = "skyward:node"
IMAGE_TAG = "skyward:image"
WARM_NAME = "skyward-warm-{tag}"
SSH_USER = "ubuntu"
FLEET_STRATEGY = "price-capacity-optimized"

SETTLING = 60.0
"""Seconds a zone just bought in is believed over an EC2 that lists no machine there.

``DescribeInstances`` filtered by tag trails ``CreateFleet`` by seconds, and the next
fleet of the same compute is often that close behind: asked then, EC2 says the compute
lives nowhere, and the fleet is free to pick a second zone. Past this, an EC2 that
lists nothing is believed — every machine is gone, and so is the reason to stay.
"""


@dataclass(frozen=True, slots=True)
class _Order:
    """One launch waiting to be bought: who it is for, on which market, and where the answer goes."""

    node: str
    market: Market
    sold: asyncio.Future[Machine]


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

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return True

    def __init__(
        self,
        provider_id: str,
        name: str,
        access_key_id: str,
        secret_access_key: str,
        config: AWS,
    ) -> None:
        self._id = provider_id
        self._name = name
        self._config = config
        self._session = aioboto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=config.session_token,
        )
        """Always the explicit credentials — never boto3's default chain.

        The default chain would fall back to the caller's environment and
        ``~/.aws``, which is exactly what makes two AWS accounts in one process
        impossible.
        """
        self._orders: dict[str, list[_Order]] = {}
        self._buying: dict[str, asyncio.Lock] = {}
        self._bought: dict[str, tuple[str, float]] = {}

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        settings = msgspec.convert({**credentials, **config}, AWS)
        if not settings.access_key_id or not settings.secret_access_key:
            raise CapabilityMismatchError("aws requires access_key_id and secret_access_key credentials", provider=name)
        return cls(provider_id, name, settings.access_key_id, settings.secret_access_key, settings)

    @property
    def _regions(self) -> tuple[str, ...]:
        return self._config.regions

    def _client_config(self) -> AioConfig:
        timeout = self._config.request_timeout
        return AioConfig(connect_timeout=timeout, read_timeout=timeout)

    @asynccontextmanager
    async def _ec2(self, region: str) -> AsyncIterator[EC2Client]:
        async with self._session.client(
            "ec2",
            region_name=region,
            config=self._client_config(),
        ) as client:
            yield client

    async def offers(self) -> AsyncIterator[Offer]:
        regions = self._regions
        results = await asyncio.gather(*(self._fetch_region(region) for region in regions))

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for region, (instance_types, spot, on_demand) in zip(regions, results, strict=True):
            for raw in instance_types:
                if self._config.exclude_burstable and raw.get("BurstablePerformanceSupported"):
                    continue
                instance_type = raw["InstanceType"]
                gpu = _gpu(raw)
                network = raw.get("NetworkInfo") or {}
                storage = raw.get("InstanceStorageInfo") or {}
                disk_gb = storage.get("TotalSizeInGB")

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
                    disk_gb=float(disk_gb) if disk_gb else None,
                    architecture=_architecture(raw),
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

    async def _fetch_region(self, region: str) -> tuple[list[Mapping[str, Any]], dict[str, float], dict[str, float]]:
        instance_types, spot, on_demand = await asyncio.gather(
            self._instance_types(self._session, region),
            self._spot_prices(self._session, region),
            self._on_demand_prices(self._session, region),
        )
        return instance_types, spot, on_demand

    async def _instance_types(self, session: aioboto3.Session, region: str) -> list[Mapping[str, Any]]:
        types: list[Mapping[str, Any]] = []
        async with session.client("ec2", region_name=region, config=self._client_config()) as ec2:
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
        async with session.client("ec2", region_name=region, config=self._client_config()) as ec2:
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
        filters: list[PricingFilterTypeDef] = [
            {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
            {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            {"Type": "TERM_MATCH", "Field": "licenseModel", "Value": "No License required"},
            {"Type": "TERM_MATCH", "Field": "marketoption", "Value": "OnDemand"},
        ]
        async with session.client("pricing", region_name=PRICING_REGION, config=self._client_config()) as pricing:
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
        async with self._session.client("ec2", region_name=region, config=self._client_config()) as ec2:
            if configured_subnet := self._config.subnet_id:
                described = await ec2.describe_subnets(SubnetIds=[configured_subnet])
                subnet = described["Subnets"][0]
                vpc = str(subnet["VpcId"])
                subnets = {str(subnet["AvailabilityZone"]): str(configured_subnet)}
            else:
                vpc = await self._vpc(ec2)
                subnets = await self._subnets(ec2, vpc, offer.instance_type)

            async with asyncio.TaskGroup() as group:
                key = group.create_task(self._key_pair(ec2, name, public_key))
                image = group.create_task(self._image(self._session, region, offer))
                security_group = (
                    None
                    if self._config.security_group_id
                    else group.create_task(self._security_group(ec2, "skyward-sg", vpc))
                )

        if configured_group := self._config.security_group_id:
            security_group_id = str(configured_group)
        elif security_group is not None:
            security_group_id = security_group.result()
        else:
            raise RuntimeError("AWS security group resolution produced no group")

        return {
            "compute_id": compute_id,
            "region": region,
            "instance_type": offer.instance_type,
            "image": image.result(),
            "key_name": key.result(),
            "security_group_id": security_group_id,
            "subnets": subnets,
            "multi_subnet": self._config.multi_subnet,
            "heterogeneous_instances": self._config.heterogeneous_instances,
            "user": self._config.username or SSH_USER,
            "disk_gb": self._config.disk_gb,
            "instance_profile_arn": self._config.instance_profile_arn,
            "allocation_strategy": self._config.allocation_strategy,
            "instance_timeout": self._config.instance_timeout,
        }

    async def launch(self, binding: Binding, market: Market, node: str) -> Machine:
        """Ask for one machine, and buy it together with every other asked for within the window.

        EC2 throttles ``RunInstances`` by the call and a pool opens by asking for all of
        its machines at once, so a launch is an order and not a request: it joins the
        compute's queue, and the first to arrive waits ``launch_window`` for the rest and
        buys the whole queue as one fleet. Every order gets its own answer — a machine,
        or the fleet's reason for not covering it.
        """
        order = _Order(node, market, asyncio.get_running_loop().create_future())
        queue = self._orders.setdefault(binding["compute_id"], [])
        queue.append(order)
        if len(queue) == 1:
            await self._fill(binding)
        return await order.sold

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        """Every machine of the compute — never in the middle of a purchase.

        A fleet's machines are born without a claim and get it a call later. One seen in
        between names no row, and a machine that names no row is terminated as a stray,
        so the question waits for the purchase in flight to finish claiming.
        """
        async with self._lock(binding), self._ec2(binding["region"]) as ec2:
            return {str(raw["InstanceId"]): _machine(raw, binding["user"]) async for raw in _instances(ec2, binding)}

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

    async def bake(self, binding: Binding, machine_id: str, tag: str) -> str:
        """Register an AMI from the instance, and leave the instance alone.

        ``NoReboot`` because this is a node that is serving: rebooting it to quiesce
        the filesystem is exactly the thing that must not happen to it. The cost is
        that the image is of a running machine, which for a bootstrap that has finished
        writing is what it is anyway.

        Both the image and the snapshot under it are tagged with the environment, which
        is how :meth:`baked` finds them and how anybody who wants the storage back finds
        them. Nothing here ever deregisters one.
        """
        name = WARM_NAME.format(tag=tag)
        tags: list[TagTypeDef] = [*_tags(name), {"Key": IMAGE_TAG, "Value": tag}]

        async with self._ec2(binding["region"]) as ec2:
            created = await ec2.create_image(
                InstanceId=machine_id,
                Name=name,
                NoReboot=True,
                TagSpecifications=[
                    {"ResourceType": "image", "Tags": tags},
                    {"ResourceType": "snapshot", "Tags": tags},
                ],
            )
            return str(created["ImageId"])

    async def baked(self, binding: Binding, tag: str) -> str | None:
        """This account's own AMI for this environment in this region, once it can be booted.

        ``Owners=["self"]`` because the tag is ours and a public image carrying it is
        somebody else's claim about our environment. An image still being registered is
        filtered out rather than returned: a fleet pointed at a ``pending`` AMI fails
        outright, which is worse than bootstrapping from scratch.
        """
        async with self._ec2(binding["region"]) as ec2:
            response = await ec2.describe_images(
                Owners=["self"],
                Filters=[
                    {"Name": f"tag:{IMAGE_TAG}", "Values": [tag]},
                    {"Name": "state", "Values": ["available"]},
                ],
            )

        images = response.get("Images", [])
        return str(images[0]["ImageId"]) if images else None

    async def mount(self, binding: Binding, volumes: tuple[Volume, ...]) -> Mount:
        """Reach the buckets with the machine's own identity, so no key is minted or shipped.

        S3 in the region the compute was launched in, signed for by the instance
        profile the machine already carries. Nothing here leaves the account: there
        is no access key to create, none to write to the node's disk, and none to
        revoke when the compute goes away.
        """
        endpoint = Endpoint(url=f"https://s3.{binding['region']}.amazonaws.com")
        return Mount(phases=(bootstrap.mounts(tuple((volume, endpoint) for volume in volumes)),))

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        """Terminate the batch; if one of them is already gone, terminate the rest.

        ``TerminateInstances`` rejects the whole call when a single id is unknown to
        it — an id EC2 has finished purging. Swallowing that error would be reading
        "one of these was already dead" as "all of these are now dead", and would
        leave the others running and billed.
        """
        if not machine_ids:
            return

        async with self._session.client(
            "ec2",
            region_name=binding["region"],
            config=self._client_config(),
        ) as ec2:
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
        self._buying.pop(binding["compute_id"], None)
        self._bought.pop(binding["compute_id"], None)
        async with self._ec2(binding["region"]) as ec2:
            await _ignoring(ec2.delete_key_pair(KeyName=binding["key_name"]), "InvalidKeyPair.NotFound")

    def _lock(self, binding: Binding) -> asyncio.Lock:
        return self._buying.setdefault(binding["compute_id"], asyncio.Lock())

    async def _fill(self, binding: Binding) -> None:
        """Wait the window out, then buy what queued up — one fleet per market, one at a time.

        Whatever stops the purchase is every order's answer, so nobody is left waiting
        on a fleet that will not come; the one leading it is no exception, and reads its
        own answer where the others do.
        """
        try:
            await asyncio.sleep(self._config.launch_window)
        finally:
            orders = tuple(self._orders.pop(binding["compute_id"], ()))

        try:
            async with self._lock(binding), self._ec2(binding["region"]) as ec2:
                markets: dict[Market, None] = dict.fromkeys(order.market for order in orders)
                for market in markets:
                    await self._purchase(ec2, binding, market, tuple(order for order in orders if order.market == market))
        except Exception as failure:
            _refuse(orders, failure)
        except BaseException:
            _refuse(orders, CapabilityMismatchError("the fleet this launch was part of was cancelled", provider=self._name))
            raise

    async def _purchase(self, ec2: EC2Client, binding: Binding, market: Market, orders: tuple[_Order, ...]) -> None:
        """One instant fleet for the orders of one market, in the zone the compute already lives in.

        Peers in different zones pay for their own traffic and cannot reach each other
        on a private address, so a compute lives in one zone. With machines alive, that
        zone is theirs and the fleet is offered its subnet alone. With none, the fleet is
        offered every subnet and keeps to a single zone of its own choosing — Fleet knows
        where the capacity is, and nothing asked beforehand does. A ``multi_subnet``
        compute asked for neither: its fleet is always offered every subnet, and spreads
        over them as the allocation strategy sees fit.

        The launch template lives only for the length of the call: Fleet takes no inline
        instance config, so the shape is written to a template, spent once, and deleted
        whether or not the fleet came up. A template is shared by the whole fleet, so the
        claim cannot ride on it: each machine is tagged with its node as soon as it has
        an id, under the lock :meth:`machines` waits on.
        """
        subnets: Mapping[str, str] = binding["subnets"]
        spread = bool(binding.get("multi_subnet"))
        zone = None if spread else await self._zone(ec2, binding)

        template = await ec2.create_launch_template(
            LaunchTemplateName=f"skyward-{binding['compute_id']}-{uuid.uuid4().hex[:6]}",
            LaunchTemplateData=_template(binding),
        )
        template_id = str(template["LaunchTemplate"]["LaunchTemplateId"])
        try:
            offered = {zone: subnets[zone]} if zone is not None and zone in subnets else subnets
            fleet = await ec2.create_fleet(**_fleet(binding, market, template_id, offered, len(orders), spread))
        finally:
            with suppress(ClientError):
                await ec2.delete_launch_template(LaunchTemplateId=template_id)

        ids = [str(iid) for sold in fleet.get("Instances", []) for iid in sold.get("InstanceIds", [])]
        zones = {subnet: name for name, subnet in subnets.items()}
        landed = next(
            (zones[subnet] for sold in fleet.get("Instances", []) if (subnet := _subnet(sold)) in zones),
            None,
        )
        logger.bind(component="aws", compute_id=binding["compute_id"]).debug(
            "a {} fleet of {} {} offered {} sold {} in {}",
            market, len(orders), binding["instance_type"], zone or "every zone", len(ids), landed or "no zone",
        )
        if landed is not None and not spread:
            self._bought[binding["compute_id"]] = (landed, asyncio.get_running_loop().time())

        async with asyncio.TaskGroup() as group:
            for order, instance in zip(orders, ids, strict=False):
                group.create_task(ec2.create_tags(Resources=[instance], Tags=_tags(f"skyward-{order.node}", node=order.node)))

        for order, instance in zip(orders, ids, strict=False):
            order.sold.set_result(Machine(id=instance, state="pending", user=binding["user"], node=order.node))

        errors = "; ".join(f"{error.get('ErrorCode')}: {error.get('ErrorMessage')}" for error in fleet.get("Errors", []))
        _refuse(
            orders[len(ids):],
            CapabilityMismatchError(
                f"fleet launched {len(ids)} of {len(orders)} {binding['instance_type']} in {zone or 'any zone'}: {errors}",
                provider=self._name,
            ),
        )

    async def _zone(self, ec2: EC2Client, binding: Binding) -> str | None:
        """The zone the compute's machines live in: what EC2 lists, else what was bought moments ago.

        A compute already split across zones — it was ``multi_subnet`` once, or EC2 was
        slower than :data:`SETTLING` — goes on in the zone holding most of it.
        """
        listed = Counter([str(raw["Placement"]["AvailabilityZone"]) async for raw in _instances(ec2, binding)])
        if listed:
            return listed.most_common(1)[0][0]
        match self._bought.get(binding["compute_id"]):
            case (zone, at) if asyncio.get_running_loop().time() - at < SETTLING:
                return zone
            case _:
                return None

    async def _vpc(self, ec2: EC2Client) -> str:
        response = await ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
        vpcs = response["Vpcs"]
        if not vpcs:
            raise CapabilityMismatchError("the account has no default vpc to launch into", provider=self._name)
        return str(vpcs[0]["VpcId"])

    async def _key_pair(self, ec2: EC2Client, name: str, public_key: str) -> str:
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

    async def _security_group(self, ec2: EC2Client, name: str, vpc: str) -> str:
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

    async def _subnets(self, ec2: EC2Client, vpc: str, instance_type: str) -> dict[str, str]:
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
        return self._config.ami or await self._latest(session, region, offer)

    async def _latest(self, session: aioboto3.Session, region: str, offer: Offer) -> str:
        """The current AMI, from the SSM public parameters rather than from a hardcoded id.

        AMI ids are per region and change on every rebuild, so the only stable name
        for "current Ubuntu" or "current NVIDIA driver base" is the parameter that
        points at it. A GPU offer gets the Deep Learning base AMI, which is the one
        that ships the driver.
        """
        architectures = offer.specific.get("architectures") or ()
        arm = "arm64" in architectures
        version = self._config.ubuntu_version

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

        async with session.client("ssm", region_name=region, config=self._client_config()) as ssm:
            response = await ssm.get_parameter(Name=parameter)
            return str(response["Parameter"]["Value"])


def _template(binding: Binding) -> RequestLaunchTemplateDataTypeDef:
    """The instance shape, minus what Fleet varies per subnet.

    ``ImageId`` and ``InstanceType`` live in the fleet overrides, not here, because
    Fleet is the thing choosing the subnet and needs the pair alongside each one.
    Everything a machine of this compute shares is what remains; the node's claim is
    the one thing it does not share, and is tagged on after the fleet.
    """
    template: RequestLaunchTemplateDataTypeDef = {
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
    if profile := binding.get("instance_profile_arn"):
        template["IamInstanceProfile"] = {"Arn": profile}
    return template


def _fleet(binding: Binding, market: Market, template_id: str, subnets: Mapping[str, str], capacity: int, spread: bool) -> dict[str, Any]:
    """An instant fleet: the machines come back with the call, not through a callback.

    ``MinTargetCapacity`` stays at one, so a zone that can sell part of the fleet
    sells that part rather than nothing.
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
            "TotalTargetCapacity": capacity,
            "DefaultTargetCapacityType": "spot" if spot else "on-demand",
            "SpotTargetCapacity": capacity if spot else 0,
            "OnDemandTargetCapacity": 0 if spot else capacity,
        },
        "SpotOptions": {
            "AllocationStrategy": binding.get("allocation_strategy") or FLEET_STRATEGY,
            "SingleAvailabilityZone": not spread,
            "SingleInstanceType": True,
            "MinTargetCapacity": 1,
        },
        "OnDemandOptions": {
            "AllocationStrategy": "lowest-price",
            "SingleAvailabilityZone": not spread,
            "SingleInstanceType": True,
            "MinTargetCapacity": 1,
        },
    }


def _tags(name: str, compute_id: str | None = None, node: str | None = None) -> list[TagTypeDef]:
    tags: list[TagTypeDef] = [{"Key": "Name", "Value": name}, {"Key": MANAGED_TAG, "Value": "true"}]
    if compute_id:
        tags.append({"Key": COMPUTE_TAG, "Value": compute_id})
    if node:
        tags.append({"Key": NODE_TAG, "Value": node})
    return tags


async def _instances(ec2: EC2Client, binding: Binding) -> AsyncIterator[Mapping[str, Any]]:
    """The compute's instances that are, or are about to be, machines."""
    pages = ec2.get_paginator("describe_instances").paginate(
        Filters=[
            {"Name": f"tag:{COMPUTE_TAG}", "Values": [binding["compute_id"]]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]},
        ],
    )
    async for page in pages:
        for reservation in page.get("Reservations", []):
            for raw in reservation.get("Instances", []):
                yield raw


def _subnet(sold: Mapping[str, Any]) -> str:
    """The subnet one entry of a fleet's ``Instances`` was launched in."""
    return str(((sold.get("LaunchTemplateAndOverrides") or {}).get("Overrides") or {}).get("SubnetId") or "")


def _refuse(orders: tuple[_Order, ...], failure: Exception) -> None:
    for order in orders:
        if not order.sold.done():
            order.sold.set_exception(failure)


def _machine(raw: Mapping[str, Any], user: str) -> Machine:
    return Machine(
        id=str(raw["InstanceId"]),
        state="running" if raw["State"]["Name"] == "running" else "pending",
        host=raw.get("PublicIpAddress"),
        user=user,
        private_host=raw.get("PrivateIpAddress"),
        node=next((str(tag["Value"]) for tag in raw.get("Tags", []) if tag.get("Key") == NODE_TAG), None),
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


@dataclass(frozen=True, slots=True)
class _Gpu:
    name: str | None
    count: int
    vram: float | None
    manufacturer: str | None


def _gpu(raw: Mapping[str, Any]) -> _Gpu:
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


def _architecture(raw: Mapping[str, Any]) -> str | None:
    """The first architecture AWS reports that the vocabulary has a name for.

    ``SupportedArchitectures`` is a list because the older x86 families still
    advertise 32-bit alongside 64-bit, and it is ordered by nothing in
    particular. The mac families report an architecture nobody builds wheels
    against, and an instance type that only offers one of those comes back as no
    answer rather than a wrong one.
    """
    supported = (raw.get("ProcessorInfo") or {}).get("SupportedArchitectures") or ()
    return next((named for arch in supported if (named := architecture(arch))), None)


def _hourly(product: dict[str, Any]) -> float | None:
    terms = (product.get("terms") or {}).get("OnDemand") or {}
    for term in terms.values():
        for dimension in (term.get("priceDimensions") or {}).values():
            price = (dimension.get("pricePerUnit") or {}).get("USD")
            if price and float(price) > 0:
                return float(price)
    return None
