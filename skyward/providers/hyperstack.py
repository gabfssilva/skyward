import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx
import msgspec

from skyward.shared.errors import CapabilityMismatchError
from skyward.shared.provider import Binding, Machine, MachineState, Mount, claimed
from skyward.shared.providers import Hyperstack
from skyward.shared.schemas import ComputeSpec, Endpoint, Market, Offer, Volume
from skyward.worker import bootstrap

BASE_URL = "https://infrahub-api.nexgencloud.com/v1"
FLAVORS_PATH = "/core/flavors"
PRICEBOOK_PATH = "/pricebook"
IMAGES_PATH = "/core/images"
ENVIRONMENTS_PATH = "/core/environments"
KEYPAIRS_PATH = "/core/keypairs"
VMS_PATH = "/core/virtual-machines"
ACCESS_KEYS_PATH = "/object-storage/access-keys"

CPU_VCPU_RATE = "vCPU (cpu-only-flavors)"
CPU_RAM_RATE = "RAM (cpu-only-flavors)"

SSH_USER = "ubuntu"
SSH_RULE = ("tcp", 22, 22, "0.0.0.0/0")


class HyperstackProvider:
    """Hyperstack (NexGen Cloud) — a fixed fleet of flavors with a published pricebook.

    Flavors and prices change on the order of weeks, not minutes, so the catalog is
    cached for an hour; stock is the only fast-moving dimension and it is exposed as
    a hint in ``specific`` rather than as a price.
    """

    kind: ClassVar[str] = "hyperstack"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(hours=1)

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return True

    def __init__(self, provider_id: str, name: str, api_key: str, config: Hyperstack) -> None:
        self._id = provider_id
        self._name = name
        self._api_key = api_key
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        settings = msgspec.convert({**credentials, **config}, Hyperstack)
        if not settings.api_key:
            raise CapabilityMismatchError("hyperstack requires an api_key credential", provider=name)
        return cls(provider_id, name, settings.api_key, settings)

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=self._config.request_timeout,
            headers={"api_key": self._api_key, "Accept": "application/json"},
        ) as client:
            flavors_response = await client.get(FLAVORS_PATH)
            flavors_response.raise_for_status()
            pricebook_response = await client.get(PRICEBOOK_PATH)
            pricebook_response.raise_for_status()

        groups = flavors_response.json().get("data", [])
        prices = _price_map(pricebook_response.json())

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for group in groups:
            for flavor in group.get("flavors", []):
                gpu = str(flavor.get("gpu") or "")
                gpu_count = int(flavor.get("gpu_count") or 0)
                cpus = int(flavor.get("cpu") or 0)
                memory_gb = float(flavor.get("ram") or 0)
                spot = gpu.lower().endswith("-spot")
                price = _hourly_price(gpu, gpu_count, cpus, memory_gb, prices)
                region = flavor.get("region_name") or group.get("region_name")
                features = flavor.get("features") or {}
                regions = self._config.regions
                if regions and region not in regions:
                    continue
                if self._config.network_optimised and (
                    not features.get("network_optimised")
                    or region not in self._config.network_optimised_regions
                ):
                    continue

                yield Offer(
                    id=str(flavor["id"]),
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    billing_unit="hour",
                    instance_type=str(flavor["name"]),
                    accelerator=gpu or None,
                    accelerator_count=gpu_count,
                    cpus=cpus,
                    memory_gb=memory_gb,
                    region=region,
                    disk_gb=float(flavor.get("disk") or 0),
                    spot_price=price if spot else None,
                    on_demand_price=None if spot else price,
                    available=None,
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "flavor_name": flavor["name"],
                        "region": region,
                        "gpu": gpu or None,
                        "stock_available": bool(flavor.get("stock_available", True)),
                        "network_optimised": bool(features.get("network_optimised")),
                        "ephemeral_gb": flavor.get("ephemeral"),
                    },
                )

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        """One environment per compute — it is the region pin, the keypair's scope, and the tag.

        A Hyperstack VM carries no label of its own; what it carries is the environment
        it was created in. So the environment name is what :meth:`machines` recognises a
        compute by, and it is derived from ``compute_id`` rather than invented, which is
        what makes a binding that was never committed recoverable.

        ``market`` buys nothing here: spot is not a purchase mode but a separate flavor,
        whose GPU name ends in ``-spot``. The decision was already spent when the offer
        was picked, and the flavor carries it.
        """
        name = _name(compute_id)
        region = str(offer.specific.get("region") or offer.region or "")
        if not region:
            raise CapabilityMismatchError(f"hyperstack offer {offer.id} carries no region", provider=self._name)

        environment = await self._environment(name, region)
        await self._keypair(name, public_key)

        return {
            "compute_id": compute_id,
            "name": name,
            "environment_id": environment,
            "region": region,
            "flavor": str(offer.specific.get("flavor_name") or offer.instance_type),
            "image": self._config.image or await self._image(region),
            "instance_timeout": self._config.instance_timeout,
        }

    async def launch(self, binding: Binding, market: Market, node: str) -> Machine:
        """One VM, named for the node: Hyperstack numbers the VMs of a ``count`` request off one name, so a node is one call."""
        created = await self._request(
            "POST",
            VMS_PATH,
            {
                "name": f"{binding['name']}-{node}",
                "environment_name": binding["name"],
                "image_name": binding["image"],
                "flavor_name": binding["flavor"],
                "key_name": binding["name"],
                "count": 1,
                "assign_floating_ip": True,
            },
        )
        vm = next(iter(created.get("instances", [])), None)
        if vm is None:
            raise CapabilityMismatchError(f"hyperstack created no VM in {binding['region']}", provider=self._name)
        return Machine(id=str(vm["id"]), state="pending", user=SSH_USER, node=node)

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        """Two ways of recognising the compute's VMs, because both of them are the same tag.

        A Hyperstack VM has no label. What it does carry is the environment it was
        created in and the name it was given, and both are derived from ``compute_id``:
        a VM is this compute's whichever of the two the account-wide listing reports.
        The node's claim is the tail of the name.
        """
        listed = await self._request("GET", VMS_PATH)
        vms = [
            vm
            for vm in listed.get("virtual_machines") or []
            if _belongs(vm, binding["name"])
        ]

        await self._firewall(vms)

        found = (_machine(vm, f"{binding['name']}-") for vm in vms)
        return {machine.id: machine for machine in found if machine is not None}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        async with asyncio.TaskGroup() as group:
            for machine_id in machine_ids:
                group.create_task(self._delete(f"{VMS_PATH}/{machine_id}"))

    async def mount(self, binding: Binding, volumes: tuple[Volume, ...]) -> Mount:
        """Create an object-storage key for this compute and nothing else.

        Hyperstack's object storage is reached with a key minted through the same
        API that sells the machines, so the compute never needs one configured: it
        gets one that is created at bind time, used by every node, and deleted with
        the environment. Path-style addressing, because the endpoint does not serve
        buckets as subdomains.
        """
        region = self._config.object_storage_region
        created = await self._request("POST", ACCESS_KEYS_PATH, {"region": region})

        endpoint = Endpoint(
            url=self._config.object_storage_endpoint,
            access_key=created["access_key"],
            secret_key=created["secret_key"],
            path_style=True,
        )
        return Mount(
            binding_patch={"access_key_id": created["id"]},
            phases=(bootstrap.mounts(tuple((volume, endpoint) for volume in volumes)),),
        )

    async def release(self, binding: Binding) -> None:
        """Deleting the environment takes the keypair with it: the keypair is scoped to it.

        The object-storage key is not scoped to anything, so it is deleted by hand —
        one left behind is a standing credential to the account's buckets.
        """
        if access_key_id := binding.get("access_key_id"):
            with suppress(httpx.HTTPError):
                await self._delete(f"{ACCESS_KEYS_PATH}/{access_key_id}")
        await self._delete(f"{ENVIRONMENTS_PATH}/{binding['environment_id']}")
        timeout = self._config.teardown_timeout
        interval = self._config.teardown_poll_interval
        async with asyncio.timeout(timeout):
            while await self._environment_id(str(binding["name"])) is not None:
                await asyncio.sleep(interval)

    async def _firewall(self, vms: Sequence[Mapping[str, Any]]) -> None:
        """A Hyperstack VM comes up with no ingress at all — SSH and peer traffic are opened here.

        Not at creation: the rules endpoint only answers once the VM is ACTIVE, and the
        peers' private addresses are not known until they are. Rules are diffed against
        the ones the VM already reports, so polling every few seconds does not turn into
        a pile of duplicates.
        """
        peers = tuple(str(vm["fixed_ip"]) for vm in vms if vm.get("fixed_ip"))

        async with asyncio.TaskGroup() as group:
            for vm in vms:
                if str(vm.get("status") or "").upper() == "ACTIVE":
                    group.create_task(self._rules(vm, peers))

    async def _rules(self, vm: Mapping[str, Any], peers: tuple[str, ...]) -> None:
        existing = {
            (
                str(rule.get("protocol")),
                int(rule.get("port_range_min") or 0),
                int(rule.get("port_range_max") or 0),
                str(rule.get("remote_ip_prefix")),
            )
            for rule in vm.get("security_rules") or ()
        }
        wanted = {SSH_RULE} | {
            (protocol, 1, 65535, f"{peer}/32") for peer in peers for protocol in ("tcp", "udp")
        }

        for protocol, port_min, port_max, remote in sorted(wanted - existing):
            with suppress(httpx.HTTPStatusError):
                await self._request(
                    "POST",
                    f"{VMS_PATH}/{vm['id']}/sg-rules",
                    {
                        "remote_ip_prefix": remote,
                        "direction": "ingress",
                        "ethertype": "IPv4",
                        "protocol": protocol,
                        "port_range_min": port_min,
                        "port_range_max": port_max,
                    },
                )

    async def _environment(self, name: str, region: str) -> int:
        if found := await self._environment_id(name):
            return found

        try:
            created = await self._request("POST", ENVIRONMENTS_PATH, {"name": name, "region": region})
        except httpx.HTTPStatusError:
            if found := await self._environment_id(name):
                return found
            raise

        return int((created.get("environment") or created)["id"])

    async def _environment_id(self, name: str) -> int | None:
        listed = await self._request("GET", ENVIRONMENTS_PATH)
        found = next((env for env in listed.get("environments", []) if env.get("name") == name), None)
        return int(found["id"]) if found else None

    async def _keypair(self, name: str, public_key: str) -> None:
        if await self._has_keypair(name):
            return

        try:
            await self._request(
                "POST",
                KEYPAIRS_PATH,
                {"name": name, "environment_name": name, "public_key": public_key},
            )
        except httpx.HTTPStatusError:
            if not await self._has_keypair(name):
                raise

    async def _has_keypair(self, name: str) -> bool:
        listed = await self._request("GET", KEYPAIRS_PATH)
        return any(key.get("name") == name for key in listed.get("keypairs", []))

    async def _image(self, region: str) -> str:
        """Newest Ubuntu that already carries CUDA: installing a driver is minutes of bootstrap."""
        listed = await self._request("GET", IMAGES_PATH, params={"region": region})
        images = [image for group in listed.get("images", []) for image in group.get("images", [])]
        names = [
            str(image["name"])
            for image in sorted(images, key=lambda image: str(image.get("created_at") or ""), reverse=True)
        ]

        cuda = next((name for name in names if "ubuntu" in name.lower() and "cuda" in name.lower()), None)
        ubuntu = next((name for name in names if "ubuntu" in name.lower()), None)

        if chosen := cuda or ubuntu or next(iter(names), None):
            return chosen
        raise CapabilityMismatchError(f"hyperstack has no image in {region}", provider=self._name)

    async def _delete(self, path: str) -> None:
        try:
            await self._request("DELETE", path)
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 404:
                raise

    async def _request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=self._config.request_timeout,
            headers={"api_key": self._api_key, "Accept": "application/json"},
        ) as client:
            response = await client.request(method, path, json=body, params=params)
            response.raise_for_status()
            return response.json() if response.content else {}


def _name(compute_id: str) -> str:
    return f"skyward-{compute_id.replace('_', '-')}"


def _belongs(vm: Mapping[str, Any], name: str) -> bool:
    environment = str((vm.get("environment") or {}).get("name") or "")
    return environment == name or str(vm.get("name") or "").startswith(f"{name}-")


def _machine(vm: Mapping[str, Any], prefix: str) -> Machine | None:
    def at(state: MachineState) -> Machine:
        return Machine(
            id=str(vm["id"]),
            state=state,
            host=vm.get("floating_ip") or None,
            user=SSH_USER,
            private_host=vm.get("fixed_ip") or None,
            node=claimed(vm.get("name"), prefix),
        )

    match str(vm.get("status") or "").upper():
        case "ERROR" | "SHUTOFF" | "HIBERNATED" | "DELETING" | "DELETED":
            return None
        case "ACTIVE" if vm.get("floating_ip"):
            return at("running")
        case _:
            return at("pending")


def _price_map(pricebook: Any) -> dict[str, float]:
    if not isinstance(pricebook, list):
        return {}
    prices: dict[str, float] = {}
    for entry in pricebook:
        name = entry.get("name")
        raw = entry.get("value")
        if not name or raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        prices[name.upper()] = value
    return prices


def _hourly_price(
    gpu: str,
    gpu_count: int,
    cpus: int,
    memory_gb: float,
    prices: dict[str, float],
) -> float | None:
    if gpu and gpu_count:
        per_gpu = prices.get(gpu.upper())
        if per_gpu is None or per_gpu <= 0:
            return None
        return round(per_gpu * gpu_count, 6)

    vcpu_rate = prices.get(CPU_VCPU_RATE.upper(), 0.0)
    ram_rate = prices.get(CPU_RAM_RATE.upper(), 0.0)
    total = vcpu_rate * cpus + ram_rate * memory_gb
    return round(total, 6) if total > 0 else None
