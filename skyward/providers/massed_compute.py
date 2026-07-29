import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward.application.errors import CapabilityMismatchError
from skyward.application.provider import Binding, Machine, MachineState
from skyward.protocol.schemas import ComputeSpec, Market, Offer

BASE_URL = "https://vm.massedcompute.com/api/v1"
INVENTORY_PATH = "/gpu-inventory"
INSTANCES_PATH = "/instance"
LAUNCH_PATH = "/instance/launch"
TERMINATE_PATH = "/instance/terminate"
KEYS_PATH = "/ssh-keys"

DEFAULT_IMAGE_ID = 184
DEFAULT_USER = "Ubuntu"
ANY_REGION = "any"

PRODUCT_PATTERN = re.compile(r"^gpu_(\d+)x_(.+)$", re.IGNORECASE)
VARIANT_SUFFIXES = ("_spot", "_low_ram", "_high_ram")


class MassedComputeProvider:
    """Massed Compute — a fixed US fleet with a small, slow-moving product catalog.

    Prices are published per product and change on the order of weeks, but the
    per-product capacity counters move whenever someone rents a box, so the TTL
    is minutes rather than hours: the numbers a user sorts by are stable, the
    availability they act on is not.
    """

    kind: ClassVar[str] = "massed_compute"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=10)

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return False

    def __init__(self, provider_id: str, name: str, api_key: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._api_key = api_key
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        api_key = credentials.get("api_key")
        if not api_key:
            raise CapabilityMismatchError("massed_compute requires an api_key credential", provider=name)
        return cls(provider_id, name, api_key, config)

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=int(self._config.get("request_timeout", 30))) as client:
            response = await client.get(
                INVENTORY_PATH,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()
            inventory: dict[str, Any] = response.json().get("gpu_inventory", {})

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for product, item in inventory.items():
            instance_type = item["instance_type"]
            specs = instance_type["specs"]
            price = float(instance_type["price_cents_per_hour"]) / 100.0
            spot = _is_spot(product)
            accelerator, count = _gpu(product)
            capacity = int(item.get("capacity_available") or 0)
            regions = [region["name"] for region in item.get("regions_with_capacity_available", [])]

            for region in regions or [None]:
                yield Offer(
                    id=f"{product}:{region}" if region else product,
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    billing_unit="minute",
                    instance_type=product,
                    accelerator=accelerator,
                    accelerator_count=count,
                    cpus=int(specs["vcpu_count"]),
                    memory_gb=float(specs["memory_gib"]),
                    region=region,
                    disk_gb=float(specs["storage_gb"]),
                    spot_price=price if spot else None,
                    on_demand_price=None if spot else price,
                    available=capacity,
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "product_name": product,
                        "description": instance_type.get("description"),
                        "spot": spot,
                        "regions_with_capacity": regions,
                        "price_cents_per_hour": instance_type["price_cents_per_hour"],
                    },
                )

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        """Register the compute's key and pin the product it will buy.

        The market is not a launch parameter here: spot is a separate product
        (``gpu_8x_h100_spot``), so it was already decided when the offer was
        picked, and buying the product the offer names is buying the price it
        quoted.

        The key is named after the compute, not after the key's own fingerprint:
        two computes then never share one, and :meth:`release` can delete it
        without asking whether anybody else is still holding it.
        """
        async with self._client() as client:
            key_name = _key_name(compute_id)
            key_id = await self._key(client, key_name, public_key)

        return {
            "compute_id": compute_id,
            "product": offer.instance_type,
            "image_id": int(self._config.get("image_id", DEFAULT_IMAGE_ID)),
            "ssh_key_id": key_id,
            "ssh_key_name": key_name,
            "prefix": _prefix(compute_id),
        }

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        """Ask for one machine per request, and read a refusal as one we did not get.

        A failure cannot be allowed to cancel its siblings: a launch cancelled
        mid-flight is a machine that may well exist and whose uuid nobody ever
        read, which is why the exceptions are collected instead of raised. What
        makes a partial fleet fatal is ``min_count``, and nothing else.
        """
        async with self._client() as client:
            attempts = [self._launch(client, binding) for _ in range(count)]
            results = await asyncio.gather(*attempts, return_exceptions=True)

        machines = tuple(result for result in results if isinstance(result, Machine))
        if len(machines) < min_count:
            failures = "; ".join(str(result) for result in results if isinstance(result, BaseException))
            raise CapabilityMismatchError(
                f"massed_compute launched {len(machines)} of the {min_count} machines required: {failures}",
                provider=self._name,
            )

        return binding, machines

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        """Massed Compute has no tags, so the instance name carries the compute id.

        It is the only carrier there is, and it is what makes a machine created by
        a process that died before it could commit the uuid findable at all.
        """
        async with self._client() as client:
            response = await client.get(INSTANCES_PATH)
            response.raise_for_status()
            instances: list[dict[str, Any]] = response.json().get("runningInstances", [])

        mine = (data for data in instances if str(data.get("name") or "").startswith(binding["prefix"]))
        found = (machine for data in mine if (machine := _machine(data)))
        return {machine.id: machine for machine in found}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        if not machine_ids:
            return

        async with self._client() as client:
            response = await client.post(TERMINATE_PATH, json={"instanceUuids": list(machine_ids)})
            if response.status_code != 404:
                response.raise_for_status()

    async def release(self, binding: Binding) -> None:
        async with self._client() as client:
            response = await client.delete(f"{KEYS_PATH}/{binding['ssh_key_id']}")
            if response.status_code != 404:
                response.raise_for_status()

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=int(self._config.get("request_timeout", 30)),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            },
        )

    async def _key(self, client: httpx.AsyncClient, name: str, public_key: str) -> str:
        """Find the compute's key, or make it — and take losing the race as finding it.

        Two processes reconciling the same compute both create, one of them is
        told the name is taken, and that answer is the id it was after.
        """
        if key_id := await self._find_key(client, name):
            return key_id

        created = await client.post(KEYS_PATH, json={"name": name, "publicKey": public_key})
        if not created.is_error:
            return str(created.json()["sshKey"]["id"])

        if key_id := await self._find_key(client, name):
            return key_id

        raise CapabilityMismatchError(
            f"massed_compute refused the compute's ssh key: {created.status_code} {created.text}",
            provider=self._name,
        )

    async def _find_key(self, client: httpx.AsyncClient, name: str) -> str | None:
        listed = await client.get(KEYS_PATH)
        listed.raise_for_status()
        keys: list[dict[str, Any]] = listed.json().get("sshKeys", [])
        return next((str(key["id"]) for key in keys if key.get("name") == name), None)

    async def _launch(self, client: httpx.AsyncClient, binding: Binding) -> Machine:
        response = await client.post(
            LAUNCH_PATH,
            json={
                "productName": binding["product"],
                "regionName": ANY_REGION,
                "imageId": binding["image_id"],
                "instanceName": f"{binding['prefix']}{uuid.uuid4().hex[:8]}",
                "sshKeys": [binding["ssh_key_name"]],
            },
        )
        if response.is_error:
            raise CapabilityMismatchError(
                f"massed_compute refused the launch: {response.status_code} {response.text}",
                provider=self._name,
            )

        match response.json().get("response"):
            case str() as instance_id if instance_id:
                return Machine(id=instance_id, state="pending")
            case other:
                raise CapabilityMismatchError(
                    f"massed_compute launched an instance without a uuid: {other}",
                    provider=self._name,
                )


def _prefix(compute_id: str) -> str:
    return f"skyward-{compute_id}-"


def _key_name(compute_id: str) -> str:
    """Massed Compute SSH key names are alphanumeric — hyphens are rejected."""
    return f"skyward{re.sub(r'[^a-zA-Z0-9]', '', compute_id)}"


def _machine(data: Mapping[str, Any]) -> Machine | None:
    """Massed Compute reports an address before the box has finished booting.

    ``os_booted`` is the flag that says the image came up, and it is the one the
    machine is called running on; the ip alone is not, and neither is a status
    that only says the rental exists.
    """
    host = data.get("ip") or None

    match data.get("status"):
        case "terminated" | "error" | "Stopped":
            return None
        case "Running" if data.get("os_booted") == 1 and host:
            state: MachineState = "running"
        case _:
            state = "pending"

    return Machine(
        id=str(data["uuid"]),
        state=state,
        host=host,
        user=str(data.get("username") or DEFAULT_USER),
    )


def _is_spot(product: str) -> bool:
    return product.lower().endswith("_spot")


def _gpu(product: str) -> tuple[str | None, int]:
    """Pull the GPU token and its count out of a product id (``gpu_8x_a6000_spot``).

    A Massed Compute product id is not a GPU name — the count, the rental
    variant and the model are packed into one string, and unpacking that is
    provider knowledge. Naming the GPU is not: the token comes out raw
    (``h100_nvl``, ``a6000``) and the shared vocabulary canonicalizes it.
    """
    match = PRODUCT_PATTERN.match(product)
    if not match:
        return None, 0

    count = int(match.group(1))
    model = match.group(2).lower()

    changed = True
    while changed:
        changed = False
        for suffix in VARIANT_SUFFIXES:
            if model.endswith(suffix):
                model = model[: -len(suffix)]
                changed = True

    return model or None, count
