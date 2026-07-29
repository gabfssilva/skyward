import asyncio
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward.application.errors import CapabilityMismatchError
from skyward.application.provider import Binding, Machine
from skyward.protocol.schemas import ComputeSpec, Market, Offer

BASE_URL = "https://api.verda.com/v1"
TOKEN_PATH = "/oauth2/token"
INSTANCE_TYPES_PATH = "/instance-types"
AVAILABILITY_PATH = "/instance-availability"
SSH_KEYS_PATH = "/ssh-keys"
INSTANCES_PATH = "/instances"

DEFAULT_CUDA = "13.0"
CUDA_IMAGE = "ubuntu-24.04-cuda-{cuda}-open"
CPU_IMAGE = "ubuntu-24.04"


class VerdaProvider:
    """Verda — a fixed fleet with a published price list, so the catalog is stable.

    Prices only change when Verda changes them; what actually moves is regional
    availability, which is why the TTL is 15 minutes rather than hours or minutes.
    """

    kind: ClassVar[str] = "verda"
    credential_fields: ClassVar[tuple[str, ...]] = ("client_id", "client_secret")
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=15)

    def __init__(
        self,
        provider_id: str,
        name: str,
        client_id: str,
        client_secret: str,
        config: Mapping[str, Any],
    ) -> None:
        self._id = provider_id
        self._name = name
        self._client_id = client_id
        self._client_secret = client_secret
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        client_id = credentials.get("client_id")
        client_secret = credentials.get("client_secret")
        if not client_id or not client_secret:
            raise CapabilityMismatchError(
                "verda requires client_id and client_secret credentials", provider=name
            )
        return cls(provider_id, name, client_id, client_secret, config)

    async def _token(self, client: httpx.AsyncClient) -> str:
        response = await client.post(
            TOKEN_PATH,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
        )
        response.raise_for_status()
        return response.json()["access_token"]

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=int(self._config.get("request_timeout", 30))) as client:
            headers = {"Authorization": f"Bearer {await self._token(client)}"}

            types_response = await client.get(INSTANCE_TYPES_PATH, headers=headers)
            types_response.raise_for_status()
            instance_types = types_response.json() or []

            on_demand_regions = await _availability(client, headers, is_spot=False)
            spot_regions = await _availability(client, headers, is_spot=True)

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for entry in instance_types:
            name = str(entry["instance_type"])
            gpu = entry.get("gpu") or {}
            gpu_count = int(gpu.get("number_of_gpus") or 0)
            total_vram = float((entry.get("gpu_memory") or {}).get("size_in_gigabytes") or 0)
            vram_per_card = total_vram / gpu_count if gpu_count else None
            on_demand_price = _price(entry.get("price_per_hour"))
            spot_price = _price(entry.get("spot_price"))

            regions = on_demand_regions.get(name, frozenset()) | spot_regions.get(name, frozenset())
            for region in sorted(regions) or [None]:
                spot_here = spot_price if region is None or region in spot_regions.get(name, frozenset()) else None
                on_demand_here = (
                    on_demand_price
                    if region is None or region in on_demand_regions.get(name, frozenset())
                    else None
                )
                yield Offer(
                    id=f"{name}:{region}" if region else name,
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    billing_unit="hour",
                    instance_type=name,
                    accelerator=gpu.get("description"),
                    accelerator_count=gpu_count,
                    vram=vram_per_card,
                    cpus=int((entry.get("cpu") or {}).get("number_of_cores") or 0),
                    memory_gb=float((entry.get("memory") or {}).get("size_in_gigabytes") or 0),
                    region=region,
                    disk_gb=None,
                    spot_price=spot_here,
                    on_demand_price=on_demand_here,
                    available=None,
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "gpu_description": gpu.get("description"),
                        "gpu_memory_gb_per_card": vram_per_card,
                        "storage": (entry.get("storage") or {}).get("description"),
                        "supported_os": entry.get("supported_os", []),
                        "location_code": region,
                        "spot_available": region in spot_regions.get(name, frozenset()),
                    },
                )

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        """Verda has no tags, so the compute is written into every hostname it creates.

        A hostname is the only field of an instance the account-wide listing gives
        back and that we get to choose, which makes it the only way to find a machine
        that was created by a process that died before it could write the id down.
        """
        name = f"skyward-{compute_id}"
        image = self._config.get("image") or (
            CUDA_IMAGE.format(cuda=self._config.get("cuda", DEFAULT_CUDA))
            if offer.accelerator_count
            else CPU_IMAGE
        )

        async with self._session() as (client, headers):
            configured_key = self._config.get("ssh_key_id")
            key_id = str(configured_key) if configured_key else await self._ssh_key(client, headers, name, public_key)
            region = await self._region(client, headers, offer, market)

        return {
            "compute_id": compute_id,
            "prefix": _prefix(compute_id),
            "ssh_key_id": key_id,
            "instance_type": offer.instance_type,
            "image": image,
            "region": region,
            "managed_ssh_key": configured_key is None,
            "instance_timeout": int(self._config.get("instance_timeout", 300)),
        }

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        async with self._session() as (client, headers), asyncio.TaskGroup() as group:
            created = [group.create_task(self._create(client, headers, binding, market)) for _ in range(count)]

        machines: list[Machine] = []
        refused: list[str] = []
        for task in created:
            match task.result():
                case Machine() as machine:
                    machines.append(machine)
                case str() as reason:
                    refused.append(reason)

        if len(machines) < min_count:
            raise CapabilityMismatchError(
                f"verda created {len(machines)} of the {min_count} machines the compute cannot start "
                f"without: {'; '.join(refused)}",
                provider=self._name,
            )
        return binding, tuple(machines)

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        async with self._session() as (client, headers):
            response = await client.get(INSTANCES_PATH, headers=headers)
            response.raise_for_status()
            listed = response.json() or []

        found = (
            _machine(entry)
            for entry in listed
            if str(entry.get("hostname") or "").startswith(binding["prefix"])
        )
        return {machine.id: machine for machine in found if machine is not None}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        async with self._session() as (client, headers), asyncio.TaskGroup() as group:
            for machine_id in machine_ids:
                group.create_task(self._delete(client, headers, machine_id))

    async def release(self, binding: Binding) -> None:
        if not binding.get("managed_ssh_key", True):
            return
        async with self._session() as (client, headers):
            deleted = await client.delete(f"{SSH_KEYS_PATH}/{binding['ssh_key_id']}", headers=headers)
            if deleted.is_client_error:
                return
            deleted.raise_for_status()

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[tuple[httpx.AsyncClient, dict[str, str]]]:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=int(self._config.get("request_timeout", 30))) as client:
            yield client, {"Authorization": f"Bearer {await self._token(client)}"}

    async def _ssh_key(
        self, client: httpx.AsyncClient, headers: dict[str, str], name: str, public_key: str
    ) -> str:
        """The key is named after the compute, so a second initialize finds the first one's.

        The lookup is repeated after a refused create because the two daemons that can
        both be initializing this compute are not serialized against each other, and a
        name Verda already holds is the answer, not a failure.
        """
        if existing := await self._find_key(client, headers, name):
            return existing

        created = await client.post(
            SSH_KEYS_PATH, headers=headers, json={"name": name, "key": public_key}
        )
        if created.is_error:
            if existing := await self._find_key(client, headers, name):
                return existing
            created.raise_for_status()
        return _identifier(created)

    async def _find_key(
        self, client: httpx.AsyncClient, headers: dict[str, str], name: str
    ) -> str | None:
        listed = await client.get(SSH_KEYS_PATH, headers=headers)
        listed.raise_for_status()
        found = next((key for key in listed.json() or [] if key.get("name") == name), None)
        return str(found["id"]) if found else None

    async def _region(
        self, client: httpx.AsyncClient, headers: dict[str, str], offer: Offer, market: Market
    ) -> str:
        available = await _availability(client, headers, is_spot=market == "spot")
        regions = available.get(offer.instance_type, frozenset())
        preferred = offer.region or self._config.get("region")

        if preferred in regions:
            return str(preferred)
        if fallback := next(iter(sorted(regions)), None):
            return fallback
        raise CapabilityMismatchError(
            f"no verda region has {offer.instance_type} available on the {market} market",
            provider=self._name,
        )

    async def _create(
        self, client: httpx.AsyncClient, headers: dict[str, str], binding: Binding, market: Market
    ) -> Machine | str:
        """A refusal comes back as its own reason rather than as an exception.

        One POST buys one instance, so a fleet that comes back short is a fleet of
        failed POSTs, and ``min_count`` only means something if the ones that
        succeeded survive the ones that did not.
        """
        response = await client.post(
            INSTANCES_PATH,
            headers=headers,
            json={
                "instance_type": binding["instance_type"],
                "image": binding["image"],
                "ssh_key_ids": [binding["ssh_key_id"]],
                "location_code": binding["region"],
                "is_spot": market == "spot",
                "hostname": f"{binding['prefix']}{uuid.uuid4().hex[:8]}",
                "description": f"skyward compute {binding['compute_id']}",
            },
        )
        if response.is_error:
            return f"{response.status_code} {response.text.strip()}"
        return Machine(id=_identifier(response), state="pending")

    async def _delete(self, client: httpx.AsyncClient, headers: dict[str, str], machine_id: str) -> None:
        """Volumes outlive the instance that carried them unless they are named in the delete.

        ``delete_permanently`` is only honoured together with ``volume_ids``, which is
        why the instance has to be read back before it can be taken down for good.

        A refused read is an instance already gone; a broken one is not, and swallowing
        it would leave a machine billing behind a node the control plane has written off.
        """
        instance = await client.get(f"{INSTANCES_PATH}/{machine_id}", headers=headers)
        if instance.is_client_error:
            return
        instance.raise_for_status()

        volume_ids = (instance.json() or {}).get("volume_ids") or []
        body: dict[str, Any] = {"id": machine_id, "action": "delete"}
        if volume_ids:
            body |= {"volume_ids": volume_ids, "delete_permanently": True}

        deleted = await client.put(INSTANCES_PATH, headers=headers, json=body)
        if deleted.is_client_error:
            return
        deleted.raise_for_status()


def _prefix(compute_id: str) -> str:
    return f"skyward-{compute_id.replace('_', '-')}-"


def _identifier(response: httpx.Response) -> str:
    """Verda answers a creation with the bare id, quoted, and not always as JSON."""
    body = response.text.strip()
    if not body.startswith(("{", "[")):
        return body.strip('"')
    return str(response.json()["id"])


def _machine(entry: Mapping[str, Any]) -> Machine | None:
    host = entry.get("ip") or None
    match entry.get("status"):
        case "error" | "discontinued" | "deleted":
            return None
        case "running":
            return Machine(id=str(entry["id"]), state="running", host=host)
        case _:
            return Machine(id=str(entry["id"]), state="pending", host=host)


async def _availability(
    client: httpx.AsyncClient, headers: Mapping[str, str], is_spot: bool
) -> dict[str, frozenset[str]]:
    response = await client.get(
        AVAILABILITY_PATH, headers=dict(headers), params={"is_spot": str(is_spot).lower()}
    )
    response.raise_for_status()
    by_type: dict[str, set[str]] = {}
    for region in response.json() or []:
        code = region["location_code"]
        for instance_type in region.get("availabilities", []):
            by_type.setdefault(instance_type, set()).add(code)
    return {k: frozenset(v) for k, v in by_type.items()}


def _price(raw: object) -> float | None:
    if raw is None:
        return None
    try:
        return float(str(raw))
    except ValueError:
        return None
