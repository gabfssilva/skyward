import asyncio
import uuid
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward2.application.errors import CapabilityMismatchError
from skyward2.application.provider import Binding, Machine
from skyward2.protocol.schemas import ComputeSpec, Market, Offer

BASE_URL = "https://backendprod.jarvislabs.net"
SERVER_META_PATH = "/misc/server_meta"

EUROPE_REGION = "europe-01"
EUROPE_GPU_COUNTS = (1, 8)
EUROPE_MIN_STORAGE_GB = 100

DEFAULT_REGION = "india-noida-01"
CLOSED_REGIONS = frozenset({"india-01"})
"""Still served, still billed, and closed to new instances — so it is not on offer.

The old India region answers the account endpoints and keeps running what is
already on it, which is why it stays reachable in ``REGION_URLS`` while never
being sold: an offer that cannot be created is a compute that fails at launch.
"""

REGION_URLS = {
    "india-01": "https://backendprod.jarvislabs.net",
    DEFAULT_REGION: "https://backendn.jarvislabs.net",
    EUROPE_REGION: "https://backendeu.jarvislabs.net",
}

TEMPLATE = "pytorch"
DEFAULT_STORAGE_GB = 50
NAME_PREFIX = "skyward"


class JarvisLabsProvider:
    """Jarvis Labs — an owned fleet, so the catalog is a price list, not a market.

    Its offers TTL is fifteen minutes: prices are set by Jarvis Labs and barely
    move, and only the free-device counts drift as instances come and go.

    The regional backends each mirror the whole fleet, so one call to the India
    backend returns Europe and Chennai too — there is no per-region fan-out.
    """

    kind: ClassVar[str] = "jarvislabs"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=15)

    def __init__(self, provider_id: str, name: str, api_key: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._api_key = api_key
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        api_key = credentials.get("api_key")
        if not api_key:
            raise CapabilityMismatchError("jarvislabs requires an api_key credential", provider=name)
        return cls(provider_id, name, api_key, config)

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
            response = await client.get(
                SERVER_META_PATH,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            entries = response.json().get("server_meta", [])

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for entry in entries:
            gpu_type = str(entry.get("gpu_type") or "")
            if not gpu_type:
                continue

            region = str(entry.get("region") or "")
            if region in CLOSED_REGIONS:
                continue

            workload = entry.get("workload_type")
            unit_price = entry.get("price_per_hour")
            unit_spot = entry.get("spot_price")
            cpus_per_gpu = int(entry.get("cpus_per_gpu") or 0)
            ram_per_gpu = float(entry.get("ram_per_gpu") or 0)
            free = int(entry.get("effective_num_free_devices") or entry.get("num_free_devices") or 0)
            vram = entry.get("vram")

            for count in _gpu_counts(entry, region):
                yield Offer(
                    id=f"{region}:{workload or 'default'}:{gpu_type}:{count}",
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    instance_type=f"{gpu_type}x{count}",
                    accelerator=gpu_type,
                    accelerator_count=count,
                    vram=float(vram) if vram else None,
                    cpus=cpus_per_gpu * count,
                    memory_gb=ram_per_gpu * count,
                    region=region,
                    disk_gb=None,
                    spot_price=float(unit_spot) * count if unit_spot is not None else None,
                    on_demand_price=float(unit_price) * count if unit_price is not None else None,
                    available=free // count,
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "gpu_type": gpu_type,
                        "num_gpus": count,
                        "region": region,
                        "workload_type": workload,
                        "vram_gb": int(vram) if vram else None,
                        "architecture": entry.get("arc"),
                        "spot_only_server": entry.get("spot_only_server"),
                        "reserved_pricing": entry.get("reserved_pricing"),
                    },
                )

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        region = str(offer.specific.get("region") or DEFAULT_REGION)
        storage = int(self._config.get("storage_gb", DEFAULT_STORAGE_GB))
        if region == EUROPE_REGION:
            storage = max(storage, EUROPE_MIN_STORAGE_GB)

        key_name = f"{NAME_PREFIX}-{compute_id}"
        key_id = await self._ensure_key(key_name, public_key)

        return {
            "compute_id": compute_id,
            "region": region,
            "gpu_type": str(offer.specific.get("gpu_type") or offer.accelerator or ""),
            "num_gpus": int(offer.specific.get("num_gpus") or offer.accelerator_count or 1),
            "storage_gb": storage,
            "template": TEMPLATE,
            "ssh_key_id": key_id,
            "ssh_key_name": key_name,
        }

    async def launch(self, binding: Binding, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        async with asyncio.TaskGroup() as group:
            created = [group.create_task(self._create(binding)) for _ in range(count)]

        machines: list[Machine] = []
        refusals: list[Exception] = []
        for task in created:
            match task.result():
                case Machine() as machine:
                    machines.append(machine)
                case Exception() as refusal:
                    refusals.append(refusal)

        if len(machines) < min_count:
            raise CapabilityMismatchError(
                f"jarvislabs created {len(machines)} of the {min_count} machines the compute cannot do without",
                provider=self._name,
            ) from next(iter(refusals), None)

        return binding, tuple(machines)

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        """Jarvis Labs has no tags, so the compute id rides in the instance name.

        The name is the only field the backend echoes back that we get to write,
        which makes it the only way to recognize a machine we created and never
        managed to write down.

        A refusal is raised rather than read as an empty account. The caller reads
        absence as death — a fetch that answers ``success: false`` and is believed
        would report every machine of the compute gone at once, and the compute
        would tear down a fleet that is running perfectly well.
        """
        fetched = await self._request("GET", "/users/fetch")
        if not _succeeded(fetched):
            raise CapabilityMismatchError(
                f"jarvislabs would not list the account's instances: {_message(fetched)}",
                provider=self._name,
            )

        prefix = f"{NAME_PREFIX}-{binding['compute_id']}-"
        found = (
            machine
            for entry in fetched.get("instances", [])
            if str(entry.get("name") or entry.get("instance_name") or "").startswith(prefix)
            if (machine := _machine(entry))
        )
        return {machine.id: machine for machine in found}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        async with asyncio.TaskGroup() as group:
            for machine_id in machine_ids:
                group.create_task(self._destroy(binding, machine_id))

    async def release(self, binding: Binding) -> None:
        """A key that outlives its compute costs nothing, so a refusal here is dropped.

        The opposite of :meth:`terminate`, and for the opposite reason: wedging a
        delete forever on a key that will not go away is the more expensive bug.
        """
        try:
            await self._request("DELETE", f"/ssh/{binding['ssh_key_id']}")
        except httpx.HTTPStatusError:
            return

    async def _destroy(self, binding: Binding, machine_id: str) -> None:
        """The backend refuses in two dialects — an HTTP error, and ``success: false`` at 200.

        Neither is proof the machine is gone, so the machine is asked. Reading a
        refusal as a death is how a machine that is merely unreachable stops being
        anybody's: the node is marked deleted, nothing points at the instance
        again, and it bills until someone reads the invoice.
        """
        try:
            destroyed = await self._request(
                "POST",
                "/misc/destroy",
                base_url=_region_url(binding["region"]),
                params={"machine_id": int(machine_id)},
            )
        except httpx.HTTPStatusError:
            destroyed = None

        if _succeeded(destroyed) or await self._gone(machine_id):
            return

        raise CapabilityMismatchError(
            f"jarvislabs would not destroy machine {machine_id}: {_message(destroyed)}",
            provider=self._name,
        )

    async def _gone(self, machine_id: str) -> bool:
        try:
            fetched = await self._request("GET", f"/users/fetch/{machine_id}")
        except httpx.HTTPStatusError as refusal:
            return refusal.response.status_code == 404
        return not _succeeded(fetched) or not fetched.get("instance")

    async def _ensure_key(self, key_name: str, public_key: str) -> str:
        """Account-wide, not per-instance: every machine gets every key on the account.

        Which is also why the key is named after the compute and deleted with it —
        nothing else distinguishes one compute's key from another's.
        """
        if key_id := await self._find_key(key_name):
            return key_id

        await self._request("POST", "/ssh/", json={"ssh_key": public_key, "key_name": key_name})

        if key_id := await self._find_key(key_name):
            return key_id
        raise CapabilityMismatchError("jarvislabs accepted the ssh key and did not list it back", provider=self._name)

    async def _find_key(self, key_name: str) -> str | None:
        match await self._request("GET", "/ssh/"):
            case list() as keys:
                return next((str(key["key_id"]) for key in keys if key.get("key_name") == key_name), None)
            case _:
                return None

    async def _create(self, binding: Binding) -> Machine | Exception:
        """One refusal out of several is a smaller fleet, not a failed launch — hence ``min_count``.

        The refusal is carried back instead of raised so that one backend saying no
        does not cancel the siblings that said yes, and so that ``launch`` can chain
        it: "created 0 of 1" is not a diagnosis, and "no free H100 in EU1" is.

        A create that reached the backend and whose answer did not reach us is not
        lost: the name carries the compute id, and :meth:`machines` finds it.
        """
        name = f"{NAME_PREFIX}-{binding['compute_id']}-{uuid.uuid4().hex[:6]}"
        try:
            created = await self._request(
                "POST",
                f"/templates/{binding['template']}/create",
                base_url=_region_url(binding["region"]),
                json={
                    "gpu_type": binding["gpu_type"],
                    "num_gpus": binding["num_gpus"],
                    "hdd": binding["storage_gb"],
                    "region": binding["region"],
                    "name": name,
                    "is_reserved": True,
                    "duration": "hour",
                    "disk_type": "ssd",
                    "http_ports": "",
                    "script_id": None,
                    "script_args": "",
                    "fs_id": None,
                    "arguments": "",
                },
            )
        except httpx.HTTPError as refusal:
            return refusal

        if not (machine_id := created.get("machine_id")):
            return CapabilityMismatchError(
                f"jarvislabs accepted the create and named no machine: {_message(created)}",
                provider=self._name,
            )
        return Machine(id=str(machine_id), state="pending")

    async def _request(
        self,
        method: str,
        path: str,
        *,
        base_url: str = BASE_URL,
        json: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        async with httpx.AsyncClient(base_url=base_url, timeout=60) as client:
            response = await client.request(
                method,
                path,
                json=json,
                params=params,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            return response.json()

def _succeeded(payload: object) -> bool:
    """The backend spells the flag two ways — one of them misspelled — and sometimes as a string."""
    match payload:
        case {"success": flag} | {"sucess": flag}:
            return str(flag).lower() == "true"
        case _:
            return False


def _message(payload: object) -> str:
    match payload:
        case {"message": str() as reason} | {"error": str() as reason} | {"detail": str() as reason}:
            return reason
        case _:
            return "unexpected error"


def _machine(entry: Mapping[str, Any]) -> Machine | None:
    host, port = _endpoint(entry)

    match str(entry.get("status") or ""):
        case "Failed":
            return None
        case "Running" if host:
            return Machine(id=str(entry["machine_id"]), state="running", host=host, port=port)
        case _:
            return Machine(id=str(entry["machine_id"]), state="pending")


def _endpoint(entry: Mapping[str, Any]) -> tuple[str | None, int]:
    """The address only ever arrives spelled as a shell command.

    ``ssh -o StrictHostKeyChecking=no -p 11021 root@sshb.jarvislabs.ai``, and the
    port is not the public ip's — the machine sits behind a shared SSH gateway.
    """
    command = str(entry.get("ssh_command") or entry.get("ssh_str") or "")
    if not command:
        return None, 22

    parts = command.split()
    host = next((part.split("@", 1)[1] for part in parts if "@" in part and not part.startswith("-")), None)
    port = next((part for flag, part in zip(parts, parts[1:], strict=False) if flag == "-p"), "22")
    return host, int(port) if port.isdigit() else 22


def _region_url(region: str) -> str:
    return REGION_URLS.get(region, BASE_URL)


def _gpu_counts(entry: Mapping[str, Any], region: str) -> Iterable[int]:
    """GPU counts a single machine of this type can be rented with.

    Jarvis Labs picks the count at provision time, so one catalog entry becomes
    one offer per rentable count. Europe machines only accept 1 or 8 GPUs; the
    other regions expose their per-machine maximum in ``num_gpus``.
    """
    if region == EUROPE_REGION:
        return EUROPE_GPU_COUNTS
    max_gpus = int(entry.get("num_gpus") or 1)
    return range(1, max_gpus + 1)
