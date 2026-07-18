import asyncio
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward.application.errors import CapabilityMismatchError
from skyward.application.provider import Binding, Machine, MachineState
from skyward.protocol.schemas import ComputeSpec, Market, Offer

BASE_URL = "https://dashboard.tensordock.com"
LOCATIONS_PATH = "/api/v2/locations"
INSTANCES_PATH = "/api/v2/instances"

DEFAULT_STORAGE_GB = 100
GPU_IMAGE = "ubuntu2404_ml_everything"
CPU_IMAGE = "ubuntu2404"
SSH_USER = "user"


class TensorDockProvider:
    """TensorDock — a marketplace of independent hosts, so its catalog is volatile.

    Hosts join, leave and get rented out by other tenants without notice, so the
    TTL is short (5 minutes): the GPU count a location advertises is a snapshot
    of somebody else's fleet, not a reservation.

    Pricing is à la carte — TensorDock bills a per-GPU rate plus per-vCPU,
    per-GB-RAM and per-GB-storage rates. An Offer here is the *whole* available
    slice of one GPU model at one location: all ``max_count`` GPUs, the maximum
    vCPU and RAM a VM of that shape may claim, and ``storage_gb`` of disk. Its
    ``on_demand_price`` is the sum of those four line items, so the price, the
    ``accelerator_count``, the ``cpus`` and the ``memory_gb`` of an Offer all
    describe the same single machine.
    """

    kind: ClassVar[str] = "tensordock"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_token",)
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=5)

    def __init__(self, provider_id: str, name: str, api_token: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._api_token = api_token
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        api_token = credentials.get("api_token")
        if not api_token:
            raise CapabilityMismatchError("tensordock requires an api_token credential", provider=name)
        return cls(provider_id, name, api_token, config)

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
            response = await client.get(
                LOCATIONS_PATH,
                headers={"Authorization": f"Bearer {self._api_token}", "Accept": "application/json"},
            )
            response.raise_for_status()
            locations = response.json().get("data", {}).get("locations", [])

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl
        storage_gb = int(self._config.get("storage_gb", DEFAULT_STORAGE_GB))

        for location in locations:
            for gpu in location.get("gpus", []):
                model = str(gpu.get("v0Name") or "")
                count = int(gpu.get("max_count") or 0)
                if not model or count < 1:
                    continue

                resources = gpu.get("resources") or {}
                pricing = gpu.get("pricing") or {}
                cpus = int(resources.get("max_vcpus") or 0)
                memory_gb = float(resources.get("max_ram_gb") or 0)
                disk_gb = float(min(storage_gb, int(resources.get("max_storage_gb") or storage_gb)))

                price = (
                    count * float(gpu.get("price_per_hr") or 0)
                    + cpus * float(pricing.get("per_vcpu_hr") or 0)
                    + memory_gb * float(pricing.get("per_gb_ram_hr") or 0)
                    + disk_gb * float(pricing.get("per_gb_storage_hr") or 0)
                )

                yield Offer(
                    id=f"{location['id']}:{model}",
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    billing_unit="second",
                    instance_type=str(gpu.get("displayName") or model),
                    accelerator=_accelerator(model),
                    accelerator_count=count,
                    cpus=cpus,
                    memory_gb=memory_gb,
                    region=_region(location),
                    disk_gb=disk_gb,
                    spot_price=None,
                    on_demand_price=price,
                    available=count,
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "location_id": location["id"],
                        "gpu_model": model,
                        "tier": location.get("tier"),
                        "price_per_gpu_hr": gpu.get("price_per_hr"),
                        "per_vcpu_hr": pricing.get("per_vcpu_hr"),
                        "per_gb_ram_hr": pricing.get("per_gb_ram_hr"),
                        "per_gb_storage_hr": pricing.get("per_gb_storage_hr"),
                        "max_vcpus": resources.get("max_vcpus"),
                        "max_ram_gb": resources.get("max_ram_gb"),
                        "max_storage_gb": resources.get("max_storage_gb"),
                        "network_features": gpu.get("network_features"),
                    },
                )

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        """Resolve the machine TensorDock was priced for, and remember it.

        Nothing is created here. TensorDock has no keypair endpoint, no network and
        no security group: the public key travels inline in every create request,
        and the only thing machines of one compute share is the name prefix that
        :meth:`machines` finds them by. So initialize is arithmetic, and being
        called twice costs nothing.

        The shape is copied off the offer rather than off the spec because
        TensorDock bills à la carte: the GPU count, the vCPUs, the RAM and the disk
        that went into ``on_demand_price`` are the ones that must be bought, or the
        machine is one nobody priced.
        """
        return {
            "compute_id": compute_id,
            "prefix": _prefix(compute_id),
            "location_id": offer.specific["location_id"],
            "gpu_model": offer.specific["gpu_model"],
            "gpu_count": offer.accelerator_count,
            "vcpus": offer.cpus,
            "ram_gb": int(offer.memory_gb),
            "storage_gb": int(offer.disk_gb or DEFAULT_STORAGE_GB),
            "image": str(self._config.get("image") or (GPU_IMAGE if offer.accelerator_count else CPU_IMAGE)),
            "public_key": public_key,
        }

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        """Create ``count`` VMs, keeping whatever ``min_count`` of them came up.

        Gathered rather than grouped: a TaskGroup cancels its siblings on the first
        failure, which is the opposite of partial readiness. A create that fails
        after the VM exists is not lost — the name carries the compute id, so the
        next :meth:`machines` finds it and the control plane adopts it.
        """
        created = await asyncio.gather(
            *(self._create(binding) for _ in range(count)), return_exceptions=True,
        )
        machines = tuple(item for item in created if isinstance(item, Machine))
        if len(machines) >= min_count:
            return binding, machines

        failures = [item for item in created if isinstance(item, BaseException)]
        if not failures:
            raise CapabilityMismatchError(
                f"tensordock created {len(machines)} of the {min_count} machines needed", provider=self._name,
            )
        raise failures[0]

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        listed = await self._request("GET", INSTANCES_PATH)
        found = (
            _machine(instance)
            for instance in listed.get("data", {}).get("instances", [])
            if str(instance.get("name") or "").startswith(binding["prefix"])
        )
        return {machine.id: machine for machine in found if machine is not None}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        async with asyncio.TaskGroup() as group:
            for machine_id in machine_ids:
                group.create_task(self._delete(machine_id))

    async def release(self, binding: Binding) -> None:
        """Nothing to give back: :meth:`initialize` created nothing."""

    async def _create(self, binding: Binding) -> Machine:
        body = {
            "data": {
                "type": "virtualmachine",
                "attributes": {
                    "name": f"{binding['prefix']}{uuid.uuid4().hex[:8]}",
                    "type": "virtualmachine",
                    "image": binding["image"],
                    "resources": {
                        "vcpu_count": binding["vcpus"],
                        "ram_gb": binding["ram_gb"],
                        "storage_gb": binding["storage_gb"],
                        "gpus": {binding["gpu_model"]: {"count": binding["gpu_count"]}},
                    },
                    "location_id": binding["location_id"],
                    "ssh_key": binding["public_key"],
                    "port_forwards": [
                        {"internal_port": 22, "external_port": 0},
                        {"internal_port": 25520, "external_port": 0},
                    ],
                },
            },
        }
        created = await self._request("POST", INSTANCES_PATH, json=body)
        instance = created.get("data", created)

        if not (instance_id := str(instance.get("id") or "")):
            raise CapabilityMismatchError("tensordock created an instance and named no id", provider=self._name)
        return Machine(id=instance_id, state="pending", user=SSH_USER)

    async def _delete(self, machine_id: str) -> None:
        await self._request("DELETE", f"{INSTANCES_PATH}/{machine_id}", allow=(404,))

    async def _request(
        self, method: str, path: str, json: Mapping[str, Any] | None = None, allow: tuple[int, ...] = (),
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as client:
            response = await client.request(
                method, path, json=json,
                headers={"Authorization": f"Bearer {self._api_token}", "Accept": "application/json"},
            )
            if response.status_code in allow:
                return {}
            response.raise_for_status()
            return response.json() if response.content else {}


def _machine(instance: Mapping[str, Any]) -> Machine | None:
    """A VM as TensorDock reports it, or nothing at all if it is on its way out.

    TensorDock keeps a stopped or failed VM in the listing, and the control plane
    has no state for that: a machine it cannot use is a machine that is gone.
    """
    state: MachineState
    match str(instance.get("status") or "").lower():
        case "stopped" | "error" | "terminated" | "deleted":
            return None
        case "running":
            state = "running"
        case _:
            state = "pending"

    forwards = instance.get("portForwards") or []
    ssh = next((f for f in forwards if f.get("internal_port") == 22), {})

    return Machine(
        id=str(instance["id"]),
        state=state,
        host=instance.get("ipAddress") or None,
        port=int(ssh.get("external_port") or 22),
        user=SSH_USER,
    )


def _prefix(compute_id: str) -> str:
    """The name is the only tag TensorDock has: it carries no labels, and ids are not remembered.

    Every VM of a compute is named from this, and :meth:`TensorDockProvider.machines`
    finds them by it — including the one whose id was never written down. The
    separator is part of the prefix so that one compute id cannot match another's
    machines by being a prefix of it.
    """
    slug = "".join(char if char.isalnum() else "-" for char in compute_id.lower())
    return f"skyward-{slug}-"


_MODELS = (
    ("h100", "h100"),
    ("h200", "h200"),
    ("a100", "a100"),
    ("l40s", "l40s"),
    ("l40", "l40"),
    ("v100", "v100"),
)


def _accelerator(model: str) -> str | None:
    """Normalize a TensorDock GPU id like ``geforcertx4090-pcie-24gb`` to ``rtx-4090``."""
    if not model:
        return None
    head = model.split("-", 1)[0].lower()
    for needle, name in _MODELS:
        if needle in head:
            return name
    if "rtx" in head:
        return f"rtx-{head.split('rtx', 1)[1]}"
    return head


def _region(location: Mapping[str, Any]) -> str | None:
    parts = [location.get("city"), location.get("stateprovince"), location.get("country")]
    return ", ".join(p for p in parts if p) or None
