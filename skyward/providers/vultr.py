import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward.application.errors import CapabilityMismatchError
from skyward.application.provider import Binding, Machine
from skyward.protocol.schemas import ComputeSpec, Market, Offer

BASE_URL = "https://api.vultr.com/v2"
PLANS_PATH = "/plans"
METAL_PLANS_PATH = "/plans-metal"
SSH_KEYS_PATH = "/ssh-keys"
INSTANCES = ("/instances", "instances", "instance")
BARE_METALS = ("/bare-metals", "bare_metals", "bare_metal")
PAGE_SIZE = 500
DEFAULT_OS_ID = 2284

VRAM_SUFFIX = re.compile(r"(\d+)vram$")


class VultrProvider:
    """Vultr — a fixed fleet with a published plan list.

    Its offers TTL is hours, not minutes: the catalog is a price list that changes
    when Vultr ships a new plan, not a live market like Vast.ai. The endpoints carry
    no per-region availability at all, so there is nothing here that goes stale fast.
    """

    kind: ClassVar[str] = "vultr"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(hours=6)

    def __init__(self, provider_id: str, name: str, api_key: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._api_key = api_key
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        api_key = credentials.get("api_key")
        if not api_key:
            raise CapabilityMismatchError("vultr requires an api_key credential", provider=name)
        return cls(provider_id, name, api_key, config)

    async def _pages(self, client: httpx.AsyncClient, path: str, key: str) -> AsyncIterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"per_page": PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor
            response = await client.get(path, params=params)
            response.raise_for_status()
            payload = response.json()
            for plan in payload.get(key, []):
                yield plan
            cursor = payload.get("meta", {}).get("links", {}).get("next") or None
            if not cursor:
                return

    async def offers(self) -> AsyncIterator[Offer]:
        headers = {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}
        async with httpx.AsyncClient(
            base_url=BASE_URL,
            headers=headers,
            timeout=int(self._config.get("request_timeout", 30)),
        ) as client:
            now = datetime.now(UTC)
            expires_at = now + self.offers_ttl

            mode = str(self._config.get("mode", "cloud"))
            if mode == "cloud":
                async for plan in self._pages(client, PLANS_PATH, "plans"):
                    for offer in self._offers(plan, cpus=int(plan.get("vcpu_count") or 0), bare_metal=False, now=now, expires_at=expires_at):
                        yield offer
            if mode == "bare-metal":
                async for plan in self._pages(client, METAL_PLANS_PATH, "plans_metal"):
                    for offer in self._offers(plan, cpus=int(plan.get("cpu_count") or 0), bare_metal=True, now=now, expires_at=expires_at):
                        yield offer

    def _offers(
        self,
        plan: dict[str, Any],
        *,
        cpus: int,
        bare_metal: bool,
        now: datetime,
        expires_at: datetime,
    ) -> Iterator[Offer]:
        plan_id = str(plan["id"])
        gpu_type = plan.get("gpu_type") or None
        whole, fraction = _accelerator_count(plan, gpu_type)
        total_vram = float(plan.get("gpu_vram_gb") or 0) or float(_vram_from_id(plan_id))
        disks = max(int(plan.get("disk_count") or 1), 1)
        preemptible = bool(plan.get("deploy_preemptible"))
        locations: list[str | None] = list(plan.get("locations") or []) or [None]

        specific = {
            "plan_type": plan.get("type"),
            "bare_metal": bare_metal,
            "instance_timeout": int(self._config.get("instance_timeout", 300)),
            "gpu_brand": plan.get("gpu_brand"),
            "gpu_type": plan.get("gpu_type"),
            "gpu_vram_gb": plan.get("gpu_vram_gb") or _vram_from_id(plan_id) or None,
            "gpu_fraction": fraction,
            "bandwidth_gb": plan.get("bandwidth"),
            "disk_type": plan.get("disk_type") or (plan.get("type") if bare_metal else None),
            "disk_count": disks,
            "monthly_cost": plan.get("monthly_cost"),
            "deploy_ondemand": bool(plan.get("deploy_ondemand", True)),
            "deploy_preemptible": preemptible,
        }

        for region in locations:
            cost = _cost(plan, region)
            yield Offer(
                id=f"vultr-{plan_id}-{region or 'any'}",
                provider_id=self._id,
                provider_name=self._name,
                kind=self.kind,
                billing_unit="hour",
                instance_type=plan_id,
                accelerator=gpu_type,
                accelerator_count=whole,
                vram=_vram_per_card(total_vram, whole),
                cpus=cpus,
                memory_gb=float(plan.get("ram") or 0) / 1024,
                region=region,
                disk_gb=float(plan.get("disk") or 0) * disks,
                spot_price=cost.get("hourly_cost_preemptible") if preemptible else None,
                on_demand_price=cost.get("hourly_cost"),
                fetched_at=now,
                expires_at=expires_at,
                specific=specific,
            )

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        """Vultr has no tags this adapter can trust on both endpoints, so the compute is written into every label.

        A label is a field ``/instances`` and ``/bare-metals`` both return, which is
        what makes :meth:`machines` able to find a machine whose id nobody wrote down.
        """
        bare_metal = bool(offer.specific.get("bare_metal"))
        region = offer.region or self._config.get("region")

        if not region:
            raise CapabilityMismatchError("vultr needs a region and the offer carries none", provider=self._name)

        if market == "spot" and bare_metal:
            raise CapabilityMismatchError("vultr sells preemptible capacity on cloud plans only", provider=self._name)

        label = _label(compute_id)

        async with self._client() as client:
            ssh_key_id = await self._ssh_key(client, label, public_key)

        return {
            "compute_id": compute_id,
            "label": label,
            "ssh_key_id": ssh_key_id,
            "plan": offer.instance_type,
            "region": region,
            "os_id": int(self._config.get("os_id", DEFAULT_OS_ID)),
            "bare_metal": bare_metal,
        }

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        async with self._client() as client:
            created = await asyncio.gather(
                *(self._create(client, binding, market) for _ in range(count)), return_exceptions=True,
            )

        machines = tuple(item for item in created if isinstance(item, Machine))
        if len(machines) < min_count:
            raise next(item for item in created if isinstance(item, BaseException))

        return binding, machines

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        path, listed, _ = _endpoint(binding)
        prefix = f"{binding['label']}-"

        async with self._client() as client:
            found = [
                _machine(data)
                async for data in self._pages(client, path, listed)
                if str(data.get("label") or "").startswith(prefix)
            ]

        return {machine.id: machine for machine in found}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        path, _, _ = _endpoint(binding)

        async with self._client() as client, asyncio.TaskGroup() as group:
            for machine_id in machine_ids:
                group.create_task(self._delete(client, f"{path}/{machine_id}"))

    async def release(self, binding: Binding) -> None:
        """The keypair is the compute's, not the account's, so it goes with the compute.

        Collected by name as well as by the id in the binding: a crash between the
        create and the commit leaves a key nothing points at, and the name is the
        only thing left that still says whose it was.
        """
        async with self._client() as client:
            stale = [
                str(key["id"])
                async for key in self._pages(client, SSH_KEYS_PATH, "ssh_keys")
                if key.get("name") == binding["label"] or str(key["id"]) == binding["ssh_key_id"]
            ]

            async with asyncio.TaskGroup() as group:
                for key_id in stale:
                    group.create_task(self._delete(client, f"{SSH_KEYS_PATH}/{key_id}"))

    async def _ssh_key(self, client: httpx.AsyncClient, name: str, public_key: str) -> str:
        """The key found under the compute's name is overwritten, not reused as it stands.

        An :meth:`initialize` that runs a second time is one whose binding was lost,
        and the control plane hands it a keypair it has just generated: a key of that
        name left holding the previous public half is a machine nobody can log into.
        """
        async for key in self._pages(client, SSH_KEYS_PATH, "ssh_keys"):
            if key.get("name") != name:
                continue

            key_id = str(key["id"])
            if str(key.get("ssh_key") or "").strip() != public_key.strip():
                updated = await client.patch(f"{SSH_KEYS_PATH}/{key_id}", json={"ssh_key": public_key})
                updated.raise_for_status()
            return key_id

        response = await client.post(SSH_KEYS_PATH, json={"name": name, "ssh_key": public_key})
        response.raise_for_status()
        return str(response.json()["ssh_key"]["id"])

    async def _create(self, client: httpx.AsyncClient, binding: Binding, market: Market) -> Machine:
        path, _, created = _endpoint(binding)
        name = f"{binding['label']}-{uuid.uuid4().hex[:8]}"
        body: dict[str, Any] = {
            "region": binding["region"],
            "plan": binding["plan"],
            "os_id": binding["os_id"],
            "label": name,
            "hostname": name,
            "sshkey_id": [binding["ssh_key_id"]],
            **({"preemptible": True} if market == "spot" else {}),
        }

        response = await client.post(path, json=body)
        response.raise_for_status()

        return _machine(response.json()[created])

    async def _delete(self, client: httpx.AsyncClient, path: str) -> None:
        response = await client.delete(path)
        if response.status_code != 404:
            response.raise_for_status()

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=BASE_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=int(self._config.get("request_timeout", 30)),
        )


def _label(compute_id: str) -> str:
    """The label is also the hostname, and a hostname has no underscore in it."""
    return f"skyward-{compute_id.replace('_', '-')}"


def _endpoint(binding: Binding) -> tuple[str, str, str]:
    return BARE_METALS if binding["bare_metal"] else INSTANCES


def _machine(data: dict[str, Any]) -> Machine:
    host = _routable(data.get("main_ip"))
    ready = data.get("status") == "active" and data.get("power_status") == "running"

    return Machine(
        id=str(data["id"]),
        state="running" if ready and host else "pending",
        host=host,
        private_host=_routable(data.get("internal_ip")),
    )


def _routable(ip: str | None) -> str | None:
    """A Vultr machine that has no address yet reports one anyway, as ``0.0.0.0``."""
    return ip if ip and ip != "0.0.0.0" else None


def _cost(plan: dict[str, Any], region: str | None) -> dict[str, Any]:
    override = (plan.get("location_cost") or {}).get(region) if region else None
    return override or plan


def _vram_per_card(total_vram: float, whole: int) -> float | None:
    """Vultr reports ``gpu_vram_gb`` as the total across the plan's cards.

    An 8x H100 plan says 640; a card has 80. A sliced plan has no whole card to
    divide by and its VRAM already is the slice, so it passes through as it is.
    """
    if not total_vram:
        return None
    return total_vram / whole if whole else total_vram


def _vram_from_id(plan_id: str) -> int:
    match = VRAM_SUFFIX.search(plan_id)
    return int(match.group(1)) if match else 0


def _accelerator_count(plan: dict[str, Any], accelerator: str | None) -> tuple[int, float | None]:
    """Whole GPUs attached, plus the fraction when the plan is a GPU slice.

    Vultr reports ``gpu_count`` as ``"1/8"`` on sliced Cloud GPU plans. A slice gets
    ``0`` whole GPUs — it must not compete with a plan that really hands over a
    device — and carries its share in ``specific``.
    """
    if not accelerator:
        return 0, None

    match plan.get("gpu_count"):
        case int(count):
            return count, float(count)
        case str(count) if "/" in count:
            numerator, _, denominator = count.partition("/")
            return 0, int(numerator) / int(denominator)
        case str(count) if count.isdigit():
            return int(count), float(count)

    return 1, 1.0
