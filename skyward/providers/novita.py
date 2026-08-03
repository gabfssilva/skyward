import asyncio
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx
import msgspec

from skyward.shared.errors import CapabilityMismatchError
from skyward.shared.provider import Binding, Machine
from skyward.shared.providers import Novita
from skyward.shared.schemas import ComputeSpec, Market, Offer

BASE_URL = "https://api.novita.ai/gpu-instance/openapi/v1"
PRODUCTS_PATH = "/products"
CREATE_PATH = "/gpu/instance/create"
DELETE_PATH = "/gpu/instance/delete"
INSTANCE_PATH = "/gpu/instance"
INSTANCES_PATH = "/gpu/instances"

PRICE_DIVISOR = 100_000
DEFAULT_IMAGE = "nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04"
PAGE_SIZE = 100
GONE = ("removed", "exited", "terminated", "error", "failed")


class NovitaProvider:
    """Novita.ai — a fixed GPU fleet with a small, slow-moving product catalog.

    The product list itself barely changes; only ``inventoryState`` and the spot
    price move, so a 15 minute TTL keeps availability roughly honest without
    re-querying for every question.
    """

    kind: ClassVar[str] = "novita"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=15)

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return False

    def __init__(self, provider_id: str, name: str, api_key: str, config: Novita) -> None:
        self._id = provider_id
        self._name = name
        self._api_key = api_key
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        settings = msgspec.convert({**credentials, **config}, Novita)
        if not settings.api_key:
            raise CapabilityMismatchError("novita requires an api_key credential", provider=name)
        return cls(provider_id, name, settings.api_key, settings)

    async def offers(self) -> AsyncIterator[Offer]:
        params: dict[str, Any] = {}
        if cluster_id := self._config.cluster_id:
            params["clusterId"] = cluster_id

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=self._config.request_timeout) as client:
            response = await client.get(
                PRODUCTS_PATH,
                params=params,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            products = response.json().get("data", [])

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for product in products:
            regions = product.get("regions") or [None]
            for region in regions:
                yield self._to_offer(product, region, now, expires_at)

    def _to_offer(
        self,
        product: dict[str, Any],
        region: str | None,
        now: datetime,
        expires_at: datetime,
    ) -> Offer:
        raw_name = str(product.get("name") or "unknown")
        billing = product.get("billingMethods") or []
        spot = _price(product.get("spotPrice")) if "spot" in billing else None
        return Offer(
            id=f"{product['id']}:{region or 'any'}",
            provider_id=self._id,
            provider_name=self._name,
            kind=self.kind,
            billing_unit="hour",
            instance_type=raw_name,
            accelerator=raw_name,
            accelerator_count=1,
            cpus=int(product.get("cpuPerGpu") or 0),
            memory_gb=float(product.get("memoryPerGpu") or 0),
            region=region,
            disk_gb=float(product.get("diskPerGpu") or 0),
            spot_price=spot,
            on_demand_price=_price(product.get("price")),
            available=_available(product),
            fetched_at=now,
            expires_at=expires_at,
            specific={
                "product_id": str(product["id"]),
                "cluster_name": region,
                "billing_methods": billing,
                "inventory_state": product.get("inventoryState"),
                "available_deploy": product.get("availableDeploy"),
                "min_rootfs_gb": product.get("minRootFS"),
                "max_rootfs_gb": product.get("maxRootFS"),
            },
        )

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        """Nothing is created here: Novita has no keypair, network or image to register.

        SSH is not the instance's, it is Novita's — a proxy in front of the
        container, reachable with a root password the instance hands back. The
        public key has nowhere to go, and the binding is only the launch recipe,
        which is what makes this trivially idempotent and :meth:`release` a no-op.
        """
        rootfs_gb = self._config.rootfs_size
        min_rootfs_gb = int(offer.specific.get("min_rootfs_gb") or 0)

        return {
            "compute_id": compute_id,
            "prefix": f"skyward-{compute_id}-",
            "product_id": str(offer.specific["product_id"]),
            "cluster_id": self._config.cluster_id or offer.region,
            "image": spec.image.base or self._config.docker_image or DEFAULT_IMAGE,
            "gpu_num": offer.accelerator_count or 1,
            "rootfs_gb": max(rootfs_gb, min_rootfs_gb),
            "min_cuda_version": self._config.min_cuda_version,
        }

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        async with (
            httpx.AsyncClient(
                base_url=BASE_URL,
                timeout=self._config.request_timeout,
                headers=self._headers(),
            ) as client,
            asyncio.TaskGroup() as group,
        ):
            attempts = [group.create_task(self._create(client, binding, market)) for _ in range(count)]

        results = [task.result() for task in attempts]
        machines = tuple(result for result in results if isinstance(result, Machine))
        if len(machines) < min_count:
            raise ExceptionGroup(
                f"novita created {len(machines)} instances, {min_count} were required",
                [result for result in results if isinstance(result, Exception)],
            )
        return binding, machines

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        """Novita has no tags, so the instance name carries the compute id.

        The list endpoint is not trusted for connection details — the proxy's
        host, port and password are only reliably on the instance detail — so the
        name scan answers "which are mine" and one detail call per instance
        answers "where is it".
        """
        prefix = binding["prefix"]

        async with httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=self._config.request_timeout,
            headers=self._headers(),
        ) as client:
            listed = await self._list(client)
            mine = [
                str(instance["id"])
                for instance in listed
                if instance.get("id") and str(instance.get("name") or "").startswith(prefix)
            ]
            if not mine:
                return {}

            async with asyncio.TaskGroup() as group:
                detailed = [group.create_task(self._detail(client, instance_id)) for instance_id in mine]

        instances = [instance for task in detailed if (instance := task.result())]
        found = (_machine(instance) for instance in instances)
        return {machine.id: machine for machine in found if machine is not None}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        async with (
            httpx.AsyncClient(
                base_url=BASE_URL,
                timeout=self._config.request_timeout,
                headers=self._headers(),
            ) as client,
            asyncio.TaskGroup() as group,
        ):
            for machine_id in machine_ids:
                group.create_task(self._delete(client, machine_id))

    async def release(self, binding: Binding) -> None:
        """Nothing to give back: :meth:`initialize` bought nothing."""

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    async def _create(self, client: httpx.AsyncClient, binding: Binding, market: Market) -> Machine | Exception:
        """Create one instance, and hand a failure back rather than raise it.

        A launch of ``count`` instances is allowed to come back with fewer, and a
        product that has run out of hosts refuses one request at a time: raising
        inside the task group would cancel the siblings that did get a host, whose
        ids would then be lost while they billed.
        """
        body: dict[str, Any] = {
            "productId": binding["product_id"],
            "name": f"{binding['prefix']}{uuid.uuid4().hex[:8]}",
            "imageUrl": binding["image"],
            "gpuNum": binding["gpu_num"],
            "rootfsSize": binding["rootfs_gb"],
            "billingMode": "spot" if market == "spot" else "onDemand",
        }
        if cluster_id := binding["cluster_id"]:
            body["clusterId"] = cluster_id
        if min_cuda := binding["min_cuda_version"]:
            body["minCudaVersion"] = min_cuda

        try:
            response = await client.post(CREATE_PATH, json=body)
            response.raise_for_status()
        except httpx.HTTPError as error:
            return error

        created = response.json() or {}

        match created:
            case {"error": error} | {"message": error} if error:
                return RuntimeError(f"novita refused the instance: {error}")
            case {"id": str(instance_id)} | {"instanceId": str(instance_id)} if instance_id:
                return Machine(id=instance_id, state="pending")
            case _:
                return RuntimeError(f"novita created an instance without an id: {created}")

    async def _list(self, client: httpx.AsyncClient) -> list[dict[str, Any]]:
        instances: list[dict[str, Any]] = []
        page = 1
        while True:
            response = await client.get(INSTANCES_PATH, params={"pageSize": PAGE_SIZE, "pageNumber": page})
            response.raise_for_status()
            payload = response.json() or {}
            batch = payload.get("data") or payload.get("instances") or []
            instances.extend(batch)
            if len(batch) < PAGE_SIZE:
                return instances
            page += 1

    async def _detail(self, client: httpx.AsyncClient, instance_id: str) -> dict[str, Any] | None:
        response = await client.get(INSTANCE_PATH, params={"instanceId": instance_id})
        if response.status_code == 404:
            return None
        response.raise_for_status()

        match response.json() or {}:
            case {"instance": dict() as instance}:
                return instance
            case {"id": str()} as instance:
                return instance
            case _:
                return None

    async def _delete(self, client: httpx.AsyncClient, instance_id: str) -> None:
        response = await client.post(DELETE_PATH, json={"instanceId": instance_id})
        if response.status_code == 404:
            return
        response.raise_for_status()


def _machine(instance: dict[str, Any]) -> Machine | None:
    if instance.get("status") in GONE:
        return None

    ssh = instance.get("connectComponentSSH") or {}
    password = ssh.get("password") or instance.get("sshPassword")
    host, port = _endpoint(ssh.get("sshCommand"))

    if instance.get("status") != "running" or not host:
        return Machine(id=instance["id"], state="pending", password=password)

    return Machine(id=instance["id"], state="running", host=host, port=port, password=password)


def _endpoint(ssh_command: str | None) -> tuple[str | None, int]:
    """Novita publishes the proxy address as a copy-pasteable ``ssh`` command, not as fields."""
    if not ssh_command:
        return None, 22

    parts = ssh_command.split()
    host = next((part.split("@", 1)[1] for part in parts if "@" in part), None)
    port = next((int(value) for flag, value in zip(parts, parts[1:], strict=False) if flag == "-p"), 22)
    return host, port


def _price(raw: str | int | float | None) -> float | None:
    """Novita quotes prices as integer strings in 1/100,000 USD per hour."""
    match raw:
        case str() as s if s.strip():
            try:
                return float(s) / PRICE_DIVISOR
            except ValueError:
                return None
        case int() | float() as n:
            return n / PRICE_DIVISOR
        case _:
            return None


def _available(product: dict[str, Any]) -> int | None:
    if product.get("availableDeploy") is False:
        return 0
    match product.get("inventoryState"):
        case "none":
            return 0
        case _:
            return None
