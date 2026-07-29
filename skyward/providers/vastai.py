import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward.shared.errors import CapabilityMismatchError
from skyward.shared.provider import Binding, Machine
from skyward.shared.schemas import ComputeSpec, Market, Offer

BASE_URL = "https://console.vast.ai"
SEARCH_PATH = "/api/v0/bundles/"
LABEL_PREFIX = "skyward-"
DEFAULT_IMAGE = "nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04"
DEFAULT_DISK_GB = 100.0
DEFAULT_BID_MULTIPLIER = 1.2

ONSTART = """#!/bin/bash
set -e
mkdir -p /root/.ssh
chmod 700 /root/.ssh
echo "{public_key}" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
chown -R root:root /root/.ssh
tail -f /dev/null &
"""


class VastAIProvider:
    """Vast.ai — a marketplace, so its catalog moves by the minute.

    Its offers TTL is short for that reason: a bundle that was there five
    minutes ago is often gone. Providers with a fixed fleet (AWS) can cache for
    hours; that is exactly why the TTL belongs to the provider and not to the
    cache.
    """

    kind: ClassVar[str] = "vastai"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=2)

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
            raise CapabilityMismatchError("vastai requires an api_key credential", provider=name)
        return cls(provider_id, name, api_key, config)

    def _query(self) -> dict[str, Any]:
        query: dict[str, Any] = {
            "verified": {"eq": bool(self._config.get("verified_only", False))},
            "rentable": {"eq": True},
            "reliability2": {"gte": float(self._config.get("min_reliability", 0.9))},
            "num_gpus": {"gte": 1},
            "order": [["score", "desc"]],
            "type": "on-demand",
            "limit": int(self._config.get("limit", 500)),
        }
        for field, value in (
            ("geolocation", self._config.get("geolocation")),
            ("inet_down", self._config.get("min_inet_down")),
            ("inet_up", self._config.get("min_inet_up")),
        ):
            if value is not None:
                query[field] = {"gte": value} if field.startswith("inet_") else {"eq": value}
        if self._config.get("direct_port"):
            query["direct_port_count"] = {"gte": 1}
        return query

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=int(self._config.get("request_timeout", 30))) as client:
            response = await client.post(
                SEARCH_PATH,
                json=self._query(),
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            bundles = response.json().get("offers", [])

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for bundle in bundles:
            if float(bundle.get("cuda_max_good") or 0) < float(self._config.get("min_cuda", 0)):
                continue
            gpus = int(bundle.get("num_gpus") or 0)
            on_demand = bundle.get("dph_total")
            gpu_name = bundle.get("gpu_name")
            gpu_ram = bundle.get("gpu_ram")
            yield Offer(
                id=str(bundle["id"]),
                provider_id=self._id,
                provider_name=self._name,
                kind=self.kind,
                billing_unit="hour",
                instance_type=str(gpu_name or "cpu"),
                accelerator=gpu_name,
                accelerator_count=gpus,
                vram=float(gpu_ram) / 1024 if gpu_ram else None,
                cpus=int(bundle.get("cpu_cores_effective") or 0),
                memory_gb=float(bundle.get("cpu_ram") or 0) / 1024,
                region=bundle.get("geolocation"),
                disk_gb=float(bundle.get("disk_space") or 0),
                spot_price=bundle.get("min_bid"),
                on_demand_price=float(on_demand) if on_demand is not None else None,
                available=gpus,
                fetched_at=now,
                expires_at=expires_at,
                specific={
                    "gpu_name": gpu_name,
                    "machine_id": bundle.get("machine_id"),
                    "cuda_max_good": bundle.get("cuda_max_good"),
                    "reliability": bundle.get("reliability2"),
                    "direct_port_count": bundle.get("direct_port_count"),
                },
            )

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        """Register the compute's key on the account and pin what every ask will be rented against.

        Nothing else is created: Vast.ai has no network and no security group to
        make, and the offer that was priced is an ask that any other tenant may
        take before we get to it — so the offer id is not pinned, only the shape
        of hardware it stood for.
        """
        async with self._client() as client:
            key_id = await self._register_key(client, public_key)

        return {
            "compute_id": compute_id,
            "label": f"{LABEL_PREFIX}{compute_id}",
            "ssh_key_id": key_id,
            "public_key": public_key,
            "image": spec.image.base or self._config.get("docker_image") or DEFAULT_IMAGE,
            "disk_gb": float(self._config.get("disk_gb", DEFAULT_DISK_GB)),
            "gpu_name": offer.specific.get("gpu_name") or offer.instance_type,
            "gpu_count": offer.accelerator_count,
            "region": self._config.get("geolocation") or offer.region,
            "direct": bool(self._config.get("direct_port", False)),
            "instance_timeout": int(self._config.get("instance_timeout", 300)),
        }

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        """Rent one ask per machine, walking down the list when an ask is taken from under us.

        A bundle is a listing, not a reservation: between the search and the PUT
        another tenant can have it, and the only answer to that is the next ask
        with the same hardware.
        """
        async with self._client() as client:
            asks = await self._asks(client, binding, market)

            machines: list[Machine] = []
            for ask in asks:
                if len(machines) == count:
                    break
                if (machine := await self._rent(client, binding, ask, market)) is not None:
                    machines.append(machine)

            if len(machines) < min_count:
                await self._destroy(client, tuple(machine.id for machine in machines))
                raise CapabilityMismatchError(
                    f"vastai could rent {len(machines)} of {min_count} required {binding['gpu_name']} machines",
                    provider=self._name,
                )

        return binding, tuple(machines)

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        async with self._client() as client:
            response = await client.get("/api/v0/instances/", params={"owner": "me"})
            response.raise_for_status()
            instances = response.json().get("instances") or []

        found = (
            _machine(instance)
            for instance in instances
            if instance.get("label") == binding["label"]
        )
        return {machine.id: machine for machine in found if machine is not None}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        async with self._client() as client:
            await self._destroy(client, machine_ids)

    async def release(self, binding: Binding) -> None:
        """Drop the key the compute was given. It is the only account-wide thing it owned."""
        if (key_id := binding.get("ssh_key_id")) is None:
            return

        async with self._client() as client:
            await client.delete(f"/api/v0/ssh/{key_id}/")

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=int(self._config.get("request_timeout", 30)),
            headers={"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"},
        )

    async def _register_key(self, client: httpx.AsyncClient, public_key: str) -> int | None:
        """Register the key, and read "duplicate" as "registered": initialize runs again after a crash."""
        created = await client.post("/api/v0/ssh/", json={"ssh_key": public_key})

        if created.status_code < 400:
            match created.json():
                case {"ssh_key_id": int() | str() as key_id} | {"id": int() | str() as key_id}:
                    return int(key_id)
                case _:
                    pass

        listed = await client.get("/api/v0/ssh/")
        listed.raise_for_status()

        blob = _blob(public_key)
        return next(
            (int(key["id"]) for key in listed.json() if _blob(key["public_key"]) == blob),
            None,
        )

    async def _asks(self, client: httpx.AsyncClient, binding: Binding, market: Market) -> list[dict[str, Any]]:
        """Sorting and region are done here rather than in the query: the geolocation a bundle reports
        ("US, TX") is not the code the search filter takes, and only one of the two prices is the one
        this market will be billed at.
        """
        spot = market == "spot"
        query: dict[str, Any] = {
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "gpu_name": {"eq": binding["gpu_name"]},
            "num_gpus": {"gte": binding["gpu_count"]},
            "reliability": {"gt": float(self._config.get("min_reliability", 0.95))},
            "disk_space": {"gte": binding["disk_gb"]},
            "allocated_storage": 5.0,
            "type": "bid" if spot else "on-demand",
            "order": [["score", "desc"]],
            "limit": 64,
        }
        if self._config.get("verified_only", True):
            query["verified"] = {"eq": True}
        if binding["direct"]:
            query["direct_port_count"] = {"gte": 1}
        if (minimum := self._config.get("min_inet_down")) is not None:
            query["inet_down"] = {"gte": minimum}
        if (minimum := self._config.get("min_inet_up")) is not None:
            query["inet_up"] = {"gte": minimum}

        response = await client.post(SEARCH_PATH, json=query)
        response.raise_for_status()

        region = binding["region"]
        price = "min_bid" if spot else "dph_total"
        asks = [
            ask for ask in response.json().get("offers", [])
            if (region is None or ask.get("geolocation") == region)
            and float(ask.get("cuda_max_good") or 0) >= float(self._config.get("min_cuda", 0))
        ]
        return sorted(asks, key=lambda ask: ask.get(price) or float("inf"))

    async def _rent(self, client: httpx.AsyncClient, binding: Binding, ask: dict[str, Any], market: Market) -> Machine | None:
        body: dict[str, Any] = {
            "client_id": "me",
            "image": binding["image"],
            "disk": binding["disk_gb"],
            "runtype": "ssh_direc" if binding["direct"] else "ssh_proxy",
            "label": binding["label"],
            "onstart": ONSTART.format(public_key=binding["public_key"]),
        }

        if market == "spot":
            if (min_bid := ask.get("min_bid")) is None:
                return None
            body["price"] = float(min_bid) * float(self._config.get("bid_multiplier", DEFAULT_BID_MULTIPLIER))

        response = await client.put(f"/api/v0/asks/{ask['id']}/", json=body)
        if response.status_code >= 400:
            return None

        created = response.json()
        match created.get("new_contract") or created.get("id"):
            case int() | str() as contract:
                return Machine(id=str(contract), state="pending")
            case _:
                return None

    async def _destroy(self, client: httpx.AsyncClient, machine_ids: tuple[str, ...]) -> None:
        async with asyncio.TaskGroup() as group:
            for machine_id in machine_ids:
                group.create_task(
                    client.request("DELETE", f"/api/v0/instances/{machine_id}/", json={}),
                )


def _blob(public_key: str) -> str:
    """Compare keys on the material only: Vast.ai keeps the comment, and the comment is not the key."""
    parts = public_key.strip().split()
    return parts[1] if len(parts) >= 2 else public_key.strip()


def _machine(instance: dict[str, Any]) -> Machine | None:
    machine_id = str(instance["id"])
    ports = instance.get("ports") or {}
    mapped = (ports.get("22/tcp") or [{}])[0].get("HostPort")

    match (instance.get("public_ipaddr"), mapped):
        case (str() as address, int() | str() as mapped_port) if address:
            host, port = address, int(mapped_port)
        case _:
            host, port = instance.get("ssh_host") or "", int(instance.get("ssh_port") or 22)

    match instance.get("actual_status"):
        case "exited" | "error" | "destroyed":
            return None
        case "running" if host:
            return Machine(id=machine_id, state="running", host=host, port=port)
        case _:
            return Machine(id=machine_id, state="pending")
