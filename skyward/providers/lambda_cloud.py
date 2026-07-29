from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward.application.errors import CapabilityMismatchError
from skyward.application.provider import Binding, Machine, MachineState
from skyward.protocol.schemas import ComputeSpec, Market, Offer

BASE_URL = "https://cloud.lambda.ai/api/v1"
INSTANCE_TYPES_PATH = "/instance-types"
SSH_USER = "ubuntu"


class LambdaProvider:
    """Lambda Cloud — a fixed fleet with a published price list, but scarce capacity.

    Its TTL is short not because prices move (they almost never do) but because
    ``available`` encodes ``regions_with_capacity_available``, which flips within
    minutes as instances are grabbed; the price half of the offer would happily
    cache for a day.
    """

    kind: ClassVar[str] = "lambda"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=5)

    def __init__(self, provider_id: str, name: str, api_key: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._api_key = api_key
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        api_key = credentials.get("api_key")
        if not api_key:
            raise CapabilityMismatchError("lambda requires an api_key credential", provider=name)
        return cls(provider_id, name, api_key, config)

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=int(self._config.get("request_timeout", 30))) as client:
            response = await client.get(
                INSTANCE_TYPES_PATH,
                auth=httpx.BasicAuth(self._api_key, ""),
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            catalog: dict[str, Any] = response.json().get("data", {})

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for type_name, entry in catalog.items():
            info = entry["instance_type"]
            specs = info["specs"]
            gpu_description = info.get("gpu_description")
            regions = [region["name"] for region in entry.get("regions_with_capacity_available", [])]

            for region in regions or [None]:
                yield Offer(
                    id=f"{type_name}:{region}" if region else type_name,
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    billing_unit="minute",
                    instance_type=type_name,
                    accelerator=gpu_description if gpu_description != "N/A" else None,
                    accelerator_count=int(specs.get("gpus") or 0),
                    cpus=int(specs.get("vcpus") or 0),
                    memory_gb=float(specs.get("memory_gib") or 0),
                    region=region,
                    disk_gb=float(specs.get("storage_gib") or 0),
                    spot_price=None,
                    on_demand_price=int(info["price_cents_per_hour"]) / 100.0,
                    available=1 if region else 0,
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "instance_type_name": type_name,
                        "description": info.get("description"),
                        "gpu_description": gpu_description,
                        "regions_with_capacity": regions,
                    },
                )

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        """Import the keypair and pin the region; Lambda has nothing else to create.

        There is no network and no security group to make: an instance comes up with
        a public address and port 22 open. The compute's name is the only carrier
        Lambda offers — it has no tags — and it is stamped on the instances at launch,
        which is what lets :meth:`machines` find one nobody wrote down.
        """
        name = f"skyward-{compute_id}"
        instance_type = str(offer.specific.get("instance_type_name") or offer.instance_type)
        region = offer.region or await self._region(instance_type)

        return {
            "compute_id": compute_id,
            "name": name,
            "instance_type": instance_type,
            "region": region,
            "ssh_key_id": await self._ssh_key(name, public_key),
            "ssh_key_name": name,
        }

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        launched = await self._request(
            "POST",
            "/instance-operations/launch",
            {
                "region_name": binding["region"],
                "instance_type_name": binding["instance_type"],
                "ssh_key_names": [binding["ssh_key_name"]],
                "quantity": count,
                "name": binding["name"],
            },
        )
        ids: list[str] = launched["data"]["instance_ids"]

        if len(ids) < min_count:
            raise CapabilityMismatchError(
                f"lambda launched {len(ids)} of {count} instances in {binding['region']}",
                provider=self._name,
            )

        return binding, tuple(Machine(id=machine_id, state="pending", user=SSH_USER) for machine_id in ids)

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        listed = await self._request("GET", "/instances")
        found = (
            _machine(entry)
            for entry in listed.get("data", [])
            if entry.get("name") == binding["name"]
        )
        return {machine.id: machine for machine in found if machine is not None}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        """Terminate, and read a rejection as an answer only once the machines are shown to be gone.

        Lambda refuses the whole batch when one id is unknown, which is exactly what
        a second attempt looks like. The rejection is only forgiven after the account
        confirms none of the ids is still there — forgiving it blindly would turn a
        malformed request into a fleet that bills forever.
        """
        if not machine_ids:
            return

        try:
            await self._request("POST", "/instance-operations/terminate", {"instance_ids": list(machine_ids)})
        except httpx.HTTPStatusError as error:
            if error.response.status_code not in (400, 404):
                raise
            remaining = await self.machines(binding)
            if any(machine_id in remaining for machine_id in machine_ids):
                raise

    async def release(self, binding: Binding) -> None:
        try:
            await self._request("DELETE", f"/ssh-keys/{binding['ssh_key_id']}")
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 404:
                raise

    async def _ssh_key(self, name: str, public_key: str) -> str:
        """Import the public key under the compute's name, replacing a stale namesake.

        Lambda's keys are account-wide and named, so the name is the whole idempotency
        story — but a second ``initialize`` arrives with a freshly generated keypair,
        and a key of the right name carrying the wrong public key is worse than none:
        the machines come up, sshd answers, and the control plane can never get in.
        A namesake that does not match is therefore dropped, not reused.

        Rejecting a duplicate name is Lambda's answer to a race with another process
        doing the same, and the key it refused to overwrite is read back from the list.
        """
        match await self._find_ssh_key(name):
            case (key_id, listed) if _same_key(listed, public_key):
                return key_id
            case (key_id, _):
                await self._request("DELETE", f"/ssh-keys/{key_id}")
            case _:
                pass

        try:
            created = await self._request("POST", "/ssh-keys", {"name": name, "public_key": public_key})
        except httpx.HTTPStatusError:
            if raced := await self._find_ssh_key(name):
                return raced[0]
            raise

        return str(created["data"]["id"])

    async def _find_ssh_key(self, name: str) -> tuple[str, str] | None:
        listed = await self._request("GET", "/ssh-keys")
        return next(
            ((str(key["id"]), str(key.get("public_key") or "")) for key in listed.get("data", []) if key["name"] == name),
            None,
        )

    async def _region(self, instance_type: str) -> str:
        """Where the offer did not say, take the first region that still has capacity.

        An offer carries a region whenever the catalog had one; an offer without one
        was quoted from a type with no capacity anywhere. Capacity is volatile enough
        that the launch is still worth a shot, so a type with none anywhere falls back
        to ``us-east-3`` and lets the launch, not the lookup, be the one to refuse it.
        """
        catalog = await self._request("GET", INSTANCE_TYPES_PATH)
        entry = catalog.get("data", {}).get(instance_type, {})
        regions = entry.get("regions_with_capacity_available", [])

        if not regions:
            return "us-east-3"
        return str(regions[0]["name"])

    async def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=int(self._config.get("request_timeout", 30))) as client:
            response = await client.request(
                method,
                path,
                json=body,
                auth=httpx.BasicAuth(self._api_key, ""),
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            return response.json() if response.content else {}


def _same_key(listed: str, public_key: str) -> bool:
    """Compare the algorithm and the material, since the trailing comment is not identity."""
    return listed.split()[:2] == public_key.split()[:2] and bool(listed)


def _machine(entry: Mapping[str, Any]) -> Machine | None:
    def at(state: MachineState) -> Machine:
        return Machine(
            id=str(entry["id"]),
            state=state,
            host=entry.get("ip") or None,
            user=SSH_USER,
            private_host=entry.get("private_ip") or None,
        )

    match entry.get("status"):
        case "terminated" | "terminating" | "error" | "unhealthy":
            return None
        case "active":
            return at("running")
        case _:
            return at("pending")
