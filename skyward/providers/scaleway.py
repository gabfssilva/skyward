import asyncio
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from skyward.application.errors import CapabilityMismatchError
from skyward.application.provider import Binding, Machine
from skyward.protocol.schemas import ComputeSpec, Market, Offer

BASE_URL = "https://api.scaleway.com"
PRODUCTS_PATH = "/instance/v1/zones/{zone}/products/servers"
AVAILABILITY_PATH = "/instance/v1/zones/{zone}/products/servers/availability"
SERVERS_PATH = "/instance/v1/zones/{zone}/servers"
IMAGES_PATH = "/instance/v1/zones/{zone}/images"
VOLUMES_PATH = "/block/v1alpha1/zones/{zone}/volumes"
SSH_KEYS_PATH = "/iam/v1alpha1/ssh-keys"

COMPUTE_TAG = "skyward.compute"

DEFAULT_ZONES = (
    "fr-par-1",
    "fr-par-2",
    "fr-par-3",
    "nl-ams-1",
    "nl-ams-2",
    "nl-ams-3",
    "pl-waw-1",
    "pl-waw-2",
    "pl-waw-3",
)

GIB = 1024**3
GB = 1_000_000_000


class ScalewayProvider:
    """Scaleway — a fixed fleet, but the offer carries a per-zone availability flag.

    The catalog itself (server types, hourly prices) barely moves, so the TTL is
    driven by the volatile half of the offer: a GPU type flips between
    ``available``, ``scarce`` and ``shortage`` within a zone over minutes, and
    that flag is what ``available`` encodes — hence ten minutes, not hours.
    """

    kind: ClassVar[str] = "scaleway"
    credential_fields: ClassVar[tuple[str, ...]] = ("secret_key", "project_id")
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=10)

    def __init__(self, provider_id: str, name: str, secret_key: str, project_id: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._secret_key = secret_key
        self._project_id = project_id
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        secret_key = credentials.get("secret_key")
        if not secret_key:
            raise CapabilityMismatchError("scaleway requires a secret_key credential", provider=name)

        project_id = credentials.get("project_id")
        if not project_id:
            raise CapabilityMismatchError("scaleway requires a project_id credential", provider=name)

        return cls(provider_id, name, secret_key, project_id, config)

    @property
    def _zones(self) -> tuple[str, ...]:
        return tuple(self._config.get("zones") or DEFAULT_ZONES)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=int(self._config.get("request_timeout", 30)),
            headers={
                "X-Auth-Token": self._secret_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )

    async def _fetch_zone(self, client: httpx.AsyncClient, zone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        products, availability = await asyncio.gather(
            client.get(PRODUCTS_PATH.format(zone=zone)),
            client.get(AVAILABILITY_PATH.format(zone=zone)),
        )
        products.raise_for_status()
        availability.raise_for_status()
        return products.json().get("servers", {}), availability.json().get("servers", {})

    async def offers(self) -> AsyncIterator[Offer]:
        zones = self._zones
        async with httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=int(self._config.get("request_timeout", 30)),
            headers={"X-Auth-Token": self._secret_key, "Accept": "application/json"},
        ) as client:
            results = await asyncio.gather(*(self._fetch_zone(client, zone) for zone in zones))

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for zone, (server_types, availability) in zip(zones, results, strict=True):
            for commercial_type, spec in server_types.items():
                gpu_info = spec.get("gpu_info") or {}
                gpu_name = gpu_info.get("gpu_name")
                gpu_count = int(spec.get("gpu") or 0)
                gpu_memory_gb = float(gpu_info.get("gpu_memory") or 0) / GIB or None
                state = (availability.get(commercial_type) or {}).get("availability")
                hourly = spec.get("hourly_price")

                yield Offer(
                    id=f"scaleway-{zone}-{commercial_type}",
                    provider_id=self._id,
                    provider_name=self._name,
                    kind=self.kind,
                    billing_unit="hour",
                    instance_type=commercial_type,
                    accelerator=gpu_name if gpu_count else None,
                    accelerator_count=gpu_count,
                    vram=gpu_memory_gb if gpu_count else None,
                    cpus=int(spec.get("ncpus") or 0),
                    memory_gb=float(spec.get("ram") or 0) / GIB,
                    region=zone,
                    disk_gb=_disk_gb(spec),
                    architecture=spec.get("arch"),
                    spot_price=None,
                    on_demand_price=float(hourly) if hourly else None,
                    available=_available(state),
                    fetched_at=now,
                    expires_at=expires_at,
                    specific={
                        "commercial_type": commercial_type,
                        "zone": zone,
                        "arch": spec.get("arch"),
                        "availability": state,
                        "gpu_name": gpu_name,
                        "gpu_memory_gb": gpu_memory_gb,
                        "monthly_price": spec.get("monthly_price"),
                        "end_of_service": spec.get("end_of_service"),
                        "boot_types": (spec.get("capabilities") or {}).get("boot_types"),
                        "block_storage": (spec.get("capabilities") or {}).get("block_storage"),
                        "scratch_storage_max_size": spec.get("scratch_storage_max_size"),
                    },
                )

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        zone = str(offer.specific["zone"])
        commercial_type = str(offer.specific["commercial_type"])

        async with self._client() as client:
            ssh_key_id = await _ensure_ssh_key(client, f"skyward-{compute_id}", public_key, self._project_id)
            image_id = str(self._config.get("image") or await _resolve_image(client, zone))

        return {
            "compute_id": compute_id,
            "zone": zone,
            "project_id": self._project_id,
            "commercial_type": commercial_type,
            "image_id": image_id,
            "ssh_key_id": ssh_key_id,
            "instance_timeout": int(self._config.get("instance_timeout", 300)),
        }

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        async with self._client() as client, asyncio.TaskGroup() as group:
            created = [group.create_task(self._create(client, binding)) for _ in range(count)]

        results = [task.result() for task in created]
        machines = tuple(result for result in results if isinstance(result, Machine))

        if len(machines) < min_count:
            await self.terminate(binding, tuple(machine.id for machine in machines))
            failure = next((result for result in results if isinstance(result, Exception)), None)
            raise failure or CapabilityMismatchError(f"scaleway launched {len(machines)} of {min_count} machines")

        return binding, machines

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        """Find the compute's servers by their tag, and trust the tag over the filter.

        Scaleway's ``tags`` filter is server-side and undocumented as to whether it
        matches a tag whole or as a prefix. The tag is the identity, so it is
        re-checked here: a filter that is looser than advertised would otherwise
        hand another compute's machines to this one.
        """
        tag = _tag(binding["compute_id"])

        async with self._client() as client:
            listed = await client.get(
                SERVERS_PATH.format(zone=binding["zone"]),
                params={"tags": tag, "per_page": 100},
            )
            listed.raise_for_status()
            found = (
                _machine(server)
                for server in listed.json().get("servers", [])
                if tag in (server.get("tags") or [])
            )
            return {machine.id: machine for machine in found}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        if not machine_ids:
            return

        zone = binding["zone"]

        async with self._client() as client:
            async with asyncio.TaskGroup() as group:
                probed = [group.create_task(_sbs_volumes(client, zone, machine_id)) for machine_id in machine_ids]

            async with asyncio.TaskGroup() as group:
                for machine_id in machine_ids:
                    group.create_task(_destroy(client, zone, machine_id))

            volumes = {volume for probe in probed for volume in probe.result()}

            async with asyncio.TaskGroup() as group:
                for volume_id in volumes:
                    group.create_task(_delete_volume(client, zone, volume_id))

    async def release(self, binding: Binding) -> None:
        async with self._client() as client:
            deleted = await client.delete(f"{SSH_KEYS_PATH}/{binding['ssh_key_id']}")
            if deleted.status_code != 404:
                deleted.raise_for_status()

    async def _create(self, client: httpx.AsyncClient, binding: Binding) -> Machine | Exception:
        """Create one server, power it on, and undo the half of it that succeeded.

        Scaleway creates a server stopped: one that is never powered on is still a
        server, and is still billed.
        """
        zone = binding["zone"]
        servers = SERVERS_PATH.format(zone=zone)
        machine_id = ""

        try:
            created = await client.post(
                servers,
                json={
                    "name": f"skyward-{uuid.uuid4().hex[:12]}",
                    "commercial_type": binding["commercial_type"],
                    "image": binding["image_id"],
                    "project": binding["project_id"],
                    "dynamic_ip_required": True,
                    "volumes": {},
                    "tags": ["skyward", _tag(binding["compute_id"])],
                },
            )
            created.raise_for_status()

            server = created.json()["server"]
            machine_id = server["id"]

            powered = await client.post(f"{servers}/{machine_id}/action", json={"action": "poweron"})
            powered.raise_for_status()

            return _machine(server)
        except httpx.HTTPError as error:
            if machine_id:
                with suppress(Exception):
                    await self.terminate(binding, (machine_id,))
            return error


def _tag(compute_id: str) -> str:
    return f"{COMPUTE_TAG}={compute_id}"


def _machine(server: Mapping[str, Any]) -> Machine:
    public = server.get("public_ip") or next(iter(server.get("public_ips") or []), {})

    return Machine(
        id=server["id"],
        state="running" if server.get("state") == "running" else "pending",
        host=public.get("address") or None,
        private_host=server.get("private_ip") or None,
    )


async def _ensure_ssh_key(client: httpx.AsyncClient, name: str, public_key: str, project_id: str) -> str:
    """Register the compute's key with IAM, where Scaleway injects it from.

    Scaleway takes no key at server creation: it injects every key of the project
    into the server at boot. The key is therefore project-wide while it exists,
    which is why it is named after the compute and deleted in ``release``.

    A second ``initialize`` arrives with a *new* keypair — the control plane
    generates one before it knows the binding was lost. The key already sitting
    under this compute's name is then the wrong one, and returning its id would
    pin the compute to a private key nobody kept. It is replaced. Matching on the
    key material first keeps a genuine retry of the same key from churning it.
    """
    listed = await client.get(SSH_KEYS_PATH)
    listed.raise_for_status()
    keys = listed.json().get("ssh_keys", [])

    if same := next((key for key in keys if _key_body(key.get("public_key", "")) == _key_body(public_key)), None):
        return same["id"]

    if stale := next((key for key in keys if key.get("name") == name), None):
        await client.delete(f"{SSH_KEYS_PATH}/{stale['id']}")

    created = await client.post(
        SSH_KEYS_PATH,
        json={"name": name, "public_key": public_key, "project_id": project_id},
    )
    created.raise_for_status()
    return created.json()["id"]


def _key_body(public_key: str) -> str:
    """The base64 blob of an OpenSSH key, which is the only part that identifies it.

    Scaleway echoes back the key with the comment it was given, and the control
    plane's generated key may carry none.
    """
    parts = public_key.split()
    return parts[1] if len(parts) > 1 else public_key.strip()


async def _resolve_image(client: httpx.AsyncClient, zone: str) -> str:
    """Pick the newest Ubuntu image backed by a block snapshot.

    GPU commercial types refuse local SSD volumes, so an ``l_ssd``-backed image —
    which is most of the catalogue — fails at creation rather than at boot.
    """
    listed = await client.get(
        IMAGES_PATH.format(zone=zone),
        params={"public": "true", "arch": "x86_64", "per_page": 100},
    )
    listed.raise_for_status()

    images = sorted(listed.json().get("images", []), key=lambda image: image.get("creation_date", ""), reverse=True)
    sbs = [image for image in images if (image.get("root_volume") or {}).get("volume_type") == "sbs_snapshot"]
    candidates = sbs or images
    if not candidates:
        raise CapabilityMismatchError(f"scaleway has no public image in {zone}")

    ubuntu = [image for image in candidates if "ubuntu" in image.get("name", "").lower()]
    return (ubuntu or candidates)[0]["id"]


async def _destroy(client: httpx.AsyncClient, zone: str, machine_id: str) -> None:
    """Terminate the server, and delete it outright when it will not be terminated.

    The ``terminate`` action wants a server that got as far as running; one that was
    created and never powered on — what a half-failed launch leaves behind — is
    refused, and has to be deleted. Both are no-ops on a server that is already gone.
    """
    servers = SERVERS_PATH.format(zone=zone)

    terminated = await client.post(f"{servers}/{machine_id}/action", json={"action": "terminate"})
    if terminated.is_success or terminated.status_code == 404:
        return

    await client.delete(f"{servers}/{machine_id}")


async def _sbs_volumes(client: httpx.AsyncClient, zone: str, machine_id: str) -> tuple[str, ...]:
    """The block volumes a terminate leaves behind, read before it can no longer be read.

    ``terminate`` deletes the server and its local volumes; a block volume outlives
    it, attached to nothing and billed anyway.
    """
    fetched = await client.get(f"{SERVERS_PATH.format(zone=zone)}/{machine_id}")
    if fetched.status_code == 404:
        return ()
    fetched.raise_for_status()

    volumes = (fetched.json().get("server", {}).get("volumes") or {}).values()
    return tuple(volume["id"] for volume in volumes if str(volume.get("volume_type", "")).startswith("sbs"))


async def _delete_volume(client: httpx.AsyncClient, zone: str, volume_id: str) -> None:
    """Delete a block volume, waiting out the server it is still attached to.

    The terminate action is asynchronous: the volume answers 412 ``in_use`` until
    the server it belonged to is actually gone.
    """
    for attempt in range(8):
        if attempt:
            await asyncio.sleep(2.0 * 2 ** (attempt - 1))
        deleted = await client.delete(f"{VOLUMES_PATH.format(zone=zone)}/{volume_id}")
        if deleted.status_code != 412:
            return


def _disk_gb(spec: dict[str, Any]) -> float | None:
    local = (spec.get("volumes_constraint") or {}).get("max_size") or 0
    scratch = spec.get("scratch_storage_max_size") or 0
    size = local or scratch
    return float(size) / GB if size else None


def _available(state: str | None) -> int | None:
    match state:
        case "available" | "scarce":
            return 1
        case "shortage":
            return 0
        case _:
            return None
