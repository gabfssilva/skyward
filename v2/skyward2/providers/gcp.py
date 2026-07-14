"""GCP Compute Engine — a catalog API that does not know its own prices.

Compute Engine publishes machine types and accelerator types per zone, and
nothing else: `machineTypes.list` returns cores, memory and (for the
accelerator-optimized families) the GPUs welded into the machine, but never a
number in dollars. Prices live in the Cloud Billing Catalog as SKUs whose only
link back to a machine type is an English description ("N1 Predefined Instance
Core running in Americas"), priced per-core and per-GB, so an hourly figure for
`a3-highgpu-8g` has to be reassembled from three or four SKUs. v1 never did
that: its GCP offers carry `spot_price=None, on_demand_price=None`.

So the shape here is two sources. The *hardware* comes from Compute Engine,
per zone, authenticated with the service account. The *price* comes from the
Vantage GCP feed (`instances.vantage.sh/gcp/instances.json`) — public, no
credentials, keyed by machine type and region, with `linux.ondemand` and
`linux.spot` already expressed as USD/hour. That is the same feed v1's
`skyward/infra/pricing.py` targets (and never calls).

The one thing neither source gives is the price of a GPU *attached* to an N1:
GCP bills the machine and the accelerator separately, Vantage only knows the
machine. Those offers are emitted with no price rather than with the price of
the bare N1, which would be a lie of about an order of magnitude.
"""

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from skyward2.application.errors import CapabilityMismatchError
from skyward2.protocol.schemas import Offer

COMPUTE_URL = "https://compute.googleapis.com/compute/v1"
VANTAGE_URL = "https://instances.vantage.sh/gcp/instances.json"
SCOPE = "https://www.googleapis.com/auth/compute.readonly"

DEFAULT_ZONES = (
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-central1-f",
)

# GPUs that are not part of a machine type: they are attached to an N1 at
# create time, in a count the caller picks. Everything else (a2, a3, a4, g2)
# ships its GPUs inside the machine type and cannot be reconfigured.
GUEST_ATTACHABLE = frozenset(
    {
        "nvidia-tesla-t4",
        "nvidia-tesla-v100",
        "nvidia-tesla-p100",
        "nvidia-tesla-p4",
    }
)

# GCP's older names do not say how much memory the card has, and the shared
# catalog can only hold one VRAM per model. Its A100 is the 40GB part (the 80GB
# one is `nvidia-a100-80gb`) and its V100 is the 16GB part — both smaller than
# the catalog default, so they have to be stated here or they read as bigger
# cards than they are.
LEGACY_VRAM: dict[str, float] = {
    "nvidia-tesla-a100": 40,
    "nvidia-tesla-v100": 16,
    "nvidia-tesla-t4": 16,
    "nvidia-tesla-p100": 16,
    "nvidia-tesla-p4": 8,
}

N1_FOR_GPUS: dict[int, str] = {
    1: "n1-standard-8",
    2: "n1-standard-16",
    4: "n1-standard-32",
    8: "n1-standard-96",
}


class GCPProvider:
    """Google Compute Engine.

    The TTL is long because nothing an offer carries moves quickly: the zone's
    machine and accelerator catalog changes when Google launches hardware, and
    GCP spot prices are list prices — recomputed roughly monthly, not bid
    against like AWS's. Six hours is already far more often than either side
    changes; anything shorter just re-downloads a 4 MB price feed.
    """

    kind: ClassVar[str] = "gcp"
    credential_fields: ClassVar[tuple[str, ...]] = ("service_account_json",)
    offers_ttl: ClassVar[timedelta] = timedelta(hours=6)

    def __init__(self, provider_id: str, name: str, service_account: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._service_account = service_account
        self._config = config

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        raw = credentials.get("service_account_json")
        if not raw:
            raise CapabilityMismatchError("gcp requires a service_account_json credential", provider=name)

        try:
            service_account = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CapabilityMismatchError(f"gcp service_account_json is not valid JSON: {error}", provider=name) from error

        missing = [field for field in ("client_email", "private_key") if not service_account.get(field)]
        if missing:
            raise CapabilityMismatchError(f"gcp service_account_json is missing {', '.join(missing)}", provider=name)

        if not (config.get("project") or service_account.get("project_id")):
            raise CapabilityMismatchError("gcp needs a project: set config.project or use a key with project_id", provider=name)

        return cls(provider_id, name, service_account, config)

    @property
    def _project(self) -> str:
        return str(self._config.get("project") or self._service_account["project_id"])

    @property
    def _zones(self) -> tuple[str, ...]:
        return tuple(self._config.get("zones") or DEFAULT_ZONES)

    async def _token(self, client: httpx.AsyncClient) -> str:
        """Mint an access token from the service-account key.

        The two-legged JWT flow, by hand: google-auth would do this, but it is
        a heavy dependency for one signature, and it also insists on falling
        back to the environment (ADC, the metadata server) when a field is
        missing — exactly what an adapter must never do.
        """
        token_uri = str(self._service_account.get("token_uri") or "https://oauth2.googleapis.com/token")
        issued_at = int(time.time())
        claims = {
            "iss": self._service_account["client_email"],
            "scope": SCOPE,
            "aud": token_uri,
            "iat": issued_at,
            "exp": issued_at + 3600,
        }
        header: dict[str, Any] = {"alg": "RS256", "typ": "JWT"}
        if key_id := self._service_account.get("private_key_id"):
            header["kid"] = key_id

        signing_input = b".".join((_b64(header), _b64(claims)))
        key = serialization.load_pem_private_key(self._service_account["private_key"].encode(), password=None)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise CapabilityMismatchError("gcp service account key is not an RSA key", provider=self._name)
        signature = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
        assertion = b".".join((signing_input, _b64url(signature))).decode()

        response = await client.post(
            token_uri,
            data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion},
        )
        response.raise_for_status()
        return str(response.json()["access_token"])

    async def _list(self, client: httpx.AsyncClient, path: str, token: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        params: dict[str, str] = {"maxResults": "500"}
        while True:
            response = await client.get(path, params=params, headers={"Authorization": f"Bearer {token}"})
            response.raise_for_status()
            page = response.json()
            items.extend(page.get("items") or ())
            if not (cursor := page.get("nextPageToken")):
                return items
            params = {"maxResults": "500", "pageToken": cursor}

    async def _fetch_zone(self, client: httpx.AsyncClient, zone: str, token: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        base = f"/projects/{self._project}/zones/{zone}"
        machines, accelerators = await asyncio.gather(
            self._list(client, f"{base}/machineTypes", token),
            self._list(client, f"{base}/acceleratorTypes", token),
        )
        return machines, accelerators

    async def offers(self) -> AsyncIterator[Offer]:
        async with httpx.AsyncClient(base_url=COMPUTE_URL, timeout=60, headers={"Accept": "application/json"}) as client:
            token = await self._token(client)
            zones = self._zones
            catalog, *per_zone = await asyncio.gather(
                _fetch_prices(client),
                *(self._fetch_zone(client, zone, token) for zone in zones),
            )

        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for zone, (machines, accelerators) in zip(zones, per_zone, strict=True):
            region = _region(zone)
            for machine in machines:
                if offer := self._machine_offer(machine, zone, region, catalog, now, expires_at):
                    yield offer

            attachable = [a["name"] for a in accelerators if a.get("name") in GUEST_ATTACHABLE]
            for accelerator in attachable:
                for count, machine_type in N1_FOR_GPUS.items():
                    if machine := next((m for m in machines if m.get("name") == machine_type), None):
                        yield self._attached_offer(machine, accelerator, count, zone, region, now, expires_at)

    def _machine_offer(
        self,
        machine: dict[str, Any],
        zone: str,
        region: str,
        catalog: Mapping[str, dict[str, Any]],
        now: datetime,
        expires_at: datetime,
    ) -> Offer | None:
        machine_type = str(machine.get("name") or "")
        priced = catalog.get(machine_type, {})

        # `accelerators` is what the machine type ships with — a2, a3, a4 and g2
        # only. It is authoritative and per-machine-type, which the family-level
        # map v1 used is not: a2-highgpu and a2-ultragpu are both "a2" and carry
        # different A100s. Vantage is the fallback for a machine type too new for
        # the field to be populated.
        bundled = (machine.get("accelerators") or [{}])[0]
        accelerator = bundled.get("guestAcceleratorType") or priced.get("GPU_model")
        count = int(bundled.get("guestAcceleratorCount") or priced.get("GPU") or 0)
        if accelerator and not count:
            return None

        prices = _prices(priced, region)
        return Offer(
            id=f"gcp-{zone}-{machine_type}",
            provider_id=self._id,
            provider_name=self._name,
            kind=self.kind,
            instance_type=machine_type,
            accelerator=accelerator if count else None,
            accelerator_count=count,
            vram=LEGACY_VRAM.get(accelerator or ""),
            cpus=int(machine.get("guestCpus") or 0),
            memory_gb=float(machine.get("memoryMb") or 0) / 1024,
            region=zone,
            disk_gb=None,
            spot_price=prices.get("spot"),
            on_demand_price=prices.get("ondemand"),
            available=None,
            fetched_at=now,
            expires_at=expires_at,
            specific={
                "project": self._project,
                "zone": zone,
                "region": region,
                "machine_type": machine_type,
                "uses_guest_accelerators": False,
                "accelerator_type": accelerator if count else None,
                "accelerator_count": count,
                "is_shared_cpu": bool(machine.get("isSharedCpu")),
                "deprecated": (machine.get("deprecated") or {}).get("state"),
                "price_source": "vantage" if prices else None,
            },
        )

    def _attached_offer(
        self,
        machine: dict[str, Any],
        accelerator: str,
        count: int,
        zone: str,
        region: str,
        now: datetime,
        expires_at: datetime,
    ) -> Offer:
        """An N1 with GPUs bolted on — the only case where the count is a choice.

        Priced at nothing on purpose. GCP bills the N1 and each GPU as separate
        SKUs; Vantage prices the N1 alone. Publishing that number would make an
        8×V100 look like a bare 96-core machine.
        """
        machine_type = str(machine.get("name") or "")
        return Offer(
            id=f"gcp-{zone}-{machine_type}-{accelerator}-x{count}",
            provider_id=self._id,
            provider_name=self._name,
            kind=self.kind,
            instance_type=machine_type,
            accelerator=accelerator,
            accelerator_count=count,
            vram=LEGACY_VRAM.get(accelerator),
            cpus=int(machine.get("guestCpus") or 0),
            memory_gb=float(machine.get("memoryMb") or 0) / 1024,
            region=zone,
            disk_gb=None,
            spot_price=None,
            on_demand_price=None,
            available=None,
            fetched_at=now,
            expires_at=expires_at,
            specific={
                "project": self._project,
                "zone": zone,
                "region": region,
                "machine_type": machine_type,
                "uses_guest_accelerators": True,
                "accelerator_type": accelerator,
                "accelerator_count": count,
                "is_shared_cpu": bool(machine.get("isSharedCpu")),
                "deprecated": (machine.get("deprecated") or {}).get("state"),
                "price_source": None,
            },
        )


async def _fetch_prices(client: httpx.AsyncClient) -> dict[str, dict[str, Any]]:
    """Index the public Vantage feed by machine type.

    A failure here costs the prices, not the catalog: an unpriced offer is still
    a machine you can ask GCP for, and the alternative is that one flaky CDN
    read takes the whole provider offline.
    """
    try:
        response = await client.get(VANTAGE_URL, timeout=60)
        response.raise_for_status()
    except httpx.HTTPError:
        return {}
    return {str(entry["instance_type"]): entry for entry in response.json() if entry.get("instance_type")}


def _prices(entry: Mapping[str, Any], region: str) -> dict[str, float]:
    linux = ((entry.get("pricing") or {}).get(region) or {}).get("linux") or {}
    return {key: price for key in ("ondemand", "spot") if (price := _float(linux.get(key))) is not None}


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _region(zone: str) -> str:
    return zone.rsplit("-", 1)[0]


def _b64(payload: Mapping[str, Any]) -> bytes:
    return _b64url(json.dumps(payload, separators=(",", ":")).encode())


def _b64url(raw: bytes) -> bytes:
    return base64.urlsafe_b64encode(raw).rstrip(b"=")


__all__: Iterable[str] = ("GCPProvider",)
