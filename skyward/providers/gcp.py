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
import re
import time
import uuid
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from skyward.shared.errors import CapabilityMismatchError
from skyward.shared.provider import Binding, Machine, MachineState, Mount
from skyward.shared.schemas import ComputeSpec, Endpoint, Market, Offer, Volume
from skyward.worker import bootstrap

COMPUTE_URL = "https://compute.googleapis.com/compute/v1"
STORAGE_URL = "https://storage.googleapis.com"
VANTAGE_URL = "https://instances.vantage.sh/gcp/instances.json"
SCOPE = "https://www.googleapis.com/auth/compute https://www.googleapis.com/auth/devstorage.full_control"
"""What the minted token may be used for: machines, and the buckets a compute mounts.

A ceiling and not a grant — the service account's own IAM decides what it may
actually do, and one that was never given storage simply fails the HMAC call.
Asking for the narrower scope instead would mean a second token, a second
signature, and a second cache for the one call per compute that needs it.
"""

COMPUTE_LABEL = "skyward-compute"
NODE_TAG = "skyward-node"

GPU_IMAGE = "projects/deeplearning-platform-release/global/images/family/common-cu128-ubuntu-2204-nvidia-570"
CPU_IMAGE = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2404-lts-amd64"

DEFAULT_DISK_GB = 200
DEFAULT_DISK_TYPE = "pd-balanced"
DEFAULT_NETWORK = "default"

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

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return True

    def __init__(self, provider_id: str, name: str, service_account: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._service_account = service_account
        self._config = config
        self._access_token: str | None = None
        self._token_expiry = 0.0

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

        The token is cached until shortly before it expires: minting one costs
        an RSA signature and a round trip, and every API call funnels through
        here. The signature itself runs off the loop for the same reason.
        """
        if self._access_token and time.time() < self._token_expiry - 300:
            return self._access_token

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

        def sign() -> bytes:
            key = serialization.load_pem_private_key(self._service_account["private_key"].encode(), password=None)
            if not isinstance(key, rsa.RSAPrivateKey):
                raise CapabilityMismatchError("gcp service account key is not an RSA key", provider=self._name)
            return key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())

        signature = await asyncio.to_thread(sign)
        assertion = b".".join((signing_input, _b64url(signature))).decode()

        response = await client.post(
            token_uri,
            data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion},
        )
        response.raise_for_status()
        payload = response.json()
        self._access_token = str(payload["access_token"])
        self._token_expiry = issued_at + int(payload.get("expires_in", 3600))
        return self._access_token

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
            billing_unit="second",
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
            billing_unit="second",
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

    @asynccontextmanager
    async def _api(self) -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(base_url=COMPUTE_URL, timeout=60, headers={"Accept": "application/json"}) as client:
            token = await self._token(client)
            client.headers["Authorization"] = f"Bearer {token}"
            yield client

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        specific = offer.specific or {}
        zone = str(specific.get("zone") or "")
        project = str(specific.get("project") or self._project)
        if not zone:
            raise CapabilityMismatchError(f"gcp offer {offer.id} carries no zone", provider=self._name)

        label = _label(compute_id)
        tag = _name(label)
        network = str(self._config.get("network") or DEFAULT_NETWORK)
        subnet = self._config.get("subnet")
        service_account = self._config.get("service_account")
        count = int(specific.get("accelerator_count") or 0)
        accelerator = specific.get("accelerator_type") if specific.get("uses_guest_accelerators") else None

        binding: dict[str, Any] = {
            "compute_id": compute_id,
            "label": label,
            "tag": tag,
            "project": project,
            "zone": zone,
            "region": _region(zone),
            "network": network,
            "subnet": str(subnet) if subnet else None,
            "firewall": tag,
            "machine_type": str(specific.get("machine_type") or offer.instance_type),
            "image": GPU_IMAGE if count else CPU_IMAGE,
            "accelerator_type": str(accelerator) if accelerator else None,
            "accelerator_count": count,
            "disk_gb": int(self._config.get("disk_gb") or DEFAULT_DISK_GB),
            "disk_type": str(self._config.get("disk_type") or DEFAULT_DISK_TYPE),
            "service_account": str(service_account) if service_account else None,
            "public_key": public_key,
            "instance_timeout": int(self._config.get("instance_timeout", 300)),
        }

        async with self._api() as client:
            response = await client.post(
                f"/projects/{project}/global/firewalls",
                json={
                    "name": tag,
                    "network": f"global/networks/{network}",
                    "direction": "INGRESS",
                    "sourceRanges": ["0.0.0.0/0"],
                    "targetTags": [NODE_TAG],
                    "allowed": [{"IPProtocol": "tcp", "ports": ["22"]}],
                },
            )
            _accept(response, 409)

        return binding

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        async with self._api() as client, asyncio.TaskGroup() as group:
            launched = [group.create_task(self._insert(client, binding, market)) for _ in range(count)]

        results = [task.result() for task in launched]
        machines = tuple(machine for machine, _ in results if machine)
        if len(machines) < min_count:
            refused = "; ".join(reason for _, reason in results if reason)
            raise CapabilityMismatchError(
                f"gcp launched {len(machines)} of {min_count} machines in {binding['zone']}: {refused}",
                provider=self._name,
            )
        return binding, machines

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        async with self._api() as client:
            response = await client.get(
                f"/projects/{binding['project']}/zones/{binding['zone']}/instances",
                params={"filter": f"labels.{COMPUTE_LABEL}={binding['label']}", "maxResults": "500"},
            )
            response.raise_for_status()
            items: list[dict[str, Any]] = response.json().get("items") or []

        found = (_machine(item) for item in items)
        return {machine.id: machine for machine in found if machine is not None}

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        async with self._api() as client, asyncio.TaskGroup() as group:
            for machine_id in machine_ids:
                group.create_task(self._delete(client, binding, machine_id))

    async def mount(self, binding: Binding, volumes: tuple[Volume, ...]) -> Mount:
        """Mint one HMAC key for the compute, because GCS speaks S3 only when signed that way.

        Google Cloud Storage's S3-compatible endpoint authenticates with an HMAC
        key pair and nothing else — the service-account token the rest of this
        adapter uses is not a thing geesefs can present. So a key is created here,
        lives exactly as long as the compute does, and its id rides the binding so
        that :meth:`release` can take it back.
        """
        email = self._service_account["client_email"]
        async with httpx.AsyncClient(base_url=STORAGE_URL, timeout=60, headers={"Accept": "application/json"}) as client:
            token = await self._token(client)
            response = await client.post(
                f"/storage/v1/projects/{binding['project']}/hmacKeys",
                params={"serviceAccountEmail": email},
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            created = response.json()

        endpoint = Endpoint(
            url=STORAGE_URL,
            access_key=created["metadata"]["accessId"],
            secret_key=created["secret"],
        )
        return Mount(
            binding_patch={"hmac_access_id": created["metadata"]["accessId"]},
            phases=(bootstrap.mounts(tuple((volume, endpoint) for volume in volumes)),),
        )

    async def release(self, binding: Binding) -> None:
        await self._revoke(binding)
        async with self._api() as client:
            response = await client.delete(f"/projects/{binding['project']}/global/firewalls/{binding['firewall']}")
            _accept(response, 404)

    async def _revoke(self, binding: Binding) -> None:
        """Take back the HMAC key the compute was mounted with, if it had one.

        Two calls because Google refuses to delete an active key: it is deactivated
        first, then removed. A key that outlives its compute is a standing
        credential to the project's buckets that nothing will ever use again.
        """
        access_id = binding.get("hmac_access_id")
        if not access_id:
            return

        path = f"/storage/v1/projects/{binding['project']}/hmacKeys/{access_id}"
        async with httpx.AsyncClient(base_url=STORAGE_URL, timeout=60, headers={"Accept": "application/json"}) as client:
            token = await self._token(client)
            client.headers["Authorization"] = f"Bearer {token}"
            _accept(await client.put(path, json={"state": "INACTIVE"}), 404)
            _accept(await client.delete(path), 404)

    async def _insert(self, client: httpx.AsyncClient, binding: Binding, market: Market) -> tuple[Machine | None, str | None]:
        """Create one instance, under a name chosen here rather than by GCP.

        The name is the identity: it is what `instances.delete` takes, and it is
        known before the call is made, so a request whose reply is lost still
        leaves a machine that :meth:`machines` finds by its label.

        A refusal comes back as its text rather than as an exception: one machine
        out of stock must not cancel the siblings that were granted, and the
        caller's ``min_count`` is the only thing entitled to call that fatal.
        """
        name = f"skyward-{uuid.uuid4().hex[:16]}"
        response = await client.post(
            f"/projects/{binding['project']}/zones/{binding['zone']}/instances",
            json=_instance_body(name, binding, market),
        )
        if response.status_code >= 400 and response.status_code != 409:
            return None, f"{response.status_code} {response.text[:400]}"

        return Machine(id=name, state="pending"), None

    async def _delete(self, client: httpx.AsyncClient, binding: Binding, machine_id: str) -> None:
        response = await client.delete(
            f"/projects/{binding['project']}/zones/{binding['zone']}/instances/{machine_id}"
        )
        _accept(response, 404)


def _instance_body(name: str, binding: Binding, market: Market) -> dict[str, Any]:
    """The instance GCP is asked for, as one JSON document.

    v1 went through an instance template plus `bulkInsert`; a template is a
    second thing to create, to name, to find again and to delete, and it buys
    nothing here — the caller asks for one machine at a time.
    """
    network_interface: dict[str, Any] = {
        "network": f"global/networks/{binding['network']}",
        "accessConfigs": [{"name": "External NAT", "type": "ONE_TO_ONE_NAT"}],
    }
    if subnet := binding["subnet"]:
        network_interface["subnetwork"] = f"regions/{binding['region']}/subnetworks/{subnet}"

    scheduling: dict[str, Any] = (
        {"provisioningModel": "SPOT", "instanceTerminationAction": "DELETE", "onHostMaintenance": "TERMINATE"}
        if market == "spot"
        else {"onHostMaintenance": "TERMINATE", "automaticRestart": True}
    )

    body: dict[str, Any] = {
        "name": name,
        "machineType": f"zones/{binding['zone']}/machineTypes/{binding['machine_type']}",
        "labels": {COMPUTE_LABEL: binding["label"]},
        "tags": {"items": [NODE_TAG]},
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "initializeParams": {
                    "sourceImage": binding["image"],
                    "diskSizeGb": binding["disk_gb"],
                    "diskType": f"zones/{binding['zone']}/diskTypes/{binding['disk_type']}",
                },
            }
        ],
        "networkInterfaces": [network_interface],
        "scheduling": scheduling,
        "metadata": {
            "items": [
                {"key": "ssh-keys", "value": f"root:{binding['public_key']}"},
                {"key": "block-project-ssh-keys", "value": "true"},
                {"key": "startup-script", "value": _startup(str(binding["public_key"]))},
            ]
        },
    }

    if binding["accelerator_type"] and binding["accelerator_count"]:
        body["guestAccelerators"] = [
            {
                "acceleratorType": f"zones/{binding['zone']}/acceleratorTypes/{binding['accelerator_type']}",
                "acceleratorCount": binding["accelerator_count"],
            }
        ]

    if service_account := binding["service_account"]:
        body["serviceAccounts"] = [
            {"email": service_account, "scopes": ["https://www.googleapis.com/auth/cloud-platform"]}
        ]

    return body


def _machine(item: Mapping[str, Any]) -> Machine | None:
    interface: Mapping[str, Any] = next(iter(item.get("networkInterfaces") or ()), {})
    access: Mapping[str, Any] = next(iter(interface.get("accessConfigs") or ()), {})
    host = access.get("natIP")
    private_host = interface.get("networkIP")

    match item.get("status"):
        case "TERMINATED" | "STOPPING" | "SUSPENDED":
            return None
        case "RUNNING" if host:
            state: MachineState = "running"
        case _:
            state = "pending"

    return Machine(id=str(item["name"]), state=state, host=host, private_host=private_host)


def _accept(response: httpx.Response, *tolerated: int) -> None:
    """Read "already exists" and "already gone" as the successes they are."""
    if response.status_code in tolerated:
        return
    response.raise_for_status()


def _label(compute_id: str) -> str:
    """A compute id as GCP will take it, in a form that is legal as both a label value and a name.

    Label values would allow underscores; firewall rules and network tags would
    not, and the same string has to serve as all three.
    """
    return re.sub(r"[^a-z0-9-]", "-", compute_id.lower())[:55].strip("-")


def _name(label: str) -> str:
    """A firewall rule and a network tag are one RFC1035 name: a letter first, no trailing dash."""
    return f"skyward-{label}"[:63].rstrip("-")


def _startup(public_key: str) -> str:
    """Make root reachable over SSH, which on a Google image it is not by default.

    The bootstrap the control plane ships writes to /opt and installs into
    /root, and no machine here has a sudo channel. The guest agent would happily
    write an ``ssh-keys`` metadata entry into ``/root/.ssh/authorized_keys``, but
    Google's images ship an sshd that refuses root, so the key alone buys
    nothing. This runs before the first connection attempt, and the node is
    retrying until it succeeds.
    """
    return f"""#!/bin/bash
set -e
mkdir -p /root/.ssh /etc/ssh/sshd_config.d
chmod 700 /root/.ssh
printf '%s\\n' '{public_key.strip()}' > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
printf 'PermitRootLogin prohibit-password\\n' > /etc/ssh/sshd_config.d/60-skyward.conf
sed -i 's/^[#[:space:]]*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
systemctl restart ssh 2>/dev/null || systemctl restart sshd
"""


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
