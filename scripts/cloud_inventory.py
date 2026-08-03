#!/usr/bin/env python3
"""What every cloud account is holding right now, asked of the clouds themselves.

Skyward's adapters clean up after themselves; this does not trust them, and it
does not import them. It talks to each provider's API directly and lists the
kinds of thing a run creates — machines, but also the keypairs, volumes,
snapshots, images, firewalls, addresses, storage keys and container groups that
outlive them, plus the local docker objects an e2e run leaves behind.

Read-only. Nothing here deletes, creates or modifies anything.

    uv run python scripts/cloud_inventory.py
    uv run python scripts/cloud_inventory.py -p vastai -p runpod --mine
    uv run python scripts/cloud_inventory.py --json > before.json
    uv run python scripts/cloud_inventory.py --diff before.json
    uv run python scripts/cloud_inventory.py --audit

Credentials come from the same environment variables the sanity tests read; a
provider with none is skipped, not failed. The tables are what an account
actually answers, and ``--audit`` is how you tell an empty account from a
question that was never asked: it prints every endpoint and its reply.
"""

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

type Payload = Mapping[str, Any]
type Scan = Callable[[float], Awaitable["Found"]]

MARKER = "skyward"
QUIET = frozenset({400, 403, 404, 405, 409, 422, 501})
AWS_KINDS = ("instance", "ssh-key", "security-group", "launch-template", "volume", "image", "snapshot", "elastic-ip", "placement-group")

ID_KEYS = (
    "id", "uuid", "instance_id", "key_id", "container_group_id", "machine_id",
    "InstanceId", "KeyPairId", "GroupId", "VolumeId", "ImageId", "SnapshotId", "AllocationId", "LaunchTemplateId",
    "hostname", "name", "Name",
)
NAME_KEYS = (
    "name", "label", "key_name", "hostname", "display_name", "instance_name", "friendly_name", "bucket",
    "Name", "KeyName", "GroupName", "LaunchTemplateName", "Repository",
)
STATE_KEYS = ("status", "state", "current_state", "desired_status", "actual_status", "power_state", "State", "Status")
WHERE_KEYS = ("region", "zone", "location", "region_name", "datacenter", "datacenter_id", "data_center_id", "country", "geolocation")
CREATED_KEYS = (
    "created_at", "createdAt", "CreatedAt", "create_time", "createTime", "creation_time", "creationTimestamp",
    "date_created", "created", "timeCreated", "start_date",
    "LaunchTime", "CreateTime", "CreationDate", "StartTime",
)


class SkipError(Exception):
    """This provider was not asked: no credential, no sdk, no daemon."""


@dataclass(frozen=True, slots=True, order=True)
class Resource:
    kind: str
    id: str
    name: str = ""
    state: str = ""
    where: str = ""
    created: str = ""
    mine: bool = False


@dataclass(frozen=True, slots=True)
class Found:
    resources: tuple[Resource, ...] = ()
    notes: tuple[str, ...] = ()
    probes: tuple[str, ...] = ()

    def __add__(self, other: "Found") -> "Found":
        return Found(self.resources + other.resources, self.notes + other.notes, self.probes + other.probes)


@dataclass(frozen=True, slots=True)
class Inventory:
    provider: str
    resources: tuple[Resource, ...] = ()
    notes: tuple[str, ...] = ()
    probes: tuple[str, ...] = ()
    skipped: str = ""


@dataclass(frozen=True, slots=True)
class Endpoint:
    kind: str
    path: str
    pick: str = ""
    params: tuple[tuple[str, str], ...] = ()
    where: str = ""


def credential(*names: str) -> tuple[str, ...]:
    if missing := [name for name in names if not os.environ.get(name)]:
        raise SkipError(f"no {', '.join(missing)}")
    return tuple(os.environ[name] for name in names)


def session(base: str, headers: Mapping[str, str] | None = None, timeout: float = 30.0, auth: httpx.Auth | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base,
        headers={"Accept": "application/json", **(headers or {})},
        auth=auth,
        timeout=timeout,
        follow_redirects=True,
    )


def text(value: Any) -> str:
    match value:
        case None | "":
            return ""
        case str():
            return value
        case bool():
            return str(value).lower()
        case Mapping():
            return text(next((value[key] for key in ("name", "Name", "id", "Id", "code") if value.get(key)), None))
        case list() | tuple():
            return ", ".join(part for part in (text(item) for item in value) if part)
        case _:
            return str(value)


def pluck(entry: Payload, keys: Sequence[str]) -> str:
    return next((text(entry[key]) for key in keys if entry.get(key) not in (None, "", [], {})), "")


def tagged(entry: Payload) -> str:
    match entry.get("Tags") or entry.get("tags"):
        case [*items]:
            named = (str(item.get("Value") or "") for item in items if isinstance(item, Mapping) and item.get("Key") == "Name")
            return next((name for name in named if name), text(items))
        case _:
            return ""


def resource(kind: str, entry: Payload, where: str = "") -> Resource:
    return Resource(
        kind=kind,
        id=pluck(entry, ID_KEYS),
        name=pluck(entry, NAME_KEYS) or tagged(entry),
        state=pluck(entry, STATE_KEYS),
        where=(where or pluck(entry, WHERE_KEYS)).rsplit("/", 1)[-1],
        created=pluck(entry, CREATED_KEYS)[:19],
        mine=MARKER in str(entry).lower(),
    )


def dig(payload: Any, pick: str) -> list[Any]:
    """Walk a dotted path, mapping over any list and flattening — ``*`` takes a mapping's values.

    A path that leads nowhere falls back to the payload itself when it is already
    a list, and otherwise to its only list of objects. That fallback is what lets
    an endpoint be listed without knowing the envelope it answers in.
    """
    found: Any = payload
    for key in filter(None, pick.split(".")):
        match found:
            case Mapping() if key == "*":
                found = list(found.values())
            case Mapping():
                found = found.get(key)
            case [*items] if key == "*":
                found = items
            case [*items]:
                reached = (item.get(key) for item in items if isinstance(item, Mapping))
                found = [entry for group in reached if isinstance(group, list) for entry in group]
            case _:
                found = None

    match found:
        case [_, *_] as items:
            return items
        case _ if isinstance(payload, list):
            return payload
        case _ if isinstance(payload, Mapping):
            lists = (value for value in payload.values() if isinstance(value, list))
            return next((value for value in lists if all(isinstance(item, Mapping) for item in value)), [])
        case _:
            return []


async def read(client: httpx.AsyncClient, endpoint: Endpoint) -> Found:
    def probe(outcome: str) -> tuple[str, ...]:
        return (f"{endpoint.kind:<16} {endpoint.path} {outcome}",)

    try:
        response = await client.get(endpoint.path, params=dict(endpoint.params))
    except httpx.HTTPError as error:
        return Found(notes=(f"{endpoint.path}: {type(error).__name__}",), probes=probe(type(error).__name__))

    if response.is_error:
        notes = () if response.status_code in QUIET else (f"{endpoint.path}: HTTP {response.status_code}",)
        return Found(notes=notes, probes=probe(f"HTTP {response.status_code}"))

    try:
        payload = response.json()
    except ValueError:
        return Found(notes=(f"{endpoint.path}: not json",), probes=probe("not json"))

    entries = [entry for entry in dig(payload, endpoint.pick) if isinstance(entry, Mapping)]
    return Found(
        tuple(resource(endpoint.kind, entry, endpoint.where) for entry in entries),
        probes=probe(f"{response.status_code} × {len(entries)}"),
    )


async def collect(client: httpx.AsyncClient, endpoints: Iterable[Endpoint]) -> Found:
    return sum(await asyncio.gather(*(read(client, endpoint) for endpoint in endpoints)), Found())


async def aws(timeout: float) -> Found:
    try:
        import aioboto3
    except ImportError as error:
        raise SkipError("aioboto3 is not installed") from error

    home = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    account = aioboto3.Session()

    try:
        async with account.client("ec2", region_name=home) as ec2:
            regions = tuple(str(entry["RegionName"]) for entry in (await ec2.describe_regions())["Regions"])
    except Exception as error:
        raise SkipError(f"no usable credentials ({type(error).__name__})") from error

    limit = asyncio.Semaphore(8)
    swept = await asyncio.gather(*(_aws_region(account, region, limit) for region in regions), _aws_buckets(account, home))
    found = sum(swept, Found())
    counted = Counter(item.kind for item in found.resources)
    return Found(found.resources, found.notes, tuple(f"{kind:<16} {len(regions)} regions × {counted[kind]}" for kind in (*AWS_KINDS, "bucket")))


async def _aws_region(account: Any, region: str, limit: asyncio.Semaphore) -> Found:
    alive = [{"Name": "instance-state-name", "Values": ["pending", "running", "shutting-down", "stopping", "stopped"]}]

    async with limit, account.client("ec2", region_name=region) as ec2:
        calls = (
            ("instance", ec2.describe_instances(Filters=alive), "Reservations.Instances"),
            ("ssh-key", ec2.describe_key_pairs(), "KeyPairs"),
            ("security-group", ec2.describe_security_groups(), "SecurityGroups"),
            ("launch-template", ec2.describe_launch_templates(), "LaunchTemplates"),
            ("volume", ec2.describe_volumes(), "Volumes"),
            ("image", ec2.describe_images(Owners=["self"]), "Images"),
            ("snapshot", ec2.describe_snapshots(OwnerIds=["self"]), "Snapshots"),
            ("elastic-ip", ec2.describe_addresses(), "Addresses"),
            ("placement-group", ec2.describe_placement_groups(), "PlacementGroups"),
        )
        payloads = await asyncio.gather(*(call for _, call, _ in calls), return_exceptions=True)

    found = Found()
    for (kind, _, pick), payload in zip(calls, payloads, strict=True):
        match payload:
            case BaseException():
                found += Found(notes=(f"{region} {kind}: {type(payload).__name__}",))
            case _:
                entries = [entry for entry in dig(payload, pick) if isinstance(entry, Mapping)]
                listed = tuple(resource(kind, entry, region) for entry in entries)
                found += Found(tuple(item for item in listed if not (item.kind == "security-group" and item.name == "default")))
    return found


async def _aws_buckets(account: Any, region: str) -> Found:
    try:
        async with account.client("s3", region_name=region) as s3:
            listed = (await s3.list_buckets()).get("Buckets") or []
    except Exception as error:
        return Found(notes=(f"buckets: {type(error).__name__}",))
    return Found(tuple(resource("bucket", entry, "global") for entry in listed))


async def docker(timeout: float) -> Found:
    return await asyncio.to_thread(_docker_scan)


def _docker_scan() -> Found:
    if not shutil.which("docker"):
        raise SkipError("docker is not on PATH")

    listings = (
        ("container", ("ps", "-a"), "ID", "Names", "State", "CreatedAt"),
        ("volume", ("volume", "ls"), "Name", "Name", "Driver", "CreatedAt"),
        ("network", ("network", "ls"), "ID", "Name", "Driver", "CreatedAt"),
        ("image", ("images", "--filter", f"reference=*{MARKER}*"), "ID", "Repository", "Tag", "CreatedAt"),
    )

    found = Found()
    for kind, command, id_key, name_key, state_key, created_key in listings:
        try:
            output = subprocess.run(("docker", *command, "--format", "{{json .}}"), capture_output=True, text=True, timeout=30, check=True).stdout
        except (subprocess.SubprocessError, OSError) as error:
            found += Found(notes=(f"docker {command[0]}: {type(error).__name__}",))
            continue

        entries = [json.loads(line) for line in output.splitlines() if line.strip()]
        found += Found(probes=(f"{kind:<16} docker {command[0]} × {len(entries)}",))
        found += Found(tuple(
            Resource(
                kind=kind,
                id=str(entry.get(id_key, ""))[:12],
                name=str(entry.get(name_key, "")),
                state=str(entry.get(state_key, "")),
                where="local",
                created=str(entry.get(created_key, ""))[:19],
                mine=MARKER in str(entry).lower(),
            )
            for entry in entries
        ))
    return found


async def gcp(timeout: float) -> Found:
    (raw,) = credential("GCP_SERVICE_ACCOUNT_JSON")
    account = json.loads(Path(raw).read_text() if Path(raw).expanduser().exists() else raw)
    project = str(account["project_id"])

    async with session("https://compute.googleapis.com/compute/v1", timeout=timeout) as client:
        token = await _google_token(client, account)
        client.headers["Authorization"] = f"Bearer {token}"
        compute = await collect(client, (
            Endpoint("instance", f"/projects/{project}/aggregated/instances", "items.*.instances"),
            Endpoint("disk", f"/projects/{project}/aggregated/disks", "items.*.disks"),
            Endpoint("address", f"/projects/{project}/aggregated/addresses", "items.*.addresses"),
            Endpoint("firewall", f"/projects/{project}/global/firewalls", "items"),
            Endpoint("image", f"/projects/{project}/global/images", "items"),
            Endpoint("network", f"/projects/{project}/global/networks", "items"),
        ))

    async with session("https://storage.googleapis.com/storage/v1", {"Authorization": f"Bearer {token}"}, timeout) as client:
        storage = await collect(client, (
            Endpoint("bucket", "/b", "items", (("project", project),)),
            Endpoint("hmac-key", f"/projects/{project}/hmacKeys", "items"),
        ))

    return compute + storage


async def _google_token(client: httpx.AsyncClient, account: Payload) -> str:
    """The two-legged JWT flow by hand, so the script owes nothing to google-auth."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    def encode(part: Payload) -> bytes:
        return base64.urlsafe_b64encode(json.dumps(part).encode()).rstrip(b"=")

    uri = str(account.get("token_uri") or "https://oauth2.googleapis.com/token")
    issued = int(time.time())
    header = {"alg": "RS256", "typ": "JWT", "kid": account.get("private_key_id", "")}
    claims = {
        "iss": account["client_email"],
        "scope": "https://www.googleapis.com/auth/cloud-platform.read-only",
        "aud": uri,
        "iat": issued,
        "exp": issued + 3600,
    }

    key = serialization.load_pem_private_key(str(account["private_key"]).encode(), password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise SkipError("the service account key is not RSA")

    payload = b".".join((encode(header), encode(claims)))
    signature = base64.urlsafe_b64encode(key.sign(payload, padding.PKCS1v15(), hashes.SHA256())).rstrip(b"=")

    response = await client.post(uri, data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": b".".join((payload, signature)).decode()})
    if response.is_error:
        raise SkipError(f"the service account would not mint a token: HTTP {response.status_code}")
    return str(response.json()["access_token"])


async def hyperstack(timeout: float) -> Found:
    (api_key,) = credential("HYPERSTACK_API_KEY")
    async with session("https://infrahub-api.nexgencloud.com/v1", {"api_key": api_key}, timeout) as client:
        return await collect(client, (
            Endpoint("vm", "/core/virtual-machines", "instances"),
            Endpoint("keypair", "/core/keypairs", "keypairs"),
            Endpoint("environment", "/core/environments", "environments"),
            Endpoint("volume", "/core/volumes", "volumes"),
            Endpoint("snapshot", "/core/snapshots", "snapshots"),
            Endpoint("firewall", "/core/firewalls", "firewalls"),
            Endpoint("bucket", "/object-storage/buckets", "buckets"),
            Endpoint("storage-key", "/object-storage/access-keys", "access_keys"),
        ))


async def jarvislabs(timeout: float) -> Found:
    (api_key,) = credential("JL_API_KEY")
    async with session("https://backendprod.jarvislabs.net", {"Authorization": f"Bearer {api_key}"}, timeout) as client:
        return await collect(client, (
            Endpoint("instance", "/users/fetch", "instances"),
            Endpoint("ssh-key", "/ssh/"),
        ))


async def lambda_cloud(timeout: float) -> Found:
    (api_key,) = credential("LAMBDA_API_KEY")
    async with session("https://cloud.lambda.ai/api/v1", timeout=timeout, auth=httpx.BasicAuth(api_key, "")) as client:
        return await collect(client, (
            Endpoint("instance", "/instances", "data"),
            Endpoint("ssh-key", "/ssh-keys", "data"),
            Endpoint("filesystem", "/file-systems", "data"),
            Endpoint("firewall-rule", "/firewall-rules", "data"),
        ))


async def massed_compute(timeout: float) -> Found:
    (api_key,) = credential("MASSED_API_KEY")
    async with session("https://vm.massedcompute.com/api/v1", {"Authorization": f"Bearer {api_key}"}, timeout) as client:
        return await collect(client, (
            Endpoint("instance", "/instance", "runningInstances"),
            Endpoint("ssh-key", "/ssh-keys", "sshKeys"),
        ))


async def novita(timeout: float) -> Found:
    (api_key,) = credential("NOVITA_API_KEY")
    async with session("https://api.novita.ai/gpu-instance/openapi/v1", {"Authorization": f"Bearer {api_key}"}, timeout) as client:
        return await collect(client, (
            Endpoint("instance", "/gpu/instances", "instances", (("pageSize", "100"), ("pageNumber", "1"))),
        ))


async def runpod(timeout: float) -> Found:
    (api_key,) = credential("RUNPOD_API_KEY")
    async with session("https://api.runpod.io/v2", {"Authorization": f"Bearer {api_key}"}, timeout) as client:
        return await collect(client, (
            Endpoint("pod", "/pods", "pods"),
            Endpoint("network-volume", "/network-volumes", "networkVolumes"),
            Endpoint("template", "/templates", "templates"),
            Endpoint("registry", "/registries", "registries"),
        ))


async def salad(timeout: float) -> Found:
    """Salad's public API has no way to list projects, so the configured one is the whole account here."""
    api_key, organization, project = credential("SALAD_API_KEY", "SALAD_ORGANIZATION", "SALAD_PROJECT")
    async with session("https://api.salad.com/api/public", {"Salad-Api-Key": api_key}, timeout) as client:
        return await collect(client, (
            Endpoint("container-group", f"/organizations/{organization}/projects/{project.lower()}/containers", "items", where=project),
        ))


async def scaleway(timeout: float) -> Found:
    (secret_key,) = credential("SCW_SECRET_KEY")
    zones = ("fr-par-1", "fr-par-2", "fr-par-3", "nl-ams-1", "nl-ams-2", "nl-ams-3", "pl-waw-1", "pl-waw-2", "pl-waw-3")
    regions = ("fr-par", "nl-ams", "pl-waw")

    private = (("public", "false"),)

    async with session("https://api.scaleway.com", {"X-Auth-Token": secret_key}, timeout) as client:
        zoned = tuple(
            Endpoint(kind, path.format(zone=zone), pick, params, where=zone)
            for zone in zones
            for kind, path, pick, params in (
                ("server", "/instance/v1/zones/{zone}/servers", "servers", ()),
                ("ip", "/instance/v1/zones/{zone}/ips", "ips", ()),
                ("volume", "/instance/v1/zones/{zone}/volumes", "volumes", ()),
                ("snapshot", "/instance/v1/zones/{zone}/snapshots", "snapshots", ()),
                ("image", "/instance/v1/zones/{zone}/images", "images", private),
                ("security-group", "/instance/v1/zones/{zone}/security_groups", "security_groups", ()),
                ("placement-group", "/instance/v1/zones/{zone}/placement_groups", "placement_groups", ()),
                ("block-volume", "/block/v1alpha1/zones/{zone}/volumes", "volumes", ()),
                ("block-snapshot", "/block/v1alpha1/zones/{zone}/snapshots", "snapshots", ()),
            )
        )
        regional = tuple(
            Endpoint(kind, path.format(region=region), pick, where=region)
            for region in regions
            for kind, path, pick in (
                ("private-network", "/vpc/v2/regions/{region}/private-networks", "private_networks"),
                ("registry", "/registry/v1/regions/{region}/namespaces", "namespaces"),
                ("ipam-ip", "/ipam/v1/regions/{region}/ips", "ips"),
            )
        )
        compute = await collect(client, (*zoned, *regional, Endpoint("ssh-key", "/iam/v1alpha1/ssh-keys", "ssh_keys")))

    return compute + await _scaleway_buckets()


async def _scaleway_buckets() -> Found:
    """Object storage is the S3 protocol, not the Scaleway API, so it needs the other half of the credential."""
    access_key = os.environ.get("SCW_ACCESS_KEY")
    secret_key = os.environ.get("SCW_SECRET_KEY")
    if not (access_key and secret_key):
        return Found(probes=("bucket           object storage: no SCW_ACCESS_KEY",))

    try:
        import aioboto3
    except ImportError:
        return Found(probes=("bucket           object storage: aioboto3 is not installed",))

    regions = ("fr-par", "nl-ams", "pl-waw")
    account = aioboto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    async def buckets(region: str) -> Found:
        try:
            async with account.client("s3", region_name=region, endpoint_url=f"https://s3.{region}.scw.cloud") as s3:
                listed = (await s3.list_buckets()).get("Buckets") or []
        except Exception as error:
            return Found(notes=(f"object storage {region}: {type(error).__name__}",))
        return Found(tuple(resource("bucket", entry, region) for entry in listed))

    found = sum(await asyncio.gather(*(buckets(region) for region in regions)), Found())
    return found + Found(probes=(f"bucket           object storage {len(regions)} regions × {len(found.resources)}",))


async def tensordock(timeout: float) -> Found:
    (token,) = credential("TENSORDOCK_API_TOKEN")
    async with session("https://dashboard.tensordock.com", {"Authorization": f"Bearer {token}"}, timeout) as client:
        return await collect(client, (
            Endpoint("instance", "/api/v2/instances", "data.instances"),
        ))


async def vastai(timeout: float) -> Found:
    (api_key,) = credential("VAST_API_KEY")
    async with session("https://console.vast.ai", {"Authorization": f"Bearer {api_key}"}, timeout) as client:
        return await collect(client, (
            Endpoint("instance", "/api/v0/instances/", "instances", (("owner", "me"),)),
            Endpoint("ssh-key", "/api/v0/ssh/"),
            Endpoint("volume", "/api/v0/volumes/", "volumes", (("owner", "me"),)),
        ))


async def verda(timeout: float) -> Found:
    client_id, client_secret = credential("VERDA_CLIENT_ID", "VERDA_CLIENT_SECRET")
    async with session("https://api.verda.com/v1", timeout=timeout) as client:
        minted = await client.post("/oauth2/token", data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret})
        if minted.is_error:
            raise SkipError(f"the credentials would not mint a token: HTTP {minted.status_code}")

        client.headers["Authorization"] = f"Bearer {minted.json()['access_token']}"
        return await collect(client, (
            Endpoint("instance", "/instances"),
            Endpoint("ssh-key", "/ssh-keys"),
            Endpoint("volume", "/volumes"),
            Endpoint("startup-script", "/scripts"),
        ))


async def vultr(timeout: float) -> Found:
    (api_key,) = credential("VULTR_API_KEY")
    page = (("per_page", "500"),)
    async with session("https://api.vultr.com/v2", {"Authorization": f"Bearer {api_key}"}, timeout) as client:
        return await collect(client, (
            Endpoint("instance", "/instances", "instances", page),
            Endpoint("bare-metal", "/bare-metals", "bare_metals", page),
            Endpoint("ssh-key", "/ssh-keys", "ssh_keys", page),
            Endpoint("block", "/blocks", "blocks", page),
            Endpoint("snapshot", "/snapshots", "snapshots", page),
            Endpoint("reserved-ip", "/reserved-ips", "reserved_ips", page),
            Endpoint("vpc", "/vpcs", "vpcs", page),
            Endpoint("firewall", "/firewalls", "firewall_groups", page),
            Endpoint("load-balancer", "/load-balancers", "load_balancers", page),
        ))


PROVIDERS: Mapping[str, Scan] = {
    "aws": aws,
    "docker": docker,
    "gcp": gcp,
    "hyperstack": hyperstack,
    "jarvislabs": jarvislabs,
    "lambda_cloud": lambda_cloud,
    "massed_compute": massed_compute,
    "novita": novita,
    "runpod": runpod,
    "salad": salad,
    "scaleway": scaleway,
    "tensordock": tensordock,
    "vastai": vastai,
    "verda": verda,
    "vultr": vultr,
}


async def sweep(names: Sequence[str], timeout: float) -> list[Inventory]:
    async def one(name: str) -> Inventory:
        try:
            found = await PROVIDERS[name](timeout)
        except SkipError as skipped:
            return Inventory(name, skipped=str(skipped))
        except Exception as error:
            return Inventory(name, notes=(f"{type(error).__name__}: {error}",))
        return Inventory(name, tuple(sorted(found.resources)), found.notes, tuple(sorted(found.probes)))

    return list(await asyncio.gather(*(one(name) for name in names)))


def audit(inventories: Sequence[Inventory], console: Console) -> None:
    """Every question that was asked and what came back — an empty provider is not the same as an unasked one."""
    for inventory in inventories:
        console.print(f"[bold]{inventory.provider}[/]" + (f" [dim]— skipped: {inventory.skipped}[/]" if inventory.skipped else ""))
        for probe in inventory.probes:
            console.print(f"  [dim]{probe}[/]")
        console.print()


def render(inventories: Sequence[Inventory], console: Console, mine_only: bool) -> None:
    for inventory in inventories:
        shown = tuple(item for item in inventory.resources if item.mine or not mine_only)
        if inventory.skipped or not shown:
            continue

        table = Table(title=f"[bold]{inventory.provider}[/]  [dim]{len(shown)}[/]", title_justify="left", header_style="dim", box=None, pad_edge=False)
        for column in ("", "kind", "id", "name", "state", "where", "created"):
            table.add_column(column, no_wrap=column != "name", overflow="ellipsis")
        for item in shown:
            table.add_row("[green]•[/]" if item.mine else "", item.kind, item.id, item.name, item.state, item.where, item.created)
        console.print(table)
        console.print()

    empty = [inventory.provider for inventory in inventories if not inventory.skipped and not inventory.resources]
    skipped = [f"{inventory.provider} ({inventory.skipped})" for inventory in inventories if inventory.skipped]
    notes = [f"{inventory.provider}: {note}" for inventory in inventories for note in inventory.notes]

    if empty:
        console.print(f"[dim]empty: {', '.join(sorted(empty))}[/]")
    if skipped:
        console.print(f"[dim]skipped: {', '.join(sorted(skipped))}[/]")
    for note in notes:
        console.print(f"[yellow]![/] [dim]{note}[/]")


def as_json(inventories: Sequence[Inventory]) -> str:
    listed = {
        inventory.provider: [asdict(item) for item in inventory.resources]
        for inventory in sorted(inventories, key=lambda inventory: inventory.provider)
        if not inventory.skipped
    }
    return json.dumps(listed, indent=2)


def compare(previous: Payload, inventories: Sequence[Inventory], console: Console) -> None:
    def keyed(provider: str, entries: Iterable[Payload]) -> dict[tuple[str, str, str], str]:
        return {(provider, str(entry["kind"]), str(entry["id"])): str(entry.get("name") or "") for entry in entries}

    before = {key: name for provider, entries in previous.items() for key, name in keyed(str(provider), entries).items()}
    after = {
        key: name
        for inventory in inventories
        if not inventory.skipped
        for key, name in keyed(inventory.provider, (asdict(item) for item in inventory.resources)).items()
    }

    seen = {inventory.provider for inventory in inventories if not inventory.skipped}
    added = sorted(key for key in after if key not in before)
    gone = sorted(key for key in before if key not in after and key[0] in seen)

    for provider, kind, identifier in added:
        console.print(f"[green]+[/] {provider:<14} {kind:<16} {identifier} {after[provider, kind, identifier]}")
    for provider, kind, identifier in gone:
        console.print(f"[red]-[/] {provider:<14} {kind:<16} {identifier} {before[provider, kind, identifier]}")
    if not added and not gone:
        console.print("[dim]no change[/]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--provider", action="append", choices=sorted(PROVIDERS), metavar="NAME", help="only this provider; repeatable")
    parser.add_argument("--mine", action="store_true", help="only resources whose payload mentions skyward")
    parser.add_argument("--json", action="store_true", help="print the inventory as json, for diffing between runs")
    parser.add_argument("--diff", metavar="FILE", type=Path, help="compare against a json inventory saved earlier")
    parser.add_argument("--audit", action="store_true", help="print every endpoint asked and what it answered, instead of the inventory")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request timeout in seconds (default 30)")
    arguments = parser.parse_args()

    console = Console(stderr=arguments.json)
    names = tuple(arguments.provider or sorted(PROVIDERS))

    with console.status(f"asking {len(names)} providers"):
        inventories = asyncio.run(sweep(names, arguments.timeout))

    if arguments.audit:
        audit(inventories, console)
    elif arguments.diff:
        compare(json.loads(arguments.diff.read_text()), inventories, console)
    elif arguments.json:
        print(as_json(inventories))
    else:
        render(inventories, console, arguments.mine)

    return 0


if __name__ == "__main__":
    sys.exit(main())
