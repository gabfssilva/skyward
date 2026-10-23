"""Salad Cloud, reached through its container gateway or its SSH relay.

Salad rents containers, not machines, and a container has two ways in, neither
of them inbound TCP to the node. The Container Gateway is an HTTP reverse proxy
in front of one port that reaches the container over IPv6; on a node where that
path never forms it answers 503 for as long as the node lives, with the
container running and Salad calling it ready. The SSH relay the portal shows is
an ``exec`` into the container from Salad's side, admitted on the key registered
in the portal; on a node whose relay agent fails it drops the connection before
the banner. Each fails on nodes the other reaches, so the account's
``transport`` says which to use: the relay unless told otherwise, the gateway,
or ``any`` for whichever answers first.

Either way is made to look like a socket. The container runs sshd and a
WebSocket-to-TCP bridge on the gateway port, and the daemon runs one loopback
listener per node whose accepted connections become links to that node: a
WebSocket to its gateway, or a session of the relay running websocat as a
stdio-to-TCP bridge to the same sshd. Above :meth:`SaladProvider.machines`
nothing knows: it hands back a host and a port, and ``asyncssh`` dials them like
any other provider's.

Two constraints of the gateway shape the design. It cannot carry the
``Salad-Api-Key`` header on a WebSocket upgrade, so the gateway is opened with
``auth=False`` and the security of a node rests on its unguessable domain name
and on sshd taking public keys only. And its domain name addresses a *container
group*, load balancing across that group's replicas, so a group with two
replicas has no way to reach one of them: every node is its own group of one,
which is also what makes a node individually terminable.

Nodes still cannot reach each other — the gateway is the only inbound and it is
the daemon's — so :meth:`allows_cluster_formation` stays false and a compute
here is a fleet of independent nodes.

Salad quotes one catalogue and sells two. ``list_gpu_classes`` is the only thing
it prices over the API, and a GPU class's price is the whole of a node: Salad
bills no vCPU or RAM beside one. A CPU-only group is the same container without
a GPU class, and it is billed per vCPU-hour and per GB-hour at rates that live
on the account here, because no endpoint answers for them.

A group is found by listing the project and matching the compute's prefix, never
by reading the binding back: a group is named for the compute and the node it
was launched for before Salad is asked for it, so a launch whose reply is lost
still leaves a group the listing names, and the daemon writes the id down from
there, or from the retried launch, which Salad answers ``name_conflict``. The name
carries the compute and the claim, and the listing is the record.

The sdk's timeout is set on it after it is built, not handed to its constructor:
the async flavour of the sdk builds its services after the constructor has
applied the timeout to the ones it is replacing, so the argument is lost and a
request waits the sdk's own minute. Every call goes through the sdk's retry
handler, which tries four times, so a request that fails costs four timeouts.
"""

import asyncio
import base64
import contextlib
import re
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any, ClassVar, Self

import asyncssh
import httpx
import msgspec
from salad_cloud_sdk import SaladCloudSdkAsync
from salad_cloud_sdk.models import (
    ContainerConfiguration,
    ContainerGroupCreationRequest,
    ContainerGroupPriority,
    ContainerNetworkingProtocol,
    ContainerRestartPolicy,
    CountryCode,
    CreateContainerGroupNetworking,
    CreateContainerResourceRequirements,
)
from salad_cloud_sdk.net.environment.environment import Environment
from salad_cloud_sdk.net.transport.api_error import ApiError
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from skyward.providers.network import tls
from skyward.shared.errors import CapabilityMismatchError
from skyward.shared.observability import logger
from skyward.shared.provider import Binding, Machine, MachineState, claimed
from skyward.shared.providers import Salad
from skyward.shared.schemas import ComputeSpec, Market, Offer

SSH_USER = "root"
DEFAULT_IMAGE = "saladtechnologies/misc:ubuntu24-dev"
GATEWAY_PORT = 8888
WEBSOCAT_URL = "https://github.com/vi/websocat/releases/download/v1.13.0/websocat.x86_64-unknown-linux-musl"


def _one_word(script: str) -> str:
    """``script`` as one word the relay passes through whole, which ``sh -c`` runs as written.

    The relay splits the command it is given on whitespace and ignores quoting, so
    ``sh -c 'a; b'`` reaches the container as ``sh -c "'a;"``. The script travels
    base64-encoded inside a word with no whitespace of its own, and the shell puts
    the spaces back: ``${IFS}`` expands to one where the word needs it, and ``eval``
    runs what ``base64 -d`` gives back. The script itself may then say anything.
    """
    return 'eval${IFS}"$(echo${IFS}' + base64.b64encode(script.encode()).decode() + '|base64${IFS}-d)"'


RELAY_SCRIPT = "stty raw -echo; exec /usr/local/bin/websocat --binary - tcp:127.0.0.1:22"
RELAY_COMMAND = f"sh -c {_one_word(RELAY_SCRIPT)}"
"""What the relay is asked to exec: the bridge, in a word it passes through whole."""
RELAY_CONNECT_SECONDS = 10.0
RELAY_PROBE_SECONDS = 15.0

MAX_VCPUS = 16
MAX_MEMORY_GB = 32
CPU_SIZES = tuple((cpus, memory_gb) for cpus in range(1, MAX_VCPUS + 1) for memory_gb in range(1, MAX_MEMORY_GB + 1))
"""Every CPU-only node the account may be sold, in vCPUs and whole GiB.

A container group is sized by the request that creates it, so a CPU-only node has
no catalogue behind it: Salad takes whatever size it is asked for and bills it per
vCPU-hour and per GB-hour. The shelf is therefore not a selection of sizes, it is
all of them — the market picks one offer and the node is created at its size, so a
size that is not on the shelf is a size nobody can buy, and every arbitrary one is
a size Salad would have taken.

The bounds are the largest group this adapter sells, with a GPU or without one.
Enumerating between them is what makes 5 vCPUs and 7 GB a thing to ask for rather
than something rounded up to the next entry in a list somebody chose.
"""

APT_LIST = "/tmp/skyward-apt.list"
BANDWIDTH_CHECK = """transfer() {
  w="-sSo /dev/null --connect-timeout 5 --max-time 15 -w %{http_code}:%{size_$1}:%{time_total}:"
  case $1 in
    download) out=$(curl $w%{time_starttransfer} 'https://speed.cloudflare.com/__down?bytes=25000000');;
    upload) out=$(head -c 25000000 /dev/zero | curl $w%{time_pretransfer} -T - -X POST https://speed.cloudflare.com/__up);;
  esac
  echo $out:$?
}
measure() {
  for connection in 1 2 3 4 5 6 7 8; do transfer $1 & done | awk -F: '
    { code = $1 }
    $1 == 200 || $5 == 28 && $2 > 0 { bytes += $2; if ($3 > ended) ended = $3; if (!n++ || $4 < began) began = $4 }
    END { print (ended > began ? int(bytes / (ended - began)) : -1), code }'
}
check() {
  [ -z "$2" ] && return 0
  best=-1
  for attempt in 1 2 3; do
    [ $attempt = 1 ] || sleep 20
    out=$(measure $1); speed=${out% *}
    if [ $speed -lt 0 ]; then
      echo "bandwidth check $attempt/3: the $1 test did not answer (http ${out#* })"; continue
    fi
    echo "bandwidth check $attempt/3: $((speed / 125000)) Mbps $1, $2 required"
    [ $speed -ge $(($2 * 125000)) ] && return 0
    [ $speed -gt $best ] && best=$speed
  done
  [ $best -lt 0 ] && { echo "$1 bandwidth unmeasured, carrying on"; return 0; }
  reason="insufficient $1 bandwidth: best of 3 attempts $((best / 125000)) Mbps, $2 Mbps required"
  echo "$reason, asking salad to reallocate"
  curl -sS --max-time 10 http://169.254.169.254/v1/reallocate -H 'Content-Type: application/json' -H 'Metadata: true' --data "{\\"reason\\":\\"$reason\\"}"
  return 1
}
check download "$MIN_DOWNLOAD_MBPS" && check upload "$MIN_UPLOAD_MBPS"
"""
"""What a node does about its connection, when the account names a speed.

Salad filters nodes by nothing about their network, and its own answer is this
one: measure from inside the container and ask for a reallocation (``POST
/v1/reallocate`` on the instance metadata service) when the node falls short.
The script follows Salad's example, in the ``sh`` the command runs under, run
once per direction the account names, three attempts twenty seconds apart.

An attempt is eight transfers at once, and the node's speed is the bytes they
moved together over the window they moved them in: from the first transfer's
first byte to the last one's end. One TCP connection carries a fraction of a
link, so one transfer measures the connection rather than the node — from one
home link, one transfer read 69 Mbps, eight read 343, and fast.com 820. The
window leaves the handshakes out, a download's starting at its first byte and an
upload's once TLS is up, because they move nothing. Every transfer is cut at
fifteen seconds, and ``curl`` reports what a cut transfer moved and for how long,
so a node too slow to finish is still measured.

Cloudflare's endpoint serves the 25 MB Salad's script asks for and refuses some
other sizes — 12.5 MB and 100 MB answered 403 when tried — and the upload sends
as much, streamed rather than held in memory eight times. An address that asks
for gigabytes in a few minutes is throttled: Cloudflare answered 429 and held
connections ten seconds. A speed test that does not answer is not held against
the node: a reallocation costs the account that node for 48 hours, and an
endpoint that is down or throttling says nothing about the connection.

What counts as an answer differs by direction. A download cut by the time limit
has had its ``200`` already, since the headers come before the body; an upload's
``200`` comes only once the whole body has arrived, so a cut upload has none and
is measured by having been cut (curl's exit 28) with something sent. Without
that, the slow uploads the check exists to turn away would read as a speed test
that did not answer, and pass.
"""
NODE_COMMAND = (
    "([ -x /usr/sbin/sshd ] && command -v curl > /dev/null || ("
    ". /etc/os-release && APT_OPTS='' && "
    'if [ "$ID" = ubuntu ]; then '
    "printf 'deb http://archive.ubuntu.com/ubuntu %s main\\ndeb http://archive.ubuntu.com/ubuntu %s-updates main\\n"
    "deb http://security.ubuntu.com/ubuntu %s-security main\\n' "
    f'"$VERSION_CODENAME" "$VERSION_CODENAME" "$VERSION_CODENAME" > {APT_LIST} && '
    f"APT_OPTS='-o Dir::Etc::SourceList={APT_LIST} -o Dir::Etc::SourceParts=/dev/null'; fi && "
    "apt-get $APT_OPTS -o Acquire::Languages=none update && "
    "DEBIAN_FRONTEND=noninteractive apt-get $APT_OPTS install -y --no-install-recommends openssh-server curl ca-certificates"
    ")) && "
    'sh -c "$1" && '
    "mkdir -p /run/sshd /root/.ssh && "
    'printf \'%s\\n\' "$PUBLIC_KEY" > /root/.ssh/authorized_keys && '
    "chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && "
    "ssh-keygen -A && "
    "/usr/sbin/sshd -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no && "
    "([ -x /usr/local/bin/websocat ] || ("
    f"curl -fsSL -o /usr/local/bin/websocat {WEBSOCAT_URL} && "
    "chmod +x /usr/local/bin/websocat"
    ")) && "
    f"exec /usr/local/bin/websocat --binary ws-l:[::]:{GATEWAY_PORT} tcp:127.0.0.1:22"
)
"""What the container runs.

sshd is started in the background and the bridge is what the container *is*, so
the gateway's port is served by PID 1 and a bridge that dies takes the node with
it rather than leaving one Salad still believes in.

The two guards are what an image is allowed to satisfy in advance: one that
already carries sshd, curl and websocat reaches the bridge without a package
manager.

The bandwidth check sits between the install and sshd: after it, so curl is
there on any image, and before it, so a node that is turned away has cost the
install's few megabytes and not the bootstrap. Each line it prints is the last
line of the container's output, which is what a booting node progresses by.

The check is the command's first argument, ``$1``, rather than part of it: Salad
refuses a container group any element of whose command is longer than 2497
characters, and the two together are. The refusal is a ``400`` that does not say
so — its type was ``name_conflict`` or ``container_group_update_exception`` when
measured, depending on what the element held.

An image that does not is installed into over the node's own connection, and a
Salad node is somebody's home computer: ``apt-get update`` against the sources an
Ubuntu image ships fetches 40 MB of indexes, of which ``universe`` alone is 19 MB,
and on a slow uplink that is longer than any connect timeout. Everything the
bridge needs is in ``main``, so on Ubuntu apt is pointed at a list of ``main``
alone — a few megabytes — written to ``/tmp`` rather than ``/etc/apt`` so the
image's own sources are untouched. Any other distribution keeps its sources.

The bridge binds ``[::]`` because the gateway reaches the container over IPv6: a
listener on ``0.0.0.0`` passes every probe inside the container and still leaves
the gateway answering 503, having found nothing to route to.

Writing to a file under ``/etc/ssh`` is avoided on purpose — Salad's WAF rejects
a container group whose command contains that path — so sshd's two settings
arrive as flags.
"""


type Carry = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]
"""How one accepted connection reaches the container's sshd: over the gateway, or over the relay."""


@dataclass(frozen=True, slots=True)
class _Relay:
    """Where Salad's SSH relay answers for one instance, and the host key it must show."""

    host: str
    port: int
    fingerprint: str


class _Bridge:
    """A loopback port that carries one node's SSH, one link to the container per accepted connection."""

    def __init__(self, carry: Carry) -> None:
        self._carry = carry
        self._server: asyncio.Server | None = None

    @property
    def port(self) -> int:
        if self._server is None:
            raise CapabilityMismatchError("salad bridge was never started")
        return int(self._server.sockets[0].getsockname()[1])

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._pump, host="127.0.0.1", port=0)

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        with contextlib.suppress(Exception):
            await self._server.wait_closed()

    async def _pump(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await self._carry(reader, writer)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


async def _over_gateway(url: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Carry one SSH connection over one WebSocket, until either end lets go.

    How the socket closed is logged: the gateway ends every one of them after a
    few minutes, and its close frame is the only place that says why.
    """
    try:
        async with connect(url, ssl=_TLS, max_size=None, ping_interval=None) as socket:
            await _either(_upstream(reader, socket), _downstream(socket, writer))
        closed = socket.protocol
        logger.debug(
            "salad bridge to {} closed: received {}, sent {}, received first: {}",
            url,
            closed.close_rcvd,
            closed.close_sent,
            closed.close_rcvd_then_sent,
        )
    except (OSError, WebSocketException) as error:
        logger.debug("salad bridge to {} failed: {}", url, error)


async def _over_relay(relay: _Relay, key: asyncssh.SSHKey, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Carry one SSH connection over one session of the relay, until either end lets go.

    The relay execs the bridge on a PTY it allocates whether asked to or not, in
    cooked mode: it echoes what it is sent and rewrites line ends, which no SSH
    client survives. The script puts the PTY in raw mode before websocat starts,
    and nothing of the client's is sent until the container's sshd banner has come
    back, the first moment raw mode is known to be on — what the client wrote
    meanwhile waits in its socket. What the relay prints before that is its own,
    and is dropped.
    """
    try:
        async with asyncio.timeout(RELAY_CONNECT_SECONDS):
            conn = await _relay_connect(relay, key)
        async with conn:
            process = await conn.create_process(RELAY_COMMAND, encoding=None)
            async with asyncio.timeout(RELAY_PROBE_SECONDS):
                first = await _banner(process.stdout)
            if first is None:
                logger.debug("salad relay {}:{} ended the session before the container answered", relay.host, relay.port)
                return
            writer.write(first)
            await writer.drain()
            await _either(_to_relay(reader, process.stdin), _from_relay(process.stdout, writer))
        logger.debug("salad relay bridge to {}:{} closed", relay.host, relay.port)
    except (OSError, asyncssh.Error, TimeoutError) as error:
        logger.debug("salad relay bridge to {}:{} failed: {}", relay.host, relay.port, error)


async def _either(*directions: Coroutine[Any, Any, None]) -> None:
    """Run both directions of a link, and end it when either one ends."""
    tasks = tuple(asyncio.create_task(direction) for direction in directions)
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()


async def _to_relay(reader: asyncio.StreamReader, stdin: asyncssh.SSHWriter[bytes]) -> None:
    while chunk := await reader.read(65536):
        stdin.write(chunk)
        await stdin.drain()


async def _from_relay(stdout: asyncssh.SSHReader[bytes], writer: asyncio.StreamWriter) -> None:
    while chunk := await stdout.read(65536):
        writer.write(chunk)
        await writer.drain()


async def _banner(stdout: asyncssh.SSHReader[bytes]) -> bytes | None:
    """The session's output from the container's sshd banner on, or nothing if the session ended first.

    The relay announces the container it is entering on the same stream, so the
    first line that begins ``SSH-`` is where the container starts to speak.
    """
    seen = b""
    while chunk := await stdout.read(65536):
        seen += chunk
        at = seen.find(b"SSH-")
        if at == 0 or at > 0 and seen[at - 1] == 0x0A:
            return seen[at:]
    return None


async def _relay_connect(relay: _Relay, key: asyncssh.SSHKey) -> asyncssh.SSHClientConnection:
    """A connection to the relay, admitted only on the host key the API published for the instance."""
    host_key = await asyncssh.get_server_host_key(relay.host, relay.port)
    if host_key is None or host_key.get_fingerprint() != relay.fingerprint:
        raise asyncssh.HostKeyNotVerifiable(f"salad relay {relay.host}:{relay.port} shows a host key other than the instance's")
    return await asyncssh.connect(
        relay.host,
        port=relay.port,
        username=SSH_USER,
        client_keys=[key],
        known_hosts=([host_key], (), (), (), (), (), ()),
        keepalive_interval=30,
        keepalive_count_max=3,
    )


async def _relay_answering(relay: _Relay, key: asyncssh.SSHKey) -> bool:
    """Whether a session through the relay reaches the container's sshd.

    The relay admits a session as soon as the instance is scheduled, long before
    the container runs, and an exec into one that is not running ends at once;
    what says the node is there to dial is the banner of its sshd coming back.
    """
    try:
        async with asyncio.timeout(RELAY_PROBE_SECONDS):
            conn = await _relay_connect(relay, key)
            try:
                process = await conn.create_process(RELAY_COMMAND, encoding=None)
                answered = await _banner(process.stdout) is not None
            finally:
                conn.close()
                await conn.wait_closed()
    except (OSError, asyncssh.Error, TimeoutError) as error:
        logger.debug("salad relay {}:{} is not answering: {}", relay.host, relay.port, error)
        return False
    return answered


async def _first_answering(ways: Sequence[tuple[Callable[[], Awaitable[bool]], Carry]]) -> Carry | None:
    """The carrier of the first way that answers, the rest given up on; none when none does."""
    pending = {asyncio.ensure_future(sound()): carry for sound, carry in ways}
    try:
        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                carry = pending.pop(task)
                if task.result():
                    return carry
        return None
    finally:
        for task in pending:
            task.cancel()


_BRIDGES: dict[str, _Bridge] = {}
"""One live listener per container group, keyed by the group it dials."""

_BRIDGE_LOCK = asyncio.Lock()

_TLS = ssl.create_default_context()
"""The one TLS configuration every gateway connection is opened with.

Left to itself, a ``wss://`` connect has asyncio build a default context per
connection — the system's certificate store read again, milliseconds on the loop —
and the gateway ends every link after a few minutes, for every node at once."""


async def _upstream(reader: asyncio.StreamReader, socket: ClientConnection) -> None:
    while chunk := await reader.read(65536):
        await socket.send(chunk)
    await socket.close()


async def _downstream(socket: ClientConnection, writer: asyncio.StreamWriter) -> None:
    async for message in socket:
        writer.write(message if isinstance(message, bytes) else message.encode())
        await writer.drain()


async def _bridge(group_name: str, carry: Carry) -> _Bridge:
    async with _BRIDGE_LOCK:
        if (running := _BRIDGES.get(group_name)) is not None:
            return running
        bridge = _Bridge(carry)
        await bridge.start()
        _BRIDGES[group_name] = bridge
        return bridge


async def _answering(dns: str) -> bool:
    """Whether the gateway has something to route to.

    The gateway answers 503 for as long as nothing in the container listens on its
    port, and Salad calls the instance running from the moment the container's
    command starts — before sshd is even installed. A handshake that completes is
    the bridge itself, so this is the only test that means "dial it".
    """
    try:
        async with connect(f"wss://{dns}/", ssl=_TLS, ping_interval=None, open_timeout=10):
            return True
    except (OSError, WebSocketException) as error:
        logger.debug("salad gateway {} is not answering: {}", dns, error)
        return False


async def _release_bridge(group_name: str) -> None:
    async with _BRIDGE_LOCK:
        bridge = _BRIDGES.pop(group_name, None)
    if bridge is not None:
        await bridge.close()


class SaladProvider:
    kind: ClassVar[str] = "salad"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key", "ssh_key")
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=10)

    def __init__(self, provider_id: str, name: str, api_key: str, config: Salad) -> None:
        self._id = provider_id
        self._name = name
        self._organization = _required(config.organization, "organization", name)
        self._project = _required(config.project, "project", name).lower()
        self._priority = config.priority
        self._transport = config.transport
        self._relay_key = asyncssh.import_private_key(config.ssh_key) if config.ssh_key else None
        self._config = config
        self._api_key = api_key
        self._sdk = SaladCloudSdkAsync()
        self._sdk.set_timeout(config.request_timeout * 1000)
        self._sdk.set_api_key(api_key, "Salad-Api-Key")

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        settings = msgspec.convert({**credentials, **config}, Salad)
        if not settings.api_key:
            raise CapabilityMismatchError("salad requires an api_key credential", provider=name)
        if settings.transport != "gateway" and not settings.ssh_key:
            raise CapabilityMismatchError(
                f"salad transport {settings.transport!r} needs the ssh_key credential: register a key in the portal under SSH Keys "
                "and name its private half in SALAD_SSH_KEY_FILE, or choose transport='gateway'",
                provider=name,
            )
        return cls(provider_id, name, settings.api_key, settings)

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return False

    async def offers(self) -> AsyncIterator[Offer]:
        """What the account may buy: every GPU class it can be sold, and every CPU-only size.

        Two shelves out of one endpoint, because Salad prices only the first, and
        the two are sized quite differently. A GPU class bills no vCPUs or RAM
        beside it, so every size of it is one price and one offer, published at the
        largest so that any spec within it fits: the size is the spec's, read when
        the compute is initialized. A CPU-only node is paid for by the vCPU and by
        the GB, so every size is a different price and the shelf carries all of them.
        """
        catalog = await self._sdk.organization_data.list_gpu_classes(self._organization)
        now = datetime.now(UTC)
        expires_at = now + self.offers_ttl

        for gpu_class in catalog.items or ():
            price = next(
                (
                    _number(getattr(entry, "price", None))
                    for entry in gpu_class.prices or ()
                    if _enum_value(getattr(entry, "priority", None)) == self._priority
                ),
                None,
            )
            if price is None:
                continue
            gpu_class_id = _required_text(getattr(gpu_class, "id_", None), "GPU class id")
            gpu_class_name = _required_text(getattr(gpu_class, "name", None), "GPU class name")

            yield Offer(
                id=f"{gpu_class_id}:{self._priority}",
                provider_id=self._id,
                provider_name=self._name,
                kind=self.kind,
                instance_type=gpu_class_name,
                accelerator=gpu_class_name,
                accelerator_count=_integer(getattr(gpu_class, "gpu_count", None), 1),
                cpus=MAX_VCPUS,
                memory_gb=MAX_MEMORY_GB,
                disk_gb=_integer(getattr(gpu_class, "max_storage", None)) / 1024**3,
                spot_price=None,
                on_demand_price=price,
                billing_unit="second",
                fetched_at=now,
                expires_at=expires_at,
                specific={
                    "gpu_class_id": gpu_class_id,
                    "priority": self._priority,
                    "gpu_class_type": _enum_value(getattr(gpu_class, "gpu_class_type", None)),
                    "min_vcpu": _integer(getattr(gpu_class, "min_vcpu", None)),
                    "min_ram_mb": _integer(getattr(gpu_class, "min_ram", None)),
                    "min_storage_bytes": _integer(getattr(gpu_class, "min_storage", None)),
                },
            )

        for cpus, memory_gb in CPU_SIZES:
            yield Offer(
                id=f"cpu-{cpus}x{memory_gb}:{self._priority}",
                provider_id=self._id,
                provider_name=self._name,
                kind=self.kind,
                instance_type=f"salad.{cpus}c{memory_gb}g",
                accelerator_count=0,
                cpus=cpus,
                memory_gb=memory_gb,
                disk_gb=float(self._config.storage_gb),
                spot_price=None,
                on_demand_price=cpus * self._config.vcpu_price + memory_gb * self._config.memory_price,
                billing_unit="second",
                fetched_at=now,
                expires_at=expires_at,
                specific={"priority": self._priority},
            )

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        """What it takes to create this compute's groups, at the size the spec asked for.

        The spec's alternative is the one from this account with the shelf the offer
        came from. What it leaves out is the offer's size: on the CPU-only shelf the
        one the market picked, on a GPU class the largest, which costs nothing more
        since Salad bills no vCPU or RAM beside a GPU. A GPU class id is how the two
        shelves are told apart — an offer without one is a CPU-only node.
        """
        image = spec.image.base or self._config.image or DEFAULT_IMAGE
        gpu_class_id = _text(offer.specific.get("gpu_class_id"))
        cpus, memory_gb = next(
            (
                (wanted.cpus or offer.cpus, wanted.memory_gb or offer.memory_gb)
                for wanted in spec.specs
                if wanted.provider.kind == self.kind
                and wanted.provider.name in (None, self._name)
                and (wanted.accelerator is None) == (gpu_class_id is None)
            ),
            (offer.cpus, offer.memory_gb),
        )

        return {
            "compute_id": compute_id,
            **({"gpu_class_id": gpu_class_id} if gpu_class_id else {}),
            "priority": self._priority,
            "image": image.format(python=spec.image.python or "3.13"),
            "public_key": public_key,
            "cpu": cpus,
            "memory": int(memory_gb * 1024),
            "storage": max(1024**3, self._config.storage_gb * 1024**3),
        }

    async def launch(self, binding: Binding, market: Market, node: str) -> Machine:
        """Create one container group, named for the compute and the node.

        The name is the identity: it is minted here, so a request whose reply is
        lost still leaves a group the listing matches, and the retry of that request
        is answered ``name_conflict`` (a 400, where a 409 was expected), which names the very group
        this launch asked for. It is written down from that answer, because Salad's
        listing has been seen leaving out the group of a create that answered 504,
        for as long as it ran. Nothing waits for Salad to
        allocate — a group it never places is reported by :meth:`machines` as one
        still waiting, and the control plane's provision deadline is what gives up
        on it.
        """
        group_name = _group_name(_binding_text(binding, "compute_id"), node)
        try:
            await self._sdk.container_groups.create_container_group(
                self._group_body(group_name, binding),
                self._organization,
                self._project,
            )
        except ApiError as error:
            if error.status != 409 and _problem(error) != "name_conflict":
                raise
        return Machine(id=group_name, state="pending", user=SSH_USER, node=node)

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        """What Salad has under this compute's prefix, and how far along each of it is.

        A group short of a running instance reports what it is doing rather than only
        that it is pending: pulling an image is minutes of honest work, and a control
        plane told only ``pending`` cannot tell it from a group that will never come
        up — so it gives up at its deadline and the replacement pulls the same image
        again.

        Salad's ``running`` is the container's command started, not the bridge up,
        and between the two lies the install over the node's own uplink. A machine
        is reported running only once a way into it answers; until then it is pending
        with the container's last line of output as its progress, so a boot that is
        still moving keeps its deadline and one that has stopped meets it.

        A group is created with ``autostart_policy`` on, but the start is something
        Salad schedules a few seconds after the creation, and it is dropped when the
        account's replica quota is full at that moment — groups of a compute deleted
        minutes ago still count: the group sits ``stopped`` with no instance, and
        nothing about it says it was ever meant to run. Such a group is started here,
        at every pass until it moves, and a refusal is what its progress reads.
        """
        prefix = _prefix(_binding_text(binding, "compute_id"))
        machines: dict[str, Machine] = {}
        for group_name, dns, status in await self._groups(binding):
            node = claimed(group_name, prefix)
            if status == "stopped":
                machines[group_name] = Machine(id=group_name, state="pending", user=SSH_USER, progress=await self._start(group_name), node=node)
                continue
            instances = await self._instances(group_name)
            if instances is None:
                continue
            instance = next(iter(instances), None)
            if _state(instance) != "running":
                machines[group_name] = Machine(
                    id=group_name,
                    state="pending",
                    user=SSH_USER,
                    progress=_progress(instance),
                    completion=_completion(instance),
                    node=node,
                )
                continue
            bridge = _BRIDGES.get(group_name) or await self._reach(group_name, dns, _relay(instance))
            if bridge is None:
                machines[group_name] = Machine(
                    id=group_name,
                    state="pending",
                    user=SSH_USER,
                    progress=await self._booting(group_name),
                    node=node,
                )
                continue
            machines[group_name] = Machine(
                id=group_name,
                state="running",
                host="127.0.0.1",
                port=bridge.port,
                user=SSH_USER,
                node=node,
            )
        return machines

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        for group_name in machine_ids:
            await self._discard(group_name)
            await _release_bridge(group_name)

    async def release(self, binding: Binding) -> None:
        """Take down everything the compute still has, including what nobody recorded."""
        for group_name, _, _ in await self._groups(binding):
            await self._discard(group_name)
            await _release_bridge(group_name)

    async def _reach(self, group_name: str, dns: str, relay: _Relay | None) -> _Bridge | None:
        """A bridge to the group over the first of its ways in that answers, or none yet.

        Which ways are sounded is the account's ``transport``. The gateway is asked
        with a handshake, the relay with a session that must come back with the
        container's sshd banner; under ``any`` both are asked at once, and the first
        to answer is what the bridge is built on, and kept for the group.
        """
        key = self._relay_key
        over_gateway = [(partial(_answering, dns), partial(_over_gateway, f"wss://{dns}/"))] if dns else []
        over_relay = [(partial(_relay_answering, relay, key), partial(_over_relay, relay, key))] if relay and key else []
        ways = {"gateway": over_gateway, "ssh": over_relay, "any": [*over_gateway, *over_relay]}[self._transport]
        carry = await _first_answering(ways)
        return None if carry is None else await _bridge(group_name, carry)

    async def _groups(self, binding: Binding) -> tuple[tuple[str, str, str], ...]:
        """Every container group of this compute, by name, gateway and status, as Salad has them."""
        prefix = _prefix(_binding_text(binding, "compute_id"))
        collection = await self._sdk.container_groups.list_container_groups(self._organization, self._project)
        return tuple(
            (name, _dns(group) or "", _status(group))
            for group in collection.items or ()
            if (name := _text(getattr(group, "name", None)) or "").startswith(prefix)
        )

    async def _start(self, group_name: str) -> str:
        """Ask Salad to start a group it left stopped, and say how that went.

        A refusal is the answer the user needs — the one seen so far is the account's
        replica quota, which a creation is allowed to exceed and a start is not — and
        it holds the same progress token, so the deadline runs against it.
        """
        try:
            await self._sdk.container_groups.start_container_group(self._organization, self._project, group_name)
        except ApiError as error:
            if error.status != 400:
                raise
            return f"salad refused to start it: {_problem(error) or error.message or 'bad request'}"
        return "starting"

    async def _discard(self, group_name: str) -> None:
        try:
            await self._sdk.container_groups.delete_container_group(self._organization, self._project, group_name)
        except ApiError as error:
            if error.status != 404:
                raise

    def _group_body(self, group_name: str, binding: Binding) -> ContainerGroupCreationRequest:
        cpu = _binding_integer(binding, "cpu", 1)
        memory = _binding_integer(binding, "memory", 1024)
        storage = _binding_integer(binding, "storage", 1024**3)
        gpu_class_id = _text(binding.get("gpu_class_id"))
        resources = (
            CreateContainerResourceRequirements(cpu=cpu, memory=memory, gpu_classes=[gpu_class_id], storage_amount=storage)
            if gpu_class_id
            else CreateContainerResourceRequirements(cpu=cpu, memory=memory, storage_amount=storage)
        )
        container = ContainerConfiguration(
            image=_binding_text(binding, "image"),
            command=["sh", "-c", NODE_COMMAND, "sh", BANDWIDTH_CHECK],
            environment_variables={
                "PUBLIC_KEY": _binding_text(binding, "public_key"),
                **({"MIN_DOWNLOAD_MBPS": str(self._config.min_download_mbps)} if self._config.min_download_mbps else {}),
                **({"MIN_UPLOAD_MBPS": str(self._config.min_upload_mbps)} if self._config.min_upload_mbps else {}),
            },
            image_caching=True,
            priority=ContainerGroupPriority(self._priority),
            resources=resources,
        )
        networking = CreateContainerGroupNetworking(
            auth=False,
            port=GATEWAY_PORT,
            protocol=ContainerNetworkingProtocol.HTTP,
        )
        country_codes = _country_codes(self._config.countries)
        if country_codes:
            return ContainerGroupCreationRequest(
                autostart_policy=True,
                container=container,
                name=group_name,
                networking=networking,
                replicas=1,
                restart_policy=ContainerRestartPolicy.ALWAYS,
                country_codes=country_codes,
                display_name=group_name,
            )
        return ContainerGroupCreationRequest(
            autostart_policy=True,
            container=container,
            name=group_name,
            networking=networking,
            replicas=1,
            restart_policy=ContainerRestartPolicy.ALWAYS,
            display_name=group_name,
        )

    async def _instances(self, group_name: str) -> tuple[object, ...] | None:
        try:
            collection = await self._sdk.container_groups.list_container_group_instances(
                self._organization,
                self._project,
                group_name,
            )
        except ApiError as error:
            if error.status == 404:
                return None
            if error.status is None or not 500 <= error.status < 600:
                raise
            return await self._instances_from_system_logs(group_name)
        return tuple(collection.instances or ())

    async def _booting(self, group_name: str) -> str:
        """The container's last line of output, as the token a booting group progresses by.

        Asked over plain HTTP: the sdk has this endpoint, but its model of the reply
        demands a field the reply does not carry and refuses every page.
        """
        now = datetime.now(UTC).replace(microsecond=0)
        query = {
            "start_time": _stamp(now - timedelta(hours=1)),
            "end_time": _stamp(now),
            "query": f'resource.labels.container_group_name = "{group_name}" and resource.type = "container"',
            "page_size": 1,
            "sort_order": "desc",
        }
        async with httpx.AsyncClient(timeout=self._config.request_timeout, verify=tls()) as client:
            response = await client.post(
                f"{Environment.DEFAULT.url}/organizations/{self._organization}/log-entries",
                headers={"Salad-Api-Key": self._api_key},
                json=query,
            )
        if response.is_server_error:
            return "booting"
        response.raise_for_status()
        entry = next(iter(response.json().get("items", ())), None)
        line = _text(entry.get("text_log")) if isinstance(entry, Mapping) else None
        return f"booting: {line.strip()[:80]}" if line else "booting"

    async def _instances_from_system_logs(self, group_name: str) -> tuple[object, ...]:
        try:
            logs = await self._sdk.system_logs.get_system_logs(self._organization, self._project, group_name)
        except ApiError as error:
            if error.status == 404 or error.status is not None and 500 <= error.status < 600:
                return ()
            raise
        except TypeError:
            response = getattr(self._sdk.system_logs, "_last_response", None)
            body = getattr(response, "body", None)
            if not isinstance(body, Mapping) or not isinstance(body.get("items"), list):
                raise
            logs_items = body["items"]
        else:
            logs_items = logs.items or ()

        instance_ids = tuple(
            dict.fromkeys(
                instance_id
                for log in logs_items
                if (
                    instance_id := _text(
                        log.get("instance_id") if isinstance(log, Mapping) else getattr(log, "instance_id", None)
                    )
                )
            )
        )
        instances: list[object] = []
        for instance_id in instance_ids:
            try:
                instance = await self._sdk.container_groups.get_container_group_instance(
                    self._organization,
                    self._project,
                    group_name,
                    instance_id,
                )
            except ApiError as error:
                if error.status == 404 or error.status is not None and 500 <= error.status < 600:
                    continue
                raise
            instances.append(instance)
        return tuple(instances)


def _state(instance: object) -> MachineState:
    if instance is None:
        return "pending"
    ready = bool(getattr(instance, "ready", False))
    return "running" if _enum_value(getattr(instance, "state", None)) == "running" and ready else "pending"


def _relay(instance: object) -> _Relay | None:
    """Where the relay reaches this instance, which Salad says once it has scheduled it on a node."""
    host = _text(getattr(instance, "ssh_ip", None))
    port = _integer(getattr(instance, "ssh_port", None))
    fingerprint = _text(getattr(instance, "ssh_host_key_fingerprint", None))
    return _Relay(host, port, fingerprint.split()[0]) if host and port and fingerprint else None


def _stamp(moment: datetime) -> str:
    """A time the log query accepts: millisecond precision, and nothing finer."""
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _dns(group: object) -> str | None:
    return _text(getattr(getattr(group, "networking", None), "dns", None))


def _status(group: object) -> str:
    return _enum_value(getattr(getattr(group, "current_state", None), "status", None)) or "pending"


def _prefix(compute_id: str) -> str:
    """What every one of a compute's container groups is called before its suffix.

    Salad has no tags, so the name is the only thing an adapter controls and can
    match an account-wide listing on. It is how a compute recognises its own.
    """
    return re.sub(r"[^a-z0-9-]", "-", f"skyward-{compute_id.lower()}").strip("-")[:44].rstrip("-") + "-"


def _group_name(compute_id: str, node: str) -> str:
    """The prefix and the node's claim, within the 63 characters a group name is allowed."""
    return f"{_prefix(compute_id)}{node}"[:63].rstrip("-")


def _progress(instance: object) -> str:
    """What Salad is doing with a group that has no address yet, in one word.

    The state's own name, which is what a reader wants above the bar: how far the
    pull has got is :func:`_completion`, and the two are put together by whoever is
    rendering them.
    """
    if instance is None:
        return "waiting for salad to allocate a machine"
    return _enum_value(getattr(instance, "state", None)) or "pending"


def _completion(instance: object) -> float | None:
    """How much of the image is pulled, as a fraction, for the state that pulls one.

    Whole percent, because every change is a line on somebody's console and an event
    on the wire: a pull is polled every couple of seconds and a tenth of a percent is
    not news. It is still a hundred times finer than the deadline needs.

    ``pulling_progress`` arrives as a fraction — ``0.43`` is 43% of the image — while
    the sdk's own model says it is a percentage and validates it to 100. Only one
    reading of ``0.43`` is sane, and a value above one can only be the other, so both
    are accepted and the answer is a fraction either way.
    """
    if instance is None or _enum_value(getattr(instance, "state", None)) != "downloading":
        return None
    pulled = _number(getattr(instance, "pulling_progress", None))
    return None if pulled is None else round(pulled if pulled <= 1 else pulled / 100, 2)


def _required(value: str | None, key: str, provider: str) -> str:
    if not value:
        raise CapabilityMismatchError(f"salad needs config.{key}", provider=provider)
    return value


def _binding_text(binding: Binding, key: str) -> str:
    value = binding.get(key)
    if not isinstance(value, str) or not value:
        raise CapabilityMismatchError(f"salad binding has no {key}")
    return value


def _binding_integer(binding: Binding, key: str, default: int) -> int:
    value = binding.get(key)
    return int(value) if isinstance(value, int | float) else default


def _enum_value(value: object) -> str | None:
    member = getattr(value, "value", value)
    return member if isinstance(member, str) else None


def _number(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _integer(value: object, default: int = 0) -> int:
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _required_text(value: object, field: str) -> str:
    text = _text(value)
    if text is None or not text:
        raise CapabilityMismatchError(f"salad returned no {field}")
    return text


def _country_codes(codes: tuple[str, ...]) -> list[CountryCode]:
    try:
        return [CountryCode(code) for code in codes]
    except ValueError as error:
        raise CapabilityMismatchError("salad country_codes contains an invalid country code") from error


def _problem(error: ApiError) -> str | None:
    """The problem type Salad answered with, read off the body the sdk kept."""
    body = getattr(error.response, "body", None)
    return _text(body.get("type")) if isinstance(body, Mapping) else None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


__all__ = ["SaladProvider"]
