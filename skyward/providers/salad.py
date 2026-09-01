"""Salad Cloud, reached through its HTTP gateway rather than its SSH relay.

Salad rents containers, not machines, and gives a container exactly one way in:
the Container Gateway, an HTTP reverse proxy in front of one port. There is no
inbound TCP to a node, and the SSH relay the portal advertises is a preview
feature that runs on Salad's side of the container — on a node whose relay agent
fails, it drops the connection before the banner and only a reallocation fixes
it. Neither the daemon nor the node can do anything about that from here.

So the gateway is made to look like a socket. The container runs sshd and a
WebSocket-to-TCP bridge on the gateway port, and the daemon runs one loopback
listener per node whose accepted connections become WebSockets to that node's
gateway. Above :meth:`SaladProvider.machines` nothing knows: it hands back a
host and a port, and ``asyncssh`` dials them like any other provider's.

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

A group is found by listing the project and matching the compute's prefix, never
by reading the binding back. The binding is written after the group is created,
and a launch waits on Salad to allocate before it returns, so the window between
a group existing and the daemon having written it down is minutes wide — and a
group nobody wrote down is a container nothing can name and nothing will ever
stop. The name carries the compute, and the listing is the record.
"""

import asyncio
import contextlib
import re
import secrets
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

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
from salad_cloud_sdk.net.transport.api_error import ApiError
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from skyward.shared.errors import CapabilityMismatchError
from skyward.shared.provider import Binding, Machine, MachineState
from skyward.shared.providers import Salad
from skyward.shared.schemas import ComputeSpec, Market, Offer

SSH_USER = "root"
DEFAULT_IMAGE = "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04"
GATEWAY_PORT = 8888
WEBSOCAT_URL = "https://github.com/vi/websocat/releases/download/v1.13.0/websocat.x86_64-unknown-linux-musl"

NODE_COMMAND = (
    "([ -x /usr/sbin/sshd ] && command -v curl > /dev/null || ("
    "apt-get update && "
    "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openssh-server curl ca-certificates"
    ")) && "
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
manager, which is the difference between a cold start and a dice roll.

The bridge binds ``[::]`` because the gateway reaches the container over IPv6: a
listener on ``0.0.0.0`` passes every probe inside the container and still leaves
the gateway answering 503, having found nothing to route to.

Writing to a file under ``/etc/ssh`` is avoided on purpose — Salad's WAF rejects
a container group whose command contains that path — so sshd's two settings
arrive as flags.
"""


class _Bridge:
    """A loopback port that carries one node's SSH over its gateway."""

    def __init__(self, url: str) -> None:
        self._url = url
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
            async with connect(self._url, max_size=None, ping_interval=None) as socket:
                streams = (
                    asyncio.create_task(_upstream(reader, socket)),
                    asyncio.create_task(_downstream(socket, writer)),
                )
                try:
                    await asyncio.wait(streams, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for stream in streams:
                        stream.cancel()
        except (OSError, WebSocketException):
            pass
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


_BRIDGES: dict[str, _Bridge] = {}
"""One live listener per container group, keyed by the group it dials."""

_BRIDGE_LOCK = asyncio.Lock()


async def _upstream(reader: asyncio.StreamReader, socket: ClientConnection) -> None:
    while chunk := await reader.read(65536):
        await socket.send(chunk)
    await socket.close()


async def _downstream(socket: ClientConnection, writer: asyncio.StreamWriter) -> None:
    async for message in socket:
        writer.write(message if isinstance(message, bytes) else message.encode())
        await writer.drain()


async def _bridge(group_name: str, dns: str) -> _Bridge:
    async with _BRIDGE_LOCK:
        if (running := _BRIDGES.get(group_name)) is not None:
            return running
        bridge = _Bridge(f"wss://{dns}/")
        await bridge.start()
        _BRIDGES[group_name] = bridge
        return bridge


async def _release_bridge(group_name: str) -> None:
    async with _BRIDGE_LOCK:
        bridge = _BRIDGES.pop(group_name, None)
    if bridge is not None:
        await bridge.close()


class SaladProvider:
    kind: ClassVar[str] = "salad"
    credential_fields: ClassVar[tuple[str, ...]] = ("api_key",)
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=10)

    def __init__(self, provider_id: str, name: str, api_key: str, config: Salad) -> None:
        self._id = provider_id
        self._name = name
        self._organization = _required(config.organization, "organization", name)
        self._project = _required(config.project, "project", name).lower()
        self._priority = config.priority
        self._config = config
        self._sdk = SaladCloudSdkAsync(timeout=config.request_timeout * 1000)
        self._sdk.set_api_key(api_key, "Salad-Api-Key")

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        settings = msgspec.convert({**credentials, **config}, Salad)
        if not settings.api_key:
            raise CapabilityMismatchError("salad requires an api_key credential", provider=name)
        return cls(provider_id, name, settings.api_key, settings)

    def allows_cluster_formation(self, spec: ComputeSpec, offer: Offer) -> bool:
        return False

    async def offers(self) -> AsyncIterator[Offer]:
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
                cpus=_integer(getattr(gpu_class, "max_vcpu", None)),
                memory_gb=_integer(getattr(gpu_class, "max_ram", None)) / 1024,
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

    async def initialize(self, compute_id: str, spec: ComputeSpec, offer: Offer, market: Market, public_key: str) -> Binding:
        gpu_class_id = offer.specific.get("gpu_class_id")
        if not isinstance(gpu_class_id, str) or not gpu_class_id:
            raise CapabilityMismatchError("salad offer has no gpu class id", provider=self._name)

        image = spec.image.base or self._config.image or DEFAULT_IMAGE

        return {
            "compute_id": compute_id,
            "gpu_class_id": gpu_class_id,
            "priority": self._priority,
            "image": image.format(python=spec.image.python or "3.13"),
            "public_key": public_key,
            "cpu": max(1, offer.cpus),
            "memory": max(1024, int(offer.memory_gb * 1024)),
            "storage": max(1024**3, self._config.storage_gb * 1024**3),
        }

    async def launch(self, binding: Binding, market: Market, count: int, min_count: int) -> tuple[Binding, Sequence[Machine]]:
        """Create one container group per machine, and wait for Salad to place them.

        The wait is what lets a GPU class that has nothing to sell be answered with
        another one: a group Salad never allocates is a launch that failed, and the
        caller tries the next offer. Nothing is written into the binding — the group
        carries the compute in its name, which is what the listing matches on.
        """
        compute_id = _binding_text(binding, "compute_id")
        minted: list[str] = []
        try:
            for _ in range(count):
                group_name = _group_name(compute_id)
                await self._sdk.container_groups.create_container_group(
                    self._group_body(group_name, binding),
                    self._organization,
                    self._project,
                )
                minted.append(group_name)
            await self._allocated(tuple(minted), min_count)
        except BaseException:
            await self._discard_all(minted)
            raise

        return binding, tuple(Machine(id=group_name, state="pending", user=SSH_USER) for group_name in minted)

    async def machines(self, binding: Binding) -> Mapping[str, Machine]:
        """What Salad has under this compute's prefix, and how far along each of it is.

        A group short of a running instance reports what it is doing rather than only
        that it is pending: pulling an image is minutes of honest work, and a control
        plane told only ``pending`` cannot tell it from a group that will never come
        up — so it gives up at its deadline and the replacement pulls the same image
        again.
        """
        machines: dict[str, Machine] = {}
        for group_name, dns in await self._groups(binding):
            instances = await self._instances(group_name)
            if instances is None:
                continue
            instance = next(iter(instances), None)
            if _state(instance) != "running" or not dns:
                machines[group_name] = Machine(
                    id=group_name,
                    state="pending",
                    user=SSH_USER,
                    progress=_progress(instance),
                    completion=_completion(instance),
                )
                continue
            bridge = await _bridge(group_name, dns)
            machines[group_name] = Machine(
                id=group_name,
                state="running",
                host="127.0.0.1",
                port=bridge.port,
                user=SSH_USER,
            )
        return machines

    async def terminate(self, binding: Binding, machine_ids: tuple[str, ...]) -> None:
        for group_name in machine_ids:
            await self._discard(group_name)
            await _release_bridge(group_name)

    async def release(self, binding: Binding) -> None:
        """Take down everything the compute still has, including what nobody recorded."""
        for group_name, _ in await self._groups(binding):
            await self._discard(group_name)
            await _release_bridge(group_name)

    async def _groups(self, binding: Binding) -> tuple[tuple[str, str], ...]:
        """Every container group of this compute, by name and gateway, as Salad has them."""
        prefix = _prefix(_binding_text(binding, "compute_id"))
        collection = await self._sdk.container_groups.list_container_groups(self._organization, self._project)
        return tuple(
            (name, _dns(group) or "")
            for group in collection.items or ()
            if (name := _text(getattr(group, "name", None)) or "").startswith(prefix)
        )

    async def _allocated(self, group_names: tuple[str, ...], min_count: int) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._config.allocation_timeout
        allocated: set[str] = set()
        while len(allocated) < min_count:
            for group_name in group_names:
                if group_name not in allocated and await self._instances(group_name):
                    allocated.add(group_name)
            if len(allocated) >= min_count:
                return
            if loop.time() >= deadline:
                raise CapabilityMismatchError(
                    f"salad did not allocate {min_count} container instance(s)",
                    provider=self._name,
                )
            await asyncio.sleep(self._config.poll_interval)

    async def _discard(self, group_name: str) -> None:
        try:
            await self._sdk.container_groups.delete_container_group(self._organization, self._project, group_name)
        except ApiError as error:
            if error.status != 404:
                raise

    async def _discard_all(self, group_names: Sequence[str]) -> None:
        """Take back what a failed launch created, without burying why it failed.

        Best effort on purpose: the reason the launch failed is what the caller has
        to act on, and a group this could not delete is still one the listing finds.
        """
        for group_name in group_names:
            with contextlib.suppress(Exception):
                await self._discard(group_name)

    def _group_body(self, group_name: str, binding: Binding) -> ContainerGroupCreationRequest:
        resources = CreateContainerResourceRequirements(
            cpu=_binding_integer(binding, "cpu", 1),
            memory=_binding_integer(binding, "memory", 1024),
            gpu_classes=[_binding_text(binding, "gpu_class_id")],
            storage_amount=_binding_integer(binding, "storage", 1024**3),
        )
        container = ContainerConfiguration(
            image=_binding_text(binding, "image"),
            command=["sh", "-c", NODE_COMMAND],
            environment_variables={"PUBLIC_KEY": _binding_text(binding, "public_key")},
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


def _dns(group: object) -> str | None:
    return _text(getattr(getattr(group, "networking", None), "dns", None))


def _prefix(compute_id: str) -> str:
    """What every one of a compute's container groups is called before its suffix.

    Salad has no tags, so the name is the only thing an adapter controls and can
    match an account-wide listing on. It is how a compute recognises its own.
    """
    return re.sub(r"[^a-z0-9-]", "-", f"skyward-{compute_id.lower()}").strip("-")[:54].rstrip("-")


def _group_name(compute_id: str) -> str:
    return f"{_prefix(compute_id)}-{secrets.token_hex(4)}"


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


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


__all__ = ["SaladProvider"]
