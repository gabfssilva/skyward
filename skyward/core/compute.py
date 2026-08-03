"""The pool, and the thread it stands on.

The public API is synchronous because the code that uses it is: a training
script is a script. Underneath, one event loop in a daemon thread serves every
call, which is also what makes ``task() > pool`` a real future and ``a & b`` a
real overlap rather than two round trips in a row.
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Iterator, Sequence
from concurrent.futures import Future, as_completed
from contextlib import ExitStack, aclosing, suppress
from contextvars import Token
from pathlib import Path
from typing import TYPE_CHECKING, Self

import msgspec

from skyward.core import context, usercode
from skyward.core.accelerators import Accelerator
from skyward.core.client import Client
from skyward.core.console import watcher
from skyward.core.errors import SkywardError, TaskFailedError
from skyward.core.forward import TcpProxy
from skyward.core.function import Group, Pending, Streaming
from skyward.core.provider import Provider
from skyward.core.spec import Executor, Nodes, NodeSpec, Options, Port, Spec, Volume
from skyward.server.persistence.db import DEFAULT_PATH
from skyward.shared import codec
from skyward.shared.accelerators import resolve
from skyward.shared.frames import Chunk, Failed, Frame
from skyward.shared.schemas import (
    Allocation,
    ComputeCreate,
    ComputeSpec,
    Dispatch,
    Endpoint,
    Error,
    Image,
    Lease,
    LeaseClaim,
    NodeBounds,
    ProviderCreate,
    ProviderRef,
    Selection,
    Task,
    TaskCreate,
    Worker,
)
from skyward.shared.schemas import (
    Compute as ComputeView,
)
from skyward.shared.schemas import (
    Options as OptionsRef,
)
from skyward.shared.schemas import (
    Provider as ProviderView,
)
from skyward.shared.schemas import (
    Spec as SpecRef,
)
from skyward.shared.schemas import (
    Volume as VolumeRef,
)
from skyward.worker.plugins import Plugin

if TYPE_CHECKING:
    from skyward.worker.storage import Credential

DEFAULT_IMAGE = Image()
DEFAULT_EXECUTOR = Executor()
DEFAULT_OPTIONS = Options()
INLINE = 256 * 1024
POLL = 30
READY_TIMEOUT = 900
DELETE_TIMEOUT = 300
LEASE_SECONDS = 60
"""How long the compute stays owned after the last renewal.

The claim is the client's heartbeat: while this process is alive the lease never
lapses, and once it dies the reconciler is free to call the compute abandoned. Short
enough that a killed script is noticed in a minute; long enough that no renewal ever
races its own expiry.
"""


class Loop:
    """One event loop in a daemon thread; the whole synchronous API stands on it."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="skyward")
        self._thread.start()

    def start[T](self, coro: Coroutine[None, None, T]) -> Future[T]:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def run[T](self, coro: Coroutine[None, None, T]) -> T:
        return self.start(coro).result()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()


class Compute:
    """A pool of machines, for as long as the ``with`` block lasts.

    ``url`` decides where the control plane is, and nothing else changes: given
    one, the pool talks to a daemon; given none, it runs the daemon in this
    process. Both go through the same client.
    """

    def __init__(
        self,
        *specs: Spec,
        provider: Provider | None = None,
        accelerator: str | Accelerator | None = None,
        cpus: int | None = None,
        memory_gb: int | None = None,
        region: str | None = None,
        nodes: NodeSpec = 1,
        allocation: Allocation = "spot_if_available",
        selection: Selection = "cheapest",
        image: Image = DEFAULT_IMAGE,
        plugins: Sequence[Plugin] = (),
        executor: Executor = DEFAULT_EXECUTOR,
        options: Options = DEFAULT_OPTIONS,
        ports: Sequence[Port] = (),
        volumes: Sequence[Volume] = (),
        ttl: int = 600,
        name: str | None = None,
        url: str | None = None,
        database: Path = DEFAULT_PATH,
        delete_on_exit: bool = True,
        console: bool = True,
        attach: str | None = None,
    ) -> None:
        if specs and provider is not None:
            raise ValueError("a pool takes either specs or a provider, not both")
        if attach and (specs or provider is not None):
            raise ValueError("attaching takes the compute that already exists; a spec would describe a different one")
        if provider is not None:
            specs = (Spec(provider, accelerator, cpus, memory_gb, region),)
        if not specs and not attach:
            raise ValueError("a pool needs a provider, or at least one spec")

        self._attach = attach
        self._providers = {spec.provider.name: spec.provider for spec in specs}
        self._spec = ComputeSpec(
            specs=tuple(_wire(spec) for spec in specs),
            nodes=_bounds(nodes),
            allocation=allocation,
            selection=selection,
            image=image,
            worker=Worker(
                concurrency=executor.concurrency,
                executor=executor.type,
                reuse=executor.reuse,
                buffer=executor.buffer,
            ),
            options=OptionsRef(
                ssh_connect_timeout=options.ssh_timeout,
                ssh_reconnect_attempts=options.max_provision_attempts,
                ssh_retry_delay=options.provision_retry_delay,
                worker_timeout=options.worker_timeout,
                autoscale_idle_timeout=options.autoscale_idle_timeout,
                autoscale_cooldown=options.autoscale_cooldown,
                default_compute_timeout=options.default_compute_timeout,
                health_command=options.health_command,
                health_interval=options.health_checker.interval if options.health_checker else options.health_interval,
                health_failures=options.health_checker.consecutive_failures if options.health_checker else options.health_failures,
                health_function=codec.dumps(options.health_checker.fn) if options.health_checker else None,
                health_timeout=options.health_checker.timeout if options.health_checker else 15.0,
                health_initial_delay=options.health_checker.initial_delay if options.health_checker else 0.0,
                cluster=options.cluster,
            ),
            plugins=tuple(plugin.ref() for plugin in plugins),
            delete_on_exit=delete_on_exit,
            ttl=ttl,
        )
        self._name = name
        self._plugins = tuple(plugins)
        self._ports = tuple(ports)
        self._volumes = tuple(volumes)
        self._proxies: list[TcpProxy] = []
        self._client_stack = ExitStack()
        self._url = url or os.environ.get("SKYWARD_URL")
        self._database = database
        self._delete_on_exit = delete_on_exit
        self._console = console
        self._ready_timeout = options.ready_timeout
        self._shutdown_timeout = options.shutdown_timeout

        self._loop: Loop | None = None
        self._client: Client | None = None
        self._watching: Future[None] | None = None
        self._leasing: Future[None] | None = None
        self._owner = f"sdk_{uuid.uuid4().hex[:12]}"
        self._id = ""
        self._active_token: Token[context.Pool | None] | None = None

    @classmethod
    def attached(
        cls,
        ref: str,
        url: str | None = None,
        database: Path = DEFAULT_PATH,
        console: bool = True,
        delete_on_exit: bool = False,
    ) -> Compute:
        """The compute that is already there, by name or by id.

            with sky.Compute(provider=sky.AWS(), nodes=8, name="training", delete_on_exit=False) as pool:
                ...

            with sky.Compute.attached("training") as pool:   # tomorrow, another process
                more(data) >> pool

        The machines outlive the process that asked for them, which is the whole
        reason the control plane is a daemon and not a library. This is how a second
        process says so — it takes no spec, because the compute it is joining already
        has one, and a spec here could only disagree with it.

        It does not delete on exit by default. A pool somebody else is using is not a
        pool to take down on the way out.
        """
        return cls(url=url, database=database, console=console, delete_on_exit=delete_on_exit, attach=ref)

    @property
    def id(self) -> str:
        return self._id

    def __enter__(self) -> Self:
        self._loop = Loop()
        self._client = self.loop.run(
            Client.remote(self._url) if self._url else Client.embedded(self._database),
        )
        self.loop.run(self._provision())
        for plugin in self._plugins:
            self._client_stack.enter_context(plugin.client(self))
        for port in self._ports:
            proxy = TcpProxy(self.client, self._id, port)
            self.loop.run(proxy.start())
            self._proxies.append(proxy)
        self._active_token = context.enter(self)
        return self

    def __exit__(self, *_: object) -> None:
        """Tear down, and stay torn down even when a ``Ctrl-C`` lands mid-teardown.

        The interrupt arrives on this thread, blocked on a result from the loop's;
        left to propagate it would skip closing the loop, and the daemon thread would
        go on running the destroy nobody is waiting for. The event loop is what has to
        be stopped last and unconditionally, because it is the thread that keeps the
        process alive.
        """
        if self._active_token is not None:
            context.reset(self._active_token)
            self._active_token = None
        for proxy in self._proxies:
            with suppress(Exception):
                self.loop.run(proxy.stop())
        self._proxies = []
        try:
            self._client_stack.close()
            if self._delete_on_exit:
                self.loop.run(self._destroy())
        finally:
            if self._leasing:
                self._leasing.cancel()
            if self._watching:
                self._watching.cancel()
            try:
                if self._id:
                    self.loop.run(self.client.delete(f"/v1/computes/{self._id}/lease"))
            finally:
                try:
                    self.loop.run(self.client.close())
                finally:
                    self.loop.close()
                    self._loop, self._client, self._watching, self._leasing = None, None, None, None

    @property
    def loop(self) -> Loop:
        if self._loop is None:
            raise RuntimeError("the pool is only usable inside its `with` block")
        return self._loop

    @property
    def client(self) -> Client:
        if self._client is None:
            raise RuntimeError("the pool is only usable inside its `with` block")
        return self._client

    def run[T](self, pending: Pending[T]) -> T:
        return self.start(pending).result()

    def start[T](self, pending: Pending[T]) -> Future[T]:
        return self.loop.start(self._one(pending))

    def broadcast[T](self, pending: Pending[T]) -> list[T]:
        return self.loop.run(self._all(pending))

    def gather[T](self, group: Group[T]) -> list[T]:
        futures = [self.start(pending) for pending in group.pendings]
        return [future.result() for future in futures]

    def gather_stream[T](self, group: Group[T]) -> Iterator[T]:
        """Each answer as it lands, rather than all of them at the end.

        Every task is submitted up front, so they overlap; the yielding is what
        differs. ``ordered`` walks the futures as submitted and blocks on the next
        one due — a slow first call holds back the rest; the unordered path hands
        over whichever finishes first and never waits on a straggler out of turn.
        """
        futures = [self.start(pending) for pending in group.pendings]
        source = futures if group.ordered else as_completed(futures)
        for future in source:
            yield future.result()

    def stream[T](self, pending: Streaming[T]) -> Iterator[T]:
        """The items, as the machine produces them.

        The task is submitted here and dispatched by the request that reads it — the
        loop below pulls one frame at a time, and the pull reaches all the way to the
        generator on the node. A consumer that stops consuming stops it.

        The failure comes back as the last frame rather than as a status, because by
        the time a generator raises, the caller already has the items it yielded
        before it, and there is no other way to say so.
        """
        task = self.loop.run(self._submit(pending, dispatch="stream"))
        frames = self.client.frames(f"/v1/tasks/{task.id}/stream")

        try:
            while (frame := self.loop.run(_next(frames))) is not None:
                match msgspec.msgpack.decode(frame, type=Frame):
                    case Chunk(value=value):
                        yield codec.loads(value)
                    case Failed(error=error, traceback=trace):
                        raise TaskFailedError(
                            Error(code="task_failed", message=error, retryable=False, details={"traceback": trace}),
                        )
        finally:
            if self._loop is not None:
                self.loop.run(frames.aclose())

    def map[I, R](self, fn: Callable[[I], Pending[R]], items: Iterable[I]) -> list[R]:
        """One task per item, spread over the nodes, answers in the order asked."""
        futures = [self.start(fn(item)) for item in items]
        return [future.result() for future in futures]

    def current_nodes(self) -> int:
        compute = self.loop.run(self.client.call("GET", f"/v1/computes/{self._id}", ComputeView))
        return compute.status.nodes_ready

    async def _provision(self) -> None:
        if self._attach:
            found = await self.client.call("GET", f"/v1/computes/{self._attach}", ComputeView)
            self._id = found.id
        else:
            await self._ensure_providers()
            await self._upload_includes()
            await self._upload_volumes()
            compute = await self.client.call(
                "POST",
                "/v1/computes",
                ComputeView,
                body=msgspec.json.encode(ComputeCreate(spec=self._spec, name=self._name)),
                headers={"Idempotency-Key": uuid.uuid4().hex},
            )
            self._id = compute.id

        await self._claim()
        self._leasing = self.loop.start(self._renew())

        if self._console:
            # Building the watcher imports the dashboard, which probes the terminal
            # for its background colour (OSC 11) and can wait ~250ms for the reply.
            follower = await asyncio.to_thread(watcher, self.client, self._id)
            self._watching = self.loop.start(follower.follow())

        async with asyncio.timeout(self._ready_timeout):
            current = await self._reach("ready", "failed", "degraded")

        if current.status.state != "ready":
            raise RuntimeError(f"compute {self._id} is {current.status.state}: {current.status.last_error}")

    async def _ensure_providers(self) -> None:
        """The credentials live in the daemon, and this is how they get there.

        A provider is a row, not a value in the spec: the spec names a kind, and
        the row holds what it takes to log in. Registering it here is what makes
        ``Compute(provider=Container())`` enough on a store that has never seen one.
        """
        for provider in self._providers.values():
            body = ProviderCreate(
                name=provider.name,
                kind=provider.kind,
                credentials=dict(provider.credentials),
                config=dict(provider.config),
            )
            try:
                existing = await self.client.call("GET", f"/v1/providers/{provider.name}", ProviderView)
            except SkywardError as error:
                if error.code != "not_found":
                    raise
                await self.client.call(
                    "POST",
                    "/v1/providers",
                    dict[str, object],
                    body=msgspec.json.encode(body),
                )
            else:
                config = msgspec.json.decode(msgspec.json.encode(provider.config), type=dict[str, object])
                if existing.kind != provider.kind or existing.config != config:
                    await self.client.call(
                        "PUT",
                        f"/v1/providers/{provider.name}",
                        dict[str, object],
                        body=msgspec.json.encode(body),
                    )

    async def _upload_includes(self) -> None:
        """Pack the local code the image asks for, and store it where the node can reach it.

        The paths are the client's — the daemon may be somewhere else entirely — so the
        tarball is built here, uploaded as a blob, and the spec carries only its hash.
        The node reads the bytes back from the blob store when it comes up.
        """
        image = self._spec.image
        if not image.includes:
            return

        blob = await asyncio.to_thread(usercode.tarball, image.includes, image.excludes)
        sha = await codec.digest(blob)
        await self.client.upload(f"/v1/blobs/{sha}", blob)
        self._spec = msgspec.structs.replace(
            self._spec,
            image=msgspec.structs.replace(image, includes_sha256=sha),
        )

    async def _upload_volumes(self) -> None:
        """Put the volumes on the spec, and the credentials for them somewhere else.

        A volume with its own ``storage`` is one the daemon cannot reach on its
        own — an R2 bucket, a Wasabi bucket, anything belonging to an account no
        provider record describes. Its keys are resolved here, where the callables
        that produce them live, and uploaded as a blob; the spec carries the digest
        and nothing more, because the spec is written to the compute row and handed
        back by ``GET /v1/computes/{id}``.
        """
        if not self._volumes:
            return

        refs: list[VolumeRef] = []
        for volume in self._volumes:
            ref = VolumeRef(bucket=volume.bucket, mount=volume.mount, prefix=volume.prefix, read_only=volume.read_only)
            match volume.storage:
                case None:
                    refs.append(ref)
                case storage:
                    resolved = await storage.resolve()
                    blob = msgspec.json.encode(
                        Endpoint(
                            url=resolved.endpoint,
                            access_key=_credential(resolved.access_key),
                            secret_key=_credential(resolved.secret_key),
                            path_style=resolved.path_style,
                        ),
                    )
                    sha = await codec.digest(blob)
                    await self.client.upload(f"/v1/blobs/{sha}", blob)
                    refs.append(msgspec.structs.replace(ref, storage_sha256=sha))

        self._spec = msgspec.structs.replace(self._spec, volumes=tuple(refs))

    async def _claim(self) -> None:
        """Own the compute, and say so out loud.

        The lease is the only thing standing between a killed script and machines
        that bill forever: while this process renews it, the compute is somebody's;
        the moment renewals stop, the reconciler is allowed to conclude nobody is
        coming back and — with ``delete_on_exit`` — tear it down.
        """
        await self.client.call(
            "PUT",
            f"/v1/computes/{self._id}/lease",
            Lease,
            body=msgspec.json.encode(LeaseClaim(owner=self._owner, ttl_seconds=LEASE_SECONDS)),
        )

    async def _renew(self) -> None:
        while True:
            await asyncio.sleep(LEASE_SECONDS / 3)
            await self._claim()

    async def _destroy(self) -> None:
        current = await self.client.call("GET", f"/v1/computes/{self._id}", ComputeView)
        await self.client.call(
            "DELETE",
            f"/v1/computes/{self._id}",
            ComputeView,
            headers={"If-Match": f'"{current.revision}"', "Idempotency-Key": uuid.uuid4().hex},
        )

        async with asyncio.timeout(self._shutdown_timeout):
            await self._reach("deleted")

    async def _reach(self, *states: str) -> ComputeView:
        """Follow the compute's events; answer with its view once it reaches one of ``states``.

        The stream replays the log from the start before it follows, so a state reached
        before this subscription still arrives. Only a compute-level event prompts a read
        — those are the transitions worth waking for — which is what turns a poll every
        half second into one read per change.
        """
        async with aclosing(self.client.events(self._id)) as events:
            async for event, _ in events:
                if not event.startswith("compute."):
                    continue
                current = await self.client.call("GET", f"/v1/computes/{self._id}", ComputeView)
                if current.status.state in states:
                    return current
        raise RuntimeError(f"the event stream for {self._id} ended before it reached {', '.join(states)}")

    async def _one[T](self, pending: Pending[T]) -> T:
        task = await self._submit(pending, dispatch="one")
        return await codec.Pickle[T]().decode(await self._settled(task.id))

    async def _all[T](self, pending: Pending[T]) -> list[T]:
        """Every node's answer, in rank order.

        The result endpoint settles the task and raises what there is to raise —
        a broadcast where one node failed is a failed broadcast. It hands back
        one node's value, so the rest are collected from the executions.
        """
        task = await self._submit(pending, dispatch="all")
        await self._settled(task.id)

        settled = await self.client.call("GET", f"/v1/tasks/{task.id}", Task)
        blobs = [
            await self.client.blob(f"/v1/blobs/{execution.result_sha256}")
            for execution in sorted(settled.executions, key=lambda execution: execution.rank)
            if execution.result_sha256
        ]
        return [await codec.Pickle[T]().decode(blob) for blob in blobs if blob is not None]

    async def _submit[T](self, pending: Pending[T] | Streaming[T], dispatch: Dispatch) -> Task:
        code = await codec.payload.encode(pending.fn)
        function = await codec.digest(code)
        await self.client.upload(
            f"/v1/functions/{function}",
            code,
            headers={"X-Skyward-Function-Name": pending.fn.__name__},
        )

        args = await codec.payload.encode((pending.args, pending.kwargs))
        inline, stored = await self._args(args)

        return await self.client.call(
            "POST",
            "/v1/tasks",
            Task,
            body=msgspec.json.encode(
                TaskCreate(
                    compute=self._id,
                    function=function,
                    dispatch=dispatch,
                    args_inline=inline,
                    args_sha256=stored,
                    timeout_seconds=int(pending.timeout) if pending.timeout else None,
                ),
            ),
            headers={"Idempotency-Key": uuid.uuid4().hex},
        )

    async def _args(self, args: bytes) -> tuple[bytes | None, str | None]:
        """Small arguments ride along; big ones are uploaded and referenced."""
        if len(args) <= INLINE:
            return args, None

        sha = await codec.digest(args)
        await self.client.upload(f"/v1/blobs/{sha}", args)
        return None, sha

    async def _settled(self, task_id: str) -> bytes:
        """Wait for the outcome, however long the function takes.

        The server holds the request until the task settles and answers 204 when
        the window closes, so this loop is a long poll and not a poll: each turn
        costs one request, not one per second. A failure arrives as an exception
        from the client, which is why nothing here checks for one.
        """
        while (blob := await self.client.blob(f"/v1/tasks/{task_id}/result", wait=POLL)) is None:
            continue
        return blob


async def _next(frames: AsyncIterator[bytes]) -> bytes | None:
    """One frame, or nothing left. The pull the consumer's ``for`` loop turns into."""
    return await anext(frames, None)


def _wire(spec: Spec) -> SpecRef:
    accelerator, count = _accelerator(spec.accelerator)
    return SpecRef(
        provider=ProviderRef(kind=spec.provider.kind, config=dict(spec.provider.config)),
        accelerator=accelerator,
        accelerator_count=count,
        cpus=spec.cpus,
        memory_gb=spec.memory_gb,
        region=spec.region,
        disk_gb=spec.disk_gb,
        architecture=spec.architecture,
        max_hourly_cost=spec.max_hourly_cost,
    )


def _credential(value: Credential | None) -> str | None:
    """The key itself, once ``Storage.resolve`` has run every callable that produced one."""
    match value:
        case str() as key:
            return key
        case _:
            return None


def _accelerator(wanted: str | Accelerator | None) -> tuple[str | None, int]:
    """A raw name goes through the same normalization every offer went through."""
    match wanted:
        case Accelerator(name, count):
            return name, count
        case str(name):
            return resolve(name, None)[0], 1
        case None:
            return None, 1


def _bounds(nodes: NodeSpec) -> NodeBounds:
    match nodes:
        case int(desired):
            return NodeBounds(desired=desired)
        case (minimum, maximum):
            return NodeBounds(desired=maximum, min=minimum, max=maximum)
        case Nodes():
            return nodes
