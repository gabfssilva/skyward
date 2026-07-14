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
from concurrent.futures import Future
from pathlib import Path
from typing import Self

import msgspec

from skyward2.accelerators import Accelerator
from skyward2.persistence.db import DEFAULT_PATH
from skyward2.plugins import Plugin
from skyward2.protocol import codec
from skyward2.protocol.accelerators import resolve
from skyward2.protocol.frames import Chunk, Failed, Frame
from skyward2.protocol.schemas import (
    Allocation,
    ComputeCreate,
    ComputeSpec,
    Dispatch,
    Error,
    Image,
    NodeBounds,
    ProviderCreate,
    ProviderRef,
    Selection,
    Task,
    TaskCreate,
    Worker,
)
from skyward2.protocol.schemas import (
    Compute as ComputeView,
)
from skyward2.protocol.schemas import (
    Spec as SpecRef,
)
from skyward2.sdk.client import Client
from skyward2.sdk.console import Console
from skyward2.sdk.errors import SkywardError, TaskFailedError
from skyward2.sdk.function import Group, Pending, Streaming
from skyward2.sdk.provider import Provider
from skyward2.sdk.spec import Nodes, NodeSpec, Spec

DEFAULT_IMAGE = Image()
INLINE = 256 * 1024
POLL = 30
READY_TIMEOUT = 900
DELETE_TIMEOUT = 300


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
        concurrency: int | None = None,
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
            worker=Worker(concurrency=concurrency),
            plugins=tuple(plugin.ref() for plugin in plugins),
        )
        self._name = name
        self._url = url or os.environ.get("SKYWARD_URL")
        self._database = database
        self._delete_on_exit = delete_on_exit
        self._console = console

        self._loop: Loop | None = None
        self._client: Client | None = None
        self._watching: Future[None] | None = None
        self._id = ""

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
        return self

    def __exit__(self, *_: object) -> None:
        if self._delete_on_exit:
            self.loop.run(self._destroy())
        if self._watching:
            self._watching.cancel()
        self.loop.run(self.client.close())
        self.loop.close()
        self._loop, self._client, self._watching = None, None, None

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
            compute = await self.client.call(
                "POST",
                "/v1/computes",
                ComputeView,
                body=msgspec.json.encode(ComputeCreate(spec=self._spec, name=self._name)),
                headers={"Idempotency-Key": uuid.uuid4().hex},
            )
            self._id = compute.id

        if self._console:
            self._watching = self.loop.start(Console(self.client, self._id).follow())

        async with asyncio.timeout(READY_TIMEOUT):
            while True:
                current = await self.client.call("GET", f"/v1/computes/{self._id}", ComputeView)
                match current.status.state:
                    case "ready":
                        return
                    case "failed" | "degraded":
                        raise RuntimeError(f"compute {self._id} is {current.status.state}: {current.status.last_error}")
                    case _:
                        await asyncio.sleep(0.5)

    async def _ensure_providers(self) -> None:
        """The credentials live in the daemon, and this is how they get there.

        A provider is a row, not a value in the spec: the spec names a kind, and
        the row holds what it takes to log in. Registering it here is what makes
        ``Compute(provider=Container())`` enough on a store that has never seen one.
        """
        for provider in self._providers.values():
            try:
                await self.client.call("GET", f"/v1/providers/{provider.name}", dict[str, object])
            except SkywardError as error:
                if error.code != "not_found":
                    raise
                await self.client.call(
                    "POST",
                    "/v1/providers",
                    dict[str, object],
                    body=msgspec.json.encode(
                        ProviderCreate(
                            name=provider.name,
                            kind=provider.kind,
                            credentials=dict(provider.credentials),
                            config=dict(provider.config),
                        ),
                    ),
                )

    async def _destroy(self) -> None:
        current = await self.client.call("GET", f"/v1/computes/{self._id}", ComputeView)
        await self.client.call(
            "DELETE",
            f"/v1/computes/{self._id}",
            ComputeView,
            headers={"If-Match": f'"{current.revision}"', "Idempotency-Key": uuid.uuid4().hex},
        )

        async with asyncio.timeout(DELETE_TIMEOUT):
            while current.status.state != "deleted":
                await asyncio.sleep(0.5)
                current = await self.client.call("GET", f"/v1/computes/{self._id}", ComputeView)

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
    )


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
