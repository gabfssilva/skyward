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
from collections.abc import Coroutine
from concurrent.futures import Future
from pathlib import Path
from typing import Self

import msgspec

from skyward2.persistence.db import DEFAULT_PATH
from skyward2.protocol import codec
from skyward2.protocol.schemas import (
    Compute as ComputeView,
)
from skyward2.protocol.schemas import (
    ComputeCreate,
    ComputeSpec,
    Dispatch,
    Image,
    NodeBounds,
    ProviderCreate,
    ProviderRef,
    Spec,
    Task,
    TaskCreate,
    Worker,
)
from skyward2.sdk.client import Client
from skyward2.sdk.errors import SkywardError
from skyward2.sdk.function import Group, Pending
from skyward2.sdk.provider import Provider

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
        provider: Provider,
        nodes: int | tuple[int, int] = 1,
        accelerator: str | None = None,
        accelerator_count: int = 1,
        cpus: int | None = None,
        memory_gb: int | None = None,
        region: str | None = None,
        image: Image = DEFAULT_IMAGE,
        concurrency: int | None = None,
        name: str | None = None,
        url: str | None = None,
        database: Path = DEFAULT_PATH,
        delete_on_exit: bool = True,
    ) -> None:
        self._provider = provider
        self._spec = ComputeSpec(
            specs=(
                Spec(
                    provider=ProviderRef(kind=provider.kind, config=dict(provider.config)),
                    accelerator=accelerator,
                    accelerator_count=accelerator_count,
                    cpus=cpus,
                    memory_gb=memory_gb,
                    region=region,
                ),
            ),
            nodes=_bounds(nodes),
            image=image,
            worker=Worker(concurrency=concurrency),
        )
        self._name = name
        self._url = url or os.environ.get("SKYWARD_URL")
        self._database = database
        self._delete_on_exit = delete_on_exit

        self._loop: Loop | None = None
        self._client: Client | None = None
        self._id = ""

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
        self.loop.run(self.client.close())
        self.loop.close()
        self._loop, self._client = None, None

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

    async def _provision(self) -> None:
        await self._ensure_provider()

        compute = await self.client.call(
            "POST",
            "/v1/computes",
            ComputeView,
            body=msgspec.json.encode(ComputeCreate(spec=self._spec, name=self._name)),
            headers={"Idempotency-Key": uuid.uuid4().hex},
        )
        self._id = compute.id

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

    async def _ensure_provider(self) -> None:
        """The credentials live in the daemon, and this is how they get there.

        A provider is a row, not a value in the spec: the spec names a kind, and
        the row holds what it takes to log in. Registering it here is what makes
        ``Compute(provider=Container())`` enough on a store that has never seen one.
        """
        try:
            await self.client.call("GET", f"/v1/providers/{self._provider.name}", dict[str, object])
        except SkywardError as error:
            if error.code != "not_found":
                raise
            await self.client.call(
                "POST",
                "/v1/providers",
                dict[str, object],
                body=msgspec.json.encode(
                    ProviderCreate(
                        name=self._provider.name,
                        kind=self._provider.kind,
                        credentials=dict(self._provider.credentials),
                        config=dict(self._provider.config),
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

    async def _submit[T](self, pending: Pending[T], dispatch: Dispatch) -> Task:
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


def _bounds(nodes: int | tuple[int, int]) -> NodeBounds:
    match nodes:
        case int(desired):
            return NodeBounds(desired=desired)
        case (minimum, maximum):
            return NodeBounds(desired=maximum, min=minimum, max=maximum)
