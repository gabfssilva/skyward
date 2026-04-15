"""HTTP-facing facade that owns a :class:`Session` and its :class:`Store`.

``PoolHost`` is the composition root for the HTTP server: it opens the
Store, hands it to a :class:`Session`, tracks live SSE subscribers, and
arbitrates every call from HTTP routes to the underlying pool actors.
Restart recovery (Phase I) and task submission (Phase G3) layer on top
of this scaffolding.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import AsyncExitStack
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Any

from skyward.server.host.domain import Queued as _Queued

if TYPE_CHECKING:
    from skyward.core.session import Session
    from skyward.server.host.blobs import Blobs
    from skyward.server.host.store import Store

type EventQueue = asyncio.Queue[Any]


class PoolHost:
    """Owns a :class:`Session` + :class:`Store` pair for the HTTP server.

    Parameters
    ----------
    store
        Pre-opened persistence store. ``PoolHost`` assumes ownership
        across its context window and closes the connection on exit.
    blobs
        Blob service used by :meth:`submit_task` (Phase G3). Passed in
        rather than constructed so tests can swap the implementation.
    logs_dir
        Root directory for per-pool log files.
    """

    def __init__(
        self,
        store: Store,
        blobs: Blobs,
        logs_dir: Path,
    ) -> None:
        self._store: Store = store
        self._blobs: Blobs = blobs
        self._logs_dir: Path = logs_dir
        self._session: Session | None = None
        self._stack: AsyncExitStack = AsyncExitStack()
        self._subscribers: dict[str, set[EventQueue]] = defaultdict(set)
        self._session_event_subs: dict[str, set[asyncio.Queue[bytes]]] = (
            defaultdict(set)
        )
        self._main_loop: asyncio.AbstractEventLoop | None = None

    @property
    def store(self) -> Store:
        """Direct access to the persistence layer (read-only routes)."""
        return self._store

    @property
    def session(self) -> Session:
        """The running :class:`Session`; raises if ``__aenter__`` was skipped."""
        if self._session is None:
            raise RuntimeError("PoolHost is not open")
        return self._session

    @property
    def subscribers(self) -> MappingProxyType[str, frozenset[EventQueue]]:
        """Snapshot of live SSE subscriber queues keyed by aggregate."""
        return MappingProxyType(
            {k: frozenset(v) for k, v in self._subscribers.items()},
        )

    async def __aenter__(self) -> PoolHost:
        from skyward.api.projection import SessionProjection
        from skyward.core.session import Session

        self._main_loop = asyncio.get_running_loop()
        projection = SessionProjection()
        projection.subscribe(on_event=self._on_session_event)
        session = Session(
            console=False, logging=True, store=self._store,
            projection=projection,
        )
        # Session.__enter__ is synchronous (drives its own event loop).
        # Offload the blocking enter to a thread so the caller's loop
        # keeps running.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, session.__enter__)
        self._session = session
        self._stack.push_async_callback(self._stop_session)
        await self._run_recovery()
        return self

    def _on_session_event(self, event: Any) -> None:
        """Fan a domain event out to SSE subscribers for its pool.

        Runs on the Session's internal loop thread (the projection is
        invoked from the projection actor). We cross-post pickled bytes
        into per-pool queues living on the server's main loop via
        ``call_soon_threadsafe`` so the SSE route can stream them.
        """
        import cloudpickle

        pool_name = getattr(event, "pool_name", None)
        if not isinstance(pool_name, str):
            return
        loop = self._main_loop
        if loop is None or loop.is_closed():
            return
        data = cloudpickle.dumps(event)
        subs = self._session_event_subs.get(pool_name)
        if not subs:
            return
        for queue in tuple(subs):
            loop.call_soon_threadsafe(queue.put_nowait, data)

    def subscribe_session_events(self, name: str) -> asyncio.Queue[bytes]:
        """Register a new pickled-event queue for ``compute:{name}``."""
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._session_event_subs[name].add(queue)
        return queue

    def unsubscribe_session_events(
        self, name: str, queue: asyncio.Queue[bytes],
    ) -> None:
        """Remove a previously registered queue; safe if already gone."""
        subs = self._session_event_subs.get(name)
        if subs is None:
            return
        subs.discard(queue)
        if not subs:
            self._session_event_subs.pop(name, None)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            await self._stack.aclose()
        finally:
            await self._store.close()

    def subscribe(self, aggregate: str) -> EventQueue:
        """Register a new SSE queue for ``aggregate`` and return it."""
        queue: EventQueue = asyncio.Queue()
        self._subscribers[aggregate].add(queue)
        return queue

    def unsubscribe(self, aggregate: str, queue: EventQueue) -> None:
        """Remove a previously registered queue; safe if already gone."""
        subs = self._subscribers.get(aggregate)
        if subs is None:
            return
        subs.discard(queue)
        if not subs:
            self._subscribers.pop(aggregate, None)

    async def ensure_compute(
        self, name: str, compute_spec: Any, client_id: str | None = None,
    ) -> Any:
        """Provision a pool for ``name`` via the hosted :class:`Session`.

        Phase G2 delivers the blocking entry point — SSE fan-out of the
        in-flight phase events lands in G5. ``compute_spec`` is the
        authoritative :class:`~skyward.server.host.domain.ComputeSpec`
        sent by the HTTP client; ``client_id`` is the optional X-Client-Id
        header that the events table tags lineage with.
        """
        from skyward.api.spec import Options
        from skyward.core.session import _build_compute_spec as _rebuild

        _ = client_id, _rebuild  # reserved for SSE + lineage in G5.
        specs = list(compute_spec.specs) or []
        if not specs:
            raise ValueError(
                "ComputeSpec.specs must be non-empty to provision a pool",
            )
        loop = asyncio.get_running_loop()
        session = self.session

        def _call() -> Any:
            return session.compute(
                *specs,
                name=name,
                options=Options(selection=compute_spec.selection),
            )

        return await loop.run_in_executor(None, _call)

    async def submit_task(
        self,
        name: str,
        task_key: tuple[str, str],
        payload: bytes,
        timeout: float,
        client_id: str | None = None,
    ) -> tuple[str, bytes]:
        """Dispatch an opaque pickled payload to the named pool.

        The bytes flow straight through the actor tree — the server
        process never deserializes user code (design §5.4). Persists
        the execution row, asks the pool actor, and returns the
        worker's pickled return value verbatim.
        """
        from datetime import UTC, datetime, timedelta

        from skyward.actors.messages import (
            SubmitTask,
            TaskFailed,
            TaskInterrupted,
            TaskSucceeded,
        )
        from skyward.infra.ids import uuid7
        from skyward.server.host.domain import (
            Queued,
            Run,
            Task,
            TaskExecution,
        )

        session = self.session
        pool = session._pools.get(name)
        if pool is None:
            raise LookupError(f"compute {name!r} is not active")

        execution_id = str(uuid7())
        blob_id = await self._blobs.put(payload, kind="payload")
        await self._store.put_task(Task(module=task_key[0], qualname=task_key[1]))

        await self._store.put_execution(
            TaskExecution(
                id=execution_id,
                task=task_key,
                compute=name,
                kind=Run(),
                payload=blob_id,
                timeout=timedelta(seconds=timeout),
                client=client_id,
                submitted_at=datetime.now(UTC),
                status=Queued(),
            ),
        )

        system = pool._system
        pool_ref = pool._pool_ref
        if system is None or pool_ref is None:
            raise RuntimeError(f"pool {name!r} has no active actor ref")

        def _factory(reply_to: Any) -> SubmitTask:
            return SubmitTask(
                payload=payload, reply_to=reply_to, task_id=execution_id,
                timeout=timeout, task_key=task_key,
            )

        result = await system.ask(pool_ref, _factory, timeout=timeout + 5)

        match result:
            case TaskSucceeded(value=value):
                return execution_id, value
            case TaskFailed(error=error) | TaskInterrupted(error=error):
                raise error
            case _:
                raise RuntimeError(f"unexpected task result: {result!r}")

    async def broadcast(
        self,
        name: str,
        task_key: tuple[str, str],
        payload: bytes,
        timeout: float,
        client_id: str | None = None,
    ) -> tuple[str, list[bytes]]:
        """Broadcast the opaque payload to every ready node.

        Returns the execution id and one pickled reply per node, in
        node-id order. The HTTP route packs the replies into a
        length-prefixed wire frame (see ``routes/executions.py``).
        """
        from datetime import UTC, datetime, timedelta

        from skyward.actors.messages import SubmitBroadcast
        from skyward.infra.ids import uuid7
        from skyward.server.host.domain import (
            Broadcast,
            Queued,
            Task,
            TaskExecution,
        )

        session = self.session
        pool = session._pools.get(name)
        if pool is None:
            raise LookupError(f"compute {name!r} is not active")

        execution_id = str(uuid7())
        blob_id = await self._blobs.put(payload, kind="payload")
        await self._store.put_task(Task(module=task_key[0], qualname=task_key[1]))

        await self._store.put_execution(
            TaskExecution(
                id=execution_id,
                task=task_key,
                compute=name,
                kind=Broadcast(),
                payload=blob_id,
                timeout=timedelta(seconds=timeout),
                client=client_id,
                submitted_at=datetime.now(UTC),
                status=Queued(),
            ),
        )

        system = pool._system
        pool_ref = pool._pool_ref
        if system is None or pool_ref is None:
            raise RuntimeError(f"pool {name!r} has no active actor ref")

        def _factory(reply_to: Any) -> SubmitBroadcast:
            return SubmitBroadcast(
                payload=payload, reply_to=reply_to, task_id=execution_id,
                timeout=timeout, task_key=task_key,
            )

        results = await system.ask(pool_ref, _factory, timeout=timeout + 30)

        for r in results:
            if isinstance(r, BaseException):
                raise r

        return execution_id, list(results)

    async def shutdown_compute(self, name: str) -> None:
        """Stop an active pool managed by the underlying session."""
        loop = asyncio.get_running_loop()
        session = self.session

        def _call() -> None:
            pool = session._pools.get(name)
            if pool is None:
                return
            pool.__exit__(None, None, None)
            session._pools.pop(name, None)

        await loop.run_in_executor(None, _call)

    def subscribe_events(self, name: str, since: int = 0) -> Any:
        """Yield events scoped to ``compute:name`` (and child aggregates).

        Replay comes from the Store's reader connection; future
        iterations will merge live actor-emitted events via the
        :attr:`subscribers` fan-out. The returned async iterator yields
        :class:`~skyward.server.host.store.EventRow` objects.
        """
        import re

        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_.\-]{0,63}$", name):
            raise ValueError(f"invalid compute name: {name!r}")

        aggregate = f"compute:{name}"

        async def _iter() -> Any:
            async for ev in self._store.tail_events(
                since_id=since, aggregate_like=aggregate,
            ):
                if ev.aggregate == aggregate or ev.aggregate.startswith(
                    aggregate + "/",
                ):
                    yield ev

        return _iter()

    async def _run_recovery(self) -> None:
        """Restart recovery — Phases I1–I4.

        Design §9.3: in-flight work that survived a server crash cannot
        be reconciled with client futures resolved before the crash, so
        we mark it ``Interrupted`` and move on (I1). Terminal-bound
        computes (``Provisioning``/``Stopping``) cannot be resumed and
        are closed (I2). ``Ready`` pools are rehydrated when possible
        via :meth:`Session.recover_pool`; any failure marks the compute
        ``Failed`` so queued executions don't hang forever (I2 ext).
        """
        await self._interrupt_in_flight()
        await self._close_transitional_computes()
        await self._recover_ready_computes()
        await self._fail_queued_without_live_pool()
        await self._blobs.gc_orphans()

    async def _recover_ready_computes(self) -> None:
        """Best-effort rehydration of ``Ready`` computes after restart.

        For each ``Ready`` compute we try to reconstruct the provider,
        verify its instances, and hand everything to
        :meth:`Session.recover_pool`. Any error path — including
        missing SDKs, provider timeouts, or zero surviving instances —
        transitions the compute to ``Failed(reason="recovery_failed: ...")``
        so the downstream :meth:`_fail_queued_without_live_pool` sweep
        can fail queued executions with a clear cause.
        """
        from dataclasses import replace as _replace
        from datetime import UTC, datetime

        from skyward.server.host.domain import Failed, Ready

        now = datetime.now(UTC)
        for row in await self._store.list_compute():
            match row.status:
                case Ready():
                    pass
                case _:
                    continue

            try:
                await self._try_recover_one(row)
            except BaseException as exc:  # noqa: BLE001
                reason = f"recovery_failed: {exc.__class__.__name__}: {exc}"[:256]
                await self._store.put_compute(
                    _replace(
                        row,
                        status=Failed(failed_at=now, reason=reason),
                    ),
                )

    async def _try_recover_one(self, row: Any) -> None:
        """Attempt full rehydration for one compute row.

        Raises on any failure so :meth:`_recover_ready_computes` can
        convert it to ``Failed``. Deliberately keeps the surgery narrow:
        reconstruction of live :class:`Cluster` state across arbitrary
        providers is provider-specific and belongs in a follow-up.
        """
        from skyward.server.host.domain import NodeReady

        assert self._session is not None
        assert isinstance(row.status, type(row.status))

        nodes = await self._store.list_nodes(compute=row.name)
        ready_nodes = [n for n in nodes if isinstance(n.status, NodeReady)]
        if not ready_nodes:
            raise RuntimeError("no surviving NodeReady rows")

        raise NotImplementedError(
            "per-provider cluster reconstruction not yet implemented; "
            "falling through to Failed so queued executions unblock",
        )

    async def _close_transitional_computes(self) -> None:
        """Close computes stuck in ``Provisioning`` / ``Stopping`` at boot."""
        from dataclasses import replace as _replace
        from datetime import UTC, datetime

        from skyward.server.host.domain import (
            Failed,
            Provisioning,
            Stopped,
            Stopping,
        )

        now = datetime.now(UTC)
        for row in await self._store.list_compute():
            match row.status:
                case Provisioning():
                    await self._store.put_compute(
                        _replace(
                            row,
                            status=Failed(
                                failed_at=now, reason="server_restart",
                            ),
                        ),
                    )
                case Stopping(started_at=started_at):
                    await self._store.put_compute(
                        _replace(
                            row,
                            status=Stopped(
                                started_at=started_at, stopped_at=now,
                            ),
                        ),
                    )
                case _:
                    continue

    async def _fail_queued_without_live_pool(self) -> None:
        """Mark ``Queued`` executions whose compute won't be resumed.

        Only computes that actually re-entered ``session._pools`` after
        :meth:`_recover_ready_computes` are skipped — a ``Ready`` row
        without a live actor means recovery bailed and the execution
        must surface ``compute_failed_on_restart`` instead of hanging.
        """
        from datetime import UTC, datetime

        from skyward.server.host.domain import InterruptedExec, Ready

        now = datetime.now(UTC)
        live_pools = (
            set(self._session._pools.keys()) if self._session is not None else set()
        )
        computes = {c.name: c for c in await self._store.list_compute()}
        for execution in await self._store.list_executions(status="queued"):
            compute = computes.get(execution.compute)
            if compute is None:
                continue
            match compute.status:
                case Ready() if compute.name in live_pools:
                    continue
                case _:
                    await self._store.update_execution_status(
                        execution.id,
                        InterruptedExec(
                            interrupted_at=now,
                            reason="compute_failed_on_restart",
                        ),
                    )

    async def _interrupt_in_flight(self) -> None:
        from datetime import UTC, datetime

        from skyward.server.host.domain import InterruptedExec

        now = datetime.now(UTC)
        active = await self._store.list_active_executions()
        for execution in active:
            match execution.status:
                case _Queued():
                    continue
                case _:
                    await self._store.update_execution_status(
                        execution.id,
                        InterruptedExec(
                            interrupted_at=now, reason="server_restart",
                        ),
                    )

    async def _stop_session(self) -> None:
        session = self._session
        if session is None:
            return
        loop = asyncio.get_running_loop()
        self._session = None
        await loop.run_in_executor(
            None, lambda: session.__exit__(None, None, None),
        )


__all__ = ["PoolHost"]
