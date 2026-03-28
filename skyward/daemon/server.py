"""Daemon server -- persistent Session hosting pools over Unix socket.

Architecture:
    Daemon event loop (main thread)     Session event loop (background thread)
    +-- Unix socket server              +-- ActorSystem
    +-- Client handlers                 +-- Pool actors
    +-- TTL timers                      +-- Node actors, SSH, workers

Task dispatch uses pool.run_async() which returns a concurrent.futures.Future,
then asyncio.wrap_future() converts it for the daemon's event loop. This avoids
deadlocking -- the daemon's loop never blocks on the Session's loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import traceback as tb_module
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any

import cloudpickle
from casty import ActorRef, ActorSystem, EventJournal, SqliteJournal

from skyward.observability.logger import logger

from .protocol import (
    BroadcastSucceeded,
    DaemonError,
    DaemonRequest,
    DaemonResponse,
    Disconnect,
    Disconnected,
    EnsurePool,
    GetNodeCount,
    NodeCount,
    Ping,
    Pong,
    PoolFailed,
    PoolReady,
    PoolShutdown,
    ShutdownPool,
    SubmitBroadcast,
    SubmitTask,
    TaskFailed,
    TaskSucceeded,
)
from .wire import async_recv, async_send

log = logger.bind(component="daemon")

_DEFAULT_SOCKET = Path.home() / ".skyward" / "daemon.sock"
_STATE_DIR = Path.home() / ".skyward" / "state"


class DaemonServer:
    """Asyncio Unix socket server hosting a Skyward Session.

    The Session runs in a background thread (standard Skyward pattern).
    The socket server runs in its own event loop. Task dispatch bridges
    the two via run_async() + wrap_future().
    """

    def __init__(
        self,
        socket_path: Path = _DEFAULT_SOCKET,
        journal: EventJournal | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._server: asyncio.Server | None = None
        self._pools: dict[str, Any] = {}  # name -> ComputePool (live references)
        self._ttl_tasks: dict[str, asyncio.Task[None]] = {}  # name -> TTL timer
        self._pool_ttls: dict[str, int] = {}  # name -> TTL seconds
        self._session: Any = None
        self._journal = journal or SqliteJournal(str(_STATE_DIR / "daemon.db"))
        self._state_ref: ActorRef | None = None
        self._actor_system: ActorSystem | None = None

    async def __aenter__(self) -> DaemonServer:
        from .state import daemon_state_actor

        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._socket_path.unlink(missing_ok=True)

        self._actor_system = ActorSystem("daemon")
        await self._actor_system.__aenter__()
        self._state_ref = self._actor_system.spawn(
            daemon_state_actor("daemon-state", self._journal), "daemon-state",
        )

        self._server = await asyncio.start_unix_server(
            self._handle_client, path=str(self._socket_path),
        )
        log.info("Daemon listening on {path}", path=self._socket_path)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        self._socket_path.unlink(missing_ok=True)
        for name in list(self._pools):
            await self._teardown_pool(name)
        if self._actor_system is not None:
            await self._actor_system.__aexit__(None, None, None)
            self._actor_system = None
            self._state_ref = None
        if self._session is not None:
            self._session.__exit__(None, None, None)
            self._session = None
        log.info("Daemon stopped")

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle one client connection."""
        client_id = uuid.uuid4().hex[:12]
        connected_pools: set[str] = set()

        try:
            while True:
                try:
                    request = await async_recv(reader)
                except (asyncio.IncompleteReadError, ConnectionError, EOFError):
                    break

                response = await self._dispatch(request, client_id)  # type: ignore[arg-type]
                await async_send(writer, response)

                match request:
                    case EnsurePool(name=name):
                        connected_pools.add(name)
                    case Disconnect():
                        break
        except Exception as e:
            log.warning("Client {cid} handler error: {err}", cid=client_id, err=e)
        finally:
            for pool_name in connected_pools:
                await self._unregister_client(pool_name, client_id)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _dispatch(self, request: DaemonRequest, client_id: str) -> DaemonResponse:
        """Route a request to the appropriate handler."""
        try:
            match request:
                case Ping():
                    return Pong()

                case EnsurePool(name=name, project_dir=project_dir):
                    return await self._ensure_pool(name, project_dir, client_id)

                case SubmitTask(pool_name=name, payload=payload, timeout=timeout):
                    return await self._submit_task(name, payload, timeout)

                case SubmitBroadcast(pool_name=name, payload=payload, timeout=timeout):
                    return await self._submit_broadcast(name, payload, timeout)

                case GetNodeCount(pool_name=name):
                    return self._get_node_count(name)

                case Disconnect(pool_name=name):
                    await self._unregister_client(name, client_id)
                    return Disconnected()

                case ShutdownPool(pool_name=name):
                    return await self._shutdown_pool(name)

                case _:
                    return DaemonError(error=f"Unknown request: {type(request).__name__}")

        except Exception as e:
            return DaemonError(error=str(e), traceback=tb_module.format_exc())

    # -- Pool provisioning -------------------------------------------------

    async def _ensure_pool(
        self, name: str, project_dir: str | None, client_id: str,
    ) -> PoolReady | PoolFailed:
        assert self._actor_system is not None and self._state_ref is not None
        if name in self._pools:
            pool = self._pools[name]
            await self._register_client(name, client_id)
            return PoolReady(pool_name=name, node_count=pool.current_nodes())

        try:
            pool = self._provision_pool(name, project_dir)
            self._pools[name] = pool

            from .state import RegisterPool
            await self._actor_system.ask(
                self._state_ref,
                lambda r: RegisterPool(
                    pool_name=name,
                    cluster_id=pool._cluster_id,
                    instance_ids=tuple(pool._instance_ids),
                    project_dir=project_dir or str(Path.cwd()),
                    reply_to=r,
                ),
                timeout=5.0,
            )
            await self._register_client(name, client_id)
            return PoolReady(pool_name=name, node_count=pool.current_nodes())
        except Exception as e:
            return PoolFailed(pool_name=name, reason=str(e))

    def _provision_pool(self, name: str, project_dir: str | None) -> Any:
        """Resolve config and provision a ComputePool via Session."""
        from skyward.config import resolve_pool_config

        resolution = resolve_pool_config(
            name, project_dir=Path(project_dir) if project_dir else None,
        )
        pool = resolution.pool

        if self._session is None:
            from skyward.core.session import Session
            self._session = Session(console=False, logging=True)
            self._session.__enter__()

        pool.__enter__()
        self._pool_ttls[name] = pool._specs[0].ttl
        return pool

    # -- Task dispatch (async, no deadlock) --------------------------------

    async def _submit_task(
        self, name: str, payload: bytes, timeout: float,
    ) -> TaskSucceeded | TaskFailed:
        if name not in self._pools:
            return TaskFailed(error=f"Pool '{name}' not found", traceback="")

        pool = self._pools[name]
        pending = cloudpickle.loads(payload)

        try:
            future = pool.run_async(pending)
            result = await asyncio.wrap_future(future)
            return TaskSucceeded(payload=cloudpickle.dumps(result))
        except Exception as e:
            return TaskFailed(error=str(e), traceback=tb_module.format_exc())

    async def _submit_broadcast(
        self, name: str, payload: bytes, timeout: float,
    ) -> BroadcastSucceeded | TaskFailed:
        if name not in self._pools:
            return TaskFailed(error=f"Pool '{name}' not found", traceback="")

        pool = self._pools[name]
        pending = cloudpickle.loads(payload)

        try:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, pool.broadcast, pending)
            return BroadcastSucceeded(payload=cloudpickle.dumps(results))
        except Exception as e:
            return TaskFailed(error=str(e), traceback=tb_module.format_exc())

    def _get_node_count(self, name: str) -> NodeCount | DaemonError:
        if name not in self._pools:
            return DaemonError(error=f"Pool '{name}' not found")
        return NodeCount(ready=self._pools[name].current_nodes())

    async def _shutdown_pool(self, name: str) -> PoolShutdown | DaemonError:
        if name not in self._pools:
            return DaemonError(error=f"Pool '{name}' not found")
        await self._teardown_pool(name)
        return PoolShutdown()

    # -- Client tracking (via state actor) ---------------------------------

    async def _register_client(self, pool_name: str, client_id: str) -> None:
        assert self._actor_system is not None and self._state_ref is not None
        from .state import AddClient
        await self._actor_system.ask(
            self._state_ref,
            lambda r: AddClient(pool_name=pool_name, client_id=client_id, reply_to=r),
            timeout=5.0,
        )
        self._cancel_ttl(pool_name)
        log.info("Client {cid} joined pool {pool}", cid=client_id, pool=pool_name)

    async def _unregister_client(self, pool_name: str, client_id: str) -> None:
        assert self._actor_system is not None and self._state_ref is not None
        from .state import GetState, RemoveClient
        await self._actor_system.ask(
            self._state_ref,
            lambda r: RemoveClient(pool_name=pool_name, client_id=client_id, reply_to=r),
            timeout=5.0,
        )
        log.info("Client {cid} left pool {pool}", cid=client_id, pool=pool_name)

        state = await self._actor_system.ask(
            self._state_ref, lambda r: GetState(reply_to=r), timeout=5.0,
        )
        if pool_name in state.pools and not state.pools[pool_name].clients:
            self._start_ttl(pool_name)

    # -- TTL management (idle-based) ---------------------------------------

    def _start_ttl(self, name: str) -> None:
        if name not in self._pools:
            return
        ttl = self._pool_ttls.get(name, 1200)
        self._cancel_ttl(name)
        self._ttl_tasks[name] = asyncio.create_task(self._ttl_expire(name, ttl))
        log.info("Pool {name} idle TTL started: {ttl}s", name=name, ttl=ttl)

    def _cancel_ttl(self, name: str) -> None:
        if task := self._ttl_tasks.pop(name, None):
            task.cancel()
            log.info("Pool {name} TTL cancelled (client reconnected)", name=name)

    async def _ttl_expire(self, name: str, ttl: int) -> None:
        await asyncio.sleep(ttl)
        log.info("Pool {name} TTL expired, tearing down", name=name)
        await self._teardown_pool(name)

    async def _teardown_pool(self, name: str) -> None:
        self._cancel_ttl(name)
        if pool := self._pools.pop(name, None):
            try:
                pool.__exit__(None, None, None)
            except Exception as e:
                log.warning("Error tearing down pool {name}: {err}", name=name, err=e)
        self._pool_ttls.pop(name, None)

    async def serve_forever(self) -> None:
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()
