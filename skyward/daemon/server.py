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
from collections.abc import AsyncGenerator
from pathlib import Path
from types import TracebackType
from typing import Any

import cloudpickle
from casty import ActorRef, ActorSystem, EventJournal, SqliteJournal

from skyward.api.events import SessionEvent
from skyward.api.views import SessionView
from skyward.observability.logger import logger

from .protocol import (
    BroadcastSucceeded,
    DaemonError,
    DaemonRequest,
    DaemonResponse,
    DaemonStopped,
    Disconnect,
    Disconnected,
    EnsurePool,
    GetNodeCount,
    GetPoolLogs,
    GetPools,
    GetPoolView,
    NodeCount,
    Ping,
    Pong,
    PoolFailed,
    PoolLogLine,
    PoolLogs,
    PoolProvisioning,
    PoolReady,
    PoolShutdown,
    ShutdownDaemon,
    ShutdownPool,
    StreamEnd,
    SubmitBroadcast,
    SubmitTask,
    SubscribeEvents,
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
        self._subscribers: dict[str, list[asyncio.Queue[object]]] = {}
        self._log_handles: dict[str, Any] = {}
        self._logs_dir = Path.home() / ".skyward" / "logs"

        from skyward.api.projection import SessionProjection

        def _on_view_changed(old: SessionView, new: SessionView) -> None:
            for name, queues in self._subscribers.items():
                if name in new.pools:
                    for q in queues:
                        q.put_nowait(new)

        def _on_event(event: SessionEvent) -> None:
            pool_name = event.pool_name
            if pool_name in self._subscribers:
                for q in self._subscribers[pool_name]:
                    q.put_nowait(event)
            self._append_log(event)

        self._projection = SessionProjection(
            on_change=_on_view_changed,
            on_event=_on_event,
        )
        if journal is None:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)
            journal = SqliteJournal(str(_STATE_DIR / "daemon.db"))
        self._journal = journal
        self._state_ref: ActorRef | None = None
        self._actor_system: ActorSystem | None = None
        self._stop: asyncio.Event | None = None

    async def __aenter__(self) -> DaemonServer:
        from .state import daemon_state_actor

        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._socket_path.unlink(missing_ok=True)

        self._actor_system = ActorSystem("daemon")
        await self._actor_system.__aenter__()
        self._state_ref = self._actor_system.spawn(
            daemon_state_actor("daemon-state", self._journal), "daemon-state",
        )

        await self._run_recovery()

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
        for fh in self._log_handles.values():
            fh.close()
        self._log_handles.clear()
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

                match request:
                    case SubscribeEvents(pool_name=name):
                        await self._handle_subscribe(name, reader, writer)
                        break
                    case EnsurePool(name=name, spec_bytes=spec_bytes):
                        async for msg in self._ensure_pool_stream(name, spec_bytes, client_id):
                            await async_send(writer, msg)
                        connected_pools.add(name)
                    case _:
                        response = await self._dispatch(request, client_id)  # type: ignore[arg-type]
                        await async_send(writer, response)

                match request:
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

    async def _handle_subscribe(
        self,
        pool_name: str,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Stream SessionView snapshots and domain events to client until disconnect."""
        view = self._projection.view
        if pool_name not in view.pools:
            await async_send(writer, DaemonError(error=f"Pool '{pool_name}' not found"))
            return

        await async_send(writer, view)

        queue: asyncio.Queue[object] = asyncio.Queue(maxsize=256)
        if pool_name not in self._subscribers:
            self._subscribers[pool_name] = []
        self._subscribers[pool_name].append(queue)

        async def _watch_disconnect() -> None:
            """Detect client disconnect by waiting for EOF on reader."""
            with contextlib.suppress(ConnectionError, asyncio.IncompleteReadError):
                await reader.read(1)

        disconnect_task = asyncio.create_task(_watch_disconnect())

        try:
            while True:
                get_task = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {get_task, disconnect_task},
                    timeout=30.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if disconnect_task in done:
                    get_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await get_task
                    break

                if not done:
                    get_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await get_task
                    current = self._projection.view
                    if pool_name in current.pools:
                        try:
                            await async_send(writer, current)
                        except (ConnectionError, BrokenPipeError, ConnectionResetError):
                            break
                    else:
                        with contextlib.suppress(ConnectionError, BrokenPipeError, ConnectionResetError):
                            await async_send(writer, StreamEnd(reason="pool removed"))
                        break
                    continue

                try:
                    msg = get_task.result()
                    await async_send(writer, msg)
                except (ConnectionError, BrokenPipeError, ConnectionResetError):
                    break
        finally:
            disconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await disconnect_task
            self._subscribers[pool_name].remove(queue)
            if not self._subscribers[pool_name]:
                del self._subscribers[pool_name]

    async def _dispatch(self, request: DaemonRequest, client_id: str) -> DaemonResponse:
        """Route a request to the appropriate handler."""
        try:
            match request:
                case Ping():
                    return Pong()

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

                case GetPools():
                    return self._get_pools()

                case GetPoolView(pool_name=name):
                    return self._get_pool_view(name)

                case GetPoolLogs(pool_name=name, all=all_logs):
                    return self._get_pool_logs(name, all_logs)

                case ShutdownDaemon():
                    return await self._shutdown_daemon()

                case _:
                    return DaemonError(error=f"Unknown request: {type(request).__name__}")

        except Exception as e:
            return DaemonError(error=str(e), traceback=tb_module.format_exc())

    # -- Log persistence ---------------------------------------------------

    def _append_log(self, event: SessionEvent) -> None:
        from skyward.api.events import Log

        match event:
            case Log.Emitted(pool_name=name, node_id=nid, message=msg, overwrite=ow):
                fh = self._log_handles.get(name)
                if fh is not None:
                    import json

                    fh.write(json.dumps({
                        "node": nid, "message": msg, "overwrite": ow,
                    }) + "\n")
                    fh.flush()
            case _:
                pass

    def _open_log_file(self, name: str) -> None:
        from datetime import UTC, datetime

        pool_dir = self._logs_dir / name
        pool_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        log_path = pool_dir / f"{ts}.log"
        self._log_handles[name] = log_path.open("a")

    # -- Pool provisioning -------------------------------------------------

    async def _ensure_pool_stream(
        self, name: str, spec_bytes: bytes, client_id: str,
    ) -> AsyncGenerator[PoolReady | PoolFailed | PoolProvisioning | PoolLogLine, None]:
        """Provision a pool, streaming phase transitions to the client."""
        assert self._actor_system is not None and self._state_ref is not None

        if name in self._pools:
            pool = self._pools[name]
            await self._register_client(name, client_id)
            yield PoolReady(pool_name=name, node_count=pool.current_nodes())
            return

        event_queue: asyncio.Queue[PoolProvisioning | PoolLogLine] = asyncio.Queue()
        last_phase: str | None = None

        original_on_change = self._projection.on_change
        original_on_event = self._projection.on_event

        def _on_change(old: SessionView, new: SessionView) -> None:
            nonlocal last_phase
            if original_on_change:
                original_on_change(old, new)
            if name in new.pools:
                phase = new.pools[name].phase.name
                if phase != last_phase and phase not in ("READY", "STOPPED"):
                    last_phase = phase
                    event_queue.put_nowait(PoolProvisioning(pool_name=name, phase=phase))

        def _on_event(event: SessionEvent) -> None:
            if original_on_event:
                original_on_event(event)
            from skyward.api.events import Log, Node

            match event:
                case Log.Emitted(pool_name=pn, node_id=nid, message=msg) if pn == name:
                    event_queue.put_nowait(PoolLogLine(pool_name=name, node_id=nid, message=msg))
                case Node.Bootstrap.Output(pool_name=pn, node_id=nid, output=output) if pn == name:
                    event_queue.put_nowait(PoolLogLine(pool_name=name, node_id=nid, message=output))
                case _:
                    pass

        self._projection.on_change = _on_change
        self._projection.on_event = _on_event

        loop = asyncio.get_running_loop()
        provision_future: asyncio.Future[Any] = loop.run_in_executor(
            None, self._provision_pool, name, spec_bytes,
        )

        try:
            while not provision_future.done():
                try:
                    msg = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                    yield msg
                except TimeoutError:
                    continue

            pool = provision_future.result()
            self._pools[name] = pool
            self._open_log_file(name)

            while not event_queue.empty():
                yield event_queue.get_nowait()

            from .state import RegisterPool

            instance_ids = tuple(
                ni.instance.id for ni in pool._instances.values()
            )
            cluster_bytes = await asyncio.to_thread(cloudpickle.dumps, pool._cluster)
            spec_bytes_out = await asyncio.to_thread(cloudpickle.dumps, pool._spec)
            provider_bytes = await asyncio.to_thread(cloudpickle.dumps, pool._specs[0].provider)
            await self._actor_system.ask(
                self._state_ref,
                lambda r: RegisterPool(
                    pool_name=name,
                    cluster_id=pool._cluster_id,
                    instance_ids=instance_ids,
                    provider_name=pool._spec.provider or "",
                    cluster_bytes=cluster_bytes,
                    spec_bytes=spec_bytes_out,
                    provider_config_bytes=provider_bytes,
                    reply_to=r,
                ),
                timeout=5.0,
            )
            self._start_ttl(name)
            await self._register_client(name, client_id)
            yield PoolReady(pool_name=name, node_count=pool.current_nodes())

        except Exception as e:
            yield PoolFailed(pool_name=name, reason=str(e))

        finally:
            self._projection.on_change = original_on_change
            self._projection.on_event = original_on_event

    def _provision_pool(self, name: str, spec_bytes: bytes) -> Any:
        """Provision a ComputePool from serialized specs."""
        import cloudpickle

        from skyward.core.pool import ComputePool

        specs: tuple = cloudpickle.loads(spec_bytes)
        first = specs[0]
        pool = ComputePool(
            *specs,
            image=first.image,
            plugins=tuple(first.plugins),
            volumes=tuple(first.volumes),
        )
        pool._pool_name = name

        if self._session is None:
            from skyward.core.session import Session
            self._session = Session(
                console=False, logging=True,
                projection=self._projection,
            )
            self._session.__enter__()

        pool.__enter__()
        self._pool_ttls[name] = specs[0].ttl if specs else 1200
        return pool

    # -- Task dispatch (async, no deadlock) --------------------------------

    async def _submit_task(
        self, name: str, payload: bytes, timeout: float,
    ) -> TaskSucceeded | TaskFailed:
        if name not in self._pools:
            return TaskFailed(error=f"Pool '{name}' not found", traceback="")

        pool = self._pools[name]
        pending = await asyncio.to_thread(cloudpickle.loads, payload)

        try:
            future = pool.run_async(pending)
            result = await asyncio.wrap_future(future)
            return TaskSucceeded(payload=await asyncio.to_thread(cloudpickle.dumps, result))
        except Exception as e:
            return TaskFailed(error=str(e), traceback=tb_module.format_exc())

    async def _submit_broadcast(
        self, name: str, payload: bytes, timeout: float,
    ) -> BroadcastSucceeded | TaskFailed:
        if name not in self._pools:
            return TaskFailed(error=f"Pool '{name}' not found", traceback="")

        pool = self._pools[name]
        pending = await asyncio.to_thread(cloudpickle.loads, payload)

        try:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, pool.broadcast, pending)
            return BroadcastSucceeded(payload=await asyncio.to_thread(cloudpickle.dumps, results))
        except Exception as e:
            return TaskFailed(error=str(e), traceback=tb_module.format_exc())

    def _get_node_count(self, name: str) -> NodeCount | DaemonError:
        if name not in self._pools:
            return DaemonError(error=f"Pool '{name}' not found")
        return NodeCount(ready=self._pools[name].current_nodes())

    def _get_pools(self) -> DaemonResponse:
        from skyward.api.views import NodeStatus

        from .protocol import PoolList, PoolSummary

        summaries: list[PoolSummary] = []
        view = self._projection.view
        for name, pv in view.pools.items():
            ready = sum(1 for n in pv.nodes.values() if n.status.value >= NodeStatus.READY.value)

            provider = ""
            accelerator = ""
            vcpus = ""
            memory = ""
            vram = ""
            disk = ""
            if pool := self._pools.get(name):
                spec = getattr(pool, "_spec", None)
                if spec is not None:
                    provider = spec.provider or ""
                    accelerator = spec.accelerator_name or "cpu"
                    vcpus = f"{int(spec.vcpus)}" if spec.vcpus else ""
                    memory = f"{int(spec.memory_gb)}gb" if spec.memory_gb else ""
                    disk = f"{int(spec.disk_gb)}gb" if spec.disk_gb else ""
                    if spec.accelerator is not None:
                        vram_val = getattr(spec.accelerator, "vram_gb", None)
                        vram = f"{int(vram_val)}gb" if vram_val else ""

            cpu_vals = [n.metrics["cpu"] for n in pv.nodes.values() if "cpu" in n.metrics]
            mem_vals = [n.metrics["mem"] for n in pv.nodes.values() if "mem" in n.metrics]
            avg_cpu = sum(cpu_vals) / len(cpu_vals) if cpu_vals else None
            avg_mem = sum(mem_vals) / len(mem_vals) if mem_vals else None

            summaries.append(PoolSummary(
                name=name, phase=pv.phase.name,
                nodes_ready=ready, nodes_total=pv.total_nodes,
                tasks_done=pv.tasks.done, tasks_running=pv.tasks.running,
                started_at=pv.started_at,
                provider=provider, accelerator=accelerator,
                vcpus=vcpus, memory=memory, vram=vram, disk=disk,
                avg_cpu=avg_cpu, avg_mem=avg_mem,
            ))
        return PoolList(pools=tuple(summaries))

    def _get_pool_view(self, name: str) -> DaemonResponse:
        from .protocol import PoolViewResponse

        view = self._projection.view
        if name not in view.pools:
            return DaemonError(error=f"Pool '{name}' not found")
        return PoolViewResponse(view=view.pools[name])

    def _get_pool_logs(self, name: str, all_logs: bool) -> PoolLogs | DaemonError:
        pool_dir = self._logs_dir / name
        if not pool_dir.exists():
            return DaemonError(error=f"No logs for pool '{name}'")
        files = sorted(pool_dir.glob("*.log"))
        if not files:
            return DaemonError(error=f"No logs for pool '{name}'")
        if all_logs:
            return PoolLogs(paths=tuple(str(f) for f in files))
        return PoolLogs(paths=(str(files[-1]),))

    async def _shutdown_pool(self, name: str) -> PoolShutdown | DaemonError:
        if name not in self._pools:
            return DaemonError(error=f"Pool '{name}' not found")
        await self._teardown_pool(name)
        return PoolShutdown()

    async def _shutdown_daemon(self) -> DaemonStopped:
        """Tear down all pools and signal the event loop to stop."""
        log.info("Daemon shutdown requested")
        for name in list(self._pools):
            await self._teardown_pool(name)
        if self._stop is not None:
            self._stop.set()
        return DaemonStopped()

    # -- Client tracking (via state actor) ---------------------------------

    async def _register_client(self, pool_name: str, client_id: str) -> None:
        assert self._actor_system is not None and self._state_ref is not None
        from .state import AddClient
        await self._actor_system.ask(
            self._state_ref,
            lambda r: AddClient(pool_name=pool_name, client_id=client_id, reply_to=r),
            timeout=5.0,
        )
        log.info("Client {cid} joined pool {pool}", cid=client_id, pool=pool_name)

    async def _unregister_client(self, pool_name: str, client_id: str) -> None:
        assert self._actor_system is not None and self._state_ref is not None
        from .state import RemoveClient
        await self._actor_system.ask(
            self._state_ref,
            lambda r: RemoveClient(pool_name=pool_name, client_id=client_id, reply_to=r),
            timeout=5.0,
        )
        log.info("Client {cid} left pool {pool}", cid=client_id, pool=pool_name)

    # -- TTL management (hard lifetime) ------------------------------------

    def _start_ttl(self, name: str) -> None:
        if name not in self._pools:
            return
        ttl = self._pool_ttls.get(name, 1200)
        self._cancel_ttl(name)
        self._ttl_tasks[name] = asyncio.create_task(self._ttl_expire(name, ttl))
        log.info("Pool {name} TTL started: {ttl}s", name=name, ttl=ttl)

    def _cancel_ttl(self, name: str) -> None:
        if task := self._ttl_tasks.pop(name, None):
            task.cancel()

    async def _ttl_expire(self, name: str, ttl: int) -> None:
        await asyncio.sleep(ttl)
        log.info("Pool {name} TTL expired, tearing down", name=name, ttl=ttl)
        self._ttl_tasks.pop(name, None)
        await self._teardown_pool(name)
        if not self._pools and self._stop is not None:
            self._stop.set()

    async def _teardown_pool(self, name: str) -> None:
        self._cancel_ttl(name)
        if fh := self._log_handles.pop(name, None):
            fh.close()
        if pool := self._pools.pop(name, None):
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, pool.__exit__, None, None, None)
            except Exception as e:
                log.warning("Error tearing down pool {name}: {err}", name=name, err=e)
        self._pool_ttls.pop(name, None)
        if self._actor_system is not None and self._state_ref is not None:
            from .state import RemovePool

            await self._actor_system.ask(
                self._state_ref,
                lambda r: RemovePool(pool_name=name, reply_to=r),
                timeout=5.0,
            )

    # -- Crash recovery -----------------------------------------------------

    async def _run_recovery(self) -> None:
        """Replay journal and attempt to reconnect pools from previous run."""
        assert self._actor_system is not None and self._state_ref is not None
        from .state import GetState

        state = await self._actor_system.ask(
            self._state_ref, lambda r: GetState(reply_to=r), timeout=5.0,
        )
        if not state.pools:
            return

        log.info("Recovery: found {n} pools in journal", n=len(state.pools))

        for name, entry in state.pools.items():
            try:
                recovered = await self._recover_pool(name, entry)
                if recovered:
                    log.info("Pool {name} recovered successfully", name=name)
                else:
                    log.warning("Pool {name} could not be recovered", name=name)
            except Exception as e:
                log.warning(
                    "Pool {name} recovery error: {err}", name=name, err=e,
                )

        from .state import RemoveClient

        state = await self._actor_system.ask(
            self._state_ref, lambda r: GetState(reply_to=r), timeout=5.0,
        )
        for name, entry in state.pools.items():
            for stale_cid in entry.clients:
                await self._actor_system.ask(
                    self._state_ref,
                    lambda r, c=stale_cid, n=name: RemoveClient(
                        pool_name=n, client_id=c, reply_to=r,
                    ),
                    timeout=5.0,
                )

    async def _recover_pool(self, name: str, entry: Any) -> bool:
        """Verify instances and reconnect a pool from journal state."""
        assert self._actor_system is not None and self._state_ref is not None

        if not entry.cluster_bytes or not entry.spec_bytes:
            log.info(
                "Pool {name}: no recovery data in journal (legacy entry), skipping",
                name=name,
            )
            return False

        try:
            cluster = await asyncio.to_thread(cloudpickle.loads, entry.cluster_bytes)
            spec = await asyncio.to_thread(cloudpickle.loads, entry.spec_bytes)
            provider_config = await asyncio.to_thread(cloudpickle.loads, entry.provider_config_bytes)
        except Exception as e:
            log.warning(
                "Pool {name}: deserialization failed ({err}), skipping",
                name=name, err=e,
            )
            return False

        if not Path(cluster.ssh_key_path).exists():
            log.warning(
                "Pool {name}: SSH key missing at {path}, "
                "cannot reconnect -- terminating instances",
                name=name, path=cluster.ssh_key_path,
            )
            provider = await self._create_provider(provider_config)
            await self._terminate_and_cleanup(name, provider, cluster, entry.instance_ids)
            return False

        provider = await self._create_provider(provider_config)
        alive: list[Any] = []
        dead: list[str] = []

        for iid in entry.instance_ids:
            try:
                cluster, instance = await provider.get_instance(cluster, iid)
            except Exception as e:
                log.warning(
                    "Pool {name}: get_instance({iid}) failed: {err}",
                    name=name, iid=iid, err=e,
                )
                dead.append(iid)
                continue

            if instance is None or getattr(instance, "status", None) == "exited":
                dead.append(iid)
                log.info("Pool {name}: instance {iid} is gone", name=name, iid=iid)
            else:
                alive.append(instance)
                log.info(
                    "Pool {name}: instance {iid} alive (status={status})",
                    name=name, iid=iid, status=instance.status,
                )

        if len(alive) < spec.nodes.desired:
            log.warning(
                "Pool {name}: only {alive}/{desired} instances alive, "
                "terminating survivors",
                name=name, alive=len(alive), desired=spec.nodes.desired,
            )
            alive_ids = tuple(inst.id for inst in alive)
            await self._terminate_and_cleanup(name, provider, cluster, alive_ids)
            return False

        log.info(
            "Pool {name}: {alive}/{total} instances alive, recovering",
            name=name, alive=len(alive), total=len(entry.instance_ids),
        )

        pool = await self._recover_pool_via_session(
            name, spec, provider, cluster, tuple(alive),
        )
        self._pools[name] = pool
        self._open_log_file(name)
        self._pool_ttls[name] = spec.ttl

        if dead:
            from .state import RegisterPool

            alive_ids = tuple(inst.id for inst in alive)
            cluster_bytes = await asyncio.to_thread(cloudpickle.dumps, cluster)
            await self._actor_system.ask(
                self._state_ref,
                lambda r: RegisterPool(
                    pool_name=name,
                    cluster_id=cluster.id,
                    instance_ids=alive_ids,
                    provider_name=entry.provider_name,
                    cluster_bytes=cluster_bytes,
                    spec_bytes=entry.spec_bytes,
                    provider_config_bytes=entry.provider_config_bytes,
                    reply_to=r,
                ),
                timeout=5.0,
            )

        self._start_ttl(name)
        return True

    async def _create_provider(self, provider_config: Any) -> Any:
        """Instantiate a provider from a persisted config."""
        return await provider_config.create_provider()

    async def _recover_pool_via_session(
        self,
        name: str,
        spec: Any,
        provider: Any,
        cluster: Any,
        instances: tuple[Any, ...],
    ) -> Any:
        """Bridge to Session to create a recovered ComputePool.

        Uses RecoverExistingPool instead of SpawnPool to skip
        prepare()/provision() and jump to node-spawning.
        """
        if self._session is None:
            from skyward.core.session import Session
            self._session = Session(
                console=False, logging=True,
                projection=self._projection,
            )
            self._session.__enter__()

        return self._session.recover_pool(
            name=name, spec=spec, provider=provider,
            cluster=cluster, instances=instances,
        )

    async def _terminate_and_cleanup(
        self,
        name: str,
        provider: Any,
        cluster: Any,
        instance_ids: tuple[str, ...],
    ) -> None:
        """Terminate instances and remove pool from journal."""
        assert self._actor_system is not None and self._state_ref is not None
        from contextlib import suppress

        from .state import RemovePool

        if instance_ids:
            with suppress(Exception):
                await provider.terminate(cluster, instance_ids)
        with suppress(Exception):
            await provider.teardown(cluster)

        await self._actor_system.ask(
            self._state_ref,
            lambda r: RemovePool(pool_name=name, reply_to=r),
            timeout=5.0,
        )

    async def serve_forever(self) -> None:
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()
