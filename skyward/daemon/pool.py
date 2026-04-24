"""DaemonPool -- Pool protocol implementation routing through daemon."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future
from contextvars import Token
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import cloudpickle

from skyward.core.context import _active_pool
from skyward.core.function import PendingFunction, PendingFunctionGroup
from skyward.core.loop import cleanup_loop, run_loop, run_sync
from skyward.observability.logger import logger

from .client import DaemonClient

if TYPE_CHECKING:
    from casty import ActorRef, ActorSystem

    from skyward.api.spec import ConsoleMode

_DEFAULT_SOCKET = Path.home() / ".skyward" / "daemon.sock"

log = logger.bind(component="daemon-pool")


class DaemonPool:
    """Pool implementation that dispatches tasks through the daemon.

    Satisfies the ``Pool`` protocol so that operators
    (``>>``, ``@``, ``>``, ``&``) work transparently.
    """

    def __init__(
        self,
        name: str,
        socket_path: Path = _DEFAULT_SOCKET,
        *,
        shutdown_on_exit: bool = False,
        spec_bytes: bytes = b"",
        default_compute_timeout: float = 300.0,
        console: bool | ConsoleMode = True,
    ) -> None:
        self._name = name
        self._socket_path = socket_path
        self._shutdown_on_exit = shutdown_on_exit
        self._spec_bytes = spec_bytes
        self._default_timeout = default_compute_timeout
        self._console: bool | ConsoleMode = console

        self._client: DaemonClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._active: bool = False
        self._context_token: Token[Any] | None = None
        self._node_count: int = 0

        self._actor_system: ActorSystem | None = None
        self._console_ref: ActorRef | None = None
        self._subscribe_client: DaemonClient | None = None
        self._subscribe_task: asyncio.Task[None] | None = None

    @property
    def concurrency(self) -> int:
        return 1

    @property
    def is_active(self) -> bool:
        return self._active

    def __enter__(self) -> DaemonPool:
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._loop_thread = threading.Thread(
            target=lambda: run_loop(loop),
            daemon=True,
            name="skyward-daemon-client-loop",
        )
        self._loop_thread.start()

        try:
            run_sync(loop, self._start_console())
        except Exception as e:
            log.warning("Console wiring failed: {err}", err=e)

        client = DaemonClient(socket_path=self._socket_path)
        self._client = client

        try:
            run_sync(loop, client.connect())
            result = run_sync(
                loop,
                client.ensure_pool(self._name, spec_bytes=self._spec_bytes),
            )
        except BaseException:
            with contextlib.suppress(Exception):
                run_sync(loop, self._stop_console())
            raise

        self._node_count = result.node_count
        self._active = True
        self._context_token = _active_pool.set(self)
        log.info(
            "Connected to daemon pool {name} ({n} nodes)",
            name=self._name, n=self._node_count,
        )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if self._context_token is not None:
                _active_pool.reset(self._context_token)
                self._context_token = None

            if self._client is not None and self._loop is not None:
                if self._shutdown_on_exit:
                    run_sync(self._loop, self._client.shutdown_pool(self._name))
                else:
                    run_sync(self._loop, self._client.disconnect(self._name))
                if self._loop is not None:
                    run_sync(self._loop, self._stop_console())
                run_sync(self._loop, self._client.close())
        except Exception as e:
            log.warning("Error disconnecting: {err}", err=e)
        finally:
            self._active = False
            cleanup_loop(self._loop, self._loop_thread)
            self._loop = None
            self._loop_thread = None
            self._client = None
            log.info("Disconnected from daemon pool {name}", name=self._name)

    def _assert_active(self) -> None:
        if not self._active or self._client is None or self._loop is None:
            raise RuntimeError("Daemon pool is not active")

    async def _start_console(self) -> None:
        """Spawn a local ActorSystem + console actor and stream daemon events into it."""
        from casty import ActorSystem

        from skyward.actors.console import resolve_console

        factory = resolve_console(self._console)
        if factory is None:
            return

        system = ActorSystem("skyward-daemon-client")
        await system.__aenter__()
        self._actor_system = system
        self._console_ref = system.spawn(factory(), "console")

        sub = DaemonClient(socket_path=self._socket_path)
        await sub.connect()
        self._subscribe_client = sub

        self._subscribe_task = asyncio.create_task(
            self._subscribe_loop(sub, self._console_ref),
        )

    async def _subscribe_loop(self, client: DaemonClient, console_ref: ActorRef) -> None:
        """Translate the daemon event stream into console actor messages."""
        from typing import cast

        from skyward.actors.console import EventReceived, LogReceived, ViewUpdated
        from skyward.api.events import Log, SessionEvent
        from skyward.api.views import SessionView

        try:
            async for msg in client.subscribe(self._name):
                match msg:
                    case SessionView():
                        console_ref.tell(ViewUpdated(view=msg))
                    case Log.Emitted():
                        console_ref.tell(LogReceived(log=msg))
                    case _:
                        console_ref.tell(EventReceived(event=cast(SessionEvent, msg)))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("Daemon subscribe stream error: {err}", err=e)

    async def _stop_console(self) -> None:
        """Drain the subscribe stream, stop the console actor and local actor system."""
        if self._subscribe_task is not None:
            timeout = 2.0 if self._shutdown_on_exit else 0.1
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._subscribe_task), timeout=timeout,
                )
            except TimeoutError:
                self._subscribe_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._subscribe_task
            except Exception:
                pass
            self._subscribe_task = None

        if self._subscribe_client is not None:
            with contextlib.suppress(Exception):
                await self._subscribe_client.close()
            self._subscribe_client = None

        if self._actor_system is not None:
            with contextlib.suppress(Exception):
                await self._actor_system.__aexit__(None, None, None)
            self._actor_system = None
            self._console_ref = None

    def _resolve_timeout(self, pending: PendingFunction[Any]) -> float:
        return pending.timeout if pending.timeout is not None else self._default_timeout

    def run[T](self, pending: PendingFunction[T]) -> T:
        self._assert_active()
        assert self._client is not None and self._loop is not None
        payload = cloudpickle.dumps(pending)
        timeout = self._resolve_timeout(pending)
        result = run_sync(
            self._loop,
            self._client.submit_task(self._name, payload, timeout),
        )
        return cloudpickle.loads(result.payload)

    def run_async[T](self, pending: PendingFunction[T]) -> Future[T]:
        self._assert_active()
        assert self._client is not None and self._loop is not None
        payload = cloudpickle.dumps(pending)
        timeout = self._resolve_timeout(pending)

        async def _run() -> T:
            assert self._client is not None
            result = await self._client.submit_task(self._name, payload, timeout)
            return cloudpickle.loads(result.payload)

        return asyncio.run_coroutine_threadsafe(_run(), self._loop)

    def broadcast[T](self, pending: PendingFunction[T]) -> list[T]:
        self._assert_active()
        assert self._client is not None and self._loop is not None
        payload = cloudpickle.dumps(pending)
        timeout = self._resolve_timeout(pending)
        result = run_sync(
            self._loop,
            self._client.submit_broadcast(self._name, payload, timeout),
        )
        return cloudpickle.loads(result.payload)

    async def _submit_on_fresh_connection[T](self, pending: PendingFunction[T]) -> T:
        """Open a dedicated connection, submit one task, close. Safe for parallel use."""
        async with DaemonClient(socket_path=self._socket_path) as client:
            result = await client.submit_task(
                self._name,
                cloudpickle.dumps(pending),
                self._resolve_timeout(pending),
            )
            return cloudpickle.loads(result.payload)

    def run_parallel(
        self, group: PendingFunctionGroup,
    ) -> tuple[Any, ...] | Generator[Any, None, None]:
        self._assert_active()
        assert self._loop is not None

        async def _parallel() -> tuple[Any, ...]:
            tasks = [self._submit_on_fresh_connection(p) for p in group.items]
            results = await asyncio.gather(*tasks)
            return tuple(results)

        return run_sync(self._loop, _parallel())

    def map[T, R](self, fn: Callable[[T], R], items: Sequence[T]) -> list[R]:
        self._assert_active()
        assert self._loop is not None

        async def _map() -> list[R]:
            tasks = [
                self._submit_on_fresh_connection(
                    PendingFunction(fn=fn, args=(item,), kwargs={}),
                )
                for item in items
            ]
            return list(await asyncio.gather(*tasks))

        return run_sync(self._loop, _map())

    def current_nodes(self) -> int:
        self._assert_active()
        assert self._client is not None and self._loop is not None
        return run_sync(self._loop, self._client.get_node_count(self._name))

    def resize(self, *spec: Any) -> None:
        raise NotImplementedError("Dynamic resize not yet available in daemon mode")

    def dict(self, name: str, **kwargs: Any) -> Any:
        raise NotImplementedError("Distributed collections not available in daemon mode")

    def set(self, name: str, **kwargs: Any) -> Any:
        raise NotImplementedError("Distributed collections not available in daemon mode")

    def counter(self, name: str, **kwargs: Any) -> Any:
        raise NotImplementedError("Distributed collections not available in daemon mode")

    def queue(self, name: str) -> Any:
        raise NotImplementedError("Distributed collections not available in daemon mode")

    def barrier(self, name: str, n: int) -> Any:
        raise NotImplementedError("Distributed collections not available in daemon mode")

    def lock(self, name: str, timeout: float = 30) -> Any:
        raise NotImplementedError("Distributed collections not available in daemon mode")

    def __repr__(self) -> str:
        status = "active" if self._active else "inactive"
        return f"DaemonPool(name={self._name!r}, {status}, daemon)"
