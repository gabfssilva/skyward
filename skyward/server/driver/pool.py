"""``ServerPool`` — the client-side :class:`Pool`-protocol adapter.

Pickles :class:`PendingFunction` payloads, ships them through
:class:`ServerClient` over the UDS, and returns the remote worker's
value (or re-raises its exception). Owns its own asyncio event loop
running on a daemon thread — the public surface stays synchronous so
the existing operator overloads (``>>``, ``@``, ``&``, ``>``) keep
working without awaits.
"""
from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
from collections.abc import Iterable
from concurrent.futures import Future
from types import TracebackType
from typing import TYPE_CHECKING, Any

import cloudpickle

from skyward.api.function import PendingFunction
from skyward.api.spec import ConsoleMode
from skyward.core.loop import cleanup_loop, run_loop, run_sync
from skyward.server.driver.http import ServerClient

if TYPE_CHECKING:
    from casty import ActorRef, ActorSystem


class PythonVersionMismatchError(RuntimeError):
    """Raised when the client interpreter does not match the server's."""


class ServerPool:
    """Synchronous pool-protocol wrapper that talks to a UDS host.

    Parameters
    ----------
    name
        Logical pool name used on the server side.
    socket_path
        Filesystem path of the host's UDS.
    compute_spec
        Authoritative ``ComputeSpec`` sent to the server on enter.
    default_compute_timeout
        Fallback seconds when ``PendingFunction.timeout`` is ``None``.
    """

    def __init__(
        self,
        name: str,
        socket_path: str,
        compute_spec: Any,
        *,
        default_compute_timeout: float = 600.0,
        console: bool | ConsoleMode = True,
    ) -> None:
        self.name = name
        self.default_compute_timeout = default_compute_timeout
        self._compute_spec: Any = compute_spec
        self._socket_path: str = socket_path
        self._console: bool | ConsoleMode = console
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: ServerClient | None = None
        self._active: bool = False
        self._ui_system: ActorSystem | None = None
        self._ui_console_ref: ActorRef[Any] | None = None
        self._ui_unsubscribe: Any = None
        self._ui_task: asyncio.Task[None] | None = None
        self._ui_projection: Any = None

    def __enter__(self) -> ServerPool:
        self._check_python_version()
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._thread = threading.Thread(
            target=lambda: run_loop(loop),
            daemon=True,
            name=f"skyward-server-pool-{self.name}",
        )
        self._thread.start()
        self._client = ServerClient(self._socket_path)

        self._start_ui()

        async def _bring_up() -> None:
            await self._client.__aenter__()
            self._ui_task = asyncio.create_task(self._consume_session_events())
            await self._client.ensure_compute(self.name, self._compute_spec)

        run_sync(loop, _bring_up())
        self._active = True
        return self

    def _start_ui(self) -> None:
        """Spin up the local projection + console actor system."""
        from casty import ActorSystem, CastyConfig

        from skyward.actors.console import (
            EventReceived,
            LogReceived,
            ViewUpdated,
            resolve_console,
        )
        from skyward.api.projection import SessionProjection

        factory = resolve_console(self._console)
        if factory is None:
            return

        assert self._loop is not None

        async def _boot() -> tuple[ActorSystem, Any]:
            system = ActorSystem(
                f"skyward-ui-{self.name}",
                config=CastyConfig(suppress_dead_letters_on_shutdown=True),
            )
            await system.__aenter__()
            console_ref = system.spawn(factory(), "console")
            return system, console_ref

        system, console_ref = run_sync(self._loop, _boot())
        projection = SessionProjection()

        def _on_change(_old: Any, new: Any) -> None:
            console_ref.tell(ViewUpdated(view=new))

        def _on_log(log: Any) -> None:
            console_ref.tell(LogReceived(log=log))

        def _on_event(ev: Any) -> None:
            console_ref.tell(EventReceived(event=ev))

        self._ui_unsubscribe = projection.subscribe(
            on_change=_on_change, on_log=_on_log, on_event=_on_event,
        )
        self._ui_system = system
        self._ui_console_ref = console_ref
        self._ui_projection = projection

    async def _consume_session_events(self) -> None:
        """Read pickled SessionEvents off SSE and feed the local projection."""
        if self._ui_projection is None or self._client is None:
            return
        try:
            async for data in self._client.subscribe_session_events(self.name):
                try:
                    event = cloudpickle.loads(data)
                except Exception:
                    continue
                try:
                    self._ui_projection.handle(event)
                except Exception:
                    continue
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._active = False
        loop = self._loop
        client = self._client
        if loop is not None and client is not None:
            async def _tear_down() -> None:
                if self._ui_task is not None:
                    self._ui_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await self._ui_task
                    self._ui_task = None
                try:
                    await client.shutdown_compute(self.name)
                finally:
                    await client.__aexit__(exc_type, exc_val, exc_tb)

            with contextlib.suppress(Exception):
                run_sync(loop, _tear_down())
        self._stop_ui()
        cleanup_loop(self._loop, self._thread)
        self._loop = None
        self._thread = None
        self._client = None

    def _stop_ui(self) -> None:
        """Tear down the local projection + console actor system."""
        if self._ui_unsubscribe is not None:
            with contextlib.suppress(Exception):
                self._ui_unsubscribe()
            self._ui_unsubscribe = None
        system = self._ui_system
        loop = self._loop
        if system is not None and loop is not None:
            async def _leave() -> None:
                await system.__aexit__(None, None, None)
            with contextlib.suppress(Exception):
                run_sync(loop, _leave(), timeout=10.0)
        self._ui_system = None
        self._ui_console_ref = None
        self._ui_projection = None

    def _assert_active(self) -> None:
        if not self._active:
            raise RuntimeError("ServerPool is not active")

    def _check_python_version(self) -> None:
        """Fail early if client python mismatches the server's image pin.

        Cross-version invocation is a real foot-gun (cloudpickle's
        opcode set is stable but bytecode layout shifts), so we refuse
        to ship work between incompatible interpreters.
        """
        client_py = f"{sys.version_info.major}.{sys.version_info.minor}"
        specs = getattr(self._compute_spec, "specs", ())
        if not specs:
            return
        want = getattr(specs[0].image, "python", None)
        if want and not want.startswith(client_py):
            raise PythonVersionMismatchError(
                f"client is Python {client_py}, pool image pins {want!r}",
            )

    def _timeout(self, pending: PendingFunction[Any]) -> float:
        return (
            pending.timeout
            if pending.timeout is not None
            else self.default_compute_timeout
        )

    def run[T](self, pending: PendingFunction[T]) -> T:
        self._assert_active()
        payload = cloudpickle.dumps((pending.fn, pending.args, pending.kwargs))
        timeout = self._timeout(pending)
        assert self._client is not None and self._loop is not None
        task_key = (pending.fn.__module__, pending.fn.__qualname__)

        async def _call() -> bytes:
            _, result_bytes = await self._client.submit_task(
                self.name, task_key, payload, timeout,
            )
            return result_bytes

        result_bytes = run_sync(self._loop, _call(), timeout=timeout + 15)
        return cloudpickle.loads(result_bytes)

    def run_async[T](self, pending: PendingFunction[T]) -> Future[T]:
        self._assert_active()
        payload = cloudpickle.dumps((pending.fn, pending.args, pending.kwargs))
        timeout = self._timeout(pending)
        assert self._client is not None and self._loop is not None
        task_key = (pending.fn.__module__, pending.fn.__qualname__)

        async def _call() -> T:
            _, result_bytes = await self._client.submit_task(
                self.name, task_key, payload, timeout,
            )
            return cloudpickle.loads(result_bytes)

        return asyncio.run_coroutine_threadsafe(_call(), self._loop)

    def broadcast[T](self, pending: PendingFunction[T]) -> list[T]:
        self._assert_active()
        payload = cloudpickle.dumps((pending.fn, pending.args, pending.kwargs))
        timeout = self._timeout(pending)
        assert self._client is not None and self._loop is not None
        task_key = (pending.fn.__module__, pending.fn.__qualname__)

        async def _call() -> list[T]:
            _, shards = await self._client.broadcast(
                self.name, task_key, payload, timeout,
            )
            return [cloudpickle.loads(s) for s in shards]

        return run_sync(self._loop, _call(), timeout=timeout + 15)

    def current_nodes(self) -> int:
        self._assert_active()
        assert self._client is not None and self._loop is not None
        return run_sync(self._loop, self._client.node_count(self.name))

    def map[T, R](
        self, fn: Any, items: Iterable[T],
    ) -> list[R]:
        """Synchronous ``map`` implemented on top of :meth:`run_async`."""
        futures = [self.run_async(fn(x)) for x in items]
        return [f.result() for f in futures]

    def __rshift__[T](self, pending: PendingFunction[T]) -> T:
        return self.run(pending)

    def __matmul__[T](self, pending: PendingFunction[T]) -> list[T]:
        return self.broadcast(pending)

    def __gt__[T](self, pending: PendingFunction[T]) -> Future[T]:
        return self.run_async(pending)


__all__ = ["PythonVersionMismatchError", "ServerPool"]
