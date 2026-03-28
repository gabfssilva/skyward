"""DaemonPool -- Pool protocol implementation routing through daemon."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future
from contextvars import Token
from pathlib import Path
from types import TracebackType
from typing import Any

import cloudpickle

from skyward.core.context import _active_pool
from skyward.core.function import PendingFunction, PendingFunctionGroup
from skyward.core.loop import cleanup_loop, run_loop, run_sync
from skyward.observability.logger import logger

from .client import DaemonClient

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
        project_dir: str | None = None,
        default_compute_timeout: float = 300.0,
    ) -> None:
        self._name = name
        self._socket_path = socket_path
        self._shutdown_on_exit = shutdown_on_exit
        self._project_dir = project_dir
        self._default_timeout = default_compute_timeout

        self._client: DaemonClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._active: bool = False
        self._context_token: Token[Any] | None = None
        self._node_count: int = 0

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

        client = DaemonClient(socket_path=self._socket_path)
        self._client = client

        run_sync(loop, client.connect())
        result = run_sync(
            loop,
            client.ensure_pool(self._name, project_dir=self._project_dir),
        )
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
