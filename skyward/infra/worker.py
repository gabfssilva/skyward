"""Casty v2 worker services for remote nodes.

Runs as a long-lived process on each node in the cluster. Starts a casty
node (rank 0 is the seed, port 25520) hosting two stateless services:

- ``WorkerService.run`` — task execution, bounded by the framework's
  ``concurrency=`` (read from ``SKYWARD_WORKERS_PER_NODE`` at import).
- ``WorkerControl`` — unbounded control channel: discovery ping, worker
  lifecycle contexts, result reconciliation, and the pull/push endpoints
  that back generator output streams and ``Iterator`` input params.

The client is a casty lite member (``casty.connect``) that pins every
call to this node's ``Member`` (``service(Cls, at=member)``); all bytes
travel through the SSH port-forward via ``address_map``.

Payloads are lz4-compressed cloudpickle ``bytes`` end to end — casty's
msgpack wire never sees user objects.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
import types
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import casty
import cloudpickle
import lz4.frame

from skyward.observability.logger import logger

_active_task_id: ContextVar[str | None] = ContextVar("_active_task_id", default=None)
"""Task id of the currently-executing function on this worker.

The stdio writer (:class:`_UnbufferedWriter`) prepends ``[task-id=<eid>] ``
to each write while this is set, letting the client correlate output
with the originating task.
"""

DEFAULT_CASTY_PORT = 25520
STREAM_END = "end"
STREAM_ERROR = "error"
STREAM_ITEM = "item"
STREAM_PENDING = "pending"


def dumps(obj: Any) -> bytes:
    return lz4.frame.compress(cloudpickle.dumps(obj))


def loads(data: bytes) -> Any:
    try:
        return cloudpickle.loads(lz4.frame.decompress(data))
    except (ModuleNotFoundError, AttributeError, ImportError) as exc:
        logger.error(
            "Deserialization failed — likely a library version mismatch "
            "between local and remote environments. Ensure libraries like "
            "pandas, numpy, torch, etc. are pinned to the same version in "
            "Image(pip=[...]). Original error: {exc}",
            exc=exc,
        )
        raise


@dataclass(frozen=True, slots=True)
class TaskSucceeded:
    result: Any
    node_id: int


@dataclass(frozen=True, slots=True)
class TaskFailed:
    error: str
    traceback: str
    node_id: int


@dataclass(frozen=True, slots=True)
class StreamStarted:
    """The task is a generator; consume it via ``WorkerControl.next_chunk``."""

    node_id: int


type TaskResult = TaskSucceeded | TaskFailed | StreamStarted


@dataclass(frozen=True, slots=True)
class ResultPending:
    """Worker has the task but it hasn't completed yet."""


@dataclass(frozen=True, slots=True)
class ResultDone:
    """Worker completed the task; result is the original outcome."""

    result: TaskResult


@dataclass(frozen=True, slots=True)
class ResultUnknown:
    """Worker has no record of this ``task_id`` (never seen, evicted, or restarted)."""


type GetResultReply = ResultPending | ResultDone | ResultUnknown

_FEED_END = object()
_STREAM_DONE = object()


class _InFeed:
    """Worker-side buffer behind an ``Iterator`` parameter.

    The client pushes elements one at a time via ``WorkerControl.feed``
    (each push awaits the bounded queue — natural backpressure); the task
    thread pulls through the sync iterator.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=256)
        self._loop = loop

    async def put(self, item: Any) -> None:
        await self._queue.put(item)

    async def close(self) -> None:
        await self._queue.put(_FEED_END)

    def __iter__(self) -> _InFeed:
        return self

    def __next__(self) -> Any:
        fut = asyncio.run_coroutine_threadsafe(self._queue.get(), self._loop)
        item = fut.result()
        if item is _FEED_END:
            raise StopIteration
        return item


class _OutStream:
    """Worker-side pull handle over a task's generator.

    The generator body runs lazily, one element per ``next_chunk`` call,
    inside a worker thread — the client's poll cadence is the backpressure.
    A pull that outlives ``wait`` stays pending and is resumed by the next
    call, so a blocked ``next()`` never leaks a second thread.
    """

    def __init__(self, gen: types.GeneratorType, task_id: str) -> None:  # type: ignore[type-arg]
        self._gen = gen
        self._task_id = task_id
        self._pending: asyncio.Task[Any] | None = None

    def _pull(self) -> Any:
        token = _active_task_id.set(self._task_id)
        try:
            return next(self._gen)
        except StopIteration:
            return _STREAM_DONE
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            _active_task_id.reset(token)

    async def next(self, wait: float) -> tuple[str, Any]:
        if self._pending is None:
            self._pending = asyncio.ensure_future(asyncio.to_thread(self._pull))
        try:
            item = await asyncio.wait_for(asyncio.shield(self._pending), timeout=wait)
        except TimeoutError:
            return (STREAM_PENDING, None)
        except Exception as e:
            self._pending = None
            return (STREAM_ERROR, f"{e}\n{traceback.format_exc()}")
        self._pending = None
        if item is _STREAM_DONE:
            return (STREAM_END, None)
        return (STREAM_ITEM, item)


@dataclass
class _Runtime:
    """Per-process worker state shared by both services."""

    node_id: int
    loop: asyncio.AbstractEventLoop
    executor: str = "thread"
    executor_pool: Executor | None = None
    registry: Any = None
    process_hooks: list[tuple[str, Any]] = field(default_factory=list)
    active_contexts: list[Any] = field(default_factory=list)
    result_cache: dict[str, tuple[float, TaskResult | None]] = field(default_factory=dict)
    result_cache_ttl: float = 600.0
    result_cache_max: int = 2048
    out_streams: dict[str, _OutStream] = field(default_factory=dict)
    in_feeds: dict[tuple[str, int], _InFeed] = field(default_factory=dict)

    def evict_expired(self) -> None:
        now = time.monotonic()
        for k in [k for k, (deadline, _) in self.result_cache.items() if deadline < now]:
            del self.result_cache[k]
        if len(self.result_cache) > self.result_cache_max:
            oldest = sorted(self.result_cache.items(), key=lambda kv: kv[1][0])
            for k, _ in oldest[: len(self.result_cache) - self.result_cache_max]:
                self.result_cache.pop(k, None)

    def cache_result(self, task_id: str, result: TaskResult) -> None:
        if not task_id:
            return
        if isinstance(result, StreamStarted):
            self.result_cache.pop(task_id, None)
            return
        self.result_cache[task_id] = (time.monotonic() + self.result_cache_ttl, result)

    def feed_for(self, task_id: str, index: int) -> _InFeed:
        key = (task_id, index)
        feed = self.in_feeds.get(key)
        if feed is None:
            feed = _InFeed(self.loop)
            self.in_feeds[key] = feed
        return feed


_runtime: _Runtime | None = None


def _rt() -> _Runtime:
    assert _runtime is not None, "worker runtime not initialized"
    return _runtime


def _run_in_process(
    payload: bytes,
    env: dict[str, str],
    around_process_hooks: tuple[tuple[str, Any], ...] = (),
    task_id: str = "",
) -> Any:
    """Execute a pre-serialized task in a subprocess."""
    os.environ.update(env)

    from skyward.plugins.process_state import get_worker_index

    worker_idx = get_worker_index()
    if worker_idx is not None:
        import json

        pool_json = os.environ.get("COMPUTE_POOL")
        if pool_json:
            pool_data = json.loads(pool_json)
            pool_data["worker"] = worker_idx
            os.environ["COMPUTE_POOL"] = json.dumps(pool_data)

    if around_process_hooks:
        from skyward.core.runtime import instance_info
        from skyward.plugins.process_state import ensure_around_process

        info = instance_info()
        for name, factory in around_process_hooks:
            ensure_around_process(name, factory, info)

    fn, args, kwargs = cloudpickle.loads(payload)
    # ContextVars don't cross process boundaries — set explicitly so the
    # subprocess's ``_UnbufferedWriter`` (installed by ``ipc_initializer``)
    # prepends ``[task-id=<eid>] `` to each line.
    token = _active_task_id.set(task_id) if task_id else None
    try:
        return fn(*args, **kwargs)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if token is not None:
            _active_task_id.reset(token)


async def _execute(
    rt: _Runtime,
    task_id: str,
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    stream_indices: tuple[int, ...],
) -> TaskResult:
    log = logger.bind(component="worker", node_id=rt.node_id)
    try:
        from skyward.infra.streaming import is_generator_compute

        use_process = (
            rt.executor == "process"
            and not stream_indices
            and not is_generator_compute(fn)
        )

        if use_process:
            payload = await asyncio.to_thread(cloudpickle.dumps, (fn, args, kwargs))
            result = await rt.loop.run_in_executor(
                rt.executor_pool, _run_in_process, payload,
                dict(os.environ), tuple(rt.process_hooks), task_id,
            )
        else:
            resolved = list(args)
            for idx in stream_indices:
                if idx < len(resolved):
                    resolved[idx] = rt.feed_for(task_id, idx)

            def _run() -> Any:
                if rt.registry is not None:
                    from skyward.distributed import _set_active_registry

                    _set_active_registry(rt.registry)

                if rt.process_hooks:
                    from skyward.core.runtime import instance_info
                    from skyward.plugins.process_state import ensure_around_process

                    info = instance_info()
                    for name, factory in rt.process_hooks:
                        ensure_around_process(name, factory, info)

                fn_name = getattr(fn, "__name__", str(fn))
                log.info("Executing {fn_name}", fn_name=fn_name)
                result = fn(*tuple(resolved), **kwargs)
                log.info("Task {fn_name} completed", fn_name=fn_name)
                return result

            result = await asyncio.to_thread(_run)

        match result:
            case types.GeneratorType():
                rt.out_streams[task_id] = _OutStream(result, task_id)
                return StreamStarted(node_id=rt.node_id)
            case _:
                return TaskSucceeded(result=result, node_id=rt.node_id)
    except Exception as e:
        log.error("Task failed: {err}", err=e)
        return TaskFailed(
            error=str(e), traceback=traceback.format_exc(), node_id=rt.node_id,
        )


def _worker_concurrency() -> int:
    return int(os.environ.get("SKYWARD_WORKERS_PER_NODE", "1"))


@casty.service(name="skyward.Worker", concurrency=_worker_concurrency())
class WorkerService:
    """Task execution — one framework concurrency slot per running task."""

    async def run(self, task_id: str, payload: bytes) -> bytes:
        rt = _rt()
        rt.evict_expired()
        if task_id:
            rt.result_cache[task_id] = (
                time.monotonic() + rt.result_cache_ttl, None,
            )
        fn, args, kwargs, stream_indices = loads(payload)
        token = _active_task_id.set(task_id) if task_id else None
        try:
            result = await _execute(rt, task_id, fn, args, kwargs, stream_indices)
        finally:
            if token is not None:
                _active_task_id.reset(token)
        rt.cache_result(task_id, result)
        return dumps(result)


@casty.service(name="skyward.WorkerControl")
class WorkerControl:
    """Unbounded control channel: discovery, lifecycle, streams, reconciliation."""

    async def ping(self) -> int:
        return _rt().node_id

    async def enter_context(self, factory: bytes) -> bytes:
        rt = _rt()

        def _enter() -> Any:
            cm = loads(factory)()
            cm.__enter__()
            return cm

        try:
            cm = await asyncio.to_thread(_enter)
        except Exception as e:
            logger.error("Context entry failed: {error}", error=e)
            return dumps(TaskFailed(
                error=str(e), traceback=traceback.format_exc(), node_id=rt.node_id,
            ))
        rt.active_contexts.append(cm)
        return dumps(TaskSucceeded(result="ok", node_id=rt.node_id))

    async def set_process_hooks(self, hooks: bytes) -> None:
        rt = _rt()
        decoded = loads(hooks)
        rt.process_hooks.clear()
        rt.process_hooks.extend(decoded)

    async def get_result(self, task_id: str) -> bytes:
        rt = _rt()
        rt.evict_expired()
        reply: GetResultReply
        match rt.result_cache.get(task_id):
            case None:
                reply = ResultUnknown()
            case (_, None):
                reply = ResultPending()
            case (_, result):
                reply = ResultDone(result=result)
        return dumps(reply)

    async def next_chunk(self, task_id: str, wait: float) -> bytes:
        stream = _rt().out_streams.get(task_id)
        if stream is None:
            return dumps((STREAM_ERROR, f"no stream for task {task_id}"))
        kind, item = await stream.next(wait)
        if kind in (STREAM_END, STREAM_ERROR):
            _rt().out_streams.pop(task_id, None)
        return dumps((kind, item))

    async def feed(self, task_id: str, index: int, item: bytes) -> None:
        await _rt().feed_for(task_id, index).put(loads(item))

    async def feed_done(self, task_id: str, index: int) -> None:
        rt = _rt()
        await rt.feed_for(task_id, index).close()
        rt.in_feeds.pop((task_id, index), None)


async def main(
    node_id: int,
    port: int,
    seeds: list[tuple[str, int]] | None,
    num_nodes: int = 1,
    host: str = "0.0.0.0",
    workers_per_node: int = 1,
    worker_executor: str = "thread",
    reuse_processes: bool = True,
    tls_cert: str | None = None,
    tls_key: str | None = None,
    tls_ca: str | None = None,
    cluster_mode: bool = True,
) -> None:
    global _runtime

    log = logger.bind(component="worker", node_id=node_id)
    log.info("Casty worker starting, port={port} nodes={n}", port=port, n=num_nodes)

    os.environ["SKYWARD_NODE_ID"] = str(node_id)

    tls: casty.TLS | None = None
    if tls_cert and tls_ca:
        tls = casty.TLS(
            cert=tls_cert, key=tls_key or tls_cert, ca=tls_ca,
            require_client_cert=True,
        )
        log.info("mTLS enabled")

    async def _wait_for_seed(seed_host: str, seed_port: int) -> None:
        # Non-head workers race the head's listener. `casty.start` binds its
        # own port before dialing the seed and doesn't release it on a failed
        # join, so probe the seed with a raw TCP connect first.
        deadline = asyncio.get_event_loop().time() + 180.0
        while True:
            try:
                _, writer = await asyncio.open_connection(seed_host, seed_port)
                writer.close()
                await writer.wait_closed()
                return
            except OSError as e:
                if asyncio.get_event_loop().time() > deadline:
                    raise
                log.warning(
                    "Seed {h}:{p} not reachable ({err}), retrying",
                    h=seed_host, p=seed_port, err=e,
                )
                await asyncio.sleep(2.0)

    async def _start_node() -> Any:
        if seeds:
            await _wait_for_seed(*seeds[0])
        return await casty.start(
            f"0.0.0.0:{port}",
            advertise=f"{host}:{port}",
            seeds=[f"{h}:{p}" for h, p in seeds] if seeds else [],
            tls=tls,
            cluster_name="skyward",
        )

    system = await _start_node()
    try:
        from skyward.distributed import _set_active_registry
        from skyward.distributed.proxies import set_system_loop

        loop = asyncio.get_running_loop()
        ipc_queue: Any = None
        match worker_executor:
            case "process":
                import multiprocessing
                from concurrent.futures import ProcessPoolExecutor

                from loky import get_reusable_executor

                from skyward.distributed.ipc import ipc_initializer

                mp_ctx = multiprocessing.get_context("spawn")
                ipc_queue = mp_ctx.Queue()
                index_queue = mp_ctx.Queue()
                for i in range(workers_per_node):
                    index_queue.put(i)
                if reuse_processes:
                    task_executor: Executor = get_reusable_executor(
                        max_workers=workers_per_node,
                        initializer=ipc_initializer,
                        initargs=(ipc_queue, index_queue),
                        reuse="auto",
                    )
                else:
                    task_executor = ProcessPoolExecutor(
                        max_workers=workers_per_node,
                        mp_context=mp_ctx,
                        initializer=ipc_initializer,
                        initargs=(ipc_queue, index_queue),
                        max_tasks_per_child=1,
                    )
            case _:
                pool_size = max((os.cpu_count() or 1) + 4, workers_per_node)
                task_executor = ThreadPoolExecutor(max_workers=pool_size)
                loop.set_default_executor(task_executor)
        set_system_loop(loop)
        if cluster_mode:
            from skyward.distributed.registry import DistributedRegistry

            registry = DistributedRegistry(system, loop=loop, num_nodes=num_nodes)
        else:
            from skyward.distributed.disabled import DisabledRegistry

            registry = DisabledRegistry()
        _set_active_registry(registry)

        if ipc_queue is not None:
            from skyward.distributed.ipc import start_bridge

            start_bridge(ipc_queue, registry)

        _runtime = _Runtime(
            node_id=node_id,
            loop=loop,
            executor=worker_executor,
            executor_pool=task_executor,
            registry=registry,
        )

        log.info(
            "Casty worker ready (cluster={num_nodes} nodes, concurrency={concurrency})",
            num_nodes=num_nodes, concurrency=workers_per_node,
        )

        await asyncio.Event().wait()
    finally:
        rt, _runtime = _runtime, None
        if rt is not None:
            for cm in reversed(rt.active_contexts):
                with suppress(Exception):
                    cm.__exit__(None, None, None)
            rt.active_contexts.clear()
        with suppress(Exception):
            await system.close()


def _parse_seeds(seeds_str: str | None) -> list[tuple[str, int]] | None:
    if not seeds_str:
        return None
    return [
        (host, int(port_str))
        for addr in seeds_str.split(",")
        for host, port_str in [addr.rsplit(":", 1)]
    ]


class _UnbufferedWriter:
    """Text file writer that flushes after every write.

    When ``_active_task_id`` is set, each *line start* is prefixed with
    ``[task-id=<eid>] ``. ``print(a, b)`` emits multiple ``write()``
    calls (``"a"``, ``" "``, ``"b"``, ``"\\n"``) — naive prepending
    would put the prefix in the middle of the rendered line. The
    ``_at_line_start`` flag tracks whether the next char begins a fresh
    line across calls so the prefix appears once per logical line.
    """

    __slots__ = ("_at_line_start", "_f", "_pending_cr")

    def __init__(self, path: str) -> None:
        self._f = open(path, "a")  # noqa: SIM115
        self._pending_cr = False
        self._at_line_start = True

    def write(self, s: str) -> int:
        if not s:
            return 0
        if (tid := _active_task_id.get()) is not None:
            s = self._prefix_line_starts(s, f"[task-id={tid}] ")
        if self._pending_cr and not s.startswith("\r"):
            self._f.write("\r\n")
        self._pending_cr = "\r" in s and not s.endswith("\n")
        n = self._f.write(s)
        self._f.flush()
        return n

    def _prefix_line_starts(self, s: str, prefix: str) -> str:
        out: list[str] = []
        for ch in s:
            if self._at_line_start:
                out.append(prefix)
                self._at_line_start = False
            out.append(ch)
            if ch in ("\n", "\r"):
                self._at_line_start = True
        return "".join(out)

    def flush(self) -> None:
        self._f.flush()

    @property
    def encoding(self) -> str:
        return self._f.encoding

    def fileno(self) -> int:
        return self._f.fileno()

    def isatty(self) -> bool:
        return False


def _redirect_stdio_to_log(log_path: str = "/var/log/casty.log") -> None:
    log_file = _UnbufferedWriter(path=log_path)
    sys.stdout = log_file  # type: ignore[assignment]
    sys.stderr = log_file  # type: ignore[assignment]


def cli() -> None:
    _redirect_stdio_to_log(os.environ.get("SKYWARD_LOG_FILE", "/var/log/casty.log"))

    import logging as _logging

    _fmt = _logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s - %(message)s")
    _h = _logging.StreamHandler(sys.stderr)
    _h.setFormatter(_fmt)
    _h.setLevel(_logging.WARNING)
    _logging.getLogger("skyward").addHandler(_h)
    _logging.getLogger("casty").addHandler(_h)

    node_id = int(os.environ["SKYWARD_NODE_ID"])
    port = int(os.environ.get("SKYWARD_PORT", str(DEFAULT_CASTY_PORT)))
    num_nodes = int(os.environ.get("SKYWARD_NUM_NODES", "1"))
    host = os.environ.get("SKYWARD_HOST", "0.0.0.0")
    cluster_mode = os.environ.get("SKYWARD_CLUSTER", "true").lower() != "false"
    seeds = _parse_seeds(os.environ.get("SKYWARD_SEEDS")) if cluster_mode else None
    workers_per_node = int(os.environ.get("SKYWARD_WORKERS_PER_NODE", "1"))
    worker_executor = os.environ.get("SKYWARD_WORKER_EXECUTOR", "thread")
    reuse_processes = os.environ.get("SKYWARD_REUSE_PROCESSES", "true").lower() != "false"
    tls_cert = os.environ.get("SKYWARD_TLS_CERT")
    tls_key = os.environ.get("SKYWARD_TLS_KEY")
    tls_ca = os.environ.get("SKYWARD_TLS_CA")

    asyncio.run(main(
        node_id, port, seeds, num_nodes, host,
        workers_per_node=workers_per_node,
        worker_executor=worker_executor,
        reuse_processes=reuse_processes,
        tls_cert=tls_cert,
        tls_key=tls_key,
        tls_ca=tls_ca,
        cluster_mode=cluster_mode,
    ))


if __name__ == "__main__":
    # Force-import the module under its canonical name so all class
    # references (TaskSucceeded, WorkerService, etc.) use
    # 'skyward.infra.worker' — not '__main__'. Without this, pickle
    # sends __main__.TaskSucceeded which the client can't resolve.
    import importlib

    _proper = importlib.import_module("skyward.infra.worker")
    _proper.cli()
