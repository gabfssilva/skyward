"""Casty worker service for remote nodes.

Runs as a long-lived process on each node in the cluster.
Starts a ClusteredActorSystem and spawns discoverable worker actors
for direct task execution via ClusterClient.

Architecture:
- Each node spawns a local worker actor wrapped in Behaviors.discoverable().
- Workers register with WORKER_KEY for service discovery.
- The executor discovers workers via ClusterClient.lookup(WORKER_KEY).
- All nodes are symmetric (no head-only HTTP server).
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
import types
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from casty import (
    ActorContext,
    ActorRef,
    Behavior,
    Behaviors,
    CastyConfig,
    CloudPickleSerializer,
    ClusteredActorSystem,
    FailureDetectorConfig,
    HeartbeatConfig,
    Lz4CompressedSerializer,
    ServiceKey,
)

from skyward.observability.logger import logger

_background_tasks: set[asyncio.Task[None]] = set()


@dataclass(frozen=True, slots=True)
class TaskSucceeded:
    result: Any
    node_id: int


@dataclass(frozen=True, slots=True)
class TaskFailed:
    error: str
    traceback: str
    node_id: int


type TaskResult = TaskSucceeded | TaskFailed


class _DiagnosticSerializer:
    """Wraps a Casty serializer to surface deserialization errors clearly.

    Casty catches all deserialization exceptions as warnings (invisible when
    log level is ERROR). This wrapper logs at ERROR via Skyward's logger
    before re-raising, so version-mismatch failures are immediately visible.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Lz4CompressedSerializer) -> None:
        self._inner = inner

    def serialize[M](self, obj: M) -> bytes:
        return self._inner.serialize(obj)

    def deserialize[M, R](
        self,
        data: bytes,
        *,
        ref_factory: Any = None,
    ) -> M:
        try:
            return self._inner.deserialize(data, ref_factory=ref_factory)
        except (ModuleNotFoundError, AttributeError, ImportError) as exc:
            logger.error(
                "Deserialization failed — likely a library version mismatch "
                "between local and remote environments. Ensure libraries like "
                "pandas, numpy, torch, etc. are pinned to the same version in "
                "Image(pip=[...]). Original error: {exc}",
                exc=exc,
            )
            raise
        except Exception as exc:
            logger.error(
                "Deserialization failed: {exc}",
                exc=exc,
            )
            raise


def skyward_serializer() -> _DiagnosticSerializer:
    """Shared wire serializer for all Casty endpoints (worker, executor, instance)."""
    return _DiagnosticSerializer(Lz4CompressedSerializer(CloudPickleSerializer()))


@dataclass(frozen=True, slots=True)
class ExecuteTask:
    fn: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    reply_to: ActorRef[TaskResult]
    input_streams: tuple[tuple[int, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class EnterContext:
    factory: Any
    reply_to: ActorRef[TaskResult]


@dataclass(frozen=True, slots=True)
class _TaskDone:
    result: TaskResult
    reply_to: ActorRef[TaskResult]


@dataclass(frozen=True, slots=True)
class _TaskErrored:
    error: Exception
    reply_to: ActorRef[TaskResult]


@dataclass(frozen=True, slots=True)
class SetProcessHooks:
    hooks: tuple[tuple[str, Any], ...]
    reply_to: ActorRef[TaskResult]


@dataclass(frozen=True, slots=True)
class _ContextEntered:
    cm: Any
    reply_to: ActorRef[TaskResult]


@dataclass(frozen=True, slots=True)
class _ContextFailed:
    error: Exception
    reply_to: ActorRef[TaskResult]


type WorkerMsg = (
    ExecuteTask | EnterContext | SetProcessHooks
    | _TaskDone | _TaskErrored | _ContextEntered | _ContextFailed
)

WORKER_KEY: ServiceKey[WorkerMsg] = ServiceKey("skyward-worker")


async def _wrap_generator(
    system: ClusteredActorSystem,
    gen: types.GeneratorType,  # type: ignore[type-arg]
    node_id: int,
) -> TaskSucceeded:
    from uuid import uuid4

    from casty import GetSink, stream_producer

    from skyward.infra.stream import sync_to_async
    from skyward.infra.streaming import _StreamHandle

    slog = logger.bind(component="wrap-generator", node_id=node_id)

    producer_ref = system.spawn(
        stream_producer(buffer_size=256),
        f"out-producer-{uuid4().hex[:8]}",
    )
    sink = await system.ask(
        producer_ref,
        lambda r: GetSink(reply_to=r),
        timeout=10.0,
    )

    async def _drain() -> None:
        count = 0
        try:
            async for elem in sync_to_async(gen):
                await sink.put(elem)
                count += 1
            slog.info("drain finished, {n} elements", n=count)
        except Exception as e:
            slog.error("drain error after {n} elements: {e}", n=count, e=e)
        finally:
            await sink.complete()

    loop = asyncio.get_running_loop()
    drain_task = loop.create_task(_drain(), name=f"stream-drain-{node_id}")
    _background_tasks.add(drain_task)
    drain_task.add_done_callback(_background_tasks.discard)
    drain_task.add_done_callback(
        lambda t: slog.error("drain task failed: {e}", e=t.exception())
        if t.exception() else None
    )

    return TaskSucceeded(
        result=_StreamHandle(producer_ref=producer_ref, node_id=node_id),
        node_id=node_id,
    )


def _run_in_process(
    payload: bytes,
    env: dict[str, str],
    around_process_hooks: tuple[tuple[str, Any], ...] = (),
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

    import cloudpickle

    fn, args, kwargs = cloudpickle.loads(payload)
    try:
        return fn(*args, **kwargs)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


def worker_behavior(
    node_id: int,
    concurrency: int = 1,
    registry: Any = None,
    system: ClusteredActorSystem | None = None,
    executor: str = "thread",
    executor_pool: Executor | None = None,
) -> Behavior[WorkerMsg]:
    log = logger.bind(component="worker", node_id=node_id)
    sem = asyncio.Semaphore(concurrency)

    async def _resolve_input_streams(
        args: tuple[Any, ...],
        input_streams: tuple[tuple[int, Any], ...],
    ) -> tuple[Any, ...]:
        if not input_streams or system is None:
            return args

        from uuid import uuid4

        from casty import GetSource, stream_consumer

        from skyward.infra.streaming import _SyncSource

        loop = asyncio.get_running_loop()
        resolved = list(args)

        for idx, producer_ref in input_streams:
            if idx >= len(resolved):
                continue
            consumer_ref = system.spawn(
                stream_consumer(producer_ref, timeout=60.0, initial_demand=16),
                f"in-consumer-{idx}-{uuid4().hex[:8]}",
            )
            source = await system.ask(
                consumer_ref,
                lambda r: GetSource(reply_to=r),
                timeout=10.0,
            )
            resolved[idx] = _SyncSource(source, loop)

        return tuple(resolved)

    async def _execute(
        fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        input_streams: tuple[tuple[int, Any], ...] = (),
    ) -> TaskResult:
        async with sem:
            try:
                from skyward.infra.streaming import is_generator_compute

                use_process = (
                    executor == "process"
                    and not input_streams
                    and not is_generator_compute(fn)
                )

                if use_process:
                    import cloudpickle

                    loop = asyncio.get_running_loop()
                    payload = cloudpickle.dumps((fn, args, kwargs))
                    result = await loop.run_in_executor(
                        executor_pool, _run_in_process, payload,
                        dict(os.environ), tuple(process_hooks),
                    )
                else:
                    args = await _resolve_input_streams(args, input_streams)

                    def _run() -> Any:
                        if registry is not None:
                            from skyward.distributed import _set_active_registry

                            _set_active_registry(registry)

                        if process_hooks:
                            from skyward.core.runtime import instance_info
                            from skyward.plugins.process_state import ensure_around_process

                            info = instance_info()
                            for name, factory in process_hooks:
                                ensure_around_process(name, factory, info)

                        fn_name = getattr(fn, "__name__", str(fn))
                        log.info("Executing {fn_name}", fn_name=fn_name)
                        result = fn(*args, **kwargs)
                        log.info("Task {fn_name} completed", fn_name=fn_name)
                        return result

                    result = await asyncio.to_thread(_run)

                match result:
                    case types.GeneratorType() if system is not None:
                        return await _wrap_generator(system, result, node_id)
                    case _:
                        return TaskSucceeded(result=result, node_id=node_id)
            except Exception as e:
                log.error("Task failed: {err}", err=e)
                return TaskFailed(
                    error=str(e),
                    traceback=traceback.format_exc(),
                    node_id=node_id,
                )

    active_contexts: list[Any] = []
    process_hooks: list[tuple[str, Any]] = []

    async def receive(ctx: ActorContext[WorkerMsg], msg: WorkerMsg) -> Behavior[WorkerMsg]:
        match msg:
            case SetProcessHooks(hooks=hooks, reply_to=reply_to):
                process_hooks.clear()
                process_hooks.extend(hooks)
                log.debug("Process hooks set ({n} hooks)", n=len(hooks))
                reply_to.tell(TaskSucceeded(result="ok", node_id=node_id))
                return Behaviors.same()
            case EnterContext(factory=factory, reply_to=reply_to):
                log.debug("EnterContext received")

                async def _enter() -> Any:
                    def _do() -> Any:
                        cm = factory()
                        cm.__enter__()
                        return cm
                    return await asyncio.to_thread(_do)

                ctx.pipe_to_self(
                    coro=_enter(),
                    mapper=lambda cm: _ContextEntered(cm=cm, reply_to=reply_to),
                    on_failure=lambda e: _ContextFailed(error=e, reply_to=reply_to),
                )
                return Behaviors.same()
            case _ContextEntered(cm=cm, reply_to=reply_to):
                active_contexts.append(cm)
                log.debug("Context entered successfully")
                reply_to.tell(TaskSucceeded(result="ok", node_id=node_id))
                return Behaviors.same()
            case _ContextFailed(error=error, reply_to=reply_to):
                log.error("Context entry failed: {error}", error=error)
                reply_to.tell(TaskFailed(
                    error=str(error),
                    traceback=traceback.format_exc(),
                    node_id=node_id,
                ))
                return Behaviors.same()
            case ExecuteTask(
                fn=fn, args=args, kwargs=kwargs,
                reply_to=reply_to, input_streams=streams,
            ):
                fn_name = getattr(fn, "__name__", str(fn))
                log.debug("ExecuteTask received, fn={fn_name}", fn_name=fn_name)
                ctx.pipe_to_self(
                    coro=_execute(fn, args, kwargs, streams),
                    mapper=lambda result: _TaskDone(result=result, reply_to=reply_to),
                    on_failure=lambda e: _TaskErrored(error=e, reply_to=reply_to),
                )
                return Behaviors.same()
            case _TaskDone(result=result, reply_to=reply_to):
                log.debug("Task completed successfully")
                reply_to.tell(result)
                return Behaviors.same()
            case _TaskErrored(error=error, reply_to=reply_to):
                log.debug("Task errored: {error}", error=error)
                reply_to.tell(TaskFailed(
                    error=str(error),
                    traceback=traceback.format_exc(),
                    node_id=node_id,
                ))
                return Behaviors.same()

    async def _post_stop(_ctx: ActorContext[WorkerMsg]) -> None:
        for cm in reversed(active_contexts):
            with suppress(Exception):
                cm.__exit__(None, None, None)
        active_contexts.clear()

    return Behaviors.with_lifecycle(
        Behaviors.receive(receive),
        post_stop=_post_stop,
    )


async def main(
    node_id: int,
    port: int,
    seeds: list[tuple[str, int]] | None,
    num_nodes: int = 1,
    host: str = "0.0.0.0",
    workers_per_node: int = 1,
    worker_executor: str = "thread",
    tls_cert: str | None = None,
    tls_key: str | None = None,
    tls_ca: str | None = None,
) -> None:
    config = CastyConfig(
        heartbeat=HeartbeatConfig(interval=2.0, availability_check_interval=5.0),
        failure_detector=FailureDetectorConfig(
            threshold=16.0,
            acceptable_heartbeat_pause_ms=10_000.0,
        ),
        suppress_dead_letters_on_shutdown=True
    )

    log = logger.bind(component="worker", node_id=node_id)
    quorum = num_nodes if num_nodes > 1 else None
    log.debug("Starting casty worker, quorum={quorum} port={port}", quorum=quorum, port=port)
    log.info("Casty worker starting, quorum={quorum} port={port}", quorum=quorum, port=port)

    os.environ["SKYWARD_NODE_ID"] = str(node_id)

    tls: Any = None
    if tls_cert and tls_ca:
        from casty.remote.tls import Config as TlsConfig

        tls = TlsConfig.from_paths(certfile=tls_cert, cafile=tls_ca, keyfile=tls_key)
        log.info("mTLS enabled")

    async with ClusteredActorSystem(
        name="skyward",
        host=host,
        port=port,
        node_id=f"node-{node_id}",
        seed_nodes=tuple(seeds) if seeds else None,
        bind_host="0.0.0.0",
        config=config,
        serializer=skyward_serializer(),
        tls=tls,
    ) as system:
        from skyward.distributed import _set_active_registry
        from skyward.distributed.proxies import set_system_loop
        from skyward.distributed.registry import DistributedRegistry

        loop = asyncio.get_running_loop()
        ipc_queue: Any = None
        match worker_executor:
            case "process":
                import multiprocessing

                from loky import get_reusable_executor

                from skyward.distributed.ipc import ipc_initializer

                mp_ctx = multiprocessing.get_context("spawn")
                ipc_queue = mp_ctx.Queue()
                index_queue = mp_ctx.Queue()
                for i in range(workers_per_node):
                    index_queue.put(i)
                task_executor: Executor = get_reusable_executor(
                    max_workers=workers_per_node,
                    initializer=ipc_initializer,
                    initargs=(ipc_queue, index_queue),
                    reuse="auto",
                )
            case _:
                pool_size = max((os.cpu_count() or 1) + 4, workers_per_node)
                task_executor = ThreadPoolExecutor(max_workers=pool_size)
                loop.set_default_executor(task_executor)
        set_system_loop(loop)
        registry = DistributedRegistry(system, loop=loop)
        _set_active_registry(registry)

        if ipc_queue is not None:
            from skyward.distributed.ipc import start_bridge

            start_bridge(ipc_queue, registry)

        system.spawn(
            Behaviors.discoverable(
                worker_behavior(
                    node_id, concurrency=workers_per_node,
                    registry=registry, system=system,
                    executor=worker_executor,
                    executor_pool=task_executor,
                ),
                key=WORKER_KEY,
            ),
            "worker",
        )
        log.info(
            "Casty worker ready (cluster={num_nodes} nodes, concurrency={concurrency})",
            num_nodes=num_nodes, concurrency=workers_per_node,
        )

        await asyncio.Event().wait()


def _parse_seeds(seeds_str: str | None) -> list[tuple[str, int]] | None:
    if not seeds_str:
        return None
    return [
        (host, int(port_str))
        for addr in seeds_str.split(",")
        for host, port_str in [addr.rsplit(":", 1)]
    ]


def _redirect_stdio_to_log(log_path: str = "/var/log/casty.log") -> None:
    log_file = open(log_path, "a", buffering=1)  # noqa: SIM115
    sys.stdout = log_file
    sys.stderr = log_file


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
    port = int(os.environ.get("SKYWARD_PORT", "25520"))
    num_nodes = int(os.environ.get("SKYWARD_NUM_NODES", "1"))
    host = os.environ.get("SKYWARD_HOST", "0.0.0.0")
    seeds = _parse_seeds(os.environ.get("SKYWARD_SEEDS"))
    workers_per_node = int(os.environ.get("SKYWARD_WORKERS_PER_NODE", "1"))
    worker_executor = os.environ.get("SKYWARD_WORKER_EXECUTOR", "thread")
    tls_cert = os.environ.get("SKYWARD_TLS_CERT")
    tls_key = os.environ.get("SKYWARD_TLS_KEY")
    tls_ca = os.environ.get("SKYWARD_TLS_CA")

    asyncio.run(main(
        node_id, port, seeds, num_nodes, host,
        workers_per_node=workers_per_node,
        worker_executor=worker_executor,
        tls_cert=tls_cert,
        tls_key=tls_key,
        tls_ca=tls_ca,
    ))


if __name__ == "__main__":
    # Force-import the module under its canonical name so all class
    # references (TaskSucceeded, EnterContext, etc.) use
    # 'skyward.infra.worker' — not '__main__'. Without this, pickle
    # sends __main__.TaskSucceeded which the client can't resolve.
    import importlib

    _proper = importlib.import_module("skyward.infra.worker")
    _proper.cli()
