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

import argparse
import asyncio
import os
import pickle
import sys
import threading
import traceback
import types
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from casty import (
    ActorContext,
    ActorRef,
    Behavior,
    Behaviors,
    CastyConfig,
    ClusteredActorSystem,
    FailureDetectorConfig,
    HeartbeatConfig,
    ServiceKey,
)

from skyward.observability.logger import logger


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
class _CompressedResult:
    data: bytes


def _compress_result(result: Any) -> _CompressedResult:
    from skyward.infra.serialization import COMPRESSION_LEVEL

    raw = pickle.dumps(result, protocol=5)
    return _CompressedResult(data=zlib.compress(raw, COMPRESSION_LEVEL))


def _decompress_result(compressed: _CompressedResult) -> Any:
    return pickle.loads(zlib.decompress(compressed.data))  # noqa: S301


type TaskResult = TaskSucceeded | TaskFailed


@dataclass(frozen=True, slots=True)
class ExecuteTask:
    fn_bytes: bytes
    reply_to: ActorRef[TaskResult]
    input_streams: tuple[tuple[int, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class _TaskDone:
    result: TaskResult
    reply_to: ActorRef[TaskResult]


@dataclass(frozen=True, slots=True)
class _TaskErrored:
    error: Exception
    reply_to: ActorRef[TaskResult]


type WorkerMsg = ExecuteTask | _TaskDone | _TaskErrored

WORKER_KEY: ServiceKey[WorkerMsg] = ServiceKey("skyward-worker")


async def _wrap_generator(
    system: ClusteredActorSystem,
    gen: types.GeneratorType,  # type: ignore[type-arg]
    node_id: int,
) -> TaskSucceeded:
    from uuid import uuid4

    from casty import GetSink, stream_producer

    from skyward.infra.streaming import _StreamHandle

    producer_ref = system.spawn(
        stream_producer(buffer_size=256),
        f"out-producer-{uuid4().hex[:8]}",
    )
    sink = await system.ask(
        producer_ref,
        lambda r: GetSink(reply_to=r),
        timeout=10.0,
    )

    loop = asyncio.get_running_loop()

    def _drain() -> None:
        try:
            for elem in gen:
                asyncio.run_coroutine_threadsafe(sink.put(elem), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(sink.complete(), loop).result()

    threading.Thread(target=_drain, daemon=True).start()

    return TaskSucceeded(
        result=_StreamHandle(producer_ref=producer_ref, node_id=node_id),
        node_id=node_id,
    )


def worker_behavior(
    node_id: int,
    concurrency: int = 1,
    registry: Any = None,
    system: ClusteredActorSystem | None = None,
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
        fn_bytes: bytes,
        input_streams: tuple[tuple[int, Any], ...] = (),
    ) -> TaskResult:
        async with sem:
            from skyward.infra.serialization import deserialize

            payload = deserialize(fn_bytes)
            fn = payload["fn"]
            args = payload.get("args", ())
            kwargs = payload.get("kwargs", {})

            args = await _resolve_input_streams(args, input_streams)

            def _run() -> Any:
                if registry is not None:
                    from skyward.distributed import _set_active_registry

                    _set_active_registry(registry)

                fn_name = getattr(fn, "__name__", str(fn))
                log.info("Executing {fn_name}", fn_name=fn_name)
                result = fn(*args, **kwargs)
                log.info("Task {fn_name} completed", fn_name=fn_name)
                return result

            try:
                result = await asyncio.to_thread(_run)

                match result:
                    case types.GeneratorType() if system is not None:
                        return await _wrap_generator(system, result, node_id)
                    case _:
                        return TaskSucceeded(result=_compress_result(result), node_id=node_id)
            except Exception as e:
                fn_name = getattr(fn, "__name__", "unknown")
                log.error("Task {fn_name} failed: {err}", fn_name=fn_name, err=e)
                return TaskFailed(
                    error=str(e),
                    traceback=traceback.format_exc(),
                    node_id=node_id,
                )

    async def receive(ctx: ActorContext[WorkerMsg], msg: WorkerMsg) -> Behavior[WorkerMsg]:
        match msg:
            case ExecuteTask(fn_bytes=fn_bytes, reply_to=reply_to, input_streams=streams):
                log.debug("ExecuteTask received, payload={size} bytes", size=len(fn_bytes))
                ctx.pipe_to_self(
                    coro=_execute(fn_bytes, streams),
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

    return Behaviors.receive(receive)


async def main(
    node_id: int,
    port: int,
    seeds: list[tuple[str, int]] | None,
    num_nodes: int = 1,
    host: str = "0.0.0.0",
    workers_per_node: int = 1,
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

    async with ClusteredActorSystem(
        name="skyward",
        host=host,
        port=port,
        node_id=f"node-{node_id}",
        seed_nodes=tuple(seeds) if seeds else None,
        bind_host="0.0.0.0",
        config=config,
        required_quorum=quorum,
    ) as system:
        from skyward.distributed import _set_active_registry
        from skyward.distributed.proxies import set_system_loop
        from skyward.distributed.registry import DistributedRegistry

        loop = asyncio.get_running_loop()
        pool_size = max((os.cpu_count() or 1) + 4, workers_per_node)
        loop.set_default_executor(ThreadPoolExecutor(max_workers=pool_size))
        set_system_loop(loop)
        registry = DistributedRegistry(system, loop=loop)
        _set_active_registry(registry)

        system.spawn(
            Behaviors.discoverable(
                worker_behavior(
                    node_id, concurrency=workers_per_node,
                    registry=registry, system=system,
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
    parser = argparse.ArgumentParser(description="Casty worker service")
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--port", type=int, default=25520)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seed addresses (host:port)",
    )
    parser.add_argument("--workers-per-node", type=int, default=1)
    parser.add_argument(
        "--log-file", type=str, default="/var/log/casty.log",
        help="Redirect stdout/stderr to this file",
    )
    args = parser.parse_args()

    _redirect_stdio_to_log(args.log_file)

    seeds = _parse_seeds(args.seeds)
    asyncio.run(main(
        args.node_id, args.port, seeds, args.num_nodes, args.host,
        workers_per_node=args.workers_per_node,
    ))


if __name__ == "__main__":
    # When running as `python -m skyward.infra.worker`, the import chain
    # (skyward → api.pool → infra.executor → infra.worker) loads this module
    # as 'skyward.infra.worker' first, then Python re-executes it as __main__.
    # This creates duplicate class objects: __main__.TaskSucceeded vs
    # skyward.infra.worker.TaskSucceeded. Pickle stores the __main__ version,
    # which the client can't resolve (its __main__ is a different module).
    # Fix: delegate to the properly-imported module so all class references
    # point to 'skyward.infra.worker', not '__main__'.
    import sys

    _proper = sys.modules.get("skyward.infra.worker")
    if _proper is not None:
        _proper.cli()
    else:
        cli()
