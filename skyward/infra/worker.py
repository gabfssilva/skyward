"""Casty worker service for remote nodes.

Runs as a long-lived process on each node in the cluster.
Starts a ClusteredActorSystem, spawns worker actors for task execution,
and exposes an HTTP API for job submission via aiohttp.

Architecture:
- Each node spawns a local worker actor for single-node execution.
- A broadcasted actor handles fan-out execution across all nodes.
- The head node (node 0) runs the HTTP server; others wait forever.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from aiohttp import web
from casty import ActorContext, ActorRef, Behavior, Behaviors
from casty.config import CastyConfig, FailureDetectorConfig, HeartbeatConfig
from casty.sharding import ClusteredActorSystem


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


@dataclass(frozen=True, slots=True)
class ExecuteTask:
    fn_bytes: bytes
    reply_to: ActorRef[TaskResult]


@dataclass(frozen=True, slots=True)
class _TaskDone:
    result: TaskResult
    reply_to: ActorRef[TaskResult]


@dataclass(frozen=True, slots=True)
class _TaskErrored:
    error: Exception
    reply_to: ActorRef[TaskResult]


type WorkerMsg = ExecuteTask | _TaskDone | _TaskErrored


def worker_behavior(node_id: int, concurrency: int = 1) -> Behavior[WorkerMsg]:
    sem = asyncio.Semaphore(concurrency)

    async def _execute(fn_bytes: bytes) -> TaskResult:
        async with sem:
            def _run() -> TaskResult:
                from skyward.infra.serialization import deserialize

                try:
                    payload = deserialize(fn_bytes)
                    fn = payload["fn"]
                    args = payload.get("args", ())
                    kwargs = payload.get("kwargs", {})
                    fn_name = getattr(fn, "__name__", str(fn))

                    print(f"[worker-{node_id}] Executing {fn_name}...", flush=True)
                    result = fn(*args, **kwargs)
                    print(f"[worker-{node_id}] {fn_name} completed", flush=True)
                    return TaskSucceeded(result=result, node_id=node_id)
                except Exception as e:
                    fn_name = "unknown"
                    print(f"[worker-{node_id}] {fn_name} failed: {e}", flush=True)
                    return TaskFailed(
                        error=str(e),
                        traceback=traceback.format_exc(),
                        node_id=node_id,
                    )

            return await asyncio.to_thread(_run)

    async def receive(ctx: ActorContext[WorkerMsg], msg: WorkerMsg) -> Behavior[WorkerMsg]:
        match msg:
            case ExecuteTask(fn_bytes=fn_bytes, reply_to=reply_to):
                ctx.pipe_to_self(
                    coro=_execute(fn_bytes),
                    mapper=lambda result: _TaskDone(result=result, reply_to=reply_to),
                    on_failure=lambda e: _TaskErrored(error=e, reply_to=reply_to),
                )
                return Behaviors.same()
            case _TaskDone(result=result, reply_to=reply_to):
                reply_to.tell(result)
                return Behaviors.same()
            case _TaskErrored(error=error, reply_to=reply_to):
                reply_to.tell(TaskFailed(
                    error=str(error),
                    traceback=traceback.format_exc(),
                    node_id=node_id,
                ))
                return Behaviors.same()

    return Behaviors.receive(receive)


def create_app(
    system: ClusteredActorSystem,
    local_ref: ActorRef[WorkerMsg],
    broadcast_ref: ActorRef[WorkerMsg],
    num_nodes: int,
) -> web.Application:
    from skyward.infra.serialization import deserialize, serialize

    async def submit_job(request: web.Request) -> web.Response:
        payload_bytes = await request.read()
        payload = deserialize(payload_bytes)
        target_node = payload.get("node_id")
        fn_name = getattr(payload.get("fn"), "__name__", "?")
        print(f"[submit] {fn_name} node_id={target_node}", flush=True)

        fn_payload = {
            "fn": payload["fn"],
            "args": payload.get("args", ()),
            "kwargs": payload.get("kwargs", {}),
        }
        fn_bytes = serialize(fn_payload)

        match target_node:
            case int(nid):
                ref = system.lookup("/worker", node=f"node-{nid}")
                if ref is None:
                    ref = local_ref
            case _:
                ref = local_ref

        response = await system.ask(
            ref,
            lambda reply_to: ExecuteTask(
                fn_bytes=fn_bytes,
                reply_to=reply_to,
            ),
            timeout=600.0,
        )

        match response:
            case TaskFailed(error=error, traceback=tb):
                error_data = {"error": error, "traceback": tb}
                return web.Response(
                    body=serialize(error_data),
                    content_type="application/octet-stream",
                    status=500,
                )
            case TaskSucceeded(result=result):
                return web.Response(body=serialize(result), content_type="application/octet-stream")
            case _:
                return web.Response(status=500, text="Unexpected response")

    async def broadcast_job(request: web.Request) -> web.Response:

        payload_bytes = await request.read()
        payload = deserialize(payload_bytes)

        fn_name = getattr(payload.get("fn"), "__name__", "unknown")
        fn_payload = {
            "fn": payload["fn"],
            "args": payload.get("args", ()),
            "kwargs": payload.get("kwargs", {}),
        }
        fn_bytes = serialize(fn_payload)

        print(f"[broadcast] {fn_name} payload={len(fn_bytes)} bytes, asking...", flush=True)

        responses: tuple[TaskResult, ...] = await system.ask(
            broadcast_ref,
            lambda reply_to: ExecuteTask(
                fn_bytes=fn_bytes,
                reply_to=reply_to,  # type: ignore[reportArgumentType]
            ),
            timeout=600.0,
        )

        node_ids = [r.node_id for r in responses]
        print(
            f"[broadcast] {fn_name} got {len(responses)} responses: {node_ids}",
            flush=True,
        )

        results = [None] * len(responses)
        for resp in responses:
            match resp:
                case TaskFailed(error=error, traceback=tb, node_id=nid):
                    error_data = {
                        "error": f"Node {nid}: {error}",
                        "traceback": tb,
                    }
                    return web.Response(
                        body=serialize(error_data),
                        content_type="application/octet-stream",
                        status=500,
                    )
                case TaskSucceeded(result=result, node_id=nid):
                    if nid < len(results):
                        results[nid] = result

        return web.Response(body=serialize(results), content_type="application/octet-stream")

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ready"})

    app = web.Application()
    app.router.add_post("/jobs", submit_job)
    app.router.add_post("/jobs/broadcast", broadcast_job)
    app.router.add_get("/health", health)
    return app


async def main(
    node_id: int,
    port: int,
    seeds: list[tuple[str, int]] | None,
    http_port: int = 8265,
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

    quorum = num_nodes if num_nodes > 1 else None
    print(f"Casty worker {node_id} starting (quorum={quorum})...", flush=True)

    os.environ["SKYWARD_NODE_ID"] = str(node_id)

    async with ClusteredActorSystem(
        name="skyward",
        host=host,
        port=port,
        node_id=f"node-{node_id}",
        seed_nodes=seeds or [],
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
        _set_active_registry(DistributedRegistry(system, loop=loop))

        local_ref = system.spawn(
            worker_behavior(node_id, concurrency=workers_per_node),
            "worker",
        )
        broadcast_ref = system.spawn(
            Behaviors.broadcasted(
                worker_behavior(node_id, concurrency=workers_per_node),
            ),
            "broadcast-worker",
        )
        print(
            f"Casty worker {node_id} ready "
            f"(cluster={num_nodes} nodes, concurrency={workers_per_node})",
            flush=True,
        )

        if node_id == 0:
            app = create_app(system, local_ref, broadcast_ref, num_nodes)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", http_port)
            await site.start()
            await asyncio.Event().wait()
        else:
            await asyncio.Event().wait()


def _parse_seeds(seeds_str: str | None) -> list[tuple[str, int]] | None:
    if not seeds_str:
        return None
    return [
        (host, int(port_str))
        for addr in seeds_str.split(",")
        for host, port_str in [addr.rsplit(":", 1)]
    ]


def cli() -> None:
    parser = argparse.ArgumentParser(description="Casty worker service")
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--port", type=int, default=25520)
    parser.add_argument("--http-port", type=int, default=8265)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seed addresses (host:port)",
    )
    parser.add_argument("--workers-per-node", type=int, default=1)
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    asyncio.run(main(
        args.node_id, args.port, seeds, args.http_port, args.num_nodes, args.host,
        workers_per_node=args.workers_per_node,
    ))


if __name__ == "__main__":
    cli()
