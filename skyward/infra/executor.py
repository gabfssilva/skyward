"""Executor actor for remote function execution via Casty ClusterClient.

The executor tells this story: connecting → ready → stopped.

In the connecting state, it receives pre-built SSH tunnel information
(address_map) from the pool layer and creates a ClusterClient to discover
and communicate with remote worker actors directly.

SSH tunnels are managed by the pool layer — the executor only consumes them.

Routing is sequential (mailbox guarantees round-robin correctness).
Actor asks are concurrent (create_task + semaphore for backpressure).
The Executor class is a thin facade over the actor for the outside world.
"""

from __future__ import annotations

import asyncio
import logging as _logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from casty import ActorContext, ActorRef, ActorSystem, Behavior, Behaviors, ClusterClient

from skyward.observability.logger import logger

from .worker import (
    WORKER_KEY,
    ExecuteTask,
    TaskFailed,
    TaskResult,
    TaskSucceeded,
    skyward_serializer,
)

_logging.getLogger("casty").setLevel(_logging.DEBUG)

log = logger.bind(component="executor")


# ─── Config ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    system_name: str
    contact_points: tuple[tuple[str, int], ...]
    address_map: dict[tuple[str, int], tuple[str, int]]
    host_to_node: dict[str, int]
    num_nodes: int
    workers_per_node: int = 1
    connect_timeout: float = 120.0
    job_timeout: float = 600.0


# ─── Reply types ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Connected:
    pass


@dataclass(frozen=True, slots=True)
class Executed:
    value: Any


@dataclass(frozen=True, slots=True)
class ExecutionFailed:
    error: str
    traceback: str = ""


@dataclass(frozen=True, slots=True)
class Broadcasted:
    values: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class ClusterReady:
    pass


@dataclass(frozen=True, slots=True)
class Disconnected:
    pass


# ─── Commands ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Connect:
    config: ExecutorConfig
    reply_to: ActorRef[Connected]


@dataclass(frozen=True, slots=True)
class _Execute:
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    node_id: int | None
    reply_to: ActorRef[Executed | ExecutionFailed]


@dataclass(frozen=True, slots=True)
class _Broadcast:
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    reply_to: ActorRef[Broadcasted | ExecutionFailed]


@dataclass(frozen=True, slots=True)
class _SetupCluster:
    env_vars: dict[str, str]
    pool_infos: tuple[str, ...]
    reply_to: ActorRef[ClusterReady]


@dataclass(frozen=True, slots=True)
class _Disconnect:
    reply_to: ActorRef[Disconnected]


@dataclass(frozen=True, slots=True)
class _ExecuteDone:
    result: Executed | ExecutionFailed
    reply_to: ActorRef[Executed | ExecutionFailed]


@dataclass(frozen=True, slots=True)
class _BroadcastDone:
    result: Broadcasted | ExecutionFailed
    reply_to: ActorRef[Broadcasted | ExecutionFailed]


@dataclass(frozen=True, slots=True)
class _SetupDone:
    reply_to: ActorRef[ClusterReady]


@dataclass(frozen=True, slots=True)
class _SetupFailed:
    error: str
    reply_to: ActorRef[ClusterReady]


type ExecutorMsg = (
    _Connect | _Execute | _Broadcast | _SetupCluster | _Disconnect
    | _ExecuteDone | _BroadcastDone | _SetupDone | _SetupFailed
)


# ─── Helpers ─────────────────────────────────────────────────────────


async def _wait_for_workers(
    client: ClusterClient,
    expected: int,
    host_to_node: dict[str, int],
    timeout: float = 240.0,
) -> dict[int, ActorRef[ExecuteTask]]:
    deadline = asyncio.get_event_loop().time() + timeout
    log.debug(
        "Waiting for {n} workers, host_to_node={h2n}",
        n=expected, h2n=host_to_node,
    )
    listing = client.lookup(WORKER_KEY)
    log.debug(
        "Initial lookup: {n} instances found",
        n=len(listing.instances),
    )
    while asyncio.get_event_loop().time() < deadline:
        listing = client.lookup(WORKER_KEY)
        found = len(listing.instances)
        if found >= expected:
            log.debug(
                "Found {found}/{expected} workers, building map...",
                found=found, expected=expected,
            )
            for inst in listing.instances:
                log.debug(
                    "  instance: node=({host}:{port}), ref={ref}",
                    host=inst.node.host, port=inst.node.port,
                    ref=inst.ref,
                )
            return _build_worker_map(listing, host_to_node)

        remaining = deadline - asyncio.get_event_loop().time()

        if remaining < 10:
            log.debug(
                "Only {found}/{expected} workers, retrying ({remaining:.0f}s left)...",
                found=found, expected=expected, remaining=remaining,
            )

        await asyncio.sleep(1.0)
    raise RuntimeError(
        f"Only found {len(listing.instances)}/{expected} workers after {timeout}s"
    )


def _build_worker_map(
    listing: Any, host_to_node: dict[str, int],
) -> dict[int, ActorRef[ExecuteTask]]:
    worker_map: dict[int, ActorRef[ExecuteTask]] = {}
    for instance in listing.instances:
        host = instance.node.host
        nid = host_to_node.get(host)
        log.debug(
            "Mapping host={host} → node_id={nid} (in host_to_node={h2n})",
            host=host, nid=nid, h2n=host_to_node,
        )
        if nid is not None:
            worker_map[nid] = instance.ref
        else:
            log.warning(
                "Host {host} not found in host_to_node, skipping worker",
                host=host,
            )
    result = dict(sorted(worker_map.items()))
    log.debug("Final worker_map: {wm}", wm={k: str(v) for k, v in result.items()})
    return result


# ─── Behavior ────────────────────────────────────────────────────────


def _executor_behavior() -> Behavior[ExecutorMsg]:
    """An executor tells this story: connecting → ready → stopped."""

    def connecting() -> Behavior[ExecutorMsg]:
        async def receive(
            _ctx: ActorContext[ExecutorMsg], msg: ExecutorMsg,
        ) -> Behavior[ExecutorMsg]:
            match msg:
                case _Connect(config=cfg, reply_to=reply_to):
                    log.info(
                        "CONNECT received: system={sys}, contact_points={pts}, "
                        "address_map={am}, host_to_node={h2n}, num_nodes={n}",
                        sys=cfg.system_name, pts=cfg.contact_points,
                        am=cfg.address_map, h2n=cfg.host_to_node, n=cfg.num_nodes,
                    )

                    log.debug("Creating ClusterClient...")
                    client = ClusterClient(
                        contact_points=list(cfg.contact_points),
                        system_name=cfg.system_name,
                        address_map=cfg.address_map,
                        serializer=skyward_serializer(),
                    )
                    log.debug("ClusterClient created, entering async context...")
                    await client.__aenter__()
                    log.info("ClusterClient entered, waiting for topology...")

                    worker_refs = await _wait_for_workers(
                        client, cfg.num_nodes, cfg.host_to_node, cfg.connect_timeout,
                    )
                    log.info(
                        "Discovered {n} workers: {nodes}",
                        n=len(worker_refs), nodes=list(worker_refs.keys()),
                    )

                    log.debug("Telling reply_to Connected()...")
                    reply_to.tell(Connected())
                    max_concurrent = cfg.num_nodes * (cfg.workers_per_node + 3)
                    log.debug(
                        "Transitioning to ready state (max_concurrent={mc}, job_timeout={jt})",
                        mc=max_concurrent, jt=cfg.job_timeout,
                    )
                    return ready(
                        client, worker_refs, cfg.num_nodes,
                        asyncio.Semaphore(max_concurrent), cfg.job_timeout,
                    )
                case other:
                    log.warning(
                        "Unexpected message in connecting state: {msg}",
                        msg=type(other).__name__,
                    )
                    return Behaviors.same()
        return Behaviors.receive(receive)

    async def _do_execute(
        client: ClusterClient,
        sem: asyncio.Semaphore,
        worker_ref: ActorRef[ExecuteTask],
        fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        timeout: float,
    ) -> Executed | ExecutionFailed:
        fn_name = getattr(fn, "__name__", str(fn))
        log.debug(
            "_do_execute: acquiring semaphore, ref={ref}, fn={fn}, timeout={t}",
            ref=worker_ref, fn=fn_name, t=timeout,
        )
        async with sem:
            try:
                log.debug(
                    "_do_execute: calling client.ask(ref={ref}, timeout={t})...",
                    ref=worker_ref, t=timeout,
                )
                result: TaskResult = await client.ask(
                    worker_ref,
                    lambda reply_to: ExecuteTask(
                        fn=fn, args=args, kwargs=kwargs, reply_to=reply_to,
                    ),
                    timeout=timeout,
                )
                log.debug(
                    "_do_execute: client.ask returned: {typ}",
                    typ=type(result).__name__,
                )
                match result:
                    case TaskSucceeded(result=value):
                        log.debug("_do_execute: TaskSucceeded")
                        return Executed(value=value)
                    case TaskFailed(error=error, traceback=tb):
                        log.error("_do_execute: TaskFailed: {err}", err=error)
                        return ExecutionFailed(error=error, traceback=tb)
                    case _:
                        log.error("_do_execute: Unexpected result type: {r}", r=result)
                        return ExecutionFailed(error=f"Unexpected result: {result}")
            except Exception as e:
                log.error("_do_execute: exception: {err}", err=e, exc_info=True)
                return ExecutionFailed(error=str(e))

    async def _do_broadcast(
        client: ClusterClient,
        sem: asyncio.Semaphore,
        worker_refs: dict[int, ActorRef[ExecuteTask]],
        fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        timeout: float,
    ) -> Broadcasted | ExecutionFailed:
        fn_name = getattr(fn, "__name__", str(fn))
        log.debug(
            "_do_broadcast: {n} workers, fn={fn}",
            n=len(worker_refs), fn=fn_name,
        )
        async with sem:
            try:
                tasks = [
                    client.ask(
                        ref,
                        lambda reply_to: ExecuteTask(
                            fn=fn, args=args, kwargs=kwargs, reply_to=reply_to,
                        ),
                        timeout=timeout,
                    )
                    for _nid, ref in sorted(worker_refs.items())
                ]
                log.debug("_do_broadcast: awaiting {n} tasks...", n=len(tasks))
                results: list[TaskResult] = list(await asyncio.gather(*tasks))
                log.debug("_do_broadcast: all tasks returned")

                values: list[Any] = [None] * len(results)
                for resp in results:
                    match resp:
                        case TaskFailed(error=error, traceback=tb, node_id=nid):
                            log.error(
                                "_do_broadcast: node {nid} failed: {err}",
                                nid=nid, err=error,
                            )
                            return ExecutionFailed(
                                error=f"Node {nid}: {error}", traceback=tb,
                            )
                        case TaskSucceeded(result=value, node_id=nid):
                            if nid < len(values):
                                values[nid] = value

                return Broadcasted(values=tuple(values))
            except Exception as e:
                log.error("_do_broadcast: exception: {err}", err=e, exc_info=True)
                return ExecutionFailed(error=str(e))

    async def _do_setup(
        client: ClusterClient,
        worker_refs: dict[int, ActorRef[ExecuteTask]],
        env_vars: dict[str, str],
        pool_infos: tuple[str, ...],
        timeout: float,
    ) -> None:
        log.info(
            "_do_setup: env_vars={n_env} keys, pool_infos={n_pi} entries, "
            "workers={n_w}, timeout={t}",
            n_env=len(env_vars), n_pi=len(pool_infos),
            n_w=len(worker_refs), t=timeout,
        )

        def setup_env(all_infos: list[str], extra: dict[str, str]) -> str:
            import os
            nid = int(os.environ.get("SKYWARD_NODE_ID", "0"))
            if all_infos and nid < len(all_infos):
                os.environ["COMPUTE_POOL"] = all_infos[nid]
            for k, v in extra.items():
                os.environ[k] = v
            return "ok"

        setup_args = (list(pool_infos), env_vars)
        setup_kwargs: dict[str, Any] = {}

        for nid, ref in sorted(worker_refs.items()):
            log.debug(
                "_do_setup: will ask node {nid}, ref={ref}",
                nid=nid, ref=ref,
            )

        tasks = []
        for nid, ref in sorted(worker_refs.items()):
            log.debug("_do_setup: creating ask task for node {nid}...", nid=nid)
            task = client.ask(
                ref,
                lambda reply_to: ExecuteTask(
                    fn=setup_env, args=setup_args, kwargs=setup_kwargs, reply_to=reply_to,
                ),
                timeout=timeout,
            )
            tasks.append(task)

        log.info("_do_setup: awaiting {n} setup tasks...", n=len(tasks))
        try:
            results = await asyncio.gather(*tasks)
            log.info(
                "_do_setup: all tasks completed, results={r}",
                r=[type(r).__name__ for r in results],
            )
        except Exception as e:
            log.error("_do_setup: gather failed: {err}", err=e, exc_info=True)
            raise

    def ready(
        client: ClusterClient,
        worker_refs: dict[int, ActorRef[ExecuteTask]],
        num_nodes: int,
        sem: asyncio.Semaphore,
        job_timeout: float,
        next_node: int = 0,
    ) -> Behavior[ExecutorMsg]:
        async def receive(
            ctx: ActorContext[ExecutorMsg], msg: ExecutorMsg,
        ) -> Behavior[ExecutorMsg]:
            log.debug("ready: received {msg_type}", msg_type=type(msg).__name__)
            match msg:
                case _Execute(fn=fn, args=args, kwargs=kwargs, node_id=nid, reply_to=reply_to):
                    target = nid if nid is not None else next_node % num_nodes
                    ref = worker_refs.get(target)
                    log.debug(
                        "ready._Execute: target={t}, ref={r}, fn={fn}",
                        t=target, r=ref,
                        fn=getattr(fn, "__name__", str(fn)),
                    )
                    if ref is None:
                        log.error("ready._Execute: No worker for node {t}", t=target)
                        reply_to.tell(ExecutionFailed(
                            error=f"No worker found for node {target}",
                        ))
                        return Behaviors.same()

                    ctx.pipe_to_self(
                        coro=_do_execute(client, sem, ref, fn, args, kwargs, job_timeout),
                        mapper=lambda result: _ExecuteDone(result=result, reply_to=reply_to),
                        on_failure=lambda e: _ExecuteDone(
                            result=ExecutionFailed(error=str(e)), reply_to=reply_to,
                        ),
                    )
                    return ready(
                        client, worker_refs, num_nodes, sem,
                        job_timeout, next_node=target + 1,
                    )

                case _ExecuteDone(result=result, reply_to=reply_to):
                    log.debug(
                        "ready._ExecuteDone: result={typ}",
                        typ=type(result).__name__,
                    )
                    reply_to.tell(result)
                    return Behaviors.same()

                case _Broadcast(fn=fn, args=args, kwargs=kwargs, reply_to=reply_to):
                    ctx.pipe_to_self(
                        coro=_do_broadcast(
                            client, sem, worker_refs, fn, args, kwargs, job_timeout,
                        ),
                        mapper=lambda result: _BroadcastDone(result=result, reply_to=reply_to),
                        on_failure=lambda e: _BroadcastDone(
                            result=ExecutionFailed(error=str(e)), reply_to=reply_to,
                        ),
                    )
                    return Behaviors.same()

                case _BroadcastDone(result=result, reply_to=reply_to):
                    log.debug(
                        "ready._BroadcastDone: result={typ}",
                        typ=type(result).__name__,
                    )
                    reply_to.tell(result)
                    return Behaviors.same()

                case _SetupCluster(env_vars=env_vars, pool_infos=pool_infos, reply_to=reply_to):
                    log.info(
                        "ready._SetupCluster: env_vars={n_env}, pool_infos={n_pi}, "
                        "piping _do_setup to self...",
                        n_env=len(env_vars), n_pi=len(pool_infos),
                    )
                    ctx.pipe_to_self(
                        coro=_do_setup(
                            client, worker_refs, env_vars, pool_infos, job_timeout,
                        ),
                        mapper=lambda _: _SetupDone(reply_to=reply_to),
                        on_failure=lambda e: _SetupFailed(error=str(e), reply_to=reply_to),
                    )
                    return Behaviors.same()

                case _SetupDone(reply_to=reply_to):
                    log.info("ready._SetupDone: cluster setup complete")
                    reply_to.tell(ClusterReady())
                    return Behaviors.same()

                case _SetupFailed(error=error, reply_to=reply_to):
                    log.error("ready._SetupFailed: {err}", err=error)
                    reply_to.tell(ClusterReady())
                    return Behaviors.same()

                case _Disconnect(reply_to=reply_to):
                    log.debug("ready._Disconnect: closing ClusterClient...")
                    with suppress(Exception):
                        await client.__aexit__(None, None, None)
                    log.debug("ready._Disconnect: done")
                    reply_to.tell(Disconnected())
                    return Behaviors.stopped()

                case other:
                    log.warning(
                        "ready: unexpected message: {msg}",
                        msg=type(other).__name__,
                    )
                    return Behaviors.same()

        return Behaviors.receive(receive)

    return connecting()


# ─── Executor (facade over the actor) ────────────────────────────────


class Executor:
    def __init__(self, system: ActorSystem, config: ExecutorConfig) -> None:
        self._system = system
        self._config = config
        self._ref: ActorRef[ExecutorMsg] | None = None

    async def connect(self) -> None:
        log.debug("Executor.connect: spawning executor actor...")
        self._ref = self._system.spawn(_executor_behavior(), "executor")
        log.debug("Executor.connect: sending _Connect message...")
        await self._system.ask(
            self._ref,
            lambda reply_to: _Connect(config=self._config, reply_to=reply_to),
            timeout=self._config.connect_timeout,
        )
        log.info("Executor.connect: connected successfully")

    async def execute[T](
        self,
        fn: Callable[..., T],
        *args: Any,
        node_id: int | None = None,
        **kwargs: Any,
    ) -> T:
        assert self._ref is not None
        fn_name = getattr(fn, "__name__", str(fn))
        log.debug(
            "Executor.execute: fn={fn}, node_id={nid}",
            fn=fn_name, nid=node_id,
        )
        result = await self._system.ask(
            self._ref,
            lambda reply_to: _Execute(
                fn=fn, args=args, kwargs=kwargs,
                node_id=node_id, reply_to=reply_to,
            ),
            timeout=self._config.job_timeout,
        )
        log.debug(
            "Executor.execute: result={typ}",
            typ=type(result).__name__,
        )
        match result:
            case Executed(value=v):
                return v
            case ExecutionFailed(error=e, traceback=tb):
                raise RuntimeError(f"Execution error: {e}\n{tb}")
            case _:
                raise RuntimeError(f"Unexpected result: {result}")

    async def broadcast[T](
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> list[T]:
        assert self._ref is not None
        fn_name = getattr(fn, "__name__", str(fn))
        log.debug("Executor.broadcast: fn={fn}", fn=fn_name)
        result = await self._system.ask(
            self._ref,
            lambda reply_to: _Broadcast(
                fn=fn, args=args, kwargs=kwargs,
                reply_to=reply_to,
            ),
            timeout=self._config.job_timeout,
        )
        log.debug(
            "Executor.broadcast: result={typ}",
            typ=type(result).__name__,
        )
        match result:
            case Broadcasted(values=vs):
                return list(vs)
            case ExecutionFailed(error=e, traceback=tb):
                raise RuntimeError(f"Broadcast error: {e}\n{tb}")
            case _:
                raise RuntimeError(f"Unexpected result: {result}")

    async def setup_cluster(self, env_vars: dict[str, str], pool_infos: list[str]) -> None:
        assert self._ref is not None
        log.info(
            "Executor.setup_cluster: env_vars={n_env}, pool_infos={n_pi}",
            n_env=len(env_vars), n_pi=len(pool_infos),
        )
        await self._system.ask(
            self._ref,
            lambda reply_to: _SetupCluster(
                env_vars=env_vars,
                pool_infos=tuple(pool_infos),
                reply_to=reply_to,
            ),
            timeout=60.0,
        )
        log.info("Executor.setup_cluster: completed")

    async def disconnect(self) -> None:
        if self._ref is None:
            return
        log.debug("Executor.disconnect: sending _Disconnect...")
        with suppress(Exception):
            await self._system.ask(
                self._ref,
                lambda reply_to: _Disconnect(reply_to=reply_to),
                timeout=10.0,
            )
        self._ref = None
        log.debug("Executor.disconnect: done")

    @property
    def is_connected(self) -> bool:
        return self._ref is not None
