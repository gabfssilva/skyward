"""Executor actor for remote function execution via Casty.

The executor tells this story: connecting → ready → stopped.

Routing is sequential (mailbox guarantees round-robin correctness).
HTTP calls are concurrent (create_task + semaphore for backpressure).
The Executor class is a thin facade over the actor for the outside world.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import asyncssh
import aiohttp
from casty import ActorContext, ActorRef, ActorSystem, Behavior, Behaviors
from loguru import logger

from .serialization import deserialize, serialize

CASTY_PORT = 25520
HTTP_PORT = 8265


# ─── Config ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    head_ip: str
    user: str
    key_path: str
    num_nodes: int
    workers_per_node: int = 1
    ssh_port: int = 22
    http_port: int = HTTP_PORT
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


# ─── Behavior ────────────────────────────────────────────────────────


async def _wait_for_ready(session: aiohttp.ClientSession, host: str, port: int) -> None:
    for attempt in range(30):
        try:
            async with session.get("/health") as resp:
                if resp.status == 200:
                    logger.debug("Casty HTTP API is ready")
                    return
        except (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError):
            pass
        if attempt < 29:
            logger.debug(f"Casty HTTP not ready (attempt {attempt + 1}/30)")
            await asyncio.sleep(2.0)
    raise RuntimeError(f"Casty HTTP API not ready after 60s at {host}:{port}")


def _executor_behavior() -> Behavior[ExecutorMsg]:
    """An executor tells this story: connecting → ready → stopped."""

    def connecting() -> Behavior[ExecutorMsg]:
        async def receive(ctx: ActorContext[ExecutorMsg], msg: ExecutorMsg) -> Behavior[ExecutorMsg]:
            match msg:
                case _Connect(config=cfg, reply_to=reply_to):
                    ssh_conn = await asyncssh.connect(
                        cfg.head_ip,
                        port=cfg.ssh_port,
                        username=cfg.user,
                        client_keys=[cfg.key_path],
                        known_hosts=None,
                        connect_timeout=cfg.connect_timeout,
                        keepalive_interval=15,
                        keepalive_count_max=4,
                    )
                    tunnel = await ssh_conn.forward_local_port(
                        "", 0, "127.0.0.1", cfg.http_port,
                    )
                    session = aiohttp.ClientSession(
                        base_url=f"http://127.0.0.1:{tunnel.get_port()}",
                        timeout=aiohttp.ClientTimeout(total=cfg.job_timeout, connect=30.0),
                    )
                    await _wait_for_ready(session, cfg.head_ip, cfg.http_port)
                    logger.info(f"Connected to Casty at {cfg.head_ip}:{cfg.http_port}")
                    reply_to.tell(Connected())
                    max_concurrent = cfg.num_nodes * (cfg.workers_per_node + 1)
                    return ready(ssh_conn, tunnel, session, cfg.num_nodes, asyncio.Semaphore(max_concurrent))
                case _:
                    return Behaviors.same()
        return Behaviors.receive(receive)

    async def _do_execute(
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: int,
    ) -> Executed | ExecutionFailed:
        fn_name = getattr(fn, "__name__", str(fn))
        async with sem:
            try:
                payload = serialize({"fn": fn, "args": args, "kwargs": kwargs, "node_id": target})
                async with session.post(
                    "/jobs", data=payload,
                    headers={"Content-Type": "application/octet-stream"},
                ) as resp:
                    result = deserialize(await resp.read())
                    match resp.status:
                        case 200:
                            return Executed(value=result)
                        case _:
                            match result:
                                case dict():
                                    error = result.get("error", "Unknown")
                                    tb = result.get("traceback", "")
                                case _:
                                    error = str(result)
                                    tb = ""
                            return ExecutionFailed(error=error, traceback=tb)
            except Exception as e:
                logger.error(f"execute({fn_name}) failed: {e}")
                return ExecutionFailed(error=str(e))

    async def _do_broadcast(
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        num_nodes: int,
    ) -> Broadcasted | ExecutionFailed:
        fn_name = getattr(fn, "__name__", str(fn))
        async with sem:
            try:
                payload = serialize({"fn": fn, "args": args, "kwargs": kwargs, "num_nodes": num_nodes})
                async with session.post(
                    "/jobs/broadcast", data=payload,
                    headers={"Content-Type": "application/octet-stream"},
                ) as resp:
                    result = deserialize(await resp.read())
                    match resp.status:
                        case 200:
                            return Broadcasted(values=tuple(result))
                        case _:
                            match result:
                                case dict():
                                    error = result.get("error", "Unknown")
                                    tb = result.get("traceback", "")
                                case _:
                                    error = str(result)
                                    tb = ""
                            return ExecutionFailed(error=error, traceback=tb)
            except Exception as e:
                logger.error(f"broadcast({fn_name}) failed: {e}")
                return ExecutionFailed(error=str(e))

    async def _do_setup(
        session: aiohttp.ClientSession,
        env_vars: dict[str, str],
        pool_infos: tuple[str, ...],
        num_nodes: int,
    ) -> None:
        def setup_env(all_infos: list[str], extra: dict[str, str]) -> str:
            import os
            nid = int(os.environ.get("SKYWARD_NODE_ID", "0"))
            if all_infos and nid < len(all_infos):
                os.environ["COMPUTE_POOL"] = all_infos[nid]
            for k, v in extra.items():
                os.environ[k] = v
            return "ok"

        payload = serialize({
            "fn": setup_env,
            "args": (list(pool_infos), env_vars),
            "kwargs": {},
            "num_nodes": num_nodes,
        })
        async with session.post(
            "/jobs/broadcast", data=payload,
            headers={"Content-Type": "application/octet-stream"},
        ) as resp:
            deserialize(await resp.read())

    def ready(
        ssh_conn: asyncssh.SSHClientConnection,
        tunnel: Any,
        session: aiohttp.ClientSession,
        num_nodes: int,
        sem: asyncio.Semaphore,
        next_node: int = 0,
    ) -> Behavior[ExecutorMsg]:
        async def receive(ctx: ActorContext[ExecutorMsg], msg: ExecutorMsg) -> Behavior[ExecutorMsg]:
            match msg:
                case _Execute(fn=fn, args=args, kwargs=kwargs, node_id=nid, reply_to=reply_to):
                    target = nid if nid is not None else next_node % num_nodes
                    ctx.pipe_to_self(
                        coro=_do_execute(session, sem, fn, args, kwargs, target),
                        mapper=lambda result: _ExecuteDone(result=result, reply_to=reply_to),
                        on_failure=lambda e: _ExecuteDone(
                            result=ExecutionFailed(error=str(e)), reply_to=reply_to,
                        ),
                    )
                    return ready(ssh_conn, tunnel, session, num_nodes, sem, next_node=target + 1)

                case _ExecuteDone(result=result, reply_to=reply_to):
                    reply_to.tell(result)
                    return Behaviors.same()

                case _Broadcast(fn=fn, args=args, kwargs=kwargs, reply_to=reply_to):
                    ctx.pipe_to_self(
                        coro=_do_broadcast(session, sem, fn, args, kwargs, num_nodes),
                        mapper=lambda result: _BroadcastDone(result=result, reply_to=reply_to),
                        on_failure=lambda e: _BroadcastDone(
                            result=ExecutionFailed(error=str(e)), reply_to=reply_to,
                        ),
                    )
                    return Behaviors.same()

                case _BroadcastDone(result=result, reply_to=reply_to):
                    reply_to.tell(result)
                    return Behaviors.same()

                case _SetupCluster(env_vars=env_vars, pool_infos=pool_infos, reply_to=reply_to):
                    ctx.pipe_to_self(
                        coro=_do_setup(session, env_vars, pool_infos, num_nodes),
                        mapper=lambda _: _SetupDone(reply_to=reply_to),
                        on_failure=lambda e: _SetupFailed(error=str(e), reply_to=reply_to),
                    )
                    return Behaviors.same()

                case _SetupDone(reply_to=reply_to):
                    reply_to.tell(ClusterReady())
                    return Behaviors.same()

                case _SetupFailed(error=error, reply_to=reply_to):
                    logger.error(f"Cluster setup failed: {error}")
                    reply_to.tell(ClusterReady())
                    return Behaviors.same()

                case _Disconnect(reply_to=reply_to):
                    with suppress(Exception):
                        await session.close()
                    with suppress(Exception):
                        tunnel.close()
                    with suppress(Exception):
                        ssh_conn.close()
                        await asyncio.wait_for(ssh_conn.wait_closed(), timeout=5.0)
                    logger.debug("Executor disconnected")
                    reply_to.tell(Disconnected())
                    return Behaviors.stopped()

                case _:
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
        self._ref = self._system.spawn(_executor_behavior(), "executor")
        await self._system.ask(
            self._ref,
            lambda reply_to: _Connect(config=self._config, reply_to=reply_to),
            timeout=self._config.connect_timeout,
        )

    async def execute[T](
        self,
        fn: Callable[..., T],
        *args: Any,
        node_id: int | None = None,
        **kwargs: Any,
    ) -> T:
        assert self._ref is not None
        result = await self._system.ask(
            self._ref,
            lambda reply_to: _Execute(
                fn=fn, args=args, kwargs=kwargs,
                node_id=node_id, reply_to=reply_to,
            ),
            timeout=self._config.job_timeout,
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
        result = await self._system.ask(
            self._ref,
            lambda reply_to: _Broadcast(
                fn=fn, args=args, kwargs=kwargs,
                reply_to=reply_to,
            ),
            timeout=self._config.job_timeout,
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
        await self._system.ask(
            self._ref,
            lambda reply_to: _SetupCluster(
                env_vars=env_vars,
                pool_infos=tuple(pool_infos),
                reply_to=reply_to,
            ),
            timeout=60.0,
        )

    async def disconnect(self) -> None:
        if self._ref is None:
            return
        with suppress(Exception):
            await self._system.ask(
                self._ref,
                lambda reply_to: _Disconnect(reply_to=reply_to),
                timeout=10.0,
            )
        self._ref = None

    @property
    def is_connected(self) -> bool:
        return self._ref is not None
