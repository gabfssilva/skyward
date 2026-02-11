"""Synchronous facade for v2 async pool.

This module provides the user-facing synchronous API that mirrors v1:

    import skyward as sky

    @sky.compute
    def train(data):
        return model.fit(data)

    @sky.pool(provider=sky.AWS(), accelerator="A100", nodes=4)
    def main():
        result = train(data) >> sky          # execute on one node
        results = train(data) @ sky          # broadcast to all nodes
        a, b = (task1() & task2()) >> sky    # parallel execution

Internally, this facade:
1. Runs v2's async provisioning in a sync context
2. Manages the asyncio event loop lifecycle
3. Provides >> and @ operator support via the sky singleton
"""

from __future__ import annotations

import asyncio
import functools
import threading
import types
from collections.abc import Callable, Coroutine, Iterator, Sequence
from concurrent.futures import Future
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal, overload

from casty import ActorRef, ActorSystem, Behaviors, CastyConfig
from loguru import logger

from skyward.accelerators import Accelerator
from skyward.distributed import (
    BarrierProxy,
    CounterProxy,
    DictProxy,
    DistributedRegistry,
    LockProxy,
    QueueProxy,
    SetProxy,
    _set_active_registry,
)
from skyward.distributed.types import Consistency
from skyward.infra.executor import HTTP_PORT, Executor, ExecutorConfig
from skyward.observability.logging import LogConfig, _setup_logging, _teardown_logging
from skyward.providers.aws.config import AWS
from skyward.providers.runpod.config import RunPod
from skyward.providers.vastai.config import VastAI
from skyward.providers.verda.config import Verda

from .spec import DEFAULT_IMAGE, Image, PoolSpec

type Provider = AWS | RunPod | VastAI | Verda


_active_pool: ContextVar[ComputePool | None] = ContextVar("active_pool", default=None)


def _get_active_pool() -> ComputePool:
    """Get the active pool from context."""
    pool = _active_pool.get()
    if pool is None:
        raise RuntimeError(
            "No active pool. Use within a @pool decorated function or 'with pool(...):' block."
        )
    return pool


class _Sky:
    """Singleton that captures >> and @ operators.

    This allows the v1-style API:
        result = compute_fn(args) >> sky   # execute on one node
        results = compute_fn(args) @ sky   # broadcast to all nodes
    """

    def __rrshift__(self, pending: PendingCompute[Any] | PendingComputeGroup) -> Any:
        """pending >> sky - execute computation(s)."""
        pool = _get_active_pool()
        match pending:
            case PendingComputeGroup():
                return pool.run_parallel(pending)
            case _:
                return pool.run(pending)

    def __rmatmul__(self, pending: PendingCompute[Any]) -> list[Any]:
        """pending @ sky - broadcast to all nodes."""
        pool = _get_active_pool()
        return pool.broadcast(pending)

    def __repr__(self) -> str:
        pool = _active_pool.get()
        if pool:
            return f"<sky: active pool with {pool.nodes} nodes>"
        return "<sky: no active pool>"


sky = _Sky()


@dataclass(frozen=True, slots=True)
class PendingCompute[T]:
    """Lazy computation wrapper.

    Represents a function call that will be executed remotely
    when sent to a pool via the >> or @ operator.

    Example:
        @compute
        def train(data):
            return model.fit(data)

        pending = train(data)  # Returns PendingCompute, doesn't execute
        result = pending >> sky  # Executes remotely on pool
        results = pending @ sky  # Broadcasts to all nodes
    """

    fn: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __rshift__(self, target: ComputePool | _Sky | types.ModuleType) -> T:
        match target:
            case types.ModuleType() if hasattr(target, "sky"):
                return target.sky.__rrshift__(self)  # type: ignore
            case _Sky():
                return target.__rrshift__(self)  # type: ignore
            case _:
                return target.run(self)  # type: ignore[union-attr]

    def __gt__(self, target: ComputePool | _Sky | types.ModuleType) -> Future[T]:
        match target:
            case types.ModuleType() if hasattr(target, "sky"):
                return target.sky._run_async(self)  # type: ignore
            case _Sky():
                return target._run_async(self)  # type: ignore
            case _:
                return target.run_async(self)  # type: ignore[union-attr]

    def __matmul__(self, target: ComputePool | _Sky | types.ModuleType) -> list[T] | tuple[T, ...]:
        """Broadcast to all nodes using @ operator."""
        match target:
            case types.ModuleType() if hasattr(target, "sky"):
                return target.sky.__rmatmul__(self)
            case _Sky():
                return target.__rmatmul__(self)
            case _:
                return target.broadcast(self)  # type: ignore[union-attr]

    def __and__(self, other: PendingCompute[Any] | PendingComputeGroup) -> PendingComputeGroup:
        """Combine with another computation for parallel execution."""
        match other:
            case PendingComputeGroup():
                return PendingComputeGroup(items=(self, *other.items))
            case _:
                return PendingComputeGroup(items=(self, other))


@dataclass(frozen=True, slots=True)
class PendingComputeGroup:
    """Group of computations for parallel execution.

    Created by using the & operator:
        group = task1() & task2() & task3()
        a, b, c = group >> sky

    Or using gather():
        group = gather(task1(), task2(), task3())
        results = group >> sky
    """

    items: tuple[PendingCompute[Any], ...]

    def __and__(self, other: PendingCompute[Any] | PendingComputeGroup) -> PendingComputeGroup:
        """Add another computation to the group."""
        match other:
            case PendingComputeGroup():
                return PendingComputeGroup(items=(*self.items, *other.items))
            case _:
                return PendingComputeGroup(items=(*self.items, other))

    def __rshift__(self, target: ComputePool | _Sky | types.ModuleType) -> tuple[Any, ...]:
        """Execute all computations in parallel using >> operator."""
        match target:
            case types.ModuleType() if hasattr(target, "sky"):
                return target.sky.__rrshift__(self)  # type: ignore
            case _Sky():
                return target.__rrshift__(self)  # type: ignore
            case _:
                return target.run_parallel(self)  # type: ignore[union-attr]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[PendingCompute[Any]]:
        return iter(self.items)


def gather(*pendings: PendingCompute[Any]) -> PendingComputeGroup:
    """Group computations for parallel execution.

    Example:
        results = gather(task1(), task2(), task3()) >> sky
        # results is a tuple of (result1, result2, result3)
    """
    return PendingComputeGroup(items=pendings)


def compute(fn: Callable[..., Any]) -> Callable[..., PendingCompute[Any]]:
    """Decorator to make a function lazy.

    The decorated function returns a PendingCompute instead of
    executing immediately. Use >> or @ to send it to a pool.

    Example:
        @compute
        def train(model, data):
            return model.fit(data)

        result = train(my_model, my_data) >> sky  # execute on one node
        results = train(my_model, my_data) @ sky  # broadcast to all
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> PendingCompute[Any]:
        return PendingCompute(fn=fn, args=args, kwargs=kwargs)

    return wrapper


@dataclass
class ComputePool:
    """Synchronous ComputePool facade.

    Wraps v2's async ComputePool with a synchronous API. Uses a
    dedicated background thread for the asyncio event loop.

    Args:
        provider: Cloud provider configuration (AWS, etc.).
        nodes: Number of nodes to provision.
        accelerator: GPU/accelerator type.
        image: Environment specification.
        region: Cloud region.
        allocation: Instance allocation strategy.
        timeout: Provisioning timeout in seconds.

    Example:
        with pool(provider=AWS(), accelerator="A100") as p:
            result = train(data) >> p

        # Or with decorator:
        @pool(provider=AWS(), accelerator="A100")
        def main():
            return train(data) >> sky
    """

    provider: Provider

    nodes: int = 1
    accelerator: str | Accelerator | None = None
    vcpus: int | None = None
    memory_gb: int | None = None
    architecture: Literal["x86_64", "arm64"] | None = None
    allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available"

    image: Image = field(default_factory=lambda: DEFAULT_IMAGE)
    timeout: int = 180
    ttl: int = 600

    concurrency: int = 1

    panel: bool = True

    logging: LogConfig | bool = True

    max_hourly_cost: float | None = None

    _log_handler_ids: list[int] = field(default_factory=list, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _loop_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    _active: bool = field(default=False, init=False, repr=False)
    _context_token: Any = field(default=None, init=False, repr=False)
    _registry: DistributedRegistry | None = field(default=None, init=False, repr=False)
    _system: ActorSystem | None = field(default=None, init=False, repr=False)
    _pool_ref: Any = field(default=None, init=False, repr=False)
    _cluster_id: str = field(default="", init=False, repr=False)
    _instances: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _spec: PoolSpec | None = field(default=None, init=False, repr=False)
    _executor: Executor | None = field(default=None, init=False, repr=False)
    _network_interfaces: dict[int, str] = field(default_factory=dict, init=False, repr=False)

    def __enter__(self) -> ComputePool:
        """Start pool and provision resources."""
        if self.logging:
            match self.logging:
                case True:
                    log_config = LogConfig(console=not self.panel)
                case _:
                    log_config = LogConfig(
                        level=self.logging.level,
                        file=self.logging.file,
                        console=self.logging.console and not self.panel,
                        rotation=self.logging.rotation,
                        retention=self.logging.retention,
                    )
            self._log_handler_ids = _setup_logging(log_config)

        logger.info(f"Starting pool with {self.nodes} nodes ({self.accelerator})")

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="skyward-event-loop",
        )
        self._loop_thread.start()

        try:
            self._run_sync(self._start_async())
            self._active = True
            self._context_token = _active_pool.set(self)
            logger.info("Pool ready")
        except Exception as e:
            logger.exception(f"Error starting pool: {e}")
            self._cleanup()
            raise

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop pool and release resources."""
        logger.info("Stopping pool...")
        try:
            if self._context_token is not None:
                _active_pool.reset(self._context_token)
                self._context_token = None

            if self._active:
                self._run_sync_with_timeout(self._stop_async(), timeout=30.0)
        except TimeoutError:
            logger.warning("Pool stop timed out after 30s, forcing cleanup")
        except Exception as e:
            logger.warning(f"Error stopping pool: {e}")
        finally:
            if self._registry is not None:
                self._registry.cleanup()
                _set_active_registry(None)
                self._registry = None

            self._active = False
            self._cleanup()
            logger.info("Pool stopped")

            if self._log_handler_ids:
                _teardown_logging(self._log_handler_ids)

    def run[T](self, pending: PendingCompute[T]) -> T:
        if not self._active or self._executor is None:
            raise RuntimeError("Pool is not active")

        return self._run_sync(
            self._executor.execute(
                pending.fn,
                *pending.args,
                **pending.kwargs,
            )
        )

    def run_async[T](self, pending: PendingCompute[T]) -> Future[T]:
        if not self._active or self._executor is None or self._loop is None:
            raise RuntimeError("Pool is not active")

        return asyncio.run_coroutine_threadsafe(
            self._executor.execute(pending.fn, *pending.args, **pending.kwargs),
            self._loop,
        )

    def broadcast[T](self, pending: PendingCompute[T]) -> list[T]:
        """Execute computation on all nodes."""
        if not self._active or self._executor is None:
            raise RuntimeError("Pool is not active")

        return self._run_sync(
            self._executor.broadcast(
                pending.fn,
                *pending.args,
                **pending.kwargs,
            )
        )

    def run_parallel(self, group: PendingComputeGroup) -> tuple[Any, ...]:
        """Execute multiple computations in parallel."""
        if not self._active or self._executor is None:
            raise RuntimeError("Pool is not active")

        async def _run_parallel() -> tuple[Any, ...]:
            tasks = [
                self._executor.execute(p.fn, *p.args, **p.kwargs)  # type: ignore
                for p in group.items
            ]
            results = await asyncio.gather(*tasks)
            return tuple(results)

        return self._run_sync(_run_parallel())

    def map[T, R](
        self,
        fn: Callable[[T], R],
        items: Sequence[T],
    ) -> list[R]:
        """Map function over items in parallel across nodes."""
        if not self._active or self._executor is None:
            raise RuntimeError("Pool is not active")

        async def _map_async() -> list[R]:
            tasks = [
                self._executor.execute(fn, item)  # type: ignore
                for item in items
            ]
            return list(await asyncio.gather(*tasks))

        return self._run_sync(_map_async())

    def dict(self, name: str, *, consistency: Consistency | None = None) -> DictProxy:
        """Get or create a distributed dict."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        return self._registry.dict(name, consistency=consistency)

    def set(self, name: str, *, consistency: Consistency | None = None) -> SetProxy:
        """Get or create a distributed set."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        return self._registry.set(name, consistency=consistency)

    def counter(self, name: str, *, consistency: Consistency | None = None) -> CounterProxy:
        """Get or create a distributed counter."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        return self._registry.counter(name, consistency=consistency)

    def queue(self, name: str) -> QueueProxy:
        """Get or create a distributed queue."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        return self._registry.queue(name)

    def barrier(self, name: str, n: int) -> BarrierProxy:
        """Get or create a distributed barrier."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        return self._registry.barrier(name, n)

    def lock(self, name: str) -> LockProxy:
        """Get or create a distributed lock."""
        if self._registry is None:
            raise RuntimeError("Pool is not active")
        return self._registry.lock(name)

    async def _start_async(self) -> None:
        """Start pool asynchronously using actors (zero bus)."""
        from skyward.actors.panel import panel_actor
        from skyward.actors.pool import PoolMsg, PoolStarted, StartPool, pool_actor
        from skyward.providers.registry import get_provider_for_config

        actor_factory, provider_name = get_provider_for_config(self.provider)

        region = getattr(self.provider, "region", "unknown")

        spec = PoolSpec(
            nodes=self.nodes,
            accelerator=self.accelerator,
            region=region,
            vcpus=self.vcpus,
            memory_gb=self.memory_gb,
            architecture=self.architecture,
            allocation=self.allocation,
            image=self.image,
            ttl=self.ttl,
            concurrency=self.concurrency,
            provider=provider_name,  # type: ignore[arg-type]
            max_hourly_cost=self.max_hourly_cost,
        )
        self._spec = spec

        self._system = ActorSystem("skyward", config=CastyConfig(
            suppress_dead_letters_on_shutdown=True
        ))

        await self._system.__aenter__()

        panel_ref = (
            self._system.spawn(panel_actor(spec), "panel")
            if self.panel
            else None
        )

        pool_behavior = pool_actor()
        spy = panel_ref is not None
        pool_ref: ActorRef[PoolMsg] = self._system.spawn(
            Behaviors.spy(pool_behavior, panel_ref, spy_children=True)
            if spy
            else pool_behavior,
            "pool",
        )
        self._pool_ref = pool_ref

        provider_behavior = actor_factory(self.provider, pool_ref)
        provider_ref = self._system.spawn(
            Behaviors.spy(provider_behavior, panel_ref, spy_children=True)
            if spy
            else provider_behavior,
            "provider",
        )

        started: PoolStarted = await self._system.ask(
            pool_ref,
            lambda reply_to: StartPool(
                spec=spec,
                provider_config=self.provider,
                provider_ref=provider_ref,
                reply_to=reply_to,
            ),
            timeout=float(self.timeout),
        )
        self._cluster_id = started.cluster_id
        self._instances = {inst.node: inst for inst in started.instances}

        await self._create_executor()

    async def _stop_async(self) -> None:
        """Stop pool asynchronously."""
        if self._executor is not None:
            logger.debug("Disconnecting executor...")
            await self._executor.disconnect()
            self._executor = None

        if self._pool_ref is not None and self._system is not None:
            from skyward.actors.pool import StopPool
            await self._system.ask(
                self._pool_ref,
                lambda reply_to: StopPool(reply_to=reply_to),
                timeout=30.0,
            )

        if self._system is not None:
            await self._system.__aexit__(None, None, None)
            self._system = None

    async def _create_executor(self) -> None:
        from skyward.providers.ssh_keys import get_ssh_key_path

        head_info = self._instances.get(0)
        if head_info is None:
            raise RuntimeError("Head node (node 0) not available")

        head_addr = head_info.private_ip or head_info.ip
        ssh_user = self._get_ssh_user()
        ssh_key = get_ssh_key_path()

        await self._start_casty_cluster(head_addr, ssh_user, ssh_key)

        pool_infos = [
            self._build_pool_info(info, head_addr).model_dump_json()
            if (info := self._instances.get(node_id)) is not None
            else ""
            for node_id in range(self.nodes)
        ]

        image_env = dict(self.image.env) if self.image and self.image.env else {}
        config = ExecutorConfig(
            head_ip=head_info.ip,
            ssh_port=head_info.ssh_port,
            user=ssh_user,
            key_path=ssh_key,
            num_nodes=self.nodes,
            workers_per_node=self.concurrency,
            http_port=HTTP_PORT,
        )
        self._executor = Executor(self._system, config)  # type: ignore[arg-type]
        await self._executor.connect()
        await self._executor.setup_cluster(env_vars=image_env, pool_infos=pool_infos)
        logger.debug("Executor connected and ready")

    async def _start_casty_cluster(
        self,
        head_addr: str,
        ssh_user: str,
        ssh_key: str,
    ) -> None:
        from skyward.infra.ssh import SSHTransport
        from skyward.providers.bootstrap import EMIT_SH_PATH
        from skyward.providers.bootstrap.compose import SKYWARD_DIR
        from skyward.providers.common import detect_network_interface

        venv_dir = f"{SKYWARD_DIR}/.venv"
        python_bin = f"{venv_dir}/bin/python"
        casty_port = 25520
        http_port = HTTP_PORT
        logger.debug("Starting Casty cluster via SSH...")

        head_info = self._instances.get(0)
        if head_info is None:
            raise RuntimeError("Head node not available")

        use_sudo = ssh_user != "root"

        async def _start_node(
            transport: SSHTransport,
            node_id: int,
            host: str,
            seeds: str = "",
        ) -> str:
            seeds_arg = f"--seeds {seeds} " if seeds else ""
            casty_cmd = (
                f"nohup {python_bin} -m skyward.infra.worker "
                f"--node-id {node_id} --port {casty_port} --http-port {http_port} "
                f"--num-nodes {self.nodes} --host {host} "
                f"--workers-per-node {self.concurrency} "
                f"{seeds_arg}"
                f"> /var/log/casty.log 2>&1 & echo $!"
            )
            tail_inner = (
                f"source {EMIT_SH_PATH} && "
                f"tail -f /var/log/casty.log 2>/dev/null | while IFS= read -r line; do "
                f'emit_console "[casty] $line"; done'
            )
            tail_cmd = f"nohup bash -c '{tail_inner}' </dev/null >/dev/null 2>&1 &"

            if use_sudo:
                casty_cmd = f"sudo bash -c '{casty_cmd}'"
                tail_cmd = f"sudo {tail_cmd}"

            exit_code, stdout, stderr = await transport.run(casty_cmd, timeout=60.0)
            if exit_code != 0:
                raise RuntimeError(f"Failed to start Casty node {node_id}: {stderr}")
            await transport.run(tail_cmd, timeout=10.0)
            return stdout.strip()

        async with SSHTransport(
            host=head_info.ip,
            user=ssh_user,
            key_path=ssh_key,
            port=head_info.ssh_port,
        ) as transport:
            logger.debug("Starting Casty head on node 0...")
            pid = await _start_node(transport, 0, host=head_addr)
            logger.debug(f"Casty head PID: {pid}")
            self._network_interfaces[0] = await detect_network_interface(transport)

        async def _start_worker(node_id: int, info: Any) -> None:
            async with SSHTransport(
                host=info.ip,
                user=ssh_user,
                key_path=ssh_key,
                port=info.ssh_port,
            ) as transport:
                worker_addr = info.private_ip or info.ip
                logger.debug(f"Starting Casty worker on node {node_id}...")
                seeds = f"{head_addr}:{casty_port}"
                await _start_node(
                    transport, node_id, host=worker_addr, seeds=seeds
                )
                self._network_interfaces[node_id] = await detect_network_interface(transport)

        await asyncio.gather(*(
            _start_worker(node_id, info)
            for node_id, info in self._instances.items()
            if node_id != 0
        ))

        logger.debug("Casty cluster started")

    def _build_pool_info(self, info: Any, head_addr: str) -> Any:
        from skyward.providers.pool_info import build_pool_info

        peers = [
            {"node": nid, "private_ip": ni.private_ip or ni.ip}
            for nid, ni in self._instances.items()
        ]

        accelerator_count = info.gpu_count or 1
        total_accelerators = sum(
            (ni.gpu_count or 1) for ni in self._instances.values()
        )

        accelerator_name = None
        if self._spec:
            accelerator_name = self._spec.accelerator_name

        return build_pool_info(
            node=info.node,
            total_nodes=self.nodes,
            accelerator_count=accelerator_count,
            total_accelerators=total_accelerators,
            head_addr=head_addr,
            head_port=29500,
            job_id=self._cluster_id,
            peers=peers,
            accelerator_type=accelerator_name,
            placement_group=(
                self._network_interfaces.get(info.node)
                or info.network_interface
                or None
            ),
            worker=0,
            workers_per_node=1,
        )

    def _get_ssh_user(self) -> str:
        provider_name = getattr(self._spec, "provider", None) if self._spec else None
        match provider_name:
            case "verda" | "vastai" | "runpod":
                return "root"
            case _:
                return "ubuntu"

    def _run_loop(self) -> None:
        """Run event loop in background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()  # type: ignore

    def _run_sync[T](self, coro: Coroutine[Any, Any, T]) -> T:
        """Run coroutine synchronously."""
        if self._loop is None:
            raise RuntimeError("Event loop not running")

        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def _run_sync_with_timeout[T](self, coro: Coroutine[Any, Any, T], timeout: float) -> T:
        """Run coroutine synchronously with timeout."""
        if self._loop is None:
            raise RuntimeError("Event loop not running")

        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def _cleanup(self) -> None:
        """Cleanup resources."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread:
                self._loop_thread.join(timeout=5)
            with suppress(Exception):
                self._loop.close()
            self._loop = None
            self._loop_thread = None

    @property
    def is_active(self) -> bool:
        """True if pool is ready for execution."""
        return self._active

    def __repr__(self) -> str:
        status = "active" if self._active else "inactive"
        return f"ComputePool(nodes={self.nodes}, accelerator={self.accelerator}, {status})"


class _PoolFactory:
    """Factory that can be used as context manager or decorator."""

    def __init__(
        self,
        provider: Provider | None = None,
        nodes: int = 1,
        accelerator: str | Accelerator | None = None,
        vcpus: int | None = None,
        memory_gb: int | None = None,
        architecture: Literal["x86_64", "arm64"] | None = None,
        image: Image | None = None,
        allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available",
        timeout: int = 180,
        ttl: int = 600,
        panel: bool = True,
        logging: LogConfig | bool = True,
        max_hourly_cost: float | None = None,
    ) -> None:
        self._provider = provider
        self._nodes = nodes
        self._accelerator = accelerator
        self._vcpus = vcpus
        self._memory_gb = memory_gb
        self._architecture = architecture
        self._image = image
        self._allocation = allocation
        self._timeout = timeout
        self._ttl = ttl
        self._panel = panel
        self._logging = logging
        self._max_hourly_cost = max_hourly_cost
        self._pool: ComputePool | None = None

    def _create_pool(self) -> ComputePool:
        """Create the underlying pool."""
        provider = self._provider or AWS()
        return ComputePool(
            provider=provider,
            nodes=self._nodes,
            accelerator=self._accelerator,
            vcpus=self._vcpus,
            memory_gb=self._memory_gb,
            architecture=self._architecture,  # type: ignore[arg-type]
            allocation=self._allocation,  # type: ignore[arg-type]
            image=self._image or DEFAULT_IMAGE,
            timeout=self._timeout,
            ttl=self._ttl,
            panel=self._panel,
            logging=self._logging,
            max_hourly_cost=self._max_hourly_cost,
        )

    def __enter__(self) -> ComputePool:
        """Use as context manager."""
        self._pool = self._create_pool()
        return self._pool.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        if self._pool:
            self._pool.__exit__(exc_type, exc_val, exc_tb)
            self._pool = None

    def __call__[F: Callable[..., Any]](self, fn: F) -> F:
        """Use as decorator."""

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            p = self._create_pool()
            with p:
                return fn(*args, **kwargs)

        return wrapper  # type: ignore


@overload
def pool(
    provider: Callable[..., Any],
) -> Callable[..., Any]: ...


@overload
def pool(
    *,
    provider: Provider | None = None,
    nodes: int = 1,
    accelerator: str | Accelerator | None = None,
    vcpus: int | None = None,
    memory_gb: int | None = None,
    architecture: Literal["x86_64", "arm64"] | None = None,
    image: Image | None = None,
    allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available",
    timeout: int = 180,
    ttl: int = 600,
    panel: bool = True,
    logging: LogConfig | bool = True,
    max_hourly_cost: float | None = None,
) -> _PoolFactory: ...


@overload
def pool(
    provider: Provider | None = None,
    nodes: int = 1,
    accelerator: str | Accelerator | None = None,
    vcpus: int | None = None,
    memory_gb: int | None = None,
    architecture: Literal["x86_64", "arm64"] | None = None,
    image: Image | None = None,
    allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available",
    timeout: int = 180,
    ttl: int = 600,
    panel: bool = True,
    logging: LogConfig | bool = True,
    max_hourly_cost: float | None = None,
) -> _PoolFactory: ...


def pool(
    provider: Provider | Callable[..., Any] | None = None,
    nodes: int = 1,
    accelerator: str | Accelerator | None = None,
    vcpus: int | None = None,
    memory_gb: int | None = None,
    architecture: Literal["x86_64", "arm64"] | None = None,
    image: Image | None = None,
    allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available",
    timeout: int = 180,
    ttl: int = 600,
    panel: bool = True,
    logging: LogConfig | bool = True,
    max_hourly_cost: float | None = None,
) -> _PoolFactory | Callable[..., Any]:
    """Create a compute pool (context manager or decorator).

    Can be used as:

    1. Context manager:
        with pool(provider=AWS(), accelerator="A100") as p:
            result = train(data) >> p

    2. Decorator:
        @pool(provider=AWS(), accelerator="A100")
        def main():
            return train(data) >> sky

    Args:
        provider: Provider configuration (AWS, VastAI, or Verda).
        nodes: Number of nodes.
        accelerator: GPU type.
        image: Environment specification.
        allocation: Instance allocation strategy.
        timeout: Provisioning timeout.
        ttl: Auto-shutdown timeout in seconds (0 = use provider default).
        panel: Enable Rich terminal dashboard (default: True).
        logging: Log configuration. If True, logs to .skyward/skyward.log.
        max_hourly_cost: Maximum hourly cost in USD for the entire cluster.

    Returns:
        A _PoolFactory that works as context manager or decorator.
    """
    if callable(provider):
        fn = provider
        factory = _PoolFactory(
            provider=None,
            nodes=nodes,
            accelerator=accelerator,
            vcpus=vcpus,
            memory_gb=memory_gb,
            architecture=architecture,
            image=image,
            allocation=allocation,
            timeout=timeout,
            ttl=ttl,
            panel=panel,
            logging=logging,
            max_hourly_cost=max_hourly_cost,
        )
        return factory(fn)

    return _PoolFactory(
        provider=provider,  # type: ignore[arg-type]
        nodes=nodes,
        accelerator=accelerator,
        vcpus=vcpus,
        memory_gb=memory_gb,
        architecture=architecture,
        image=image,
        allocation=allocation,
        timeout=timeout,
        ttl=ttl,
        panel=panel,
        logging=logging,
        max_hourly_cost=max_hourly_cost,
    )
