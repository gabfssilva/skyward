"""Node — one asyncio object per compute node.

The lifecycle is a single linear coroutine:

    poll → connect → bootstrap → post_bootstrap → start_worker → ready
         → join (worker discovery + env setup) → warm (health) → active

Failures anywhere collapse into one path (``_fail``): inflight tasks are
interrupted, the pool is notified via ``node_exhausted``, and the node
stops. The ``SshTransport`` object encapsulates SSH, port forwarding,
event streaming, and reconnection; the node holds it and awaits methods.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol

from skyward.api import events
from skyward.api.facts import NodeInstance
from skyward.api.health import HealthChecker, hc_loop
from skyward.api.pool import NodeFileResult
from skyward.api.spec import (
    DEFAULT_BOOTSTRAP_TIMEOUT,
    DEFAULT_PROVISION_TIMEOUT,
    DEFAULT_SSH_TIMEOUT,
)
from skyward.control.instance_monitor import StreamMsg, monitor_instance
from skyward.control.types import HeadAddressKnown, NodeInterruptedError
from skyward.infra.ssh_transport import (
    ConnectionFailed,
    ConnectionLost,
    ConnectionRestored,
    SshTransport,
)
from skyward.observability.logger import logger
from skyward.providers.provider import WarmableProvider

from .helpers import (
    _run_file_op,
    discover_own_worker,
    do_start_worker,
    execute_with_streaming,
    install_local_skyward,
    run_bootstrap,
    setup_worker_env,
    sync_user_code,
)

if TYPE_CHECKING:
    from skyward.api.facts import ProviderName
    from skyward.api.model import Cluster, Instance
    from skyward.api.pool import FileOpKind
    from skyward.infra.tls import CertificateAuthority

type NodeId = int


def _no_emit(_event: events.SessionEvent) -> None:
    pass


def _bind_to_node(
    inst: Instance,
    node_id: NodeId,
    provider: ProviderName,
    cluster: Cluster,  # type: ignore[type-arg]
) -> NodeInstance:
    return NodeInstance(
        instance=inst,
        node=node_id,
        provider=provider,
        ssh_user=cluster.ssh_user,
        ssh_key_path=cluster.ssh_key_path,
        ssh_password=inst.ssh_password,
    )


def _should_announce_idle(
    inflight_count: int,
    last_task_at: float,
    now: float,
    threshold: float,
    announced: bool,
) -> bool:
    """True iff the node is empty, quiet past ``threshold``, and unannounced."""
    return (
        inflight_count == 0
        and now - last_task_at > threshold
        and not announced
    )


def _evaluate_health(
    healthy: bool,
    reason: str | None,
    current: int,
    threshold: int,
) -> tuple[int, str | None]:
    """Decide the next failure count and preemption reason for a health tick.

    Returns ``(new_count, preempt_reason)``; ``preempt_reason`` is non-None
    iff the node must be preempted.
    """
    if healthy:
        return 0, None
    new_count = current + 1
    if new_count >= threshold:
        return new_count, f"health check failed {new_count}x: {reason or 'unspecified'}"
    return new_count, None


class IdleSink(Protocol):
    def node_idle(self, node_id: int) -> None: ...
    def node_busy(self, node_id: int) -> None: ...


class NodeListener(Protocol):
    """What a node needs from its pool."""

    def node_connected(self, node_id: int, ni: NodeInstance) -> None: ...

    async def node_ready(
        self,
        node: Node,
        ni: NodeInstance,
        *,
        local_port: int,
        private_ip: str,
        transport: SshTransport,
    ) -> None: ...

    def head_address_known(self, info: HeadAddressKnown) -> None: ...

    def node_activated(self, node: Node, slots: int) -> None: ...

    def node_exhausted(
        self, node_id: int, reason: str, instance_id: str | None,
    ) -> None: ...

    def on_stream(self, msg: StreamMsg) -> None: ...


class Node:
    def __init__(
        self,
        node_id: NodeId,
        pool: NodeListener,
        *,
        ssh_timeout: float = float(DEFAULT_SSH_TIMEOUT),
        ssh_retry_interval: float = 5.0,
        poll_interval: float = 5.0,
        poll_timeout: float = float(DEFAULT_PROVISION_TIMEOUT),
        bootstrap_timeout: float = float(DEFAULT_BOOTSTRAP_TIMEOUT),
        skip_monitor: bool = False,
        ca: CertificateAuthority | None = None,
        autoscaler: IdleSink | None = None,
        scale_down_idle_seconds: float = 60.0,
        idle_tick_interval: float = 15.0,
        pool_name: str = "",
        emit: events.Emit | None = None,
    ) -> None:
        self.node_id = node_id
        self._pool = pool
        self._ssh_timeout = ssh_timeout
        self._ssh_retry_interval = ssh_retry_interval
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout
        self._bootstrap_timeout = bootstrap_timeout
        self._skip_monitor = skip_monitor
        self._ca = ca
        self._autoscaler = autoscaler
        self._scale_down_idle_seconds = scale_down_idle_seconds
        self._idle_tick_interval = idle_tick_interval
        self._pool_name = pool_name
        self._emit = emit or _no_emit
        self._log = logger.bind(component="node", node_id=node_id)

        self._cluster: Any = None
        self._provider: Any = None
        self.ni: NodeInstance | None = None
        self.transport: SshTransport | None = None
        self._local_port = 0
        self._worker: Any = None
        self._head_info: HeadAddressKnown | None = None
        self._head_event = asyncio.Event()
        self._bootstrap_result: asyncio.Future[tuple[bool, str | None]] | None = None
        self._transport_up = asyncio.Event()
        self._inflight: dict[str, asyncio.Future[Any]] = {}
        self._task_counter = 0
        self._last_task_at = 0.0
        self._idle_announced = False
        self._health_failures = 0
        self._active = False
        self._stopped = False

        self._lifecycle_task: asyncio.Task[None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._health_task: asyncio.Task[None] | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._aux_tasks: set[asyncio.Task[Any]] = set()

    # ── lifecycle ─────────────────────────────────────────────────

    def provision(self, cluster: Any, provider: Any, instance: Any) -> None:
        self._log.info(
            "Node {nid} provisioning instance {iid}",
            nid=self.node_id, iid=instance.id,
        )
        self._start_lifecycle(cluster, provider, instance, reattach=False)

    def adopt(self, cluster: Any, provider: Any, instance: Any) -> None:
        self._log.info(
            "Node {nid} adopting instance {iid}",
            nid=self.node_id, iid=instance.id,
        )
        self._start_lifecycle(cluster, provider, instance, reattach=True)

    def _start_lifecycle(
        self, cluster: Any, provider: Any, instance: Any, *, reattach: bool,
    ) -> None:
        self._cluster = cluster
        self._provider = provider
        provider_name = cluster.spec.provider or "aws"
        self.ni = _bind_to_node(instance, self.node_id, provider_name, cluster)
        self._lifecycle_task = asyncio.create_task(
            self._run(instance.id, reattach=reattach),
        )

    async def _run(self, instance_id: str, *, reattach: bool) -> None:
        try:
            await self._poll(instance_id)
            await self._connect()
            self._start_monitor()
            if reattach:
                self._log.info(
                    "Node {nid} reattached, entering ready", nid=self.node_id,
                )
            else:
                async with asyncio.timeout(self._bootstrap_timeout):
                    await self._bootstrap()
                    await self._post_bootstrap()
                    await self._resolve_head()
                    await self._start_worker()
            ni = self.ni
            transport = self.transport
            assert ni is not None and transport is not None
            self._emit(events.Node.Ready(self._pool_name, self.node_id))
            private_ip = ni.instance.private_ip or ni.instance.ip or ""
            await self._pool.node_ready(
                self, ni,
                local_port=self._local_port,
                private_ip=private_ip,
                transport=transport,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            self._log.error("Bootstrap timed out after {t:.0f}s", t=self._bootstrap_timeout)
            self._fail(f"Bootstrap timed out after {self._bootstrap_timeout:.0f}s")
        except Exception as e:
            self._fail(str(e) or repr(e))

    async def _poll(self, instance_id: str) -> None:
        provider_name = self._cluster.spec.provider or "aws"
        deadline = time.monotonic() + self._poll_timeout
        while True:
            inst = None
            with suppress(Exception):
                updated, inst = await self._provider.get_instance(
                    self._cluster, instance_id,
                )
                if updated is not None:
                    self._cluster = updated
            match inst:
                case _ if inst is not None and inst.status == "provisioned" and inst.ip:
                    self.ni = _bind_to_node(
                        inst, self.node_id, provider_name, self._cluster,
                    )
                    self._log.info("Instance ready at {ip}", ip=inst.ip)
                    return
                case _ if inst is not None and inst.status == "exited":
                    self._log.warning("Instance {iid} exited", iid=instance_id)
                    raise RuntimeError(f"Instance {instance_id} exited")
            if time.monotonic() > deadline:
                self._log.error("Instance not ready within {t}s", t=self._poll_timeout)
                raise RuntimeError(f"Instance not ready within {self._poll_timeout}s")
            await asyncio.sleep(self._poll_interval)

    async def _connect(self) -> None:
        ni = self.ni
        assert ni is not None
        self._log.info("Creating SSH transport for {ip}", ip=ni.instance.ip)
        transport = SshTransport(
            host=ni.instance.ip or "",
            user=ni.ssh_user or self._cluster.ssh_user,
            key_path=ni.ssh_key_path or self._cluster.ssh_key_path,
            port=ni.instance.ssh_port,
            retry_max_attempts=int(self._ssh_timeout / self._ssh_retry_interval) + 1,
            retry_delay=self._ssh_retry_interval,
            connect_timeout=min(self._ssh_retry_interval * 2, 10.0),
            reconnect_max_attempts=5,
            password=ni.ssh_password,
            on_event=self._on_transport_event,
        )
        self.transport = transport
        try:
            await transport.connect()
            self._local_port = await transport.forward_port("127.0.0.1", 25520)
        except Exception as e:
            self._emit(events.Node.ConnectionFailed(self._pool_name, str(e)))
            self._log.error("SSH connection failed: {error}", error=e)
            raise
        self._transport_up.set()
        self._log.info("SSH tunnel established (port={port})", port=self._local_port)
        self._emit(events.Node.Connected(self._pool_name, self.node_id, ni.instance))
        self._pool.node_connected(self.node_id, ni)

    def _start_monitor(self) -> None:
        if self._skip_monitor:
            return
        ni = self.ni
        transport = self.transport
        assert ni is not None and transport is not None
        self._monitor_task = asyncio.create_task(monitor_instance(
            transport, ni,
            on_stream=self._pool.on_stream,
            on_bootstrap_done=self._on_bootstrap_done,
            emit=self._emit,
            pool_name=self._pool_name,
        ))

    def _on_bootstrap_done(self, success: bool, error: str | None) -> None:
        if self._bootstrap_result is not None and not self._bootstrap_result.done():
            self._bootstrap_result.set_result((success, error))

    async def _bootstrap(self) -> None:
        ni = self.ni
        transport = self.transport
        assert ni is not None and transport is not None
        if self._cluster.prebaked:
            self._log.info("Prebaked image detected, skipping bootstrap")
            return
        self._log.info("Starting bootstrap")
        self._bootstrap_result = asyncio.get_running_loop().create_future()
        await run_bootstrap(transport, ni, self._cluster, self._cluster.spec)
        self._log.info("Bootstrap script uploaded and started")
        success, error = await self._bootstrap_result
        self._emit(events.Node.Bootstrap.Done(
            self._pool_name, self.node_id, success, error,
        ))
        if not success:
            self._log.error("Bootstrap failed: {error}", error=error)
            raise RuntimeError(error or "bootstrap failed")
        self._log.info("Bootstrap completed successfully")
        match self._provider:
            case WarmableProvider() if self.node_id == 0:
                self._log.info("Snapshotting provider image...")
                self._spawn_aux(self._save_snapshot())

    async def _save_snapshot(self) -> None:
        try:
            await self._provider.save(self._cluster)
            self._log.info("Snapshot saved")
        except Exception as e:  # noqa: BLE001 — snapshot is best-effort
            self._log.warning("Snapshot failed: {error}", error=e)

    async def _post_bootstrap(self) -> None:
        ni = self.ni
        transport = self.transport
        assert ni is not None and transport is not None
        spec = self._cluster.spec
        try:
            if spec.image and getattr(spec.image, "skyward_source", None) == "local":
                await install_local_skyward(transport, ni, self._cluster)
            if spec.image and getattr(spec.image, "includes", None):
                await sync_user_code(transport, ni, spec, self._cluster)
        except Exception as e:
            self._emit(events.Error.Occurred(
                self._pool_name, f"Post-bootstrap failed: {e}",
            ))
            raise

    async def _resolve_head(self) -> None:
        spec = self._cluster.spec
        ni = self.ni
        assert ni is not None
        if not spec.cluster:
            return
        if self.node_id == 0:
            head_private = ni.instance.private_ip or ni.instance.ip or ""
            self._pool.head_address_known(HeadAddressKnown(
                head_addr=head_private,
                casty_port=25520,
                num_nodes=spec.nodes.desired,
                worker_concurrency=spec.worker.concurrency,
                worker_executor=spec.worker.resolved_executor,
                worker_reuse_processes=spec.worker.reuse_processes,
            ))
            return
        if self._head_info is None:
            self._log.info("Waiting for head address before starting worker")
            await self._head_event.wait()

    def set_head_info(self, info: HeadAddressKnown) -> None:
        self._head_info = info
        self._head_event.set()

    async def _start_worker(self) -> None:
        ni = self.ni
        transport = self.transport
        assert ni is not None and transport is not None
        role = "head" if self.node_id == 0 else "worker"
        self._log.info("Starting worker (role={role})", role=role)
        try:
            self._local_port, _ = await do_start_worker(
                transport, self._local_port, ni, self._head_info,
                self.node_id, self._cluster, self._cluster.spec, ca=self._ca,
            )
        except Exception as e:
            self._emit(events.Node.WorkerFailed(self._pool_name, str(e)))
            self._log.error("Worker failed to start: {error}", error=e)
            raise

    # ── join (called by the pool after node_ready) ────────────────

    async def join(
        self,
        client: Any,
        pool_info_json: str,
        env_vars: dict[str, str],
        around_app_hooks: tuple[tuple[str, Any], ...] = (),
        around_process_hooks: tuple[tuple[str, Any], ...] = (),
    ) -> None:
        self._log.info("Joining: discovering worker")
        self._worker = await discover_own_worker(
            client, self.ni, standalone=not self._cluster.spec.cluster,
        )
        self._log.info("Worker discovered, setting up env")
        await setup_worker_env(
            self._worker, pool_info_json,
            env_vars, around_app_hooks, around_process_hooks,
        )
        if (hc := self._cluster.spec.health_checker) is not None:
            self._log.info(
                "Node {nid} warming up, awaiting first health check",
                nid=self.node_id,
            )
            await self._warm(hc)
        self._log.info("Env setup done, entering active")
        self._activate()

    def _activate(self) -> None:
        self._active = True
        self._last_task_at = time.monotonic()
        self._idle_announced = False
        if self._autoscaler is not None:
            self._idle_task = asyncio.create_task(self._idle_loop())
        self._pool.node_activated(self, self._cluster.spec.worker.concurrency)

    # ── health ────────────────────────────────────────────────────

    async def _warm(self, hc: HealthChecker) -> None:
        first_healthy: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._health_task = asyncio.create_task(
            self._health_loop(hc, first_healthy),
        )
        await first_healthy
        self._log.info(
            "Node {nid} first health check passed, activating", nid=self.node_id,
        )

    async def _health_loop(
        self, hc: HealthChecker, first_healthy: asyncio.Future[None],
    ) -> None:
        from skyward.infra.streaming import iter_output_stream
        from skyward.infra.worker import StreamStarted, dumps, loads

        fn = partial(hc_loop, hc.fn, hc.interval, hc.timeout, hc.initial_delay)

        def _on_result(healthy: bool, reason: str | None) -> None:
            new_count, preempt_reason = _evaluate_health(
                healthy=healthy, reason=reason,
                current=self._health_failures,
                threshold=hc.consecutive_failures,
            )
            self._health_failures = new_count
            if healthy:
                if not first_healthy.done():
                    first_healthy.set_result(None)
                self._log.debug("Node {nid} health check ok", nid=self.node_id)
                return
            self._log.warning(
                "Node {nid} health check failed ({n}/{max}): {reason}",
                nid=self.node_id, n=new_count,
                max=hc.consecutive_failures, reason=reason or "(no reason)",
            )
            if preempt_reason is not None:
                self._preempt(preempt_reason)

        try:
            tid = f"health-{self.node_id}"
            raw = await asyncio.wait_for(
                self._worker.service.run(tid, dumps((fn, (), {}, ()))),
                timeout=60.0,
            )
            match loads(raw):
                case StreamStarted():
                    async for elem in iter_output_stream(self._worker.control, tid):
                        match elem:
                            case ("ok", _):
                                _on_result(True, None)
                            case ("fail", reason):
                                _on_result(False, reason)
                            case _:
                                _on_result(False, f"malformed yield: {elem!r}")
                    ended = "stream closed"
                case result:
                    ended = f"failed to start: {getattr(result, 'error', repr(result))}"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            ended = f"drain failed: {e!r}"
        self._log.error(
            "Node {nid} health stream ended: {reason}",
            nid=self.node_id, reason=ended,
        )
        self._preempt(f"health stream ended: {ended}")

    def _preempt(self, reason: str) -> None:
        self._emit(events.Node.Preempted(self._pool_name, reason))
        self._log.warning("Preempted: {reason}", reason=reason)
        self._fail(reason)

    # ── task execution ────────────────────────────────────────────

    async def execute(
        self,
        fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        timeout: float = 600.0,
        task_id: str = "",
    ) -> Any:
        if self._stopped:
            raise NodeInterruptedError(self.node_id, "node stopped")
        tid = task_id or str(self._task_counter)
        self._task_counter += 1
        self._last_task_at = time.monotonic()
        if self._idle_announced and self._autoscaler is not None:
            self._autoscaler.node_busy(self.node_id)
            self._idle_announced = False
        self._log.debug("Node {nid} dispatching task {tid}", nid=self.node_id, tid=tid)

        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._inflight[tid] = fut
        ask_task = asyncio.create_task(self._dispatch(tid, fn, args, kwargs, timeout, fut))
        try:
            return await fut
        finally:
            self._inflight.pop(tid, None)
            if not ask_task.done():
                ask_task.cancel()
            self._last_task_at = time.monotonic()

    async def _dispatch(
        self,
        tid: str,
        fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        timeout: float,
        fut: asyncio.Future[Any],
    ) -> None:
        from casty import ActorUnavailableError, ConnectionLostError, TransportError

        deadline = time.monotonic() + timeout
        try:
            await self._transport_up.wait()
            result = await execute_with_streaming(
                self._worker, fn, args, kwargs, timeout,
                task_id=tid,
            )
        except asyncio.CancelledError:
            raise
        except (ConnectionLostError, TransportError, ActorUnavailableError) as e:
            # The RPC died with the tunnel — the task may still be running
            # remotely. Wait for the transport to come back and recover the
            # result from the worker's cache.
            self._log.warning(
                "Dispatch {tid} lost mid-flight ({err}), reconciling",
                tid=tid, err=e,
            )
            await self._reconcile(tid, fut, deadline)
            return
        except Exception as e:
            if not fut.done():
                fut.set_exception(RuntimeError(str(e) or repr(e)))
            return
        if fut.done():
            return
        if hasattr(result, "result"):
            fut.set_result(result.result)
        else:
            fut.set_exception(RuntimeError(result.error))

    async def _reconcile(
        self, tid: str, fut: asyncio.Future[Any], deadline: float,
    ) -> None:
        """Poll the worker's result cache until the task resolves.

        Runs after a connection blip killed the original ``run`` RPC.
        ``ResultPending`` keeps polling (the task is still executing);
        ``ResultUnknown`` or an unreachable worker raises
        ``NodeInterruptedError`` so the TaskManager retries elsewhere.
        """
        from skyward.infra.worker import (
            ResultDone,
            ResultPending,
            TaskFailed,
            TaskSucceeded,
            loads,
        )

        while not fut.done():
            await self._transport_up.wait()
            try:
                reply = loads(await asyncio.wait_for(
                    self._worker.control.get_result(tid), timeout=10.0,
                ))
            except Exception as e:  # noqa: BLE001 — reconcile failure means retry elsewhere
                reply = None
                err = repr(e)
            else:
                err = None
            if fut.done():
                return
            match reply:
                case ResultDone(result=TaskSucceeded() as ws):
                    self._log.info(
                        "Reconcile {tid} on node {nid}: recovered TaskSucceeded",
                        tid=tid, nid=self.node_id,
                    )
                    fut.set_result(ws.result)
                    return
                case ResultDone(result=TaskFailed() as wf):
                    self._log.info(
                        "Reconcile {tid} on node {nid}: recovered TaskFailed",
                        tid=tid, nid=self.node_id,
                    )
                    fut.set_exception(RuntimeError(f"{wf.error}\n{wf.traceback}"))
                    return
                case ResultPending() if time.monotonic() < deadline:
                    self._log.debug(
                        "Reconcile {tid} on node {nid}: still running",
                        tid=tid, nid=self.node_id,
                    )
                    await asyncio.sleep(2.0)
                case _:
                    self._log.warning(
                        "Reconcile {tid} on node {nid} unrecoverable: {err}",
                        tid=tid, nid=self.node_id, err=err or "worker has no record",
                    )
                    fut.set_exception(NodeInterruptedError(
                        self.node_id, f"result lost on node {self.node_id}",
                    ))
                    return

    # ── file ops ──────────────────────────────────────────────────

    async def file_op(
        self, op: FileOpKind, path: str, content: bytes, timeout: float,
    ) -> NodeFileResult:
        if self.transport is None:
            return NodeFileResult(
                node_id=self.node_id, success=False,
                error="transport not connected",
            )
        return await _run_file_op(
            self.transport, self.node_id, op, path, content, timeout,
        )

    # ── transport events ──────────────────────────────────────────

    def _on_transport_event(self, event: Any) -> None:
        match event:
            case ConnectionLost(error=error):
                self._log.warning(
                    "Connection lost on node {nid}: {err}, transport reconnecting",
                    nid=self.node_id, err=error,
                )
                self._transport_up.clear()
            case ConnectionRestored():
                self._log.info(
                    "Transport reconnected on node {nid}: {n} inflight resume",
                    nid=self.node_id, n=len(self._inflight),
                )
                # Recovery is owned by each task's _dispatch: the failed RPC
                # falls into _reconcile, which is gated on _transport_up.
                self._transport_up.set()
            case ConnectionFailed(error=error):
                self._log.error(
                    "Transport permanently failed on node {nid}: {err}",
                    nid=self.node_id, err=error,
                )
                self._fail(f"connection failed: {error}")

    # ── idle detection ────────────────────────────────────────────

    async def _idle_loop(self) -> None:
        while True:
            await asyncio.sleep(self._idle_tick_interval)
            if self._autoscaler is not None and _should_announce_idle(
                inflight_count=len(self._inflight),
                last_task_at=self._last_task_at,
                now=time.monotonic(),
                threshold=self._scale_down_idle_seconds,
                announced=self._idle_announced,
            ):
                self._autoscaler.node_idle(self.node_id)
                self._idle_announced = True

    # ── failure & shutdown ────────────────────────────────────────

    def _interrupt_inflight(self, reason: str) -> None:
        for fut in list(self._inflight.values()):
            if not fut.done():
                fut.set_exception(NodeInterruptedError(self.node_id, reason))
        self._inflight.clear()

    def _fail(self, reason: str) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._active = False
        self._interrupt_inflight(reason)
        self._cancel_tasks()
        iid = self.ni.instance.id if self.ni else None
        self._pool.node_exhausted(self.node_id, reason, iid)
        if self.transport is not None:
            self._spawn_aux(self.transport.close(), shielded=True)

    def _cancel_tasks(self) -> None:
        current = asyncio.current_task()
        for task in (
            self._monitor_task, self._health_task, self._idle_task,
            self._lifecycle_task,
        ):
            if task is not None and task is not current:
                task.cancel()

    def _spawn_aux(self, coro: Any, *, shielded: bool = False) -> None:
        task = asyncio.create_task(coro)
        if not shielded:
            self._aux_tasks.add(task)
            task.add_done_callback(self._aux_tasks.discard)

    async def stop(self) -> None:
        """Graceful stop (drain/shutdown) — no node_exhausted notification."""
        self._stopped = True
        self._active = False
        self._interrupt_inflight("node stopped")
        self._cancel_tasks()
        for task in self._aux_tasks:
            task.cancel()
        if self.transport is not None:
            with suppress(Exception):
                await self.transport.close()
