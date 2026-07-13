"""Pool — the orchestrator, as a plain asyncio object.

Owns the provisioning flow (offers → prepare → provision → nodes), the
task manager, the reconciler/autoscaler pair, per-node transports, TCP
proxies, and the casty ``ClusterClient`` used to reach remote workers.

Nodes call back into the pool via the ``NodeListener`` methods
(``node_connected``, ``node_ready``, ``node_activated``,
``node_exhausted``, ``head_address_known``, ``on_stream``).
"""

from __future__ import annotations

import asyncio
import itertools
import logging as _logging
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NamedTuple

from casty import ClusterClient

from skyward.actors.messages import (
    BootstrapCommand,
    BootstrapFailed,
    BootstrapPhase,
    ConsoleOutput,
    HeadAddressKnown,
    NodeFileResult,
    NodeId,
    NodeInstance,
)
from skyward.actors.node.node import Node
from skyward.actors.snapshot import (
    BootstrapTimeline,
    NodeSnapshot,
    NodeStatus,
    PoolPhase,
    PoolSnapshot,
    ScalingSnapshot,
    TaskCounters,
)
from skyward.actors.task_manager.manager import TaskManager
from skyward.api import events
from skyward.api.spec import Nodes
from skyward.core.errors import ProvisioningError
from skyward.core.model import Cluster
from skyward.infra.tcp_proxy import TcpProxy
from skyward.observability.logger import logger

if TYPE_CHECKING:
    from skyward.actors.autoscaler.autoscaler import Autoscaler
    from skyward.actors.instance_monitor import StreamMsg
    from skyward.actors.messages import FileOpKind, NodeSelection, NodeTarget
    from skyward.actors.reconciler.reconciler import Reconciler
    from skyward.infra.ssh_transport import SshTransport

_logging.getLogger("casty").setLevel(_logging.ERROR)


def _no_emit(_event: events.SessionEvent) -> None:
    pass


def _format_task(fn: object, args: tuple, kwargs: dict) -> str:
    name = getattr(fn, "__name__", str(fn))
    parts = [repr(a) for a in args]
    parts.extend(f"{k}={v!r}" for k, v in kwargs.items())
    call = f"{name}({', '.join(parts)})"
    return call[:80] + "…" if len(call) > 80 else call


@dataclass(frozen=True, slots=True)
class PoolStarted:
    cluster_id: str
    instances: tuple[NodeInstance, ...]
    cluster: Cluster[Any]


class NodeCount(NamedTuple):
    total: int
    ready: int


async def _build_mount_plan(volumes: Any, cluster: Any, provider: Any) -> Any:
    """Dispatch a `Volume` tuple to the right mount strategy."""
    from skyward.api.model import MountPlan
    from skyward.providers.bootstrap.ops import mount_volumes
    from skyward.providers.provider import Mountable

    explicit: list[tuple[Any, Any]] = []
    needs_provider: list[Any] = []
    for vol in volumes:
        if vol.storage is not None:
            explicit.append((vol, await vol.storage.resolve()))
        else:
            needs_provider.append(vol)

    if explicit and needs_provider:
        raise RuntimeError(
            "Mixing volumes with explicit `storage=` and provider-managed volumes "
            "in the same pool is not supported."
        )

    if needs_provider:
        if not isinstance(provider, Mountable):
            bucket_names = [v.bucket for v in needs_provider]
            raise RuntimeError(
                f"Volumes {bucket_names} require a provider implementing Mountable, "
                "but none have explicit `storage=` and the selected provider does not."
            )
        return await provider.mount_plan(cluster, tuple(needs_provider))

    if explicit:
        return MountPlan(bootstrap=mount_volumes(tuple(explicit)))

    if isinstance(provider, Mountable):
        return await provider.mount_plan(cluster, ())
    return MountPlan()


def _build_pool_info_json(
    node_id: int, spec: Any, cluster: Any, ni: NodeInstance, head_addr: str,
) -> str:
    from skyward.providers.pool_info import build_pool_info

    accel = ni.instance.offer.instance_type.accelerator
    accelerator_count = accel.count if accel else 1
    total_accelerators = accelerator_count * spec.nodes.desired

    wpn = spec.worker.concurrency if spec.worker.executor == "process" else 1

    pool_info = build_pool_info(
        node=node_id,
        total_nodes=spec.nodes.desired,
        accelerator_count=accelerator_count,
        total_accelerators=total_accelerators,
        head_addr=head_addr,
        head_port=29500,
        job_id=cluster.id,
        peers=[],
        accelerator_type=getattr(spec, "accelerator_name", None),
        placement_group=ni.network_interface or None,
        worker=0,
        workers_per_node=wpn,
    )
    return pool_info.model_dump_json()


def apply_stream_event(
    timelines: dict[str, BootstrapTimeline],
    iid: str,
    msg: BootstrapPhase | BootstrapCommand | ConsoleOutput,
) -> None:
    """Fold a monitor stream message into the per-instance bootstrap timeline."""
    tl = timelines.get(iid)
    match msg:
        case BootstrapPhase(event="started", phase=p) if p != "bootstrap":
            if tl is None:
                new = BootstrapTimeline(
                    phases=(p,), completed=frozenset(), active=p, output="",
                )
            else:
                completed = tl.completed | {tl.active} if tl.active else tl.completed
                phases = tl.phases if p in tl.phases else (*tl.phases, p)
                new = BootstrapTimeline(
                    phases=phases, completed=completed, active=p, output="",
                )
        case BootstrapPhase(event="completed", phase=p) if tl is not None and p != "bootstrap":
            new = replace(tl, completed=tl.completed | {p})
        case BootstrapCommand(command=cmd) if tl is not None:
            new = replace(tl, output=cmd[:80])
        case ConsoleOutput(content=c) if (
            tl is not None and (stripped := c.strip()) and not stripped.startswith("#")
        ):
            new = replace(tl, output=stripped)
        case _:
            return
    timelines[iid] = new


async def _shutdown_cluster(
    provider: Any, cluster: Any, instance_ids: tuple[str, ...],
) -> None:
    log = logger.bind(actor="pool")
    if instance_ids and cluster is not None:
        log.info("Terminating {n} instances: {ids}", n=len(instance_ids), ids=instance_ids)
        try:
            await provider.terminate(cluster, instance_ids)
            log.info("Terminate OK for {n} instances", n=len(instance_ids))
        except Exception as e:
            log.error(
                "CRITICAL: FAILED to terminate instances "
                "{ids} — they may still be running! err={err}",
                ids=instance_ids, err=e,
            )
    elif not instance_ids:
        log.warning(
            "Shutdown called with 0 instance_ids — no terminate requested "
            "(cluster={cl_none}); this WILL leak pods if provider-side pods "
            "exist but pool state forgot them.",
            cl_none=(cluster is None),
        )
    if cluster is not None:
        with suppress(Exception):
            await provider.teardown(cluster)


class Pool:
    def __init__(
        self,
        *,
        pool_name: str = "",
        emit: events.Emit | None = None,
    ) -> None:
        self.name = pool_name
        self._emit = emit or _no_emit
        self._log = logger.bind(actor="pool")

        self.spec: Any = None
        self.provider: Any = None
        self.cluster: Cluster[Any] | None = None
        self.cluster_id: str = ""
        self.instances: dict[NodeId, NodeInstance] = {}
        self.nodes: dict[NodeId, Node] = {}
        self.ready_nodes: set[NodeId] = set()
        self.dead_nodes: set[NodeId] = set()
        self.instance_map: dict[NodeId, str] = {}
        self.node_statuses: dict[str, NodeStatus] = {}
        self.bootstrap_timelines: dict[str, BootstrapTimeline] = {}
        self.node_transports: dict[NodeId, SshTransport] = {}
        self.task_counters = TaskCounters()
        self.scaling = ScalingSnapshot()
        self.phase = PoolPhase.PROVISIONING
        self.pool_started_at = 0.0

        self._tm: TaskManager | None = None
        self._reconciler: Reconciler | None = None
        self._autoscaler: Autoscaler | None = None
        self._client: ClusterClient | None = None
        self._clients: dict[NodeId, ClusterClient] = {}
        self._head_addr: str | None = None
        self._ca: Any = None
        self._client_tls: Any = None
        self._spawn_gen = 0
        self._proxies: tuple[TcpProxy, ...] = ()
        self._operational = False
        self._stopped = False
        self._started_event = asyncio.Event()
        self._tunnel_map: dict[tuple[str, int], tuple[str, int]] = {}
        self._bg_tasks: set[asyncio.Task[Any]] = set()

    # ── start / recover / stop ────────────────────────────────────

    async def start(
        self, spec: Any, provider: Any, offers: tuple[Any, ...],
    ) -> PoolStarted:
        self._emit(events.Pool.Provisioning(
            self.name, spec.nodes.desired, time.monotonic(),
        ))
        if not offers:
            raise ProvisioningError(pool_name=self.name, reason="No offers available")

        from skyward.infra.tls import ensure_ca, issue_client_config

        self.spec = spec
        self.provider = provider
        self._ca = ensure_ca()
        self._client_tls = issue_client_config(self._ca)
        self.pool_started_at = time.monotonic()

        self._log.info(
            "Starting pool: {nodes} nodes, accelerator={acc}, offers={n}",
            nodes=spec.nodes.desired,
            acc=getattr(spec, "accelerator", None),
            n=len(offers),
        )

        last_reason = "No offers available"
        for i, offer in enumerate(offers):
            if self._stopped:
                raise ProvisioningError(pool_name=self.name, reason="Interrupted")
            try:
                cluster = await self.provider.prepare(self.spec, offer)
            except Exception as e:
                last_reason = str(e) or repr(e)
                self._emit(events.Pool.ProvisionFailed(self.name, last_reason))
                if i < len(offers) - 1:
                    self._log.warning(
                        "Prepare failed, trying next offer ({n} remaining)",
                        n=len(offers) - i - 1,
                    )
                continue

            cluster = await self._effective_cluster(cluster)
            try:
                enough = await self._provision_instances(cluster)
            except Exception as e:
                last_reason = str(e) or repr(e)
                enough = False
            if enough:
                break
            last_reason = (
                f"Only provisioned {len(self.nodes)}/{self.spec.nodes.desired} nodes "
                f"across attempts"
            )
            await self._abort_offer()
        else:
            self._emit(events.Pool.ProvisionFailed(self.name, last_reason))
            raise ProvisioningError(pool_name=self.name, reason=last_reason)

        if self.spec.nodes.desired == 0:
            return await self._became_operational()

        await self._started_event.wait()
        if self._stopped:
            raise ProvisioningError(pool_name=self.name, reason="Interrupted")
        return await self._became_operational()

    async def _effective_cluster(self, cluster: Cluster[Any]) -> Cluster[Any]:
        spec = self.spec
        spec_with_provider_plugins = replace(
            spec, plugins=cluster.spec.plugins,
        ) if cluster.spec.plugins != spec.plugins else spec

        transformed_image = spec_with_provider_plugins.image
        for plugin in spec_with_provider_plugins.plugins:
            if plugin.transform is not None:
                transformed_image = plugin.transform(transformed_image, cluster)
        effective_spec = replace(spec_with_provider_plugins, image=transformed_image)
        prebaked = cluster.prebaked and transformed_image == spec.image
        cluster = replace(cluster, spec=effective_spec, prebaked=prebaked)
        self.spec = effective_spec

        if effective_spec.volumes and cluster.mount_plan is None:
            plan = await _build_mount_plan(
                effective_spec.volumes, cluster, self.provider,
            )
            cluster = replace(cluster, mount_plan=plan)
        return cluster

    async def _provision_instances(self, cluster: Cluster[Any]) -> bool:
        """Provision within one offer; True when min threshold is met."""
        spec = self.spec
        min_needed = spec.nodes.min if spec.nodes.min is not None else spec.nodes.desired
        attempt = 1
        while True:
            remaining = spec.nodes.desired - len(self.nodes)
            instances: tuple[Any, ...] = ()
            if remaining > 0:
                try:
                    cluster, instances = await self.provider.provision(cluster, remaining)
                except Exception as e:
                    self._log.warning("Provision failed: {err}", err=e)
                    instances = ()
            self._emit(events.Pool.Provisioned(self.name, cluster, tuple(instances)))
            if instances:
                self._emit(events.Scaling.Spawning(self.name, len(instances), tuple(instances)))
            self._log.info(
                "Instances provisioned ({n}), attempt {attempt}/{max}",
                n=len(instances), attempt=attempt,
                max=spec.max_provision_attempts,
            )

            to_spawn = instances[: spec.nodes.desired - len(self.nodes)]
            cluster = replace(cluster, instances=(*cluster.instances, *to_spawn))
            self.cluster = cluster
            self.cluster_id = cluster.id

            threshold_will_cross = (
                self._reconciler is None
                and len(self.nodes) + len(to_spawn) >= min_needed
            )
            if threshold_will_cross:
                future_ids = frozenset(
                    range(len(self.nodes), len(self.nodes) + len(to_spawn)),
                )
                self._spawn_control_plane(
                    initial_node_ids=frozenset(self.nodes) | future_ids,
                )

            for instance in to_spawn:
                nid = len(self.nodes)
                self._spawn_node(nid, instance)
                self.instance_map[nid] = instance.id

            if len(self.nodes) >= min_needed:
                return True

            if attempt >= spec.max_provision_attempts:
                return False

            self._log.info(
                "Got {got}/{need} nodes, retrying in {delay}s (attempt {next}/{max})",
                got=len(self.nodes), need=spec.nodes.desired,
                delay=spec.provision_retry_delay,
                next=attempt + 1, max=spec.max_provision_attempts,
            )
            await asyncio.sleep(spec.provision_retry_delay)
            attempt += 1

    async def _abort_offer(self) -> None:
        """Terminate everything from a failed offer before trying the next."""
        for node in self.nodes.values():
            await node.stop()
        self.nodes.clear()
        self.instances.clear()
        self.instance_map.clear()
        self.ready_nodes.clear()
        if self.cluster is not None:
            stale_iids = tuple(inst.id for inst in self.cluster.instances)
            await _shutdown_cluster(self.provider, self.cluster, stale_iids)
        self.cluster = None
        self.cluster_id = ""

    def _spawn_control_plane(self, initial_node_ids: frozenset[int]) -> None:
        from skyward.actors.autoscaler.autoscaler import Autoscaler
        from skyward.actors.reconciler.reconciler import Reconciler

        spec = self.spec
        min_n = spec.nodes.min if spec.nodes.min is not None else spec.nodes.desired
        max_n = spec.nodes.max or spec.nodes.desired

        self._tm = TaskManager(
            retry_on_interruption=spec.retry_on_interruption,
            emit=self._emit, pool_name=self.name,
        )
        self._reconciler = Reconciler(
            self,
            min_nodes=min_n,
            desired_count=spec.nodes.desired,
            initial_node_ids=initial_node_ids,
            tick_interval=spec.reconcile_tick_interval,
            max_provision_retries=spec.max_provision_attempts,
            emit=self._emit, pool_name=self.name,
        )
        self._autoscaler = Autoscaler(
            min_nodes=min_n,
            max_nodes=max_n,
            reconciler=self._reconciler,
            slots_per_node=spec.worker.concurrency + spec.worker.buffer,
            initial_count=spec.nodes.desired,
            initial_nodes=initial_node_ids,
            cooldown=spec.autoscale_cooldown,
        )
        self._reconciler.start()
        self._autoscaler.start()

    def _spawn_node(self, nid: int, instance: Any, *, adopt: bool = False) -> Node:
        spec = self.spec
        tick_interval = max(1.0, spec.autoscale_idle_timeout / 4)
        node = Node(
            nid, self,
            pool_name=self.name, emit=self._emit,
            ssh_timeout=spec.ssh_timeout,
            ssh_retry_interval=spec.ssh_retry_interval,
            poll_timeout=spec.provision_timeout,
            bootstrap_timeout=spec.bootstrap_timeout,
            ca=self._ca,
            autoscaler=self._autoscaler,
            scale_down_idle_seconds=spec.autoscale_idle_timeout,
            idle_tick_interval=tick_interval,
        )
        self.nodes[nid] = node
        self._spawn_gen += 1
        if adopt:
            node.adopt(self.cluster, self.provider, instance)
        else:
            node.provision(self.cluster, self.provider, instance)
        if spec.cluster and self._head_addr and nid != 0:
            node.set_head_info(HeadAddressKnown(
                head_addr=self._head_addr, casty_port=25520,
                num_nodes=spec.nodes.max or spec.nodes.desired,
                worker_concurrency=spec.worker.concurrency,
                worker_executor=spec.worker.resolved_executor,
                worker_reuse_processes=spec.worker.reuse_processes,
            ))
        return node

    async def _became_operational(self) -> PoolStarted:
        assert self.cluster is not None
        self._operational = True
        self.phase = PoolPhase.READY

        nodes = self.spec.nodes
        min_n = nodes.min if nodes.min is not None else nodes.desired
        max_n = nodes.max or nodes.desired
        self.scaling = ScalingSnapshot(
            desired_nodes=nodes.desired,
            is_elastic=nodes.auto_scaling,
            min_nodes=min_n,
            max_nodes=max_n,
        )
        if self._tm is not None and self._autoscaler is not None:
            autoscaler = self._autoscaler
            self._tm.set_pressure_observer(autoscaler.report_pressure)
        self.instance_map = {
            nid: ni.instance.id for nid, ni in self.instances.items()
        }
        self._proxies = await self._start_proxies()
        return PoolStarted(
            cluster_id=self.cluster_id,
            instances=tuple(self.instances.values()),
            cluster=self.cluster,
        )

    async def recover(
        self,
        spec: Any,
        provider: Any,
        cluster: Cluster[Any],
        instances: tuple[Any, ...],
        node_ids: tuple[int, ...] = (),
    ) -> PoolStarted:
        self._emit(events.Pool.Provisioning(
            self.name, len(instances), time.monotonic(),
        ))
        self._log.info(
            "Recovering {n} instances for cluster {cid}",
            n=len(instances), cid=cluster.id,
        )
        if not instances:
            raise ProvisioningError(
                pool_name=self.name, reason="No alive instances to recover",
            )

        from skyward.infra.tls import ensure_ca, issue_client_config

        self.spec = spec
        self.provider = provider
        self.cluster = cluster
        self.cluster_id = cluster.id
        self._ca = ensure_ca()
        self._client_tls = issue_client_config(self._ca)
        self.pool_started_at = time.monotonic()
        self._tm = TaskManager(emit=self._emit, pool_name=self.name)

        for gen, instance in enumerate(instances):
            nid = node_ids[gen] if gen < len(node_ids) else len(self.nodes)
            self._spawn_node(nid, instance, adopt=True)
            self.instance_map[nid] = instance.id

        await self._started_event.wait()
        if self._stopped:
            raise ProvisioningError(pool_name=self.name, reason="Interrupted")
        return await self._became_operational()

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._emit(events.Pool.Stopped(self.name))
        self.phase = PoolPhase.STOPPED
        self._started_event.set()

        if self._reconciler is not None:
            await self._reconciler.stop()
        if self._autoscaler is not None:
            await self._autoscaler.stop()
        for proxy in self._proxies:
            await proxy.stop()
        self._proxies = ()
        for node in list(self.nodes.values()):
            await node.stop()

        instance_ids = self._collect_instance_ids()
        self._log.info(
            "Stopping pool, terminating cluster {cid}: {n} iids {ids}",
            cid=self.cluster_id, n=len(instance_ids), ids=instance_ids,
        )
        await _shutdown_cluster(self.provider, self.cluster, instance_ids)
        await self._close_clients()
        for task in list(self._bg_tasks):
            task.cancel()
        await asyncio.gather(*self._bg_tasks, return_exceptions=True)

    async def _close_clients(self) -> None:
        if self._client is not None:
            with suppress(Exception):
                await self._client.__aexit__(None, None, None)
            self._client = None
        for c in self._clients.values():
            with suppress(Exception):
                await c.__aexit__(None, None, None)
        self._clients.clear()

    def _collect_instance_ids(self) -> tuple[str, ...]:
        known_iids = {ni.instance.id for ni in self.instances.values()}
        known_iids |= {iid for iid in self.instance_map.values() if iid}
        if self.cluster is not None:
            known_iids |= {inst.id for inst in self.cluster.instances if inst.id}
        return tuple(known_iids)

    # ── task dispatch ─────────────────────────────────────────────

    def _require_tm(self) -> TaskManager:
        if self._tm is None or self._stopped:
            raise RuntimeError("Pool is not ready")
        return self._tm

    async def execute(
        self,
        fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        timeout: float = 600.0,
        task_id: str = "",
        target: NodeTarget | None = None,
    ) -> Any:
        tm = self._require_tm()
        self._emit(events.Task.Queued(
            self.name, task_id, _format_task(fn, args, kwargs), "single",
        ))
        self.task_counters = replace(
            self.task_counters, queued=self.task_counters.queued + 1,
        )
        return await tm.submit(
            fn, args, kwargs, timeout=timeout, task_id=task_id, target=target,
        )

    async def broadcast(
        self,
        fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        timeout: float = 600.0,
        task_id: str = "",
    ) -> list[Any]:
        tm = self._require_tm()
        self._emit(events.Task.Queued(
            self.name, task_id, _format_task(fn, args, kwargs), "broadcast",
        ))
        self.task_counters = replace(
            self.task_counters, queued=self.task_counters.queued + 1,
        )
        return await tm.broadcast(
            fn, args, kwargs, timeout=timeout, task_id=task_id,
        )

    # ── queries ───────────────────────────────────────────────────

    def current_nodes(self) -> NodeCount:
        return NodeCount(total=len(self.nodes), ready=len(self.ready_nodes))

    def snapshot(self) -> PoolSnapshot:
        node_snaps = tuple(
            NodeSnapshot(
                node_id=nid,
                instance_id=self.instances[nid].instance.id if nid in self.instances else "",
                status=self.node_statuses.get(
                    self.instances[nid].instance.id if nid in self.instances else "",
                    NodeStatus.WAITING,
                ),
                bootstrap=self.bootstrap_timelines.get(
                    self.instances[nid].instance.id if nid in self.instances else "",
                ),
            )
            for nid in sorted(self.nodes)
        )
        return PoolSnapshot(
            name=self.name,
            phase=self._derive_phase(),
            nodes=node_snaps,
            tasks=self.task_counters,
            scaling=self.scaling,
            cluster=self.cluster,
            instances=tuple(
                {
                    **{i.id: i for i in (self.cluster.instances if self.cluster else ())},
                    **{ni.instance.id: ni.instance for ni in self.instances.values()},
                }.values()
            ),
            started_at=self.pool_started_at,
        )

    def _derive_phase(self) -> PoolPhase:
        match self.phase:
            case PoolPhase.READY | PoolPhase.WORKERS | PoolPhase.STOPPED:
                return self.phase
        statuses = tuple(self.node_statuses.values())
        if not statuses or len(statuses) < self.spec.nodes.desired:
            return PoolPhase.PROVISIONING
        min_status = min(statuses, key=lambda v: v.value)
        match min_status:
            case NodeStatus.READY:
                return PoolPhase.WORKERS
            case NodeStatus.BOOTSTRAPPING:
                return PoolPhase.BOOTSTRAP
            case NodeStatus.SSH:
                return PoolPhase.SSH
            case _:
                return PoolPhase.PROVISIONING

    # ── file operations ───────────────────────────────────────────

    def _select_nodes(self, sel: NodeSelection) -> tuple[tuple[NodeId, Node], ...]:
        match sel:
            case "head":
                return ((0, self.nodes[0]),) if 0 in self.ready_nodes and 0 in self.nodes else ()
            case "all":
                return tuple(
                    (nid, self.nodes[nid])
                    for nid in sorted(self.ready_nodes) if nid in self.nodes
                )
            case _:
                return (
                    ((sel, self.nodes[sel]),)
                    if sel in self.ready_nodes and sel in self.nodes else ()
                )

    async def file_op(
        self,
        op: FileOpKind,
        path: str,
        *,
        content: bytes = b"",
        selection: NodeSelection = "head",
        timeout: float = 30.0,
    ) -> tuple[NodeFileResult, ...]:
        targets = self._select_nodes(selection)
        self._log.debug("File op {op} on {n} nodes", op=op, n=len(targets))

        async def _one(nid: NodeId, node: Node) -> NodeFileResult:
            try:
                async with asyncio.timeout(timeout + 5.0):
                    return await node.file_op(op, path, content, timeout)
            except Exception as e:  # noqa: BLE001 — one dead node must not sink the op
                return NodeFileResult(node_id=nid, success=False, error=str(e))

        return tuple(await asyncio.gather(*(_one(nid, n) for nid, n in targets)))

    # ── resize ────────────────────────────────────────────────────

    def resize(self, new_nodes: Nodes) -> None:
        assert self._autoscaler is not None
        assert self._reconciler is not None
        min_n = new_nodes.min if new_nodes.min is not None else new_nodes.desired
        max_n = new_nodes.max or new_nodes.desired
        desired = new_nodes.desired
        self._log.info(
            "Resize: desired={desired}, min={min}, max={max}",
            desired=desired, min=min_n, max=max_n,
        )
        self.spec = replace(self.spec, nodes=new_nodes)
        self.scaling = replace(
            self.scaling,
            desired_nodes=desired,
            is_elastic=new_nodes.auto_scaling,
            min_nodes=min_n,
            max_nodes=max_n,
        )
        self._autoscaler.set_bounds(min_n, max_n, desired)
        self._reconciler.set_desired(desired, "resize")

    # ── reconciler-facing scaling ─────────────────────────────────

    async def scale_up(self, count: int) -> int:
        if self._stopped or self.cluster is None:
            return 0
        self._log.info("Reconciler requested scale-up of {n} nodes", n=count)
        try:
            cluster, new_instances = await self.provider.provision(self.cluster, count)
        except Exception as e:
            self._log.warning("Scale-up provision failed: {err}", err=e)
            return 0
        self._emit(events.Pool.Provisioned(self.name, cluster, tuple(new_instances)))
        if not new_instances:
            return 0
        self._emit(events.Scaling.Spawning(
            self.name, len(new_instances), tuple(new_instances),
        ))
        if self._stopped:
            late_iids = tuple(inst.id for inst in new_instances)
            self._log.warning(
                "Instances provisioned during shutdown — terminating orphans {iids}",
                iids=late_iids,
            )
            with suppress(Exception):
                await self.provider.terminate(cluster, late_iids)
            return 0

        cluster = replace(cluster, instances=(*cluster.instances, *new_instances))
        self.cluster = cluster

        free_ids = list(itertools.islice(
            itertools.filterfalse(self.nodes.__contains__, itertools.count()),
            len(new_instances),
        ))
        self._log.info(
            "Scale-up: spawning {n} nodes, reusing ids {ids}",
            n=len(new_instances), ids=free_ids,
        )
        for nid, inst in zip(free_ids, new_instances, strict=True):
            self._spawn_node(nid, inst)
            self.instance_map[nid] = inst.id
        self.scaling = replace(
            self.scaling,
            pending_nodes=self.scaling.pending_nodes + len(new_instances),
        )
        return len(new_instances)

    def scale_down(self, count: int) -> int:
        self._emit(events.Scaling.Draining(self.name, count))
        self._log.info("Reconciler requested scale-down of {n} nodes", n=count)
        victims = [
            nid for nid in sorted(self.nodes, reverse=True) if nid != 0
        ][:count]
        return self._drain_victims(victims)

    def drain_nodes(self, node_ids: frozenset[NodeId]) -> int:
        self._log.info("Reconciler requested targeted drain of {n} nodes", n=len(node_ids))
        last_one = set(self.nodes) <= set(node_ids)
        allow_head = self.spec.nodes.min == 0 and last_one
        victims = [
            nid for nid in node_ids
            if (nid != 0 or allow_head) and nid in self.nodes
        ]
        return self._drain_victims(victims)

    def _drain_victims(self, victims: list[NodeId]) -> int:
        for nid in victims:
            if self._tm is not None:
                self._tm.node_unavailable(nid)
            iid = self.instance_map.get(nid, "")
            self._emit(events.Scaling.DrainCompleted(self.name, nid))
            if self._reconciler is not None:
                self._reconciler.drain_complete(nid)
            if self._autoscaler is not None:
                self._autoscaler.drain_complete(nid)
            for proxy in self._proxies:
                proxy.remove_node(nid)
            node = self.nodes.pop(nid, None)
            if node is not None:
                self._spawn_bg(node.stop())
            self.node_transports.pop(nid, None)
            self.ready_nodes.discard(nid)
            if iid and self.cluster is not None:
                self._spawn_bg(self._terminate_instance(iid))
        self.scaling = replace(
            self.scaling,
            draining_nodes=self.scaling.draining_nodes + len(victims),
        )
        if not self.nodes and self._client is not None:
            old_client = self._client
            self._client = None
            self._head_addr = None

            async def _close() -> None:
                with suppress(Exception):
                    await old_client.__aexit__(None, None, None)

            self._spawn_bg(_close())
        return len(victims)

    def reconciliation_exhausted(self, reason: str) -> None:
        if self.ready_nodes:
            self._log.warning(
                "Reconciler gave up but {n} nodes still active",
                n=len(self.ready_nodes),
            )
            return
        self._log.error(
            "Reconciler gave up with 0 ready nodes, shutting down: {reason}",
            reason=reason,
        )
        self._spawn_bg(self.stop())

    async def _terminate_instance(self, iid: str) -> None:
        try:
            await self.provider.terminate(self.cluster, (iid,))
            self._log.debug("Dead instance terminated")
        except Exception as e:  # noqa: BLE001 — best-effort terminate
            self._log.warning(
                "Terminate request failed; instance may still be "
                "alive at provider: {err}",
                err=e,
            )

    def _spawn_bg(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    # ── node listener callbacks ───────────────────────────────────

    def node_connected(self, node_id: NodeId, ni: NodeInstance) -> None:
        self.node_statuses[ni.instance.id] = NodeStatus.SSH
        self._log.debug("Node {nid} SSH connected", nid=node_id)

    def head_address_known(self, info: HeadAddressKnown) -> None:
        if not self.spec.cluster:
            return
        self._log.info("Head address known: {addr}", addr=info.head_addr)
        self._head_addr = info.head_addr
        for nid, node in self.nodes.items():
            if nid != 0:
                node.set_head_info(info)

    def on_stream(self, msg: StreamMsg) -> None:
        ni = msg.instance
        match msg:
            case BootstrapPhase(event="started", phase=p) if p != "bootstrap":
                self._emit(events.Node.Bootstrap.Started(self.name, ni.node, p))
            case BootstrapPhase(event="completed", phase=p) if p != "bootstrap":
                self._emit(events.Node.Bootstrap.Completed(self.name, ni.node, p))
            case BootstrapPhase(event="failed", phase=p, error=err):
                self._emit(events.Node.Bootstrap.Failed(self.name, ni.node, p, err or ""))
            case BootstrapFailed(phase=p, error=err):
                self._emit(events.Node.Bootstrap.Failed(self.name, ni.node, p, err))
            case BootstrapCommand(command=cmd):
                self._emit(events.Node.Bootstrap.Command(self.name, ni.node, cmd))
            case ConsoleOutput(content=c, overwrite=ow):
                stripped = c.strip()
                if stripped and not stripped.startswith("#"):
                    self._emit(events.Node.Bootstrap.Output(
                        self.name, ni.node, stripped, overwrite=ow,
                    ))
        match msg:
            case BootstrapFailed():
                pass
            case _:
                apply_stream_event(self.bootstrap_timelines, ni.instance.id, msg)

    async def node_ready(
        self,
        node: Node,
        ni: NodeInstance,
        *,
        local_port: int,
        private_ip: str,
        transport: SshTransport,
    ) -> None:
        nid = node.node_id
        iid = ni.instance.id
        self.bootstrap_timelines.pop(iid, None)
        self.instances[nid] = ni
        self.instance_map[nid] = iid
        self.node_statuses[iid] = NodeStatus.BOOTSTRAPPING
        self.node_transports[nid] = transport
        self._log.info(
            "Node {nid} ready ({n}/{total})",
            nid=nid, n=len(self.instances), total=self.spec.nodes.desired,
        )

        if self.spec.cluster:
            effective_head = self._head_addr or private_ip or ""
            self._head_addr = effective_head
            if self._client is None:
                self._client = await self._create_client(
                    private_ip, 25520, local_port,
                )
                self._log.info("ClusterClient created")
            else:
                self._tunnel_map[(private_ip, 25520)] = ("127.0.0.1", local_port)
            client = self._client
        else:
            effective_head = private_ip or ""
            client = await self._create_standalone_client(
                private_ip, 25520, local_port,
            )
            old = self._clients.get(nid)
            if old:
                with suppress(Exception):
                    await old.__aexit__(None, None, None)
            self._clients[nid] = client
            self._log.info("Standalone client created for node {nid}", nid=nid)

        if self._operational:
            if self._reconciler is not None:
                self._reconciler.node_joined(nid)
            if self._autoscaler is not None:
                self._autoscaler.node_joined(nid)

        pool_json = _build_pool_info_json(
            nid, self.spec, self.cluster, ni, effective_head,
        )
        image_env = (
            dict(self.spec.image.env)
            if self.spec.image and self.spec.image.env else {}
        )
        hooks = tuple(
            (p.name, p.around_app)
            for p in self.spec.plugins if p.around_app is not None
        )
        process_hooks = tuple(
            (p.name, p.around_process)
            for p in self.spec.plugins if p.around_process is not None
        )
        await node.join(client, pool_json, image_env, hooks, process_hooks)

    def node_activated(self, node: Node, slots: int) -> None:
        nid = node.node_id
        self._log.info("Node {nid} activated", nid=nid)
        if self._tm is not None:
            self._tm.node_available(nid, node, slots + self.spec.worker.buffer)
        self.ready_nodes.add(nid)
        if (ni := self.instances.get(nid)) is not None:
            self.node_statuses[ni.instance.id] = NodeStatus.READY
        if (transport := self.node_transports.get(nid)) is not None:
            for proxy in self._proxies:
                proxy.add_node(nid, transport)
        self.scaling = replace(
            self.scaling,
            pending_nodes=max(0, self.scaling.pending_nodes - 1),
        )
        ready_threshold = self.spec.nodes.min or self.spec.nodes.desired
        if not self._operational and len(self.ready_nodes) >= ready_threshold:
            if len(self.ready_nodes) == self.spec.nodes.desired:
                self._log.info(
                    "All {n} nodes active, pool is operational",
                    n=self.spec.nodes.desired,
                )
            else:
                self._log.info(
                    "{n}/{total} nodes active (min threshold met), pool is operational",
                    n=len(self.ready_nodes), total=self.spec.nodes.desired,
                )
            self.phase = PoolPhase.WORKERS
            self._started_event.set()

    def node_exhausted(
        self, node_id: NodeId, reason: str, instance_id: str | None,
    ) -> None:
        self._emit(events.Node.Lost(
            self.name, node_id, f"permanently lost: {reason}",
            instance_id=instance_id,
        ))
        if node_id not in self.nodes:
            return
        self._log.error(
            "Node {nid} permanently lost: {reason}", nid=node_id, reason=reason,
        )
        if self._tm is not None:
            self._tm.node_unavailable(node_id)
        self.dead_nodes.add(node_id)
        self.ready_nodes.discard(node_id)
        self.nodes.pop(node_id, None)
        self.node_transports.pop(node_id, None)
        for proxy in self._proxies:
            proxy.remove_node(node_id)

        dead_iid = instance_id or self._resolve_dead_iid(node_id)
        self.instances.pop(node_id, None)
        self.instance_map.pop(node_id, None)
        if dead_iid is not None and self.cluster is not None:
            self.cluster = replace(
                self.cluster,
                instances=tuple(
                    i for i in self.cluster.instances if i.id != dead_iid
                ),
            )
            self._spawn_bg(self._terminate_instance(dead_iid))
        if self._reconciler is not None:
            self._reconciler.node_lost(node_id, reason)

    def _resolve_dead_iid(self, nid: int) -> str | None:
        dead_ni = self.instances.get(nid)
        dead_iid = dead_ni.instance.id if dead_ni is not None else None
        if dead_iid is None:
            dead_iid = self.instance_map.get(nid)
        if dead_iid is None and self.cluster is not None and nid < len(self.cluster.instances):
            dead_iid = self.cluster.instances[nid].id
        return dead_iid

    # ── clients & proxies ─────────────────────────────────────────

    def _address_resolver(self, addr: tuple[str, int]) -> tuple[str, int]:
        return self._tunnel_map.get(addr, addr)

    async def _create_client(
        self, private_ip: str, casty_port: int, local_port: int,
    ) -> ClusterClient:
        from skyward.infra.worker import skyward_serializer

        self._tunnel_map[(private_ip, casty_port)] = ("127.0.0.1", local_port)
        client = ClusterClient(
            contact_points=[(private_ip, casty_port)],
            system_name="skyward",
            address_map=self._address_resolver,
            serializer=skyward_serializer(),
            tls=self._client_tls,
        )
        await client.__aenter__()
        return client

    async def _create_standalone_client(
        self, private_ip: str, casty_port: int, local_port: int,
    ) -> ClusterClient:
        from skyward.infra.worker import skyward_serializer

        local_map = {(private_ip, casty_port): ("127.0.0.1", local_port)}
        client = ClusterClient(
            contact_points=[(private_ip, casty_port)],
            system_name="skyward",
            address_map=lambda addr, _m=local_map: _m.get(addr, addr),
            serializer=skyward_serializer(),
            tls=self._client_tls,
        )
        await client.__aenter__()
        return client

    async def _start_proxies(self) -> tuple[TcpProxy, ...]:
        if not self.spec.ports:
            return ()
        seed = tuple(
            (nid, self.node_transports[nid])
            for nid in sorted(self.ready_nodes)
            if nid in self.node_transports
        )
        started: list[TcpProxy] = []
        for p in self.spec.ports:
            proxy = TcpProxy(
                remote_port=p.remote, local_port=p.local,
                route=p.route, initial_nodes=seed,
            )
            try:
                await proxy.start()
            except OSError as e:
                self._log.error("Port {lp} bind failed: {err}", lp=p.local, err=e)
                continue
            started.append(proxy)
        return tuple(started)
