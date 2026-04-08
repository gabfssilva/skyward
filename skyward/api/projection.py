"""Mutable projection that builds ``SessionView`` from domain events.

Reactive via ``on_change`` and ``on_log`` callbacks.  This is the single
source of truth for the current session state, consumed by console actors,
CLI dashboards, and programmatic introspection.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from types import MappingProxyType

from skyward.api.events import (
    Error,
    Log,
    Metric,
    Node,
    Pool,
    Scaling,
    SessionEvent,
    Task,
)
from skyward.api.views import (
    BootstrapView,
    NodeStatus,
    NodeView,
    PoolPhase,
    PoolView,
    ScalingView,
    SessionView,
    TaskEntry,
    TasksView,
)

__all__ = [
    "SessionProjection",
    "_advance",
    "_insert_phase",
    "_throughput",
]

# ── Constants ────────────────────────────────────────────────────

_KNOWN_PHASE_ORDER: tuple[str, ...] = (
    "connecting", "env", "apt", "uv", "deps", "skyward", "volumes", "worker",
)
_DEFAULT_PHASES: tuple[str, ...] = ("connecting", "apt", "uv", "deps", "worker")

_PHASE_MAP: dict[str, PoolPhase] = {
    "PROVISIONING": PoolPhase.PROVISIONING,
    "SSH": PoolPhase.SSH,
    "BOOTSTRAP": PoolPhase.BOOTSTRAP,
    "WORKERS": PoolPhase.WORKERS,
    "READY": PoolPhase.READY,
    "STOPPED": PoolPhase.STOPPED,
}

_SNAPSHOT_NODE_STATUS_MAP: dict[int, NodeStatus] = {
    NodeStatus.WAITING.value: NodeStatus.WAITING,
    NodeStatus.SSH.value: NodeStatus.SSH,
    NodeStatus.BOOTSTRAPPING.value: NodeStatus.BOOTSTRAPPING,
    NodeStatus.READY.value: NodeStatus.READY,
}


# ── Helper functions ─────────────────────────────────────────────


def _insert_phase(phases: tuple[str, ...], new_phase: str) -> tuple[str, ...]:
    """Insert phase respecting known order. Unknown phases go before 'worker'."""
    if new_phase in phases:
        return phases
    known_idx = {p: i for i, p in enumerate(_KNOWN_PHASE_ORDER)}
    if new_phase in known_idx:
        pos = known_idx[new_phase]
        for i, p in enumerate(phases):
            if p in known_idx and known_idx[p] > pos:
                return (*phases[:i], new_phase, *phases[i:])
        return (*phases, new_phase)
    for i, p in enumerate(phases):
        if p == "worker":
            return (*phases[:i], new_phase, *phases[i:])
    return (*phases, new_phase)


def _advance(current: PoolPhase, candidate: PoolPhase) -> PoolPhase:
    """Only advance forward (higher enum value)."""
    return candidate if candidate.value > current.value else current


def _throughput(tasks: TasksView, now: float | None = None) -> float:
    """tasks_done / elapsed_minutes."""
    if not tasks.done:
        return 0.0
    ts = now if now is not None else time.monotonic()
    elapsed_min = (ts - tasks.first_task_at) / 60
    return tasks.done / elapsed_min if elapsed_min > 0 else 0.0


# ── SessionProjection ───────────────────────────────────────────


class SessionProjection:
    """Mutable projection that builds ``SessionView`` from domain events."""

    def __init__(
        self,
        on_log: Callable[[Log.Emitted], None] | None = None,
        on_change: Callable[[SessionView, SessionView], None] | None = None,
        on_event: Callable[[SessionEvent], None] | None = None,
    ) -> None:
        self._pools: dict[str, PoolView] = {}
        self._view: SessionView = SessionView()
        self.on_log = on_log
        self.on_change = on_change
        self.on_event = on_event

    @property
    def view(self) -> SessionView:
        return self._view

    def _is_duplicate(self, event: SessionEvent) -> bool:
        match event:
            case Task.Queued(pool_name=name, task_id=tid):
                pool = self._pools.get(name)
                return pool is not None and tid in pool.tasks.inflight
            case Task.Completed(pool_name=name, task_id=tid) | Task.Failed(pool_name=name, task_id=tid):
                pool = self._pools.get(name)
                return pool is not None and tid not in pool.tasks.inflight
            case _:
                return False

    def handle(self, event: SessionEvent) -> None:
        """Dispatch a domain event to the appropriate handler."""
        if self._is_duplicate(event):
            return
        if self.on_event:
            self.on_event(event)
        match event:
            # ── Logs: fire callback only, no state change ────────
            case Log.Emitted():
                if self.on_log:
                    self.on_log(event)
                return

            # ── Pool lifecycle ───────────────────────────────────
            case Pool.Provisioning(pool_name=name, total_nodes=total, started_at=started):
                self._pools[name] = PoolView(
                    name=name,
                    phase=PoolPhase.PROVISIONING,
                    tasks=TasksView(),
                    scaling=ScalingView(desired=total),
                    total_nodes=total,
                    started_at=started,
                )

            case Pool.PhaseChanged(pool_name=name, phase=phase_str):
                if name not in self._pools:
                    return
                if phase_str in _PHASE_MAP:
                    pool = self._pools[name]
                    self._pools[name] = replace(
                        pool, phase=_advance(pool.phase, _PHASE_MAP[phase_str]),
                    )

            case Pool.Stopped(pool_name=name):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                self._pools[name] = replace(pool, phase=PoolPhase.STOPPED)

            case Pool.Provisioned(pool_name=name, cluster=cluster, instances=instances):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                self._pools[name] = replace(
                    pool,
                    phase=_advance(pool.phase, PoolPhase.SSH),
                    cluster=cluster,
                    instances=instances,
                )

            case Pool.ProvisionFailed():
                return

            case Pool.Reconciled(pool_name=name, snapshot=snapshot):
                if name not in self._pools or snapshot is None:
                    return
                self._on_reconciled(name, snapshot)

            # ── Node lifecycle ───────────────────────────────────
            case Node.Connected(pool_name=name, node_id=nid, instance=inst):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                node = self._get_node(pool, nid)
                node = replace(node, status=NodeStatus.SSH, instance=inst or node.instance)
                pool = self._set_node(pool, node)
                ssh_count = sum(
                    1 for n in pool.nodes.values()
                    if n.status.value >= NodeStatus.SSH.value
                )
                if ssh_count >= pool.total_nodes:
                    pool = replace(pool, phase=_advance(pool.phase, PoolPhase.BOOTSTRAP))
                self._pools[name] = pool

            case Node.Ready(pool_name=name, node_id=nid):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                node = self._get_node(pool, nid)
                node = replace(node, status=NodeStatus.READY, bootstrap=None)
                pool = self._set_node(pool, node)
                ready_count = sum(
                    1 for n in pool.nodes.values()
                    if n.status.value >= NodeStatus.READY.value
                )
                if ready_count >= pool.total_nodes:
                    pool = replace(
                        pool,
                        phase=_advance(pool.phase, PoolPhase.READY),
                        ready_at=pool.ready_at or time.monotonic(),
                    )
                self._pools[name] = pool

            case Node.Lost(pool_name=name, node_id=nid):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                lost_node = pool.nodes.get(nid)
                lost_iid = lost_node.instance.id if lost_node and lost_node.instance else None
                nodes = MappingProxyType({
                    k: v for k, v in pool.nodes.items() if k != nid
                })
                instances = tuple(
                    i for i in pool.instances if i.id != lost_iid
                ) if lost_iid else pool.instances
                self._pools[name] = replace(
                    pool,
                    nodes=nodes,
                    instances=instances,
                )

            # ── Bootstrap ────────────────────────────────────────
            case Node.Bootstrap.Started(pool_name=name, node_id=nid, phase=phase):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                node = self._get_node(pool, nid)
                bootstrap = node.bootstrap
                if bootstrap is None:
                    phases = _insert_phase(_DEFAULT_PHASES, phase)
                    bootstrap = BootstrapView(
                        phases=phases, completed=frozenset(), active=phase, output="",
                    )
                else:
                    completed = (
                        bootstrap.completed | {bootstrap.active}
                        if bootstrap.active
                        else bootstrap.completed
                    )
                    phases = _insert_phase(bootstrap.phases, phase)
                    bootstrap = BootstrapView(
                        phases=phases, completed=completed, active=phase, output="",
                    )
                node = replace(node, bootstrap=bootstrap)
                self._pools[name] = self._set_node(pool, node)

            case Node.Bootstrap.Completed(pool_name=name, node_id=nid, phase=phase):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                node = self._get_node(pool, nid)
                if node.bootstrap is None:
                    return
                bootstrap = replace(
                    node.bootstrap, completed=node.bootstrap.completed | {phase},
                )
                node = replace(node, bootstrap=bootstrap)
                self._pools[name] = self._set_node(pool, node)

            case Node.Bootstrap.Output(pool_name=name, node_id=nid, output=output, overwrite=ow):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                node = self._get_node(pool, nid)
                if node.bootstrap is None:
                    self.handle(Log.Emitted(name, nid, output, overwrite=ow))
                    return
                bootstrap = replace(node.bootstrap, output=output)
                node = replace(node, bootstrap=bootstrap)
                self._pools[name] = self._set_node(pool, node)

            case Node.Bootstrap.Done(pool_name=name, node_id=nid, success=success):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                node = self._get_node(pool, nid)
                if success:
                    node = replace(node, status=NodeStatus.BOOTSTRAPPING)
                    pool = self._set_node(pool, node)
                    done_count = sum(
                        1 for n in pool.nodes.values()
                        if n.status.value >= NodeStatus.BOOTSTRAPPING.value
                    )
                    if done_count >= pool.total_nodes:
                        pool = replace(
                            pool, phase=_advance(pool.phase, PoolPhase.WORKERS),
                        )
                else:
                    node = replace(node, bootstrap=None)
                    pool = self._set_node(pool, node)
                self._pools[name] = pool

            case Node.Bootstrap.Command(pool_name=name, node_id=nid, command=cmd):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                node = self._get_node(pool, nid)
                if node.bootstrap is None:
                    return
                bootstrap = replace(node.bootstrap, output=cmd[:80])
                node = replace(node, bootstrap=bootstrap)
                self._pools[name] = self._set_node(pool, node)

            # ── Tasks ────────────────────────────────────────────
            case Task.Queued(pool_name=name, task_id=tid, name=fname, kind=kind, broadcast_total=bt):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                entry = TaskEntry(
                    task_id=tid, name=fname, kind=kind,
                    started_at=time.monotonic(), broadcast_total=bt,
                )
                tasks = pool.tasks
                first = tasks.first_task_at or time.monotonic()
                inflight = MappingProxyType({**tasks.inflight, tid: entry})
                tasks = replace(
                    tasks, queued=tasks.queued + 1,
                    first_task_at=first, inflight=inflight,
                )
                self._pools[name] = replace(pool, tasks=tasks)

            case Task.Assigned(pool_name=name, task_id=tid, node_id=nid):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                if tid not in pool.tasks.inflight:
                    return
                entry = pool.tasks.inflight[tid]
                already_assigned = entry.node_id >= 0
                started = time.monotonic() if entry.node_id < 0 else entry.started_at
                updated = replace(entry, node_id=nid, started_at=started)
                tasks = replace(
                    pool.tasks,
                    queued=max(0, pool.tasks.queued - 1) if not already_assigned else pool.tasks.queued,
                    running=pool.tasks.running + 1 if not already_assigned else pool.tasks.running,
                    inflight=MappingProxyType({**pool.tasks.inflight, tid: updated}),
                )
                self._pools[name] = replace(pool, tasks=tasks)

            case Task.Completed(pool_name=name, task_id=tid, node_id=nid, elapsed=elapsed):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                entry = pool.tasks.inflight.get(tid)
                fn_name = entry.name.split("(")[0] if entry else "unknown"
                fn_stats = {**pool.tasks.fn_stats}
                fn_stats[fn_name] = (*fn_stats.get(fn_name, ()), elapsed)
                inflight = dict(pool.tasks.inflight)
                inflight.pop(tid, None)
                per_node = dict(pool.tasks.tasks_per_node)
                if entry and entry.node_id >= 0:
                    per_node[entry.node_id] = per_node.get(entry.node_id, 0) + 1
                tasks = replace(
                    pool.tasks,
                    running=max(0, pool.tasks.running - 1),
                    done=pool.tasks.done + 1,
                    latencies=(*pool.tasks.latencies, elapsed),
                    inflight=MappingProxyType(inflight),
                    fn_stats=MappingProxyType(fn_stats),
                    tasks_per_node=MappingProxyType(per_node),
                )
                self._pools[name] = replace(pool, tasks=tasks)

            case Task.Failed(pool_name=name, task_id=tid):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                entry = pool.tasks.inflight.get(tid)
                fn_name = entry.name.split("(")[0] if entry else "unknown"
                fn_failed = {**pool.tasks.fn_failed}
                fn_failed[fn_name] = fn_failed.get(fn_name, 0) + 1
                inflight = dict(pool.tasks.inflight)
                inflight.pop(tid, None)
                tasks = replace(
                    pool.tasks,
                    running=max(0, pool.tasks.running - 1),
                    failed=pool.tasks.failed + 1,
                    inflight=MappingProxyType(inflight),
                    fn_failed=MappingProxyType(fn_failed),
                )
                self._pools[name] = replace(pool, tasks=tasks)

            case Task.BroadcastPartial(pool_name=name, task_id=tid):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                if tid not in pool.tasks.inflight:
                    return
                entry = pool.tasks.inflight[tid]
                updated = replace(entry, broadcast_done=entry.broadcast_done + 1)
                tasks = replace(
                    pool.tasks,
                    inflight=MappingProxyType({**pool.tasks.inflight, tid: updated}),
                )
                self._pools[name] = replace(pool, tasks=tasks)

            # ── Metrics ──────────────────────────────────────────
            case Metric.Sampled(pool_name=name, node_id=nid, name=metric_name, value=value):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                node = self._get_node(pool, nid)
                metrics = MappingProxyType({**node.metrics, metric_name: value})
                node = replace(node, metrics=metrics)
                self._pools[name] = self._set_node(pool, node)

            # ── Scaling ──────────────────────────────────────────
            case Scaling.DesiredChanged(pool_name=name, desired=desired):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                match desired:
                    case d if d > max(pool.scaling.desired, pool.total_nodes):
                        reconciler = "scaling_up"
                    case d if d < len(pool.nodes):
                        reconciler = "draining"
                    case _:
                        reconciler = "watching"
                scaling = replace(pool.scaling, desired=desired, reconciler_state=reconciler)
                self._pools[name] = replace(pool, scaling=scaling)

            case Scaling.Spawning(pool_name=name, count=count, instances=instances):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                scaling = replace(pool.scaling, pending=pool.scaling.pending + count)
                self._pools[name] = replace(
                    pool,
                    scaling=scaling,
                    total_nodes=scaling.desired or pool.total_nodes + count,
                    instances=(*pool.instances, *instances),
                )

            case Scaling.Draining(pool_name=name, count=n):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                scaling = replace(
                    pool.scaling,
                    draining=pool.scaling.draining + n,
                    reconciler_state="draining",
                )
                self._pools[name] = replace(pool, scaling=scaling)

            case Scaling.DrainCompleted(pool_name=name, node_id=nid):
                if name not in self._pools:
                    return
                pool = self._pools[name]
                draining = max(0, pool.scaling.draining - 1)
                reconciler = "watching" if draining == 0 else pool.scaling.reconciler_state
                scaling = replace(
                    pool.scaling, draining=draining, reconciler_state=reconciler,
                )
                drained_node = pool.nodes.get(nid)
                drained_iid = drained_node.instance.id if drained_node and drained_node.instance else None
                nodes = MappingProxyType({
                    k: v for k, v in pool.nodes.items() if k != nid
                })
                instances = tuple(
                    i for i in pool.instances if i.id != drained_iid
                ) if drained_iid else pool.instances
                self._pools[name] = replace(
                    pool,
                    scaling=scaling,
                    nodes=nodes,
                    instances=instances,
                    total_nodes=max(0, pool.total_nodes - 1),
                )

            # ── Errors / unknown ─────────────────────────────────
            case Error.Occurred():
                return

            case _:
                return

        self._rebuild_view()

    # ── Internal helpers ─────────────────────────────────────────

    def _get_node(self, pool: PoolView, node_id: int) -> NodeView:
        if node_id in pool.nodes:
            return pool.nodes[node_id]
        return NodeView(node_id=node_id, status=NodeStatus.WAITING)

    def _set_node(self, pool: PoolView, node: NodeView) -> PoolView:
        nodes = MappingProxyType({**pool.nodes, node.node_id: node})
        return replace(pool, nodes=nodes)

    def _rebuild_view(self) -> None:
        old = self._view
        self._view = SessionView(
            pools=MappingProxyType(dict(self._pools)),
        )
        if self.on_change and self._view != old:
            self.on_change(old, self._view)

    def _on_reconciled(self, name: str, snapshot: object) -> None:
        from skyward.actors.snapshot import PoolSnapshot

        if not isinstance(snapshot, PoolSnapshot):
            return

        pool = self._pools[name]
        instances_by_id = {i.id: i for i in snapshot.instances} if snapshot.instances else {}

        nodes: dict[int, NodeView] = {}
        for ns in snapshot.nodes:
            status = _SNAPSHOT_NODE_STATUS_MAP.get(ns.status.value, NodeStatus.WAITING)
            bootstrap: BootstrapView | None = None
            if ns.bootstrap is not None:
                bootstrap = BootstrapView(
                    phases=ns.bootstrap.phases,
                    completed=ns.bootstrap.completed,
                    active=ns.bootstrap.active,
                    output=ns.bootstrap.output,
                )
            existing = pool.nodes.get(ns.node_id)
            instance = (
                instances_by_id.get(ns.instance_id)
                or (existing.instance if existing else None)
            )
            nodes[ns.node_id] = NodeView(
                node_id=ns.node_id, status=status, bootstrap=bootstrap,
                instance=instance,
            )

        phase = _PHASE_MAP.get(snapshot.phase.name, PoolPhase.PROVISIONING)
        tasks = replace(
            pool.tasks,
            queued=snapshot.tasks.queued,
            running=snapshot.tasks.running,
            done=snapshot.tasks.done,
            failed=snapshot.tasks.failed,
        )
        scaling = replace(
            pool.scaling,
            desired=snapshot.scaling.desired_nodes,
            pending=snapshot.scaling.pending_nodes,
            draining=snapshot.scaling.draining_nodes,
            reconciler_state=snapshot.scaling.reconciler_state,
            is_elastic=snapshot.scaling.is_elastic,
            min_nodes=snapshot.scaling.min_nodes,
            max_nodes=snapshot.scaling.max_nodes,
        )
        self._pools[name] = replace(
            pool,
            phase=phase,
            nodes=MappingProxyType(nodes),
            total_nodes=len(snapshot.nodes),
            tasks=tasks,
            scaling=scaling,
            cluster=snapshot.cluster or pool.cluster,
            instances=snapshot.instances or pool.instances,
            started_at=snapshot.started_at or pool.started_at,
        )
