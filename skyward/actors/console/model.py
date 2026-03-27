from __future__ import annotations

import time
from dataclasses import replace
from types import MappingProxyType

from skyward.core.model import Cluster, Instance

from .state import _BootstrapTimeline, _NodeStatus, _Phase, _State, _TaskEntry


def _on_start_pool(state: _State) -> _State:
    return replace(state, phase=_Phase.PROVISIONING, pool_started_at=time.monotonic())


def _on_cluster_ready(state: _State) -> _State:
    return replace(state, phase=_Phase.SSH)


def _on_instances_provisioned(
    state: _State, cluster: Cluster, instances: tuple[Instance, ...],
) -> _State:
    return replace(
        state, cluster=cluster, instances=instances,
        ssh_user=cluster.ssh_user, ssh_key_path=cluster.ssh_key_path,
    )


def _update_instance(state: _State, resolved: Instance) -> _State:
    if any(i.id == resolved.id for i in state.instances):
        updated = tuple(
            resolved if i.id == resolved.id else i for i in state.instances
        )
    else:
        updated = (*state.instances, resolved)
    return replace(state, instances=updated)


def _advance(current: _Phase, candidate: _Phase) -> _Phase:
    return candidate if candidate.value > current.value else current


_KNOWN_PHASE_ORDER: tuple[str, ...] = (
    "connecting", "env", "apt", "uv", "deps", "skyward", "volumes", "worker",
)
_DEFAULT_PHASES: tuple[str, ...] = ("connecting", "apt", "uv", "deps", "worker")


def _insert_phase(phases: tuple[str, ...], new_phase: str) -> tuple[str, ...]:
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


def _on_timeline_phase_started(state: _State, node_id: int, phase: str) -> _State:
    started = state.bootstrap_started
    if node_id not in started:
        started = MappingProxyType({**started, node_id: time.monotonic()})
    timeline = state.bootstrap_spinners.get(node_id)
    if timeline is None:
        phases = _insert_phase(_DEFAULT_PHASES, phase)
        new_tl = _BootstrapTimeline(
            phases=phases, completed=frozenset(), active=phase, output="",
        )
    else:
        completed = timeline.completed | {timeline.active} if timeline.active else timeline.completed
        phases = _insert_phase(timeline.phases, phase)
        new_tl = _BootstrapTimeline(
            phases=phases, completed=completed, active=phase, output="",
        )
    spinners = MappingProxyType({**state.bootstrap_spinners, node_id: new_tl})
    return replace(state, bootstrap_spinners=spinners, bootstrap_started=started)


def _on_timeline_phase_completed(state: _State, node_id: int, phase: str) -> _State:
    timeline = state.bootstrap_spinners.get(node_id)
    if timeline is None:
        return state
    new_tl = replace(timeline, completed=timeline.completed | {phase})
    spinners = MappingProxyType({**state.bootstrap_spinners, node_id: new_tl})
    return replace(state, bootstrap_spinners=spinners)


def _on_timeline_output(state: _State, node_id: int, output: str) -> _State:
    timeline = state.bootstrap_spinners.get(node_id)
    if timeline is None:
        return state
    new_tl = replace(timeline, output=output)
    spinners = MappingProxyType({**state.bootstrap_spinners, node_id: new_tl})
    return replace(state, bootstrap_spinners=spinners)


def _on_spinner_remove(state: _State, node_id: int) -> _State:
    spinners = MappingProxyType({k: v for k, v in state.bootstrap_spinners.items() if k != node_id})
    started = MappingProxyType({k: v for k, v in state.bootstrap_started.items() if k != node_id})
    return replace(state, bootstrap_spinners=spinners, bootstrap_started=started)


def _on_ssh_connected(state: _State, node_id: int) -> _State:
    nodes = MappingProxyType({**state.nodes, node_id: _NodeStatus.SSH})
    ssh_count = sum(1 for s in nodes.values() if s.value >= _NodeStatus.SSH.value)
    phase = _advance(state.phase, _Phase.BOOTSTRAP) if ssh_count >= state.total_nodes else state.phase
    return replace(state, nodes=nodes, phase=phase)


def _on_bootstrap_done(state: _State, node_id: int) -> _State:
    nodes = MappingProxyType({**state.nodes, node_id: _NodeStatus.BOOTSTRAPPING})
    done = sum(1 for s in nodes.values() if s.value >= _NodeStatus.BOOTSTRAPPING.value)
    phase = _advance(state.phase, _Phase.WORKERS) if done >= state.total_nodes else state.phase
    return replace(state, nodes=nodes, phase=phase)


def _on_worker_started(state: _State, node_id: int) -> _State:
    nodes = MappingProxyType({**state.nodes, node_id: _NodeStatus.READY})
    ready = sum(1 for s in nodes.values() if s.value >= _NodeStatus.READY.value)
    phase = _advance(state.phase, _Phase.READY) if ready >= state.total_nodes else state.phase
    ready_at = time.monotonic() if phase == _Phase.READY and not state.ready_at else state.ready_at
    return replace(state, nodes=nodes, phase=phase, ready_at=ready_at)


def _on_task_submitted(
    state: _State, task_id: str, name: str, kind: str,
) -> _State:
    entry = _TaskEntry(
        task_id=task_id, name=name, kind=kind, started_at=time.monotonic(),
        broadcast_total=len(state.instances) if kind == "broadcast" else 0,
    )
    first = state.first_task_at or time.monotonic()
    inflight = MappingProxyType({**state.inflight, task_id: entry})
    return replace(
        state, tasks_queued=state.tasks_queued + 1,
        first_task_at=first, inflight=inflight,
    )


def _on_task_assigned(state: _State, task_id: str, node_id: int) -> _State:
    if task_id not in state.inflight:
        return state
    entry = state.inflight[task_id]
    already_assigned = entry.node_id >= 0
    started = time.monotonic() if entry.node_id < 0 else entry.started_at
    updated = replace(entry, node_id=node_id, started_at=started)
    return replace(
        state,
        tasks_queued=max(0, state.tasks_queued - 1) if not already_assigned else state.tasks_queued,
        tasks_running=state.tasks_running + 1 if not already_assigned else state.tasks_running,
        inflight=MappingProxyType({**state.inflight, task_id: updated}),
    )


def _on_task_done(state: _State, task_id: str, elapsed: float) -> _State:
    entry = state.inflight.get(task_id)
    fn_name = entry.name.split("(")[0] if entry else "unknown"
    fn_stats = {**state.task_fn_stats}
    fn_stats[fn_name] = (*fn_stats.get(fn_name, ()), elapsed)
    inflight = dict(state.inflight)
    inflight.pop(task_id, None)
    per_node = dict(state.tasks_per_node)
    if entry and entry.node_id >= 0:
        per_node[entry.node_id] = per_node.get(entry.node_id, 0) + 1
    return replace(
        state, tasks_running=max(0, state.tasks_running - 1),
        tasks_done=state.tasks_done + 1,
        task_latencies=(*state.task_latencies, elapsed),
        inflight=MappingProxyType(inflight),
        task_fn_stats=MappingProxyType(fn_stats),
        tasks_per_node=MappingProxyType(per_node),
    )


def _on_task_failed(state: _State, task_id: str) -> _State:
    entry = state.inflight.get(task_id)
    fn_name = entry.name.split("(")[0] if entry else "unknown"
    fn_failed = {**state.task_fn_failed}
    fn_failed[fn_name] = fn_failed.get(fn_name, 0) + 1
    inflight = dict(state.inflight)
    inflight.pop(task_id, None)
    return replace(
        state, tasks_running=max(0, state.tasks_running - 1),
        tasks_failed=state.tasks_failed + 1,
        inflight=MappingProxyType(inflight),
        task_fn_failed=MappingProxyType(fn_failed),
    )


def _on_broadcast_partial(state: _State, task_id: str) -> _State:
    if task_id not in state.inflight:
        return state
    entry = state.inflight[task_id]
    updated = replace(entry, broadcast_done=entry.broadcast_done + 1)
    return replace(state, inflight=MappingProxyType({**state.inflight, task_id: updated}))


def _on_metric(state: _State, node_id: int, name: str, value: float) -> _State:
    node_metrics = dict(state.metrics.get(node_id, MappingProxyType({})))
    node_metrics[name] = value
    new_metrics = MappingProxyType({
        **state.metrics, node_id: MappingProxyType(node_metrics),
    })
    return replace(state, metrics=new_metrics)


def _on_desired_changed(state: _State, desired: int) -> _State:
    match desired:
        case d if d > state.desired_nodes:
            reconciler = "scaling_up"
        case d if d < len(state.nodes):
            reconciler = "draining"
        case _:
            reconciler = "watching"
    return replace(state, desired_nodes=desired, reconciler_state=reconciler)


def _on_spawn_nodes(
    state: _State, instances: tuple[Instance, ...], cluster: Cluster | None = None,
) -> _State:
    return replace(
        state,
        pending_nodes=state.pending_nodes + len(instances),
        instances=(*state.instances, *instances),
        total_nodes=state.total_nodes + len(instances),
        cluster=cluster or state.cluster,
    )


def _on_node_joined(state: _State) -> _State:
    pending = max(0, state.pending_nodes - 1)
    reconciler = "watching" if pending == 0 and state.draining_nodes == 0 else state.reconciler_state
    return replace(state, pending_nodes=pending, reconciler_state=reconciler)


def _on_drain_node(state: _State) -> _State:
    return replace(state, draining_nodes=state.draining_nodes + 1, reconciler_state="draining")


def _on_drain_complete(state: _State, node_id: int = -1) -> _State:
    draining = max(0, state.draining_nodes - 1)
    reconciler = "watching" if draining == 0 else state.reconciler_state
    nodes = state.nodes
    if node_id >= 0 and node_id in nodes:
        nodes = MappingProxyType({k: v for k, v in nodes.items() if k != node_id})
    return replace(
        state,
        draining_nodes=draining,
        reconciler_state=reconciler,
        nodes=nodes,
        total_nodes=max(0, state.total_nodes - 1),
    )


def _on_node_lost(state: _State, node_id: int) -> _State:
    spinners = MappingProxyType({k: v for k, v in state.bootstrap_spinners.items() if k != node_id})
    started = MappingProxyType({k: v for k, v in state.bootstrap_started.items() if k != node_id})
    progress = MappingProxyType({k: v for k, v in state.progress_lines.items() if k != node_id})
    nodes = MappingProxyType({k: v for k, v in state.nodes.items() if k != node_id})
    metrics = MappingProxyType({k: v for k, v in state.metrics.items() if k != node_id})
    return replace(
        state,
        bootstrap_spinners=spinners,
        bootstrap_started=started,
        progress_lines=progress,
        nodes=nodes,
        metrics=metrics,
    )


def _on_reconciler_node_lost(state: _State) -> _State:
    pending = max(0, state.pending_nodes - 1)
    return replace(state, pending_nodes=pending)


def _throughput(state: _State, now: float | None = None) -> float:
    if not state.tasks_done:
        return 0.0
    ts = now if now is not None else time.monotonic()
    elapsed_min = (ts - state.first_task_at) / 60
    return state.tasks_done / elapsed_min if elapsed_min > 0 else 0.0
