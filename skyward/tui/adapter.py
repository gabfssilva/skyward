"""Adapter: ``SessionView`` (real session state) → ``UiCluster`` (view-model).

Pure functions — no Textual, no Rich, no Casty.  This is the only seam
between the live :class:`~skyward.api.views.PoolView` and the UI's
:class:`~skyward.tui.model.UiCluster`, so Phase 1's renderers stay
unchanged.  Logs and per-node ready times are supplied by the caller
(:class:`~skyward.tui.sources.ProjectionSource`) because the projection
does not retain either.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from skyward.api.views import NodeStatus, NodeView, PoolView
from skyward.tui.model import UiCluster, UiLog, UiNode, UiStatus

__all__ = ["view_to_ui"]

_EMPTY_LOGS: Mapping[int, tuple[UiLog, ...]] = MappingProxyType({})
_EMPTY_TICKS: Mapping[int, int] = MappingProxyType({})


def _avg(metrics: Mapping[str, float], prefix: str) -> float:
    """Average a metric, exact-key first then ``prefix_*`` (multi-GPU).

    Mirrors the console's ``_find_metrics``: an exact key wins (single GPU
    or an already-aggregated value); otherwise indexed siblings such as
    ``gpu_util_0``/``gpu_util_1`` are averaged.  Returns ``0.0`` when absent.
    """
    if prefix in metrics:
        return metrics[prefix]
    vals = [v for k, v in metrics.items() if k.startswith(f"{prefix}_")]
    return sum(vals) / len(vals) if vals else 0.0


def _ui_status(node: NodeView) -> UiStatus:
    """Map a real :class:`NodeStatus` to a :class:`UiStatus`.

    The real lifecycle keeps a node in ``SSH`` while bootstrap phases run,
    flipping to ``BOOTSTRAPPING`` only once bootstrap finishes and the
    worker starts.  Collapse both into ``UiStatus.BOOTSTRAPPING`` so the
    phase bar shows for the whole bootstrap window, matching the mockup.
    """
    match node.status:
        case NodeStatus.READY:
            return UiStatus.READY
        case NodeStatus.BOOTSTRAPPING:
            return UiStatus.BOOTSTRAPPING
        case NodeStatus.SSH:
            return UiStatus.BOOTSTRAPPING if node.bootstrap is not None else UiStatus.SSH
        case _:
            return UiStatus.WAITING


def _ui_node(
    node: NodeView,
    *,
    logs: tuple[UiLog, ...],
    ready_tick: int | None,
) -> UiNode:
    status = _ui_status(node)
    inst = node.instance
    ip = (inst.private_ip or inst.ip) if inst else None

    bootstrap = node.bootstrap
    phases = bootstrap.phases if bootstrap else ()
    if bootstrap and bootstrap.active in phases:
        phase_idx = phases.index(bootstrap.active)
    elif bootstrap:
        phase_idx = len(bootstrap.completed)
    else:
        phase_idx = 0

    m = node.metrics
    return UiNode(
        id=f"node-{node.node_id}",
        rank=node.node_id,
        role="coord" if node.node_id == 0 else "worker",
        ip=ip or "—",
        status=status,
        gpu=_avg(m, "gpu_util"),
        mem=_avg(m, "gpu_mem_mb") / 1024,
        temp=_avg(m, "gpu_temp"),
        phases=phases,
        phase_idx=phase_idx,
        started_at=ready_tick if status is UiStatus.READY else None,
        logs=logs,
    )


def _provider(pool: PoolView) -> str:
    if pool.cluster is not None and pool.cluster.spec.provider:
        return pool.cluster.spec.provider
    return ""


def _accel(pool: PoolView) -> str:
    accel = pool.cluster.spec.accelerator if pool.cluster is not None else None
    if accel is None and pool.instances:
        accel = pool.instances[0].offer.instance_type.accelerator
    if accel is None:
        return ""
    return f"{accel.name} {accel.memory}".strip() if accel.memory else accel.name


def _region(pool: PoolView) -> str:
    if pool.instances and pool.instances[0].region:
        return pool.instances[0].region
    return pool.cluster.spec.region if pool.cluster is not None else ""


def _hourly_cost(pool: PoolView) -> float:
    return sum(
        (i.offer.spot_price if i.spot else i.offer.on_demand_price) or 0.0
        for i in pool.instances
    )


def view_to_ui(
    pool: PoolView,
    *,
    t: int,
    base_elapsed: int,
    cluster_log: tuple[UiLog, ...] = (),
    node_logs: Mapping[int, tuple[UiLog, ...]] = _EMPTY_LOGS,
    ready_ticks: Mapping[int, int] = _EMPTY_TICKS,
) -> UiCluster:
    """Map a live :class:`PoolView` onto the UI's :class:`UiCluster`.

    Parameters
    ----------
    pool : PoolView
        The pool to render.
    t : int
        Elapsed seconds since the UI attached (drives the topbar clock).
    base_elapsed : int
        Offset added to ``t`` so ``clock(elapsed)`` reads real wall-clock.
    cluster_log : tuple[UiLog, ...]
        Cluster-wide log lines (the projection does not retain these).
    node_logs : Mapping[int, tuple[UiLog, ...]]
        Per-node log lines, keyed by ``node_id``.
    ready_ticks : Mapping[int, int]
        ``node_id -> tick`` at which the node was first seen ready, used for
        the uptime display.

    Returns
    -------
    UiCluster
        The immutable snapshot the renderers draw.
    """
    nodes = tuple(
        _ui_node(
            pool.nodes[nid],
            logs=node_logs.get(nid, ()),
            ready_tick=ready_ticks.get(nid),
        )
        for nid in sorted(pool.nodes)
    )
    allocation = pool.cluster.spec.allocation if pool.cluster is not None else ""
    return UiCluster(
        name=pool.name,
        provider=_provider(pool),
        accel=_accel(pool),
        region=_region(pool),
        allocation=allocation,
        hourly_cost=_hourly_cost(pool),
        t=t,
        base_elapsed=base_elapsed,
        nodes=nodes,
        cluster_log=cluster_log,
    )
