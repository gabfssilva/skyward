"""Convert Skyward view dataclasses to JSON-serializable dicts."""

from __future__ import annotations

import re
from typing import Any

from skyward.api.views import NodeView, PoolView, TasksView


def pool_view_to_dict(
    view: PoolView, logs: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Serialize a ``PoolView`` to a plain dict matching the TS protocol.

    Parameters
    ----------
    view
        The frozen ``PoolView`` from the projection.
    logs
        Pre-serialized log entries (from ``Log.Emitted`` events).
    """
    return {
        "name": view.name,
        "phase": view.phase.name.lower(),
        "total_nodes": view.total_nodes,
        "started_at": view.started_at,
        "ready_at": view.ready_at,
        "cost_per_hour": 0.0,
        "cost_total": 0.0,
        "nodes": {str(k): _node(v) for k, v in view.nodes.items()},
        "tasks": _tasks(view.tasks),
        "scaling": {
            "desired": view.scaling.desired,
            "pending": view.scaling.pending,
            "draining": view.scaling.draining,
            "reconciler_state": view.scaling.reconciler_state,
            "is_elastic": view.scaling.is_elastic,
            "min_nodes": view.scaling.min_nodes,
            "max_nodes": view.scaling.max_nodes,
        },
        "logs": logs or [],
    }


def _node(n: NodeView) -> dict[str, Any]:
    """Serialize a single ``NodeView``."""
    d: dict[str, Any] = {
        "node_id": n.node_id,
        "status": n.status.name.lower(),
        "metrics": dict(n.metrics),
    }
    if n.instance:
        d["ip"] = n.instance.ip or n.instance.private_ip
        accel = n.instance.offer.instance_type.accelerator
        d["accelerator"] = accel.name if accel else None
    if n.bootstrap:
        d["bootstrap"] = {
            "phases": list(n.bootstrap.phases),
            "completed": list(n.bootstrap.completed),
            "active": n.bootstrap.active,
            "output": n.bootstrap.output,
        }
    return d


def _tasks(t: TasksView) -> dict[str, Any]:
    """Serialize ``TasksView`` with pre-computed stats."""
    lats = list(t.latencies) if t.latencies else []
    avg_lat = sum(lats) / len(lats) if lats else 0.0

    fn_summary: dict[str, Any] = {}
    for name, times in t.fn_stats.items():
        vals = list(times)
        failed = t.fn_failed.get(name, 0)
        fn_summary[name] = {
            "calls": len(vals),
            "avg": sum(vals) / len(vals) if vals else 0,
            "min": min(vals) if vals else 0,
            "max": max(vals) if vals else 0,
            "failed": failed,
        }

    return {
        "queued": t.queued,
        "running": t.running,
        "done": t.done,
        "failed": t.failed,
        "inflight": {
            k: {
                "task_id": e.task_id,
                "name": e.name,
                "kind": e.kind,
                "started_at": e.started_at,
                "node_id": e.node_id,
                "broadcast_total": e.broadcast_total,
                "broadcast_done": e.broadcast_done,
            }
            for k, e in t.inflight.items()
        },
        "throughput": t.throughput,
        "avg_latency": round(avg_lat, 4),
        "tasks_per_node": {str(k): v for k, v in t.tasks_per_node.items()},
        "fn_summary": fn_summary,
        "first_task_at": t.first_task_at,
    }


def event_to_dict(event: object) -> dict[str, Any] | None:
    """Convert a ``SessionEvent`` to a JSON-serializable dict.

    Returns ``None`` for events that don't carry a ``pool_name``
    (i.e. events that don't need forwarding to the extension).
    """
    pool = getattr(event, "pool_name", None)
    if pool is None:
        return None

    qualname = type(event).__qualname__.split(".")
    event_name = (
        f"{qualname[-2].lower()}.{_camel_to_snake(qualname[-1])}"
        if len(qualname) >= 2
        else _camel_to_snake(qualname[-1])
    )

    data: dict[str, Any] = {}
    for field_name in getattr(event, "__dataclass_fields__", {}):
        if field_name == "pool_name":
            continue
        val = getattr(event, field_name)
        if hasattr(val, "__dataclass_fields__"):
            data[field_name] = {
                k: getattr(val, k) for k in val.__dataclass_fields__
            }
        else:
            data[field_name] = val

    return {"event": event_name, "pool": pool, "data": data}


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
