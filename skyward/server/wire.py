"""Wire encoding for the HTTP server.

Two formats coexist:

- ``encode`` / ``decode`` — cloudpickle + lz4 for binary payloads (specs,
  pending functions, results, exceptions). Used for in-band data that
  needs full Python fidelity.
- ``event_to_json`` / ``pool_view_to_json`` and their counterparts —
  JSON-friendly dicts for the SSE event stream and the live ``view`` UI.

The JSON path is asymmetric on purpose:

- **Server → wire** (``_to_json``): generic recursive walk over dataclasses,
  ``Mapping``, sequences, ``Enum``, primitives. Anything else falls back to
  ``repr`` so payloads always serialize.
- **Wire → client** (``event_from_json``, ``pool_view_from_json``): hand-written
  reconstructors per type. No type-hint introspection, no magic. Adding a
  new event type means one branch in :data:`_EVENT_CLASS_BY_QUALNAME` and,
  if it carries a new nested dataclass, one helper.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import cloudpickle
import lz4.frame

from skyward.accelerators.spec import Accelerator
from skyward.api import events as _events
from skyward.api.model import Instance, InstanceType, Offer
from skyward.api.views import (
    BootstrapView,
    NodeStatus,
    NodeView,
    PoolPhase,
    PoolView,
    ScalingView,
    TasksView,
)

if TYPE_CHECKING:
    from skyward.api.events import SessionEvent


def encode(value: Any) -> bytes:
    """Serialize ``value`` to lz4-compressed cloudpickle bytes."""
    return lz4.frame.compress(cloudpickle.dumps(value))


def decode(payload: bytes) -> Any:
    """Reverse of :func:`encode`."""
    return cloudpickle.loads(lz4.frame.decompress(payload))


# ── Server → JSON ────────────────────────────────────────────────


def _to_json(value: Any) -> Any:
    """Recursively convert ``value`` to a JSON-friendly form.

    Parameters
    ----------
    value : Any
        Frozen dataclass, mapping, sequence, enum, or primitive.

    Returns
    -------
    Any
        ``dict`` / ``list`` / primitive that ``json.dumps`` accepts.
        Unknown types fall back to ``repr`` so events never fail to
        serialize on the wire.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple | frozenset | set):
        return [_to_json(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _to_json(v) for k, v in value.items()}
    if hasattr(value, "name") and hasattr(type(value), "__members__"):  # Enum
        return value.name
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_json(getattr(value, f.name)) for f in dataclasses.fields(value)}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except Exception:  # noqa: BLE001
            return repr(value)
    return repr(value)


def event_to_json(event: SessionEvent) -> dict:
    """Serialize a domain event to ``{type: qualname, fields: {...}}``."""
    return {"type": type(event).__qualname__, "fields": _to_json(event)}


def pool_view_to_json(view: PoolView) -> dict:
    """Serialize a ``PoolView`` snapshot via :func:`_to_json`."""
    return _to_json(view)


# ── Wire → typed (hand-written reconstructors) ───────────────────


def _accelerator_from_json(d: Mapping[str, Any] | None) -> Accelerator | None:
    if d is None:
        return None
    metadata = d.get("metadata")
    return Accelerator(
        name=d["name"],
        memory=d.get("memory", ""),
        count=d.get("count", 1),
        metadata=dict(metadata) if isinstance(metadata, Mapping) else None,
    )


def _instance_type_from_json(d: Mapping[str, Any] | None) -> InstanceType | None:
    if d is None:
        return None
    return InstanceType(
        name=d["name"],
        accelerator=_accelerator_from_json(d.get("accelerator")),
        vcpus=d["vcpus"],
        memory_gb=d["memory_gb"],
        architecture=d["architecture"],
        specific=d.get("specific"),
    )


def _offer_from_json(d: Mapping[str, Any] | None) -> Offer | None:
    if d is None:
        return None
    itype = _instance_type_from_json(d["instance_type"])
    if itype is None:
        return None
    return Offer(
        id=d["id"],
        instance_type=itype,
        spot_price=d.get("spot_price"),
        on_demand_price=d.get("on_demand_price"),
        billing_unit=d["billing_unit"],
        specific=d.get("specific"),
    )


def _instance_from_json(d: Mapping[str, Any] | None) -> Instance | None:
    if d is None:
        return None
    offer = _offer_from_json(d["offer"])
    if offer is None:
        return None
    return Instance(
        id=d["id"],
        status=d["status"],
        offer=offer,
        ip=d.get("ip"),
        private_ip=d.get("private_ip"),
        ssh_port=d.get("ssh_port", 22),
        ssh_password=d.get("ssh_password"),
        spot=d.get("spot", False),
        region=d.get("region", ""),
    )


def _bootstrap_from_json(d: Mapping[str, Any] | None) -> BootstrapView | None:
    if d is None:
        return None
    return BootstrapView(
        phases=tuple(d.get("phases", [])),
        completed=frozenset(d.get("completed", [])),
        active=d.get("active", ""),
        output=d.get("output", ""),
    )


def _node_view_from_json(d: Mapping[str, Any]) -> NodeView:
    return NodeView(
        node_id=d["node_id"],
        status=NodeStatus[d.get("status", "WAITING")],
        instance=_instance_from_json(d.get("instance")),
        bootstrap=_bootstrap_from_json(d.get("bootstrap")),
    )


def _tasks_from_json(d: Mapping[str, Any] | None) -> TasksView:
    if not d:
        return TasksView()
    return TasksView(
        queued=d.get("queued", 0),
        running=d.get("running", 0),
        done=d.get("done", 0),
        failed=d.get("failed", 0),
        first_task_at=d.get("first_task_at", 0.0),
    )


def _scaling_from_json(d: Mapping[str, Any] | None) -> ScalingView:
    if not d:
        return ScalingView()
    return ScalingView(
        desired=d.get("desired", 0),
        pending=d.get("pending", 0),
        draining=d.get("draining", 0),
        reconciler_state=d.get("reconciler_state", "watching"),
        is_elastic=d.get("is_elastic", False),
        min_nodes=d.get("min_nodes"),
        max_nodes=d.get("max_nodes"),
    )


def pool_view_from_json(payload: Mapping[str, Any]) -> PoolView:
    """Reconstruct a ``PoolView`` from its JSON snapshot.

    Heavy fields the renderer doesn't depend on (``cluster``, ``spec``)
    are dropped; ``instances`` and per-node ``instance`` are reconstructed
    so badges (cluster size, region, accelerator, cost) render fully.
    """
    nodes = {
        int(k): _node_view_from_json(v)
        for k, v in (payload.get("nodes") or {}).items()
    }
    instances = tuple(
        i for i in (
            _instance_from_json(d) for d in (payload.get("instances") or [])
        ) if i is not None
    )
    return PoolView(
        name=payload.get("name", ""),
        phase=PoolPhase[payload.get("phase", "PROVISIONING")],
        tasks=_tasks_from_json(payload.get("tasks")),
        scaling=_scaling_from_json(payload.get("scaling")),
        total_nodes=payload.get("total_nodes", 0),
        nodes=MappingProxyType(nodes),
        cluster=None,
        instances=instances,
        started_at=payload.get("started_at", 0.0),
        ready_at=payload.get("ready_at", 0.0),
    )


# ── Event dispatch ───────────────────────────────────────────────


def _resolve_event_class(qualname: str) -> type | None:
    """Walk ``Node.Bootstrap.Output`` etc. through :mod:`skyward.api.events`."""
    cls: Any = _events
    for part in qualname.split("."):
        cls = getattr(cls, part, None)
        if cls is None:
            return None
    return cls if dataclasses.is_dataclass(cls) and isinstance(cls, type) else None


def event_from_json(payload: Mapping[str, Any]) -> SessionEvent | None:
    """Reconstruct a domain event from ``{type: qualname, fields: {...}}``.

    Most events carry only primitives — those are reconstructed with
    ``cls(**fields)``. Events with nested dataclasses go through the
    explicit branches below.
    """
    qualname = payload.get("type")
    if not isinstance(qualname, str):
        return None

    cls = _resolve_event_class(qualname)
    if cls is None:
        return None

    fields: Mapping[str, Any] = payload.get("fields") or {}

    match qualname:
        case "Node.Connected":
            return _events.Node.Connected(
                pool_name=fields["pool_name"],
                node_id=fields["node_id"],
                instance=_instance_from_json(fields.get("instance")),
            )
        case "Pool.Provisioned":
            return _events.Pool.Provisioned(
                pool_name=fields["pool_name"],
                cluster=None,  # not reconstructed; renderer degrades
                instances=tuple(
                    i for i in (
                        _instance_from_json(d) for d in fields.get("instances") or []
                    ) if i is not None
                ),
            )
        case "Pool.Reconciled":
            return _events.Pool.Reconciled(
                pool_name=fields["pool_name"],
                snapshot=None,  # not reconstructed
            )

    accepted = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in fields.items() if k in accepted}
    try:
        return cls(**kwargs)
    except (TypeError, ValueError):
        return None
