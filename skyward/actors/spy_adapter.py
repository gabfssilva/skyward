"""Translates Casty SpyEvent into domain events from skyward.api.events.

The console actor already does a similar translation inline (actor.py _handle_realtime).
This module extracts that mapping into a pure function suitable for SessionProjection.
"""

from __future__ import annotations

import re
import time

from casty import SpyEvent, Terminated

from skyward.actors.messages import (
    BootstrapCommand,
    BootstrapDone,
    BootstrapPhase,
    ClusterReady,
    ConsoleOutput,
    DesiredCountChanged,
    DrainComplete,
    DrainNode,
    Error,
    ExecuteOnNode,
    Log,
    Metric,
    NodeActivated,
    NodeBecameReady,
    NodeConnected,
    NodeJoined,
    NodeLost,
    Preempted,
    Provision,
    ReconcilerNodeLost,
    ShutdownRequested,
    SpawnNodes,
    SubmitBroadcast,
    SubmitTask,
    TaskResult,
    TaskSubmitted,
)
from skyward.actors.node.messages import (
    _Connected,
    _ConnectionFailed,
    _LocalInstallDone,
    _PollResult,
    _PostBootstrapFailed,
    _UserCodeSyncDone,
    _WorkerFailed,
    _WorkerStarted,
)
from skyward.actors.pool.messages import (
    InstancesProvisioned,
    PoolStarted,
    PoolStopped,
    ProvisionFailed,
    RecoverPool,
    StartPool,
    StopPool,
    _ShutdownDone,
)
from skyward.api.events import (
    Error as ErrorEvent,
)
from skyward.api.events import (
    Log as LogEvent,
)
from skyward.api.events import (
    Metric as MetricEvent,
)
from skyward.api.events import (
    Node,
    Pool,
    Scaling,
    SessionEvent,
    Task,
)

__all__ = ["pool_name_from_path", "translate"]

_NODE_ID_RE = re.compile(r"/node-(\d+)")
_POOL_NAME_RE = re.compile(r"(?:^|/)session/pool-([^/]+)")


def _node_id_from_path(path: str) -> int | None:
    if m := _NODE_ID_RE.search(path):
        return int(m.group(1))
    return None


def pool_name_from_path(path: str) -> str | None:
    """Extract pool name from a Casty actor path.

    Parameters
    ----------
    path
        Actor path like ``/session/pool-train/node-0``.

    Returns
    -------
    str | None
        The pool name (e.g. ``"train"``), or None if no match.
    """
    if m := _POOL_NAME_RE.search(path):
        return m.group(1)
    return None


def _format_task(fn: object, args: tuple, kwargs: dict) -> str:
    name = getattr(fn, "__name__", str(fn))
    parts = [repr(a) for a in args]
    parts.extend(f"{k}={v!r}" for k, v in kwargs.items())
    call = f"{name}({', '.join(parts)})"
    return call[:80] + "\u2026" if len(call) > 80 else call


def translate(spy: SpyEvent, pool_name: str) -> SessionEvent | None:  # type: ignore[type-arg]
    """Translate a SpyEvent into a domain event, or None to skip."""
    match spy.event:
        # ── Pool init ───────────────────────────────────────
        case StartPool(spec=spec):
            return Pool.Provisioning(pool_name, spec.nodes.min, time.monotonic())

        case RecoverPool(spec=spec, instances=instances):
            return Pool.Provisioning(pool_name, len(instances), time.monotonic())

        case PoolStarted():
            return None

        # ── No-ops ──────────────────────────────────────────
        case Terminated():
            return None

        case PoolStopped() | _ShutdownDone():
            return None

        case (
            Provision()
            | NodeBecameReady()
            | NodeActivated()
            | NodeConnected()
            | _PollResult()
        ):
            return None

        case ExecuteOnNode():
            return None

        case _LocalInstallDone() | _UserCodeSyncDone():
            return None

        case NodeJoined():
            return None

        case ReconcilerNodeLost():
            return None

        # ── Pool lifecycle ──────────────────────────────────
        case ProvisionFailed(reason=reason):
            return Pool.ProvisionFailed(pool_name, reason)

        case ShutdownRequested() | StopPool():
            return Pool.Stopped(pool_name)

        case ClusterReady():
            return None

        case InstancesProvisioned(cluster=cluster, instances=instances):
            return Pool.Provisioned(pool_name, cluster, instances)

        # ── Node lifecycle ──────────────────────────────────
        case _Connected(instance=ni):
            nid = _node_id_from_path(spy.actor_path)
            if nid is None:
                return None
            return Node.Connected(pool_name, nid, ni.instance if ni else None)

        case _ConnectionFailed(error=error):
            return Node.ConnectionFailed(pool_name, str(error))

        case _WorkerStarted():
            nid = _node_id_from_path(spy.actor_path)
            if nid is None:
                return None
            return Node.Ready(pool_name, nid)

        case _WorkerFailed(error=error):
            return Node.WorkerFailed(pool_name, str(error))

        case NodeLost(node_id=nid, reason=reason):
            return Node.Lost(pool_name, nid, reason)

        case Preempted(reason=reason):
            return Node.Preempted(pool_name, reason)

        # ── Bootstrap ───────────────────────────────────────
        case BootstrapPhase(instance=ni, event="started", phase=p) if p != "bootstrap":
            return Node.Bootstrap.Started(pool_name, ni.node, p)

        case BootstrapPhase(instance=ni, event="completed", phase=p) if p != "bootstrap":
            return Node.Bootstrap.Completed(pool_name, ni.node, p)

        case BootstrapPhase(instance=ni, event="failed", phase=p, error=err):
            return Node.Bootstrap.Failed(pool_name, ni.node, p, err or "")

        case BootstrapPhase():
            return None

        case BootstrapCommand(instance=ni, command=cmd):
            return Node.Bootstrap.Command(pool_name, ni.node, cmd)

        case BootstrapDone(instance=ni, success=ok, error=err):
            return Node.Bootstrap.Done(pool_name, ni.node, ok, err)

        case ConsoleOutput(instance=ni, content=c, overwrite=ow):
            stripped = c.strip()
            if not stripped or stripped.startswith("#"):
                return None
            return Node.Bootstrap.Output(pool_name, ni.node, stripped, overwrite=ow)

        case _PostBootstrapFailed(error=err):
            return ErrorEvent.Occurred(pool_name, f"Post-bootstrap failed: {err}")

        # ── Tasks ───────────────────────────────────────────
        case SubmitTask(task_id=tid, fn=fn, args=args, kwargs=kwargs):
            return Task.Queued(pool_name, tid, _format_task(fn, args, kwargs), "single")

        case SubmitBroadcast(task_id=tid, fn=fn, args=args, kwargs=kwargs):
            return Task.Queued(pool_name, tid, _format_task(fn, args, kwargs), "broadcast")

        case TaskSubmitted(task_id=tid, node_id=nid):
            return Task.Assigned(pool_name, tid, nid)

        case TaskResult(task_id=tid, node_id=nid, error=True):
            return Task.Failed(pool_name, tid, nid, "")

        case TaskResult(task_id=tid, node_id=nid):
            return Task.Completed(pool_name, tid, nid, 0.0)

        # ── Telemetry ──────────────────────────────────────
        case Metric(instance=ni, name=name, value=value):
            return MetricEvent.Sampled(pool_name, ni.node, name, value)

        case Log(instance=ni, line=line, overwrite=ow):
            stripped = line.strip()
            if not stripped:
                return None
            return LogEvent.Emitted(pool_name, ni.node, stripped, overwrite=ow)

        # ── Errors ──────────────────────────────────────────
        case Error(message=message, fatal=fatal):
            return ErrorEvent.Occurred(pool_name, message, fatal)

        # ── Scaling ─────────────────────────────────────────
        case DesiredCountChanged(desired=desired, reason=reason):
            return Scaling.DesiredChanged(pool_name, desired, reason)

        case SpawnNodes(instances=instances):
            return Scaling.Spawning(pool_name, len(instances), instances)

        case DrainNode(node_id=nid):
            return Scaling.Draining(pool_name, nid)

        case DrainComplete(node_id=nid):
            return Scaling.DrainCompleted(pool_name, nid)

        # ── Default ─────────────────────────────────────────
        case _:
            return None
