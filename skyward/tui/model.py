"""View-model for the Textual TUI.

Pure data — no Textual, no Rich, no Skyward internals.  These frozen
dataclasses are a *superset* of the mockup: they carry everything the UI
draws (including ``temp`` and the ``PREEMPTED``/``REPLACING`` states),
decoupled from ``skyward.api.views`` so Phase 1 can render without the
real session.  Phase 2 adds a ``SessionView -> UiCluster`` adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

__all__ = [
    "BASE",
    "UiCluster",
    "UiLog",
    "UiNode",
    "UiStatus",
    "clock",
    "dur",
    "uptime",
]

BASE = datetime(2026, 5, 28, 14, 3, 12)
"""Wall-clock origin for the simulated session, mirroring the mockup."""


class UiStatus(Enum):
    """Node lifecycle state as shown in the UI.

    Superset of ``skyward.api.views.NodeStatus``: ``WAITING``/``SSH`` map
    the real lifecycle, while ``PREEMPTED``/``REPLACING`` model the
    spot-recovery visuals that the projection does not yet expose.
    """

    WAITING = "waiting"
    SSH = "ssh"
    BOOTSTRAPPING = "bootstrapping"
    READY = "ready"
    PREEMPTED = "preempted"
    REPLACING = "replacing"


@dataclass(frozen=True, slots=True)
class UiLog:
    """A single log line.

    Parameters
    ----------
    ts : int
        Seconds relative to :data:`BASE` (may be negative for backlog).
    level : str
        One of ``INFO``/``OK``/``WARN``/``ERR``.
    msg : str
        Message body.
    node : str | None
        Node id, when the line belongs to the cluster log.
    """

    ts: int
    level: str
    msg: str
    node: str | None = None


@dataclass(frozen=True, slots=True)
class UiNode:
    """Per-node state.

    Parameters
    ----------
    id : str
        Node identifier, e.g. ``node-0``.
    rank : int
        Zero-based rank (``0`` is the coordinator).
    role : str
        ``coord`` or ``worker``.
    ip : str
        Private address.
    status : UiStatus
        Lifecycle state.
    gpu, mem, temp : float
        Live metrics (gpu utilisation %, gpu memory GB, temperature °C).
    phases : tuple[str, ...]
        Ordered bootstrap phase names.
    phase_idx : int
        Number of completed phases; ``== len(phases)`` means fully ready.
    started_at : int | None
        Tick at which the node became ready, or ``None``.
    logs : tuple[UiLog, ...]
        Per-node log backlog.
    """

    id: str
    rank: int
    role: str
    ip: str
    status: UiStatus
    gpu: float
    mem: float
    temp: float
    phases: tuple[str, ...]
    phase_idx: int
    started_at: int | None
    logs: tuple[UiLog, ...]


@dataclass(frozen=True, slots=True)
class UiCluster:
    """Whole-cluster snapshot the UI renders.

    Parameters
    ----------
    name : str
        Pool name.
    provider, accel, region, allocation : str
        Display metadata for the summary header.
    hourly_cost : float
        Aggregate ``$/hr`` across nodes.
    t : int
        Elapsed ticks since the UI started.
    base_elapsed : int
        Seconds the session had already been running at ``t == 0``.
    nodes : tuple[UiNode, ...]
        All nodes, in display order.
    cluster_log : tuple[UiLog, ...]
        Cluster-wide log stream.
    """

    name: str
    provider: str
    accel: str
    region: str
    allocation: str
    hourly_cost: float
    t: int
    base_elapsed: int
    nodes: tuple[UiNode, ...]
    cluster_log: tuple[UiLog, ...]

    @property
    def ready_count(self) -> int:
        """Number of nodes in :attr:`UiStatus.READY`."""
        return sum(1 for n in self.nodes if n.status is UiStatus.READY)

    @property
    def elapsed(self) -> int:
        """Total session seconds (``base_elapsed + t``)."""
        return self.base_elapsed + self.t

    @property
    def cost(self) -> float:
        """Cumulative spend in dollars."""
        return self.hourly_cost * (self.elapsed / 3600)


def clock(sec: int) -> str:
    """Return ``HH:MM:SS`` for ``sec`` seconds past :data:`BASE`."""
    return (BASE + timedelta(seconds=sec)).strftime("%H:%M:%S")


def dur(sec: int) -> str:
    """Format a positive duration as ``Hh Mm`` / ``Mm Ss`` / ``Ss``."""
    sec = max(0, int(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def uptime(node: UiNode, t: int) -> str:
    """Return a node's uptime string, or ``—`` when not running."""
    if node.status is UiStatus.READY and node.started_at is not None:
        return dur(t - node.started_at)
    return "—"
