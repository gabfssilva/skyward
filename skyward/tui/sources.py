"""Data sources that feed the TUI.

The app is agnostic: it holds a :class:`DataSource` and calls ``tick`` /
``snapshot`` on a timer.  Phase 1 ships :class:`SimulatedSource`, a direct
port of the mockup's scripted timeline.  Phase 2 will add a
``ProjectionSource`` wrapping ``SessionProjection.subscribe``.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from skyward.api.views import NodeStatus
from skyward.tui.adapter import view_to_ui
from skyward.tui.model import BASE, UiCluster, UiLog, UiNode, UiStatus

if TYPE_CHECKING:
    from skyward.api.events import Log
    from skyward.api.projection import SessionProjection
    from skyward.api.views import SessionView

__all__ = ["DataSource", "ProjectionSource", "SimulatedSource"]

_PHASES: tuple[str, ...] = ("provision", "image", "deps", "nccl", "ready")
_TARGET: dict[str, tuple[float, float, float]] = {
    "node-0": (86, 62, 64),
    "node-1": (83, 60, 62),
    "node-2": (84, 61, 63),
    "node-3": (88, 64, 66),
}


class DataSource(Protocol):
    """Source of cluster snapshots for the UI."""

    def snapshot(self) -> UiCluster:
        """Return the current immutable cluster state."""
        ...

    def tick(self) -> None:
        """Advance the source by one second."""
        ...


@dataclass(slots=True)
class _Node:
    """Mutable per-node state used only inside the simulation."""

    id: str
    rank: int
    role: str
    ip: str
    status: UiStatus
    gpu: float
    mem: float
    temp: float
    phase_idx: int
    started_at: int | None
    logs: list[UiLog] = field(default_factory=list)


class SimulatedSource:
    """Scripted, self-driving cluster for UI-first testing.

    Ports the mockup's ``useEffect`` loop: node-2 finishes bootstrapping,
    node-1 is preempted around ``t=18`` and recovers, periodic training
    logs accrue, and ready-node metrics jitter toward fixed targets.

    Parameters
    ----------
    seed : int
        Seed for the metric/loss jitter, for reproducible demos.
    """

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)
        self._t = 0
        self._nodes: list[_Node] = [
            _Node("node-0", 0, "coord", "10.0.1.10", UiStatus.READY, 85, 61.4, 63, 5, -258),
            _Node("node-1", 1, "worker", "10.0.1.11", UiStatus.READY, 82, 59.8, 61, 5, -242),
            _Node("node-2", 2, "worker", "10.0.1.12", UiStatus.BOOTSTRAPPING, 0, 0, 41, 0, None),
            _Node("node-3", 3, "worker", "10.0.1.13", UiStatus.READY, 87, 63.1, 65, 5, -271),
        ]
        self._cluster_log: list[UiLog] = []
        self._seed_logs()

    def snapshot(self) -> UiCluster:
        nodes = tuple(
            UiNode(
                id=n.id, rank=n.rank, role=n.role, ip=n.ip, status=n.status,
                gpu=n.gpu, mem=n.mem, temp=n.temp, phases=_PHASES,
                phase_idx=n.phase_idx, started_at=n.started_at, logs=tuple(n.logs),
            )
            for n in self._nodes
        )
        return UiCluster(
            name="h100-finetune-4x", provider="VastAI", accel="H100 80GB",
            region="us-east", allocation="spot", hourly_cost=7.56,
            t=self._t, base_elapsed=258, nodes=nodes, cluster_log=tuple(self._cluster_log),
        )

    def tick(self) -> None:
        self._t += 1
        t = self._t
        self._script(t)
        if t % 5 == 0:
            self._cluster_step(t)
        if t % 9 == 0:
            self._node_chatter(t)
        self._jitter()

    def _rnd(self, a: float, b: float) -> float:
        return a + self._rng.random() * (b - a)

    def _log(self, idx: int, level: str, msg: str, ts: int) -> None:
        self._nodes[idx].logs.append(UiLog(ts, level, msg))

    def _clog(self, node: str, level: str, msg: str, ts: int) -> None:
        self._cluster_log.append(UiLog(ts, level, msg, node=node))

    def _set_status(self, idx: int, status: UiStatus, **fields: float | int | None) -> None:
        node = self._nodes[idx]
        node.status = status
        for key, value in fields.items():
            setattr(node, key, value)

    def _script(self, t: int) -> None:
        match t:
            case 2:
                self._nodes[2].phase_idx = 1
                self._log(2, "INFO", "pulling image ghcr.io/skyward/torch:cu124 …", t)
            case 3:
                self._log(2, "OK", "image ready (cached layers 7/9)", t)
            case 4:
                self._nodes[2].phase_idx = 2
                self._log(2, "INFO", "pip install torch transformers accelerate …", t)
            case 5:
                self._log(2, "OK", "dependencies resolved in 3.1s", t)
            case 6:
                self._nodes[2].phase_idx = 3
                self._log(2, "INFO", "NCCL init · rank 2 · world_size 4", t)
                self._log(2, "INFO", "peer handshake 10.0.1.10 .11 .13", t)
            case 7:
                self._nodes[2].phase_idx = 4
                self._log(2, "OK", "all-reduce probe 312 GB/s", t)
            case 8:
                self._set_status(2, UiStatus.READY, phase_idx=5, started_at=t, gpu=16, mem=9.2, temp=45)
                self._log(2, "OK", "rank 2 online — joined cluster", t)
                self._clog("node-2", "OK", "rank 2 online · joined cluster", t)
            case 18:
                self._set_status(1, UiStatus.PREEMPTED)
                self._log(1, "WARN", "SIGTERM received — spot reclaim in 120s", t)
                self._clog("node-1", "WARN", "spot reclaim notice · VastAI", t)
            case 19:
                self._set_status(1, UiStatus.REPLACING, gpu=0, mem=0, temp=39, started_at=None, phase_idx=0)
                self._log(1, "INFO", "requesting replacement · selection=cheapest", t)
            case 20:
                self._log(1, "OK", "re-bid accepted · VastAI $1.92/hr", t)
                self._clog("node-0", "INFO", "node-1 re-bid accepted · VastAI $1.92/hr", t)
                self._nodes[1].phase_idx = 1
                self._log(1, "INFO", "pulling image (cached)…", t)
            case 22:
                self._nodes[1].phase_idx = 2
                self._log(1, "INFO", "pip install (cached wheels)…", t)
            case 24:
                self._nodes[1].phase_idx = 3
                self._log(1, "INFO", "NCCL re-init · rank 1 rejoining", t)
            case 26:
                self._nodes[1].phase_idx = 4
                self._log(1, "OK", "all-reduce probe 309 GB/s", t)
            case 28:
                self._set_status(1, UiStatus.READY, phase_idx=5, started_at=t, gpu=14, mem=8.6, temp=43)
                self._log(1, "OK", "rank 1 restored — cluster healthy", t)
                self._clog("node-1", "OK", "rank 1 restored · cluster healthy", t)

    def _cluster_step(self, t: int) -> None:
        ready = [n for n in self._nodes if n.status is UiStatus.READY]
        if not ready:
            return
        step = 1240 + (t // 5) * 13
        loss = 1.88 - t * 0.0012 + self._rnd(-0.008, 0.008)
        src = ready[(t // 5) % len(ready)]
        self._clog(src.id, "INFO", f"step {step} · loss {loss:.3f} · {len(ready) * 78} tok/s/gpu", t)

    def _node_chatter(self, t: int) -> None:
        step = 1240 + (t // 5) * 13
        loss = 1.88 - t * 0.0012
        for i, node in enumerate(self._nodes):
            if node.status is not UiStatus.READY:
                continue
            variants = (
                f"step {step} · loss {loss:.3f} · 78 tok/s/gpu",
                f"allreduce 312 GB/s · grad_norm 0.9{i + 1}",
                f"gpu {node.gpu:.0f}% · mem {node.mem:.0f}G · {node.temp:.0f}°C",
            )
            self._log(i, "INFO", variants[(t // 9 + i) % len(variants)], t)

    def _jitter(self) -> None:
        for node in self._nodes:
            if node.status is not UiStatus.READY:
                continue
            tg_gpu, tg_mem, tg_temp = _TARGET[node.id]
            node.gpu = float(max(0, min(99, round(node.gpu + (tg_gpu - node.gpu) * 0.2 + self._rnd(-2, 2)))))
            node.mem = max(0.0, min(80.0, round(node.mem + (tg_mem - node.mem) * 0.2 + self._rnd(-0.6, 0.6), 1)))
            node.temp = float(max(30, min(92, round(node.temp + (tg_temp - node.temp) * 0.2 + self._rnd(-1, 1)))))

    def _seed_logs(self) -> None:
        backlog: dict[int, tuple[tuple[int, str, str], ...]] = {
            0: (
                (-258, "INFO", "instance acquired · VastAI us-east · $1.89/hr"),
                (-255, "OK", "image ready ghcr.io/skyward/torch:cu124"),
                (-252, "OK", "dependencies resolved in 2.8s"),
                (-250, "INFO", "NCCL init · rank 0 · world_size 4 (coordinator)"),
                (-249, "OK", "rank 0 online — cluster bootstrapped"),
            ),
            1: (
                (-242, "INFO", "instance acquired · VastAI us-east"),
                (-239, "OK", "image ready (cached)"),
                (-237, "OK", "all-reduce probe 311 GB/s"),
                (-236, "OK", "rank 1 online — joined cluster"),
            ),
            2: (
                (-1, "INFO", "instance acquired · VastAI us-east · $1.89/hr"),
                (0, "INFO", "starting bootstrap…"),
            ),
            3: (
                (-271, "INFO", "instance acquired · VastAI us-east"),
                (-267, "OK", "image ready (cached)"),
                (-265, "OK", "rank 3 online — joined cluster"),
            ),
        }
        for idx, lines in backlog.items():
            self._nodes[idx].logs.extend(UiLog(ts, lvl, msg) for ts, lvl, msg in lines)
        self._cluster_log.extend(
            UiLog(ts, lvl, msg, node=node)
            for ts, node, lvl, msg in (
                (-250, "node-0", "OK", "cluster bootstrapped · 4 nodes · NCCL/FSDP"),
                (-240, "node-0", "INFO", "training started · global_batch 512"),
                (-120, "node-3", "INFO", "step 1188 · loss 1.93 · 311 tok/s/gpu"),
                (-60, "node-0", "INFO", "step 1214 · loss 1.90 · 312 tok/s/gpu"),
                (-15, "node-0", "OK", "checkpoint saved · step 1238"),
            )
        )


class ProjectionSource:
    """Live cluster snapshots backed by a :class:`SessionProjection`.

    Subscribes to the projection's change and log streams, accumulates the
    log lines (the projection fires them but does not retain them), and maps
    the current ``SessionView`` to a :class:`UiCluster` on every
    :meth:`snapshot`.

    The clock is the source's own: ``t`` counts seconds since the source
    attached, anchored to :data:`BASE` so the existing renderers show real
    local wall-clock time without modification.  Per-node uptime counts from
    the first moment the node was observed ``READY``.

    Callbacks may fire on a different thread than :meth:`snapshot` (the
    in-process session runs its loop on a daemon thread); a lock guards the
    accumulated logs and ready times.

    Parameters
    ----------
    projection : SessionProjection
        The projection to observe.
    pool_name : str | None
        Which pool to render; ``None`` selects the first available pool.
    max_log_lines : int
        Cap on retained log lines per stream (cluster and each node); the
        oldest lines are dropped once exceeded.
    """

    def __init__(
        self,
        projection: SessionProjection,
        *,
        pool_name: str | None = None,
        max_log_lines: int = 1000,
    ) -> None:
        self._proj = projection
        self._pool_name = pool_name
        self._max = max_log_lines
        self._lock = threading.Lock()
        self._cluster_log: list[UiLog] = []
        self._node_logs: dict[int, list[UiLog]] = {}
        self._ready_at: dict[int, float] = {}
        self._start = time.time()
        self._unsubscribe = projection.subscribe(
            on_change=self._on_change, on_log=self._on_log,
        )

    def _resolve_name(self, view: SessionView) -> str | None:
        if self._pool_name is not None:
            return self._pool_name
        return next(iter(view.pools), None)

    def _on_change(self, _old: SessionView, new: SessionView) -> None:
        name = self._resolve_name(new)
        pool = new.pools.get(name) if name is not None else None
        if pool is None:
            return
        now = time.time()
        with self._lock:
            for nid, node in pool.nodes.items():
                if node.status is NodeStatus.READY:
                    self._ready_at.setdefault(nid, now)

    def _on_log(self, log: Log.Emitted) -> None:
        ts = int(time.time() - BASE.timestamp())
        level = log.level.upper()
        with self._lock:
            self._cluster_log.append(
                UiLog(ts, level, log.message, node=f"node-{log.node_id}"),
            )
            node_log = self._node_logs.setdefault(log.node_id, [])
            node_log.append(UiLog(ts, level, log.message))
            if (drop := len(self._cluster_log) - self._max) > 0:
                del self._cluster_log[:drop]
            if (drop := len(node_log) - self._max) > 0:
                del node_log[:drop]

    def snapshot(self) -> UiCluster:
        view = self._proj.view
        name = self._resolve_name(view)
        pool = view.pools.get(name) if name is not None else None
        now = time.time()
        t = int(now - self._start)
        base_elapsed = int(self._start - BASE.timestamp())
        if pool is None:
            return UiCluster(
                name=name or "—", provider="", accel="", region="",
                allocation="", hourly_cost=0.0, t=t, base_elapsed=base_elapsed,
                nodes=(), cluster_log=(),
            )
        with self._lock:
            for nid, node in pool.nodes.items():
                if node.status is NodeStatus.READY:
                    self._ready_at.setdefault(nid, now)
            ready_ticks = {
                nid: int(ep - self._start) for nid, ep in self._ready_at.items()
            }
            cluster_log = tuple(self._cluster_log)
            node_logs = {nid: tuple(rows) for nid, rows in self._node_logs.items()}
        return view_to_ui(
            pool, t=t, base_elapsed=base_elapsed,
            cluster_log=cluster_log, node_logs=node_logs, ready_ticks=ready_ticks,
        )

    def tick(self) -> None:
        """No-op: the clock advances from wall time in :meth:`snapshot`."""

    def close(self) -> None:
        """Unsubscribe from the projection."""
        self._unsubscribe()
