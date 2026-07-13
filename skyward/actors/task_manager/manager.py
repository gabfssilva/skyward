"""Task manager — queue, round-robin, and per-node slots as plain asyncio.

Dispatches submitted tasks to nodes with free slots, retries tasks lost
to infrastructure failures (``NodeInterruptedError``), and fans broadcasts out
to every registered node.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from skyward.actors.messages import NodeInterruptedError, NodeTarget, PressureReport
from skyward.api import events
from skyward.observability.logger import logger

log = logger.bind(actor="task_manager")


def _no_emit(_event: events.SessionEvent) -> None:
    pass


class NodeExecutor(Protocol):
    async def execute(
        self,
        fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        timeout: float,
        task_id: str,
    ) -> Any: ...


@dataclass(slots=True)
class _Slot:
    node: NodeExecutor
    total: int
    used: int = 0

    @property
    def free(self) -> bool:
        return self.used < self.total


@dataclass(slots=True)
class _Pending:
    fn: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    timeout: float
    task_id: str
    target: NodeTarget | None
    future: asyncio.Future[Any]
    attempts: int = 0


def _target_rank(target: NodeTarget) -> int:
    return 0 if target == "head" else int(target)


class TaskManager:
    """Round-robin task dispatch with per-node slots and backpressure."""

    def __init__(
        self,
        retry_on_interruption: int = 3,
        *,
        emit: events.Emit | None = None,
        pool_name: str = "",
    ) -> None:
        self._retry_on_interruption = retry_on_interruption
        self._emit = emit or _no_emit
        self._pool_name = pool_name
        self._nodes: dict[int, _Slot] = {}
        self._queue: deque[_Pending] = deque()
        self._round_robin = 0
        self._running: dict[int, set[asyncio.Task[None]]] = {}
        self._on_pressure: Callable[[PressureReport], None] | None = None
        log.info("Task manager started")

    # ── node membership ──────────────────────────────────────────

    def node_available(self, node_id: int, node: NodeExecutor, slots: int) -> None:
        log.info("Node {nid} available ({slots} slots)", nid=node_id, slots=slots)
        self._nodes[node_id] = _Slot(node=node, total=slots)
        self._drain()
        self._pressure()

    def node_unavailable(self, node_id: int) -> None:
        log.info("Node {nid} unavailable", nid=node_id)
        self._nodes.pop(node_id, None)
        self._pressure()

    def set_pressure_observer(
        self, observer: Callable[[PressureReport], None],
    ) -> None:
        self._on_pressure = observer
        self._pressure()

    # ── submission ────────────────────────────────────────────────

    async def submit(
        self,
        fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        timeout: float = 600.0,
        task_id: str = "",
        target: NodeTarget | None = None,
    ) -> Any:
        tid = task_id or uuid4().hex[:8]
        if target is not None and _target_rank(target) not in self._nodes:
            rank = _target_rank(target)
            log.warning("Targeted task to absent rank {r}", r=rank)
            raise RuntimeError(f"no such node rank {rank}")
        pending = _Pending(
            fn=fn, args=args, kwargs=kwargs, timeout=timeout,
            task_id=tid, target=target,
            future=asyncio.get_running_loop().create_future(),
        )
        self._queue.append(pending)
        self._drain()
        self._pressure()
        return await pending.future

    async def broadcast(
        self,
        fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        timeout: float = 600.0,
        task_id: str = "",
    ) -> list[Any]:
        """Run on every registered node; failures become exception values.

        Returns one entry per node in rank order — the return value on
        success, or the raised exception object on failure (matching the
        historical broadcast semantics consumed by ``ComputePool``).
        """
        tid = task_id or uuid4().hex[:8]
        targets = sorted(self._nodes)
        log.debug("Broadcasting task to {n} nodes", n=len(targets))

        async def _one(nid: int) -> Any:
            slot = self._nodes.get(nid)
            if slot is None:
                return RuntimeError(f"Node {nid} lost during broadcast")
            slot.used += 1
            self._emit(events.Task.Assigned(self._pool_name, tid, nid))
            started = time.monotonic()
            try:
                value = await slot.node.execute(
                    fn, args, kwargs, timeout=timeout, task_id=tid,
                )
            except NodeInterruptedError:
                return RuntimeError(f"Node {nid} lost during broadcast")
            except Exception as e:  # noqa: BLE001 — errors are values here
                self._emit(events.Task.Failed(self._pool_name, tid, nid, ""))
                return e
            else:
                self._emit(events.Task.Completed(
                    self._pool_name, tid, nid, time.monotonic() - started,
                ))
                return value
            finally:
                current = self._nodes.get(nid)
                if current is slot:
                    slot.used = max(0, slot.used - 1)
                self._drain()
                self._pressure()

        return list(await asyncio.gather(*(_one(nid) for nid in targets)))

    # ── internals ─────────────────────────────────────────────────

    def _pick_with_free_slot(self) -> int | None:
        node_ids = sorted(self._nodes)
        if not node_ids:
            return None
        for i in range(len(node_ids)):
            nid = node_ids[(self._round_robin + i) % len(node_ids)]
            if self._nodes[nid].free:
                return nid
        return None

    def _drain(self) -> None:
        remaining: deque[_Pending] = deque()
        while self._queue:
            pending = self._queue.popleft()
            if pending.future.cancelled():
                continue
            if pending.target is not None:
                rank = _target_rank(pending.target)
                slot = self._nodes.get(rank)
                if slot is not None and slot.free:
                    self._dispatch(rank, pending)
                else:
                    remaining.append(pending)
                continue
            nid = self._pick_with_free_slot()
            if nid is None:
                remaining.append(pending)
                continue
            self._dispatch(nid, pending)
            self._round_robin += 1
        self._queue = remaining

    def _dispatch(self, nid: int, pending: _Pending) -> None:
        slot = self._nodes[nid]
        slot.used += 1
        self._emit(events.Task.Assigned(self._pool_name, pending.task_id, nid))
        log.debug("Dispatching task {tid} to node {nid}", tid=pending.task_id, nid=nid)
        task = asyncio.create_task(self._run_one(nid, slot, pending))
        self._running.setdefault(nid, set()).add(task)
        task.add_done_callback(lambda t, _n=nid: self._running.get(_n, set()).discard(t))

    async def _run_one(self, nid: int, slot: _Slot, pending: _Pending) -> None:
        tid = pending.task_id
        started = time.monotonic()
        try:
            value = await slot.node.execute(
                pending.fn, pending.args, pending.kwargs,
                timeout=pending.timeout, task_id=tid,
            )
        except NodeInterruptedError as e:
            pending.attempts += 1
            if pending.attempts <= self._retry_on_interruption:
                log.info(
                    "Task {tid} interrupted on node {nid}, re-enqueuing (attempt {att}/{max})",
                    tid=tid, nid=nid, att=pending.attempts, max=self._retry_on_interruption,
                )
                self._queue.append(pending)
            else:
                log.warning(
                    "Task {tid} interrupted on node {nid}, retries exhausted ({max})",
                    tid=tid, nid=nid, max=self._retry_on_interruption,
                )
                self._emit(events.Task.Failed(self._pool_name, tid, nid, ""))
                if not pending.future.done():
                    pending.future.set_exception(e)
        except Exception as e:  # noqa: BLE001 — user errors surface via future
            self._emit(events.Task.Failed(self._pool_name, tid, nid, ""))
            if not pending.future.done():
                pending.future.set_exception(e)
        else:
            self._emit(events.Task.Completed(
                self._pool_name, tid, nid, time.monotonic() - started,
            ))
            if not pending.future.done():
                pending.future.set_result(value)
        finally:
            current = self._nodes.get(nid)
            if current is slot:
                slot.used = max(0, slot.used - 1)
            self._drain()
            self._pressure()

    def _pressure(self) -> None:
        if self._on_pressure is None:
            return
        self._on_pressure(PressureReport(
            queued=len(self._queue),
            inflight=sum(s.used for s in self._nodes.values()),
            total_capacity=sum(s.total for s in self._nodes.values()),
            node_count=len(self._nodes),
        ))
