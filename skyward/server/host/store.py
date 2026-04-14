"""Store lifecycle, transaction context, and repositories for persistence.

Two ``aiosqlite`` connections back the store: a writer serialized by an
``asyncio.Lock`` (all transactions run through it) and a reader used by
queries and the event tailer. WAL mode lets readers proceed concurrently
with the single writer.

Repository methods encode domain ADTs into the schema defined in
``skyward.server.host.schema`` using ``match``/``case`` to discriminate
on sum-type variants (``ComputeStatus``, ``NodeStatus``) and the wire
codec (``to_dict``/``from_dict``) to serialize complex fields
(``ComputeSpec``, chosen ``Spec``, provider config) into JSON columns.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from skyward.api.spec import Spec
from skyward.server.host.domain import (
    Blob,
    BlobId,
    BlobKind,
    Broadcast,
    CancelledExec,
    Compute,
    ComputeName,
    ComputeSpec,
    ComputeStatus,
    Dispatching,
    ErrorId,
    ExecutionId,
    ExecutionStatus,
    Failed,
    FailedExec,
    FailedRes,
    GroupId,
    GroupMember,
    InterruptedExec,
    InterruptedRes,
    Node,
    NodeBootstrapping,
    NodeConnecting,
    NodeLost,
    NodeReady,
    NodeStatus,
    NodeWaiting,
    PendingRes,
    Provider,
    ProviderType,
    Provisioning,
    Queued,
    Ready,
    ResultStatus,
    Run,
    RunningExec,
    RunningRes,
    Stopped,
    Stopping,
    SucceededExec,
    SucceededRes,
    Task,
    TaskExecution,
    TaskExecutionKind,
    TaskKey,
    TaskResult,
)
from skyward.server.host.migrate import apply_schema


def _wire() -> Any:
    """Deferred import of the wire codec to avoid circular imports."""
    import skyward.server.wire as wire

    return wire

type DBPath = str
type SqlParams = Sequence[Any]


@dataclass(frozen=True, slots=True)
class EventRow:
    """A row materialized from the ``events`` table."""

    id: int
    aggregate: str
    type: str
    payload: dict[str, Any]
    created_at: datetime


def _to_ts(dt: datetime) -> float:
    return dt.timestamp()


def _from_ts(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)


def _to_ts_opt(dt: datetime | None) -> float | None:
    return None if dt is None else dt.timestamp()


def _from_ts_opt(ts: float | None) -> datetime | None:
    return None if ts is None else datetime.fromtimestamp(ts, tz=UTC)


type _ComputeStatusCols = tuple[
    str,            # status_tag
    float | None,   # started_at
    float | None,   # stopped_at
    float | None,   # stopping_since
    float | None,   # failed_at
    str | None,     # failure_reason
    int | None,     # nodes_ready
    float | None,   # last_activity_at
    int | None,     # chosen_spec_ordinal
    str | None,     # chosen_spec_json
]


def _encode_compute_status(s: ComputeStatus) -> _ComputeStatusCols:
    match s:
        case Provisioning(started_at=t):
            return ("provisioning", _to_ts(t), None, None, None, None, None, None, None, None)
        case Ready(started_at=t, chosen=chosen, nodes_ready=n, last_activity_at=la):
            return (
                "ready",
                _to_ts(t),
                None,
                None,
                None,
                None,
                n,
                _to_ts(la),
                None,
                json.dumps(_wire().to_dict(chosen)),
            )
        case Stopping(started_at=t, stopping_since=since):
            return ("stopping", _to_ts(t), None, _to_ts(since), None, None, None, None, None, None)
        case Stopped(started_at=t, stopped_at=st):
            return ("stopped", _to_ts(t), _to_ts(st), None, None, None, None, None, None, None)
        case Failed(failed_at=fa, reason=r):
            return ("failed", None, None, None, _to_ts(fa), r, None, None, None, None)


def _decode_compute_status(row: aiosqlite.Row) -> ComputeStatus:
    tag: str = row["status_tag"]
    match tag:
        case "provisioning":
            return Provisioning(started_at=_from_ts(row["started_at"]))
        case "ready":
            chosen_json: str = row["chosen_spec_json"]
            chosen = _wire().from_dict(json.loads(chosen_json), Spec)
            return Ready(
                started_at=_from_ts(row["started_at"]),
                chosen=chosen,
                nodes_ready=int(row["nodes_ready"]),
                last_activity_at=_from_ts(row["last_activity_at"]),
            )
        case "stopping":
            return Stopping(
                started_at=_from_ts(row["started_at"]),
                stopping_since=_from_ts(row["stopping_since"]),
            )
        case "stopped":
            return Stopped(
                started_at=_from_ts(row["started_at"]),
                stopped_at=_from_ts(row["stopped_at"]),
            )
        case "failed":
            return Failed(
                failed_at=_from_ts(row["failed_at"]),
                reason=row["failure_reason"],
            )
        case _:
            raise ValueError(f"Unknown compute status_tag {tag!r}")


type _NodeStatusCols = tuple[
    str,            # status_tag
    float | None,   # status_since
    str | None,     # bootstrap_phase
    float | None,   # lost_at
    str | None,     # lost_reason
]


def _encode_node_status(s: NodeStatus) -> _NodeStatusCols:
    match s:
        case NodeWaiting():
            return ("waiting", None, None, None, None)
        case NodeConnecting(since=t):
            return ("connecting", _to_ts(t), None, None, None)
        case NodeBootstrapping(since=t, phase=p):
            return ("bootstrapping", _to_ts(t), p, None, None)
        case NodeReady(since=t):
            return ("ready", _to_ts(t), None, None, None)
        case NodeLost(at=t, reason=r):
            return ("lost", None, None, _to_ts(t), r)


def _decode_node_status(row: aiosqlite.Row) -> NodeStatus:
    tag: str = row["status_tag"]
    match tag:
        case "waiting":
            return NodeWaiting()
        case "connecting":
            return NodeConnecting(since=_from_ts(row["status_since"]))
        case "bootstrapping":
            return NodeBootstrapping(
                since=_from_ts(row["status_since"]),
                phase=row["bootstrap_phase"],
            )
        case "ready":
            return NodeReady(since=_from_ts(row["status_since"]))
        case "lost":
            return NodeLost(at=_from_ts(row["lost_at"]), reason=row["lost_reason"])
        case _:
            raise ValueError(f"Unknown node status_tag {tag!r}")


type _ExecKindCols = tuple[str, str | None]


def _encode_exec_kind(k: TaskExecutionKind) -> _ExecKindCols:
    match k:
        case Run():
            return ("run", None)
        case Broadcast():
            return ("broadcast", None)
        case GroupMember(group=g):
            return ("group_member", g)


def _decode_exec_kind(row: aiosqlite.Row) -> TaskExecutionKind:
    tag: str = row["kind_tag"]
    match tag:
        case "run":
            return Run()
        case "broadcast":
            return Broadcast()
        case "group_member":
            return GroupMember(group=row["group_id"])
        case _:
            raise ValueError(f"Unknown execution kind_tag {tag!r}")


type _ExecStatusCols = tuple[
    str,            # status_tag
    float | None,   # finished_at
    float | None,   # interrupted_at
    str | None,     # interrupt_reason
    float | None,   # cancelled_at
    str | None,     # cancel_reason
]


def _encode_execution_status(s: ExecutionStatus) -> _ExecStatusCols:
    match s:
        case Queued():
            return ("queued", None, None, None, None, None)
        case Dispatching():
            return ("dispatching", None, None, None, None, None)
        case RunningExec():
            return ("running", None, None, None, None, None)
        case SucceededExec(finished_at=t):
            return ("succeeded", _to_ts(t), None, None, None, None)
        case FailedExec(finished_at=t):
            return ("failed", _to_ts(t), None, None, None, None)
        case InterruptedExec(interrupted_at=t, reason=r):
            return ("interrupted", None, _to_ts(t), r, None, None)
        case CancelledExec(cancelled_at=t, reason=r):
            return ("cancelled", None, None, None, _to_ts(t), r)


def _decode_execution_status(row: aiosqlite.Row) -> ExecutionStatus:
    tag: str = row["status_tag"]
    match tag:
        case "queued":
            return Queued()
        case "dispatching":
            return Dispatching()
        case "running":
            return RunningExec()
        case "succeeded":
            return SucceededExec(finished_at=_from_ts(row["finished_at"]))
        case "failed":
            return FailedExec(finished_at=_from_ts(row["finished_at"]))
        case "interrupted":
            return InterruptedExec(
                interrupted_at=_from_ts(row["interrupted_at"]),
                reason=row["interrupt_reason"],
            )
        case "cancelled":
            return CancelledExec(
                cancelled_at=_from_ts(row["cancelled_at"]),
                reason=row["cancel_reason"],
            )
        case _:
            raise ValueError(f"Unknown execution status_tag {tag!r}")


type _ResultStatusCols = tuple[
    str,            # status_tag
    float | None,   # dispatched_at
    float | None,   # started_at
    float | None,   # finished_at
    float | None,   # interrupted_at
    str | None,     # interrupt_reason
    int | None,     # result_blob
    int | None,     # error_id
    str | None,     # node_id
]


def _encode_result_status(s: ResultStatus) -> _ResultStatusCols:
    match s:
        case PendingRes():
            return ("pending", None, None, None, None, None, None, None, None)
        case RunningRes(dispatched_at=da, started_at=sa, node=n):
            return ("running", _to_ts(da), _to_ts(sa), None, None, None, None, None, n)
        case SucceededRes(
            dispatched_at=da, started_at=sa, finished_at=fa, node=n, blob=b
        ):
            return (
                "succeeded", _to_ts(da), _to_ts(sa), _to_ts(fa),
                None, None, b, None, n,
            )
        case FailedRes(
            dispatched_at=da, started_at=sa, finished_at=fa, node=n, error=e
        ):
            return (
                "failed", _to_ts(da), _to_ts_opt(sa), _to_ts(fa),
                None, None, None, e, n,
            )
        case InterruptedRes(
            dispatched_at=da, started_at=sa, interrupted_at=ia, node=n, reason=r
        ):
            return (
                "interrupted", _to_ts(da), _to_ts_opt(sa), None,
                _to_ts(ia), r, None, None, n,
            )


def _decode_result_status(row: aiosqlite.Row) -> ResultStatus:
    tag: str = row["status_tag"]
    match tag:
        case "pending":
            return PendingRes()
        case "running":
            return RunningRes(
                dispatched_at=_from_ts(row["dispatched_at"]),
                started_at=_from_ts(row["started_at"]),
                node=row["node_id"],
            )
        case "succeeded":
            return SucceededRes(
                dispatched_at=_from_ts(row["dispatched_at"]),
                started_at=_from_ts(row["started_at"]),
                finished_at=_from_ts(row["finished_at"]),
                node=row["node_id"],
                blob=int(row["result_blob"]),
            )
        case "failed":
            return FailedRes(
                dispatched_at=_from_ts(row["dispatched_at"]),
                started_at=_from_ts_opt(row["started_at"]),
                finished_at=_from_ts(row["finished_at"]),
                node=row["node_id"],
                error=int(row["error_id"]),
            )
        case "interrupted":
            return InterruptedRes(
                dispatched_at=_from_ts(row["dispatched_at"]),
                started_at=_from_ts_opt(row["started_at"]),
                interrupted_at=_from_ts(row["interrupted_at"]),
                node=row["node_id"],
                reason=row["interrupt_reason"],
            )
        case _:
            raise ValueError(f"Unknown result status_tag {tag!r}")


class Transaction:
    """Handle forwarded onto the writer connection inside ``Store.tx()``.

    Commits and rollbacks are managed by the enclosing ``tx()`` context.
    """

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def execute(
        self,
        sql: str,
        params: SqlParams = (),
    ) -> aiosqlite.Cursor:
        return await self._conn.execute(sql, params)

    async def executemany(
        self,
        sql: str,
        seq_of_params: Iterable[SqlParams],
    ) -> aiosqlite.Cursor:
        return await self._conn.executemany(sql, seq_of_params)

    async def fetch_one(
        self,
        sql: str,
        params: SqlParams = (),
    ) -> aiosqlite.Row | None:
        cursor = await self._conn.execute(sql, params)
        try:
            return await cursor.fetchone()
        finally:
            await cursor.close()

    async def fetch_all(
        self,
        sql: str,
        params: SqlParams = (),
    ) -> list[aiosqlite.Row]:
        cursor = await self._conn.execute(sql, params)
        try:
            return list(await cursor.fetchall())
        finally:
            await cursor.close()


class Store:
    """Owns the writer and reader connections to the SQLite database.

    Parameters
    ----------
    path
        Filesystem path (or ``":memory:"``) to the SQLite database.
    """

    def __init__(self, path: DBPath) -> None:
        self._path: DBPath = path
        self._write_lock: asyncio.Lock = asyncio.Lock()
        self._write: aiosqlite.Connection | None = None
        self._read: aiosqlite.Connection | None = None

    async def open(self) -> None:
        """Connect writer and reader, apply schema, and enable WAL."""
        if self._write is not None or self._read is not None:
            raise RuntimeError("Store already open")
        self._write = await aiosqlite.connect(self._path)
        self._write.row_factory = aiosqlite.Row
        await apply_schema(self._write)
        self._read = await aiosqlite.connect(self._path)
        self._read.row_factory = aiosqlite.Row
        await self._read.execute("PRAGMA journal_mode=WAL")

    async def close(self) -> None:
        """Close both connections."""
        match self._write:
            case None:
                pass
            case conn:
                await conn.close()
                self._write = None
        match self._read:
            case None:
                pass
            case conn:
                await conn.close()
                self._read = None

    @asynccontextmanager
    async def tx(self) -> AsyncIterator[Transaction]:
        """Acquire the writer lock and run statements in a single transaction.

        Commits on clean exit, rolls back on exception. State and event
        writes committed inside the same ``tx`` block are atomic.
        """
        if self._write is None:
            raise RuntimeError("Store is not open")
        write = self._write
        async with self._write_lock:
            try:
                yield Transaction(write)
            except BaseException:
                await write.rollback()
                raise
            else:
                await write.commit()

    # ── providers ──────────────────────────────────────────────────

    async def put_provider(self, p: Provider) -> None:
        """Insert or replace a provider row."""
        config_json = json.dumps(_wire().to_dict(p.config))
        async with self.tx() as tx:
            await tx.execute(
                "INSERT OR REPLACE INTO providers "
                "(name, type, config, created_at, updated_at, last_used_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    p.name,
                    p.type,
                    config_json,
                    _to_ts(p.created_at),
                    _to_ts(p.updated_at),
                    _to_ts_opt(p.last_used_at),
                ),
            )

    async def get_provider(self, name: str) -> Provider | None:
        """Load a provider row by name."""
        if self._read is None:
            raise RuntimeError("Store is not open")
        cursor = await self._read.execute(
            "SELECT name, type, config, created_at, updated_at, last_used_at "
            "FROM providers WHERE name = ?",
            (name,),
        )
        try:
            row = await cursor.fetchone()
        finally:
            await cursor.close()
        if row is None:
            return None
        return _row_to_provider(row)

    async def list_providers(self) -> list[Provider]:
        """Return every provider ordered by name."""
        if self._read is None:
            raise RuntimeError("Store is not open")
        cursor = await self._read.execute(
            "SELECT name, type, config, created_at, updated_at, last_used_at "
            "FROM providers ORDER BY name"
        )
        try:
            rows = await cursor.fetchall()
        finally:
            await cursor.close()
        return [_row_to_provider(r) for r in rows]

    # ── compute ────────────────────────────────────────────────────

    async def put_compute(self, c: Compute) -> None:
        """Insert or replace a compute row."""
        spec_json = json.dumps(_wire().to_dict(c.spec))
        (
            tag,
            started_at,
            stopped_at,
            stopping_since,
            failed_at,
            failure_reason,
            nodes_ready,
            last_activity_at,
            chosen_spec_ordinal,
            chosen_spec_json,
        ) = _encode_compute_status(c.status)
        async with self.tx() as tx:
            await tx.execute(
                "INSERT OR REPLACE INTO compute "
                "(name, spec, created_at, status_tag, started_at, stopped_at, "
                "stopping_since, failed_at, failure_reason, nodes_ready, "
                "last_activity_at, chosen_spec_ordinal, chosen_spec_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    c.name,
                    spec_json,
                    _to_ts(c.created_at),
                    tag,
                    started_at,
                    stopped_at,
                    stopping_since,
                    failed_at,
                    failure_reason,
                    nodes_ready,
                    last_activity_at,
                    chosen_spec_ordinal,
                    chosen_spec_json,
                ),
            )

    async def get_compute(self, name: ComputeName) -> Compute | None:
        """Load a compute row by name."""
        if self._read is None:
            raise RuntimeError("Store is not open")
        cursor = await self._read.execute(
            "SELECT * FROM compute WHERE name = ?", (name,)
        )
        try:
            row = await cursor.fetchone()
        finally:
            await cursor.close()
        if row is None:
            return None
        return _row_to_compute(row)

    async def list_compute(
        self, *, status: str | None = None
    ) -> list[Compute]:
        """Return compute rows, optionally filtered by ``status_tag``."""
        if self._read is None:
            raise RuntimeError("Store is not open")
        match status:
            case None:
                cursor = await self._read.execute(
                    "SELECT * FROM compute ORDER BY name"
                )
            case tag:
                cursor = await self._read.execute(
                    "SELECT * FROM compute WHERE status_tag = ? ORDER BY name",
                    (tag,),
                )
        try:
            rows = await cursor.fetchall()
        finally:
            await cursor.close()
        return [_row_to_compute(r) for r in rows]

    async def delete_compute(self, name: ComputeName) -> None:
        """Delete a compute row (and cascade to nodes)."""
        async with self.tx() as tx:
            await tx.execute("DELETE FROM compute WHERE name = ?", (name,))

    # ── nodes ──────────────────────────────────────────────────────

    async def put_node(self, n: Node) -> None:
        """Insert or replace a node row."""
        (
            tag,
            status_since,
            bootstrap_phase,
            lost_at,
            lost_reason,
        ) = _encode_node_status(n.status)
        async with self.tx() as tx:
            await tx.execute(
                "INSERT OR REPLACE INTO nodes "
                "(id, compute, instance_id, provider_name, head_addr, created_at, "
                "status_tag, status_since, bootstrap_phase, lost_at, lost_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    n.id,
                    n.compute,
                    n.instance_id,
                    n.provider_name,
                    n.head_addr,
                    _to_ts(n.created_at),
                    tag,
                    status_since,
                    bootstrap_phase,
                    lost_at,
                    lost_reason,
                ),
            )

    async def list_nodes(self, *, compute: ComputeName) -> list[Node]:
        """Return every node attached to *compute* ordered by id."""
        if self._read is None:
            raise RuntimeError("Store is not open")
        cursor = await self._read.execute(
            "SELECT * FROM nodes WHERE compute = ? ORDER BY id",
            (compute,),
        )
        try:
            rows = await cursor.fetchall()
        finally:
            await cursor.close()
        return [_row_to_node(r) for r in rows]

    # ── tasks ──────────────────────────────────────────────────────

    async def put_task(self, t: Task) -> None:
        """Upsert a task row keyed by ``(module, qualname)``."""
        async with self.tx() as tx:
            await tx.execute(
                "INSERT OR IGNORE INTO tasks (module, qualname) VALUES (?, ?)",
                (t.module, t.qualname),
            )

    # ── executions ─────────────────────────────────────────────────

    async def put_execution(self, e: TaskExecution) -> None:
        """Insert or replace a task execution row."""
        kind_tag, group_id = _encode_exec_kind(e.kind)
        (
            status_tag,
            finished_at,
            interrupted_at,
            interrupt_reason,
            cancelled_at,
            cancel_reason,
        ) = _encode_execution_status(e.status)
        module, qualname = e.task
        timeout_s = None if e.timeout is None else e.timeout.total_seconds()
        async with self.tx() as tx:
            await tx.execute(
                "INSERT OR REPLACE INTO task_executions "
                "(id, task_module, task_qualname, compute, kind_tag, group_id, "
                "payload_blob, timeout_s, client_id, submitted_at, "
                "status_tag, finished_at, interrupted_at, interrupt_reason, "
                "cancelled_at, cancel_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    e.id,
                    module,
                    qualname,
                    e.compute,
                    kind_tag,
                    group_id,
                    e.payload,
                    timeout_s,
                    e.client,
                    _to_ts(e.submitted_at),
                    status_tag,
                    finished_at,
                    interrupted_at,
                    interrupt_reason,
                    cancelled_at,
                    cancel_reason,
                ),
            )

    async def get_execution(self, id: ExecutionId) -> TaskExecution | None:
        """Load a task execution row by id."""
        if self._read is None:
            raise RuntimeError("Store is not open")
        cursor = await self._read.execute(
            "SELECT * FROM task_executions WHERE id = ?", (id,)
        )
        try:
            row = await cursor.fetchone()
        finally:
            await cursor.close()
        if row is None:
            return None
        return _row_to_execution(row)

    async def list_executions(
        self,
        *,
        compute: ComputeName | None = None,
        status: str | None = None,
        task: TaskKey | None = None,
        group: GroupId | None = None,
    ) -> list[TaskExecution]:
        """Return task executions, optionally filtered by several dimensions.

        Filters compose via ``AND``. Results are ordered by ``id`` ascending.
        """
        if self._read is None:
            raise RuntimeError("Store is not open")
        clauses: list[str] = []
        params: list[Any] = []
        match compute:
            case None:
                pass
            case name:
                clauses.append("compute = ?")
                params.append(name)
        match status:
            case None:
                pass
            case tag:
                clauses.append("status_tag = ?")
                params.append(tag)
        match task:
            case None:
                pass
            case (module, qualname):
                clauses.append("task_module = ? AND task_qualname = ?")
                params.append(module)
                params.append(qualname)
        match group:
            case None:
                pass
            case gid:
                clauses.append("group_id = ?")
                params.append(gid)
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        sql = f"SELECT * FROM task_executions{where} ORDER BY id ASC"
        cursor = await self._read.execute(sql, tuple(params))
        try:
            rows = await cursor.fetchall()
        finally:
            await cursor.close()
        return [_row_to_execution(r) for r in rows]

    async def list_active_executions(self) -> list[TaskExecution]:
        """Return executions whose ``status_tag`` is queued/dispatching/running."""
        if self._read is None:
            raise RuntimeError("Store is not open")
        cursor = await self._read.execute(
            "SELECT * FROM task_executions "
            "WHERE status_tag IN ('queued','dispatching','running') "
            "ORDER BY id ASC"
        )
        try:
            rows = await cursor.fetchall()
        finally:
            await cursor.close()
        return [_row_to_execution(r) for r in rows]

    # ── results ────────────────────────────────────────────────────

    async def put_result(self, r: TaskResult) -> None:
        """Insert a task result row.

        Results are identified by ``(execution, shard)``; the ``id`` field on
        the input dataclass is ignored — the database assigns the autoincrement
        id. Re-inserting the same ``(execution, shard)`` replaces the row.
        """
        (
            status_tag,
            dispatched_at,
            started_at,
            finished_at,
            interrupted_at,
            interrupt_reason,
            result_blob,
            error_id,
            node_id,
        ) = _encode_result_status(r.status)
        async with self.tx() as tx:
            await tx.execute(
                "INSERT OR REPLACE INTO task_results "
                "(execution, shard, node_id, status_tag, dispatched_at, "
                "started_at, finished_at, interrupted_at, interrupt_reason, "
                "result_blob, error_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    r.execution,
                    r.shard,
                    node_id,
                    status_tag,
                    dispatched_at,
                    started_at,
                    finished_at,
                    interrupted_at,
                    interrupt_reason,
                    result_blob,
                    error_id,
                ),
            )

    async def list_results(self, *, execution: ExecutionId) -> list[TaskResult]:
        """Return every result row for *execution* ordered by shard."""
        if self._read is None:
            raise RuntimeError("Store is not open")
        cursor = await self._read.execute(
            "SELECT * FROM task_results WHERE execution = ? ORDER BY shard ASC",
            (execution,),
        )
        try:
            rows = await cursor.fetchall()
        finally:
            await cursor.close()
        return [_row_to_result(r) for r in rows]

    # ── blobs ──────────────────────────────────────────────────────

    async def put_blob(
        self,
        *,
        path: str,
        size: int,
        kind: BlobKind,
        sha256: str | None = None,
    ) -> BlobId:
        """Insert a blob row and return the new id.

        The filesystem is not touched — only the metadata row is persisted.
        The Blobs service (Phase E) is responsible for the content path.
        """
        ts = time.time()
        async with self.tx() as tx:
            cursor = await tx.execute(
                "INSERT INTO blobs (path, size, sha256, kind, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (path, size, sha256, kind, ts),
            )
            return _lastrowid(cursor)

    async def get_blob(self, id: BlobId) -> Blob | None:
        """Load a blob row by id."""
        if self._read is None:
            raise RuntimeError("Store is not open")
        cursor = await self._read.execute(
            "SELECT id, path, size, sha256, kind, created_at, evicted_at "
            "FROM blobs WHERE id = ?",
            (id,),
        )
        try:
            row = await cursor.fetchone()
        finally:
            await cursor.close()
        if row is None:
            return None
        return _row_to_blob(row)

    async def evict_blob(self, id: BlobId) -> None:
        """Mark a blob as evicted by setting ``evicted_at = now()``.

        The blob file is not removed; the Blobs service handles content unlink
        in Phase E.
        """
        ts = time.time()
        async with self.tx() as tx:
            await tx.execute(
                "UPDATE blobs SET evicted_at = ? WHERE id = ?",
                (ts, id),
            )

    # ── errors ─────────────────────────────────────────────────────

    async def put_error(
        self,
        *,
        type: str,
        message: str,
        traceback: str | None = None,
    ) -> ErrorId:
        """Insert an error row and return the new id."""
        ts = time.time()
        async with self.tx() as tx:
            cursor = await tx.execute(
                "INSERT INTO errors (type, message, traceback, created_at) "
                "VALUES (?, ?, ?, ?)",
                (type, message, traceback, ts),
            )
            return _lastrowid(cursor)

    # ── events ─────────────────────────────────────────────────────

    async def emit(
        self,
        aggregate: str,
        type: str,
        payload: Mapping[str, Any],
        *,
        tx: Transaction | None = None,
    ) -> int:
        """Append an event row and return its autoincrement id.

        Standalone mode (``tx is None``) opens its own transaction via
        ``self.tx()`` so the write is isolated and durable. When ``tx`` is
        supplied, the insert rides on the caller's transaction and commits
        atomically with any other state writes in that block.

        Parameters
        ----------
        aggregate
            Logical aggregate identifier (e.g. ``"compute:my-pool"``).
        type
            Event type tag (e.g. ``"Pool.Ready"``).
        payload
            Opaque JSON-serializable dict persisted as TEXT.
        tx
            Optional transaction to enlist in. When omitted, a fresh
            transaction is opened and committed around the insert.

        Returns
        -------
        int
            The ``events.id`` autoincrement value.
        """
        data = json.dumps(dict(payload))
        ts = time.time()
        sql = "INSERT INTO events (ts, aggregate, type, payload) VALUES (?, ?, ?, ?)"
        params = (ts, aggregate, type, data)
        match tx:
            case None:
                async with self.tx() as owned:
                    cursor = await owned.execute(sql, params)
                    return _lastrowid(cursor)
            case Transaction() as existing:
                cursor = await existing.execute(sql, params)
                return _lastrowid(cursor)

    async def tail_events(
        self,
        *,
        since_id: int = 0,
        aggregate_like: str | None = None,
    ) -> AsyncIterator[EventRow]:
        """Yield events with ``id > since_id`` ordered ascending by id.

        When ``aggregate_like`` is supplied, rows are filtered by
        ``aggregate LIKE <aggregate_like> || '%'`` (prefix match). The
        iterator terminates at the current tail; live fanout lands in a
        later task.

        Parameters
        ----------
        since_id
            Return rows whose id is strictly greater than this value.
        aggregate_like
            Optional aggregate prefix (not a SQL pattern). ``%`` and ``_``
            inside the prefix are interpreted by SQLite's ``LIKE``.
        """
        if self._read is None:
            raise RuntimeError("Store is not open")
        match aggregate_like:
            case None:
                sql = (
                    "SELECT id, aggregate, type, payload, ts FROM events "
                    "WHERE id > ? ORDER BY id ASC"
                )
                params: SqlParams = (since_id,)
            case prefix:
                sql = (
                    "SELECT id, aggregate, type, payload, ts FROM events "
                    "WHERE id > ? AND aggregate LIKE ? || '%' ORDER BY id ASC"
                )
                params = (since_id, prefix)
        cursor = await self._read.execute(sql, params)
        try:
            async for row in cursor:
                yield EventRow(
                    id=int(row["id"]),
                    aggregate=str(row["aggregate"]),
                    type=str(row["type"]),
                    payload=json.loads(row["payload"]),
                    created_at=_from_ts(float(row["ts"])),
                )
        finally:
            await cursor.close()


def _lastrowid(cursor: aiosqlite.Cursor) -> int:
    rid = cursor.lastrowid
    if rid is None:
        raise RuntimeError("INSERT did not return lastrowid")
    return int(rid)


def _row_to_provider(row: aiosqlite.Row) -> Provider:
    config = _wire().from_dict(json.loads(row["config"]))
    if not isinstance(config, dict):
        config = {"value": config}
    return Provider(
        name=row["name"],
        type=_as_provider_type(row["type"]),
        config=config,
        created_at=_from_ts(row["created_at"]),
        updated_at=_from_ts(row["updated_at"]),
        last_used_at=_from_ts_opt(row["last_used_at"]),
    )


def _as_provider_type(value: str) -> ProviderType:
    return value  # type: ignore[return-value]


def _row_to_compute(row: aiosqlite.Row) -> Compute:
    spec = _wire().from_dict(json.loads(row["spec"]), ComputeSpec)
    return Compute(
        name=row["name"],
        spec=spec,
        created_at=_from_ts(row["created_at"]),
        status=_decode_compute_status(row),
    )


def _row_to_execution(row: aiosqlite.Row) -> TaskExecution:
    timeout_s = row["timeout_s"]
    timeout = None if timeout_s is None else timedelta(seconds=float(timeout_s))
    return TaskExecution(
        id=row["id"],
        task=(row["task_module"], row["task_qualname"]),
        compute=row["compute"],
        kind=_decode_exec_kind(row),
        payload=int(row["payload_blob"]),
        timeout=timeout,
        client=row["client_id"],
        submitted_at=_from_ts(row["submitted_at"]),
        status=_decode_execution_status(row),
    )


def _row_to_result(row: aiosqlite.Row) -> TaskResult:
    return TaskResult(
        id=int(row["id"]),
        execution=row["execution"],
        shard=int(row["shard"]),
        status=_decode_result_status(row),
    )


def _row_to_blob(row: aiosqlite.Row) -> Blob:
    return Blob(
        id=int(row["id"]),
        path=Path(row["path"]),
        size=int(row["size"]),
        sha256=row["sha256"],
        kind=_as_blob_kind(row["kind"]),
        created_at=_from_ts(row["created_at"]),
        evicted_at=_from_ts_opt(row["evicted_at"]),
    )


def _as_blob_kind(value: str) -> BlobKind:
    return value  # type: ignore[return-value]


def _row_to_node(row: aiosqlite.Row) -> Node:
    return Node(
        id=row["id"],
        compute=row["compute"],
        instance_id=row["instance_id"],
        provider_name=row["provider_name"],
        head_addr=row["head_addr"],
        status=_decode_node_status(row),
        created_at=_from_ts(row["created_at"]),
    )


__all__ = ["DBPath", "EventRow", "Store", "Transaction"]
