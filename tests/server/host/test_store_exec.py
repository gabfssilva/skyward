"""Tests for task / execution / result repository methods on ``Store``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from skyward.api.spec import Image, Nodes, Spec
from skyward.providers.container.config import Container
from skyward.server.host.domain import (
    Broadcast,
    CancelledExec,
    Compute,
    ComputeSpec,
    Dispatching,
    ExecutionStatus,
    FailedExec,
    FailedRes,
    GroupMember,
    InterruptedExec,
    InterruptedRes,
    PendingRes,
    Provisioning,
    Queued,
    ResultStatus,
    Run,
    RunningExec,
    RunningRes,
    SucceededExec,
    SucceededRes,
    Task,
    TaskExecution,
    TaskExecutionKind,
    TaskResult,
)
from skyward.server.host.store import Store


def _t(sec: int = 0) -> datetime:
    return datetime(2026, 4, 14, 12, sec // 60, sec % 60, tzinfo=UTC)


def _simple_spec() -> Spec:
    return Spec(provider=Container(), image=Image(metrics=None))


def _compute_spec() -> ComputeSpec:
    return ComputeSpec(
        specs=(_simple_spec(),),
        selection="cheapest",
        nodes=Nodes(desired=2),
        allocation="spot-if-available",
        ttl=timedelta(hours=1),
    )


async def _open(tmp_path: Path) -> Store:
    store = Store(str(tmp_path / "test.db"))
    await store.open()
    return store


async def _seed_compute(store: Store, name: str) -> None:
    await store.put_compute(
        Compute(
            name=name,
            spec=_compute_spec(),
            created_at=_t(0),
            status=Provisioning(started_at=_t(0)),
        )
    )


async def _seed_payload(store: Store) -> int:
    return await store.put_blob(path="/tmp/payload.bin", size=42, kind="payload")


async def _seed_task(store: Store, key: tuple[str, str] = ("m", "f")) -> None:
    await store.put_task(Task(module=key[0], qualname=key[1]))


@pytest.mark.asyncio
async def test_task_upsert(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        t = Task(module="pkg.mod", qualname="fn")
        await store.put_task(t)
        await store.put_task(t)
    finally:
        await store.close()


_EXEC_STATUSES: list[tuple[str, ExecutionStatus]] = [
    ("queued", Queued()),
    ("dispatching", Dispatching()),
    ("running", RunningExec()),
    ("succeeded", SucceededExec(finished_at=_t(10))),
    ("failed", FailedExec(finished_at=_t(20))),
    ("interrupted", InterruptedExec(interrupted_at=_t(30), reason="preempted")),
    ("cancelled", CancelledExec(cancelled_at=_t(40), reason="user")),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label,status", _EXEC_STATUSES, ids=[s[0] for s in _EXEC_STATUSES]
)
async def test_execution_roundtrip_each_status(
    tmp_path: Path, label: str, status: ExecutionStatus
) -> None:
    store = await _open(tmp_path)
    try:
        await _seed_compute(store, "c1")
        await _seed_task(store)
        blob = await _seed_payload(store)
        e = TaskExecution(
            id=f"exec-{label}",
            task=("m", "f"),
            compute="c1",
            kind=Run(),
            payload=blob,
            timeout=timedelta(seconds=60),
            client="client-1",
            submitted_at=_t(0),
            status=status,
        )
        await store.put_execution(e)
        loaded = await store.get_execution(e.id)
        assert loaded == e
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_execution_kinds_roundtrip(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await _seed_compute(store, "c1")
        await _seed_task(store)
        blob = await _seed_payload(store)
        kinds: list[tuple[str, TaskExecutionKind]] = [
            ("run", Run()),
            ("broadcast", Broadcast()),
            ("group", GroupMember(group="g1")),
        ]
        for label, kind in kinds:
            e = TaskExecution(
                id=f"exec-{label}",
                task=("m", "f"),
                compute="c1",
                kind=kind,
                payload=blob,
                timeout=None,
                client=None,
                submitted_at=_t(0),
                status=Queued(),
            )
            await store.put_execution(e)
            loaded = await store.get_execution(e.id)
            assert loaded == e
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_list_executions_filter_by_compute(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await _seed_compute(store, "c1")
        await _seed_compute(store, "c2")
        await _seed_task(store)
        blob = await _seed_payload(store)
        for eid, compute in (("a", "c1"), ("b", "c2"), ("c", "c1")):
            await store.put_execution(
                TaskExecution(
                    id=eid, task=("m", "f"), compute=compute, kind=Run(),
                    payload=blob, timeout=None, client=None,
                    submitted_at=_t(0), status=Queued(),
                )
            )
        got = await store.list_executions(compute="c1")
        assert {e.id for e in got} == {"a", "c"}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_list_executions_filter_by_status(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await _seed_compute(store, "c1")
        await _seed_task(store)
        blob = await _seed_payload(store)
        await store.put_execution(
            TaskExecution(id="a", task=("m", "f"), compute="c1", kind=Run(),
                          payload=blob, timeout=None, client=None,
                          submitted_at=_t(0), status=Queued())
        )
        await store.put_execution(
            TaskExecution(id="b", task=("m", "f"), compute="c1", kind=Run(),
                          payload=blob, timeout=None, client=None,
                          submitted_at=_t(0),
                          status=SucceededExec(finished_at=_t(1)))
        )
        queued = await store.list_executions(status="queued")
        succeeded = await store.list_executions(status="succeeded")
        assert [e.id for e in queued] == ["a"]
        assert [e.id for e in succeeded] == ["b"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_list_executions_filter_by_task(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await _seed_compute(store, "c1")
        await store.put_task(Task(module="m1", qualname="f"))
        await store.put_task(Task(module="m2", qualname="g"))
        blob = await _seed_payload(store)
        await store.put_execution(
            TaskExecution(id="a", task=("m1", "f"), compute="c1", kind=Run(),
                          payload=blob, timeout=None, client=None,
                          submitted_at=_t(0), status=Queued())
        )
        await store.put_execution(
            TaskExecution(id="b", task=("m2", "g"), compute="c1", kind=Run(),
                          payload=blob, timeout=None, client=None,
                          submitted_at=_t(0), status=Queued())
        )
        got = await store.list_executions(task=("m1", "f"))
        assert [e.id for e in got] == ["a"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_list_executions_filter_by_group(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await _seed_compute(store, "c1")
        await _seed_task(store)
        blob = await _seed_payload(store)
        await store.put_execution(
            TaskExecution(id="g0", task=("m", "f"), compute="c1",
                          kind=GroupMember(group="g1"),
                          payload=blob, timeout=None, client=None,
                          submitted_at=_t(0), status=Queued())
        )
        await store.put_execution(
            TaskExecution(id="g1", task=("m", "f"), compute="c1",
                          kind=GroupMember(group="g1"),
                          payload=blob, timeout=None, client=None,
                          submitted_at=_t(0), status=Queued())
        )
        await store.put_execution(
            TaskExecution(id="nogroup", task=("m", "f"), compute="c1",
                          kind=Run(), payload=blob, timeout=None, client=None,
                          submitted_at=_t(0), status=Queued())
        )
        got = await store.list_executions(group="g1")
        assert {e.id for e in got} == {"g0", "g1"}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_list_active_executions_returns_only_active(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await _seed_compute(store, "c1")
        await _seed_task(store)
        blob = await _seed_payload(store)
        statuses: list[tuple[str, ExecutionStatus]] = [
            ("q", Queued()),
            ("d", Dispatching()),
            ("r", RunningExec()),
            ("s", SucceededExec(finished_at=_t(1))),
            ("f", FailedExec(finished_at=_t(1))),
            ("i", InterruptedExec(interrupted_at=_t(1), reason="x")),
        ]
        for eid, status in statuses:
            await store.put_execution(
                TaskExecution(id=eid, task=("m", "f"), compute="c1", kind=Run(),
                              payload=blob, timeout=None, client=None,
                              submitted_at=_t(0), status=status)
            )
        active = await store.list_active_executions()
        assert {e.id for e in active} == {"q", "d", "r"}
    finally:
        await store.close()


_RESULT_STATUSES: list[tuple[str, ResultStatus]] = [
    ("pending", PendingRes()),
    ("running", RunningRes(dispatched_at=_t(1), started_at=_t(2), node="n1")),
]


@pytest.mark.asyncio
async def test_result_roundtrip_pending_and_running(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await _seed_compute(store, "c1")
        await _seed_task(store)
        blob = await _seed_payload(store)
        await store.put_execution(
            TaskExecution(id="e1", task=("m", "f"), compute="c1", kind=Run(),
                          payload=blob, timeout=None, client=None,
                          submitted_at=_t(0), status=Queued())
        )
        shard = 0
        for label, status in _RESULT_STATUSES:
            r = TaskResult(id=0, execution="e1", shard=shard, status=status)
            await store.put_result(r)
            shard += 1
        results = await store.list_results(execution="e1")
        assert [r.status for r in results] == [s for _, s in _RESULT_STATUSES]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_result_roundtrip_succeeded(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await _seed_compute(store, "c1")
        await _seed_task(store)
        payload = await _seed_payload(store)
        result_blob = await store.put_blob(path="/tmp/r.bin", size=7, kind="result")
        await store.put_execution(
            TaskExecution(id="e1", task=("m", "f"), compute="c1", kind=Run(),
                          payload=payload, timeout=None, client=None,
                          submitted_at=_t(0), status=Queued())
        )
        status = SucceededRes(
            dispatched_at=_t(1), started_at=_t(2), finished_at=_t(3),
            node="n1", blob=result_blob,
        )
        await store.put_result(TaskResult(id=0, execution="e1", shard=0, status=status))
        got = await store.list_results(execution="e1")
        assert len(got) == 1
        assert got[0].status == status
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_result_roundtrip_failed(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await _seed_compute(store, "c1")
        await _seed_task(store)
        payload = await _seed_payload(store)
        err = await store.put_error(type="RuntimeError", message="nope", traceback="tb")
        await store.put_execution(
            TaskExecution(id="e1", task=("m", "f"), compute="c1", kind=Run(),
                          payload=payload, timeout=None, client=None,
                          submitted_at=_t(0), status=Queued())
        )
        status = FailedRes(
            dispatched_at=_t(1), started_at=_t(2), finished_at=_t(3),
            node="n1", error=err,
        )
        await store.put_result(TaskResult(id=0, execution="e1", shard=0, status=status))
        got = await store.list_results(execution="e1")
        assert len(got) == 1
        assert got[0].status == status
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_result_roundtrip_interrupted(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await _seed_compute(store, "c1")
        await _seed_task(store)
        payload = await _seed_payload(store)
        await store.put_execution(
            TaskExecution(id="e1", task=("m", "f"), compute="c1", kind=Run(),
                          payload=payload, timeout=None, client=None,
                          submitted_at=_t(0), status=Queued())
        )
        status = InterruptedRes(
            dispatched_at=_t(1), started_at=_t(2), interrupted_at=_t(3),
            node="n1", reason="preempted",
        )
        await store.put_result(TaskResult(id=0, execution="e1", shard=0, status=status))
        got = await store.list_results(execution="e1")
        assert len(got) == 1
        assert got[0].status == status
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_list_results_by_execution(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await _seed_compute(store, "c1")
        await _seed_task(store)
        payload = await _seed_payload(store)
        for eid in ("e1", "e2"):
            await store.put_execution(
                TaskExecution(id=eid, task=("m", "f"), compute="c1", kind=Run(),
                              payload=payload, timeout=None, client=None,
                              submitted_at=_t(0), status=Queued())
            )
        await store.put_result(
            TaskResult(id=0, execution="e1", shard=0, status=PendingRes())
        )
        await store.put_result(
            TaskResult(id=0, execution="e1", shard=1, status=PendingRes())
        )
        await store.put_result(
            TaskResult(id=0, execution="e2", shard=0, status=PendingRes())
        )
        e1 = await store.list_results(execution="e1")
        e2 = await store.list_results(execution="e2")
        assert [r.shard for r in e1] == [0, 1]
        assert [r.shard for r in e2] == [0]
    finally:
        await store.close()
