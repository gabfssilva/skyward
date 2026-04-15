"""Task manager persists every post-insert transition (Workstream B)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from skyward.server.host.domain import (
    FailedExec,
    Queued,
    Run,
    RunningExec,
    SucceededExec,
    Task,
    TaskExecution,
)
from skyward.server.host.store import Store

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


async def _seed_queued(store: Store, tid: str) -> None:
    await store.put_task(Task(module="mod", qualname="fn"))
    await store.put_execution(
        TaskExecution(
            id=tid,
            task=("mod", "fn"),
            compute="c",
            kind=Run(),
            payload=0,
            timeout=timedelta(seconds=60),
            client=None,
            submitted_at=datetime.now(UTC),
            status=Queued(),
        ),
    )


@pytest.mark.asyncio
async def test_persist_dispatch_writes_running_exec() -> None:
    from skyward.actors.task_manager.actor import _persist

    store = Store(":memory:")
    await store.open()
    try:
        await _seed_queued(store, "t1")
        _persist(store, "t1", RunningExec())
        # Let the fire-and-forget task flush.
        for _ in range(20):
            await asyncio.sleep(0.01)
            row = await store.get_execution("t1")
            if row is not None and not isinstance(row.status, Queued):
                break
        row = await store.get_execution("t1")
        assert row is not None
        assert isinstance(row.status, RunningExec)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_persist_succeeded_and_failed_are_terminal() -> None:
    from skyward.actors.task_manager.actor import _persist

    store = Store(":memory:")
    await store.open()
    try:
        await _seed_queued(store, "ok")
        await _seed_queued(store, "bad")
        _persist(store, "ok", SucceededExec(finished_at=datetime.now(UTC)))
        _persist(store, "bad", FailedExec(finished_at=datetime.now(UTC)))
        for _ in range(20):
            await asyncio.sleep(0.01)
            rows = {
                r.id: r
                for r in await store.list_executions()
            }
            if (
                isinstance(rows.get("ok"), TaskExecution)
                and isinstance(rows["ok"].status, SucceededExec)
                and isinstance(rows.get("bad"), TaskExecution)
                and isinstance(rows["bad"].status, FailedExec)
            ):
                break
        rows = {r.id: r for r in await store.list_executions()}
        assert isinstance(rows["ok"].status, SucceededExec)
        assert isinstance(rows["bad"].status, FailedExec)
    finally:
        await store.close()


def test_persist_with_none_store_is_noop() -> None:
    from skyward.actors.task_manager.actor import _persist

    asyncio.get_event_loop()
    _persist(None, "whatever", RunningExec())
