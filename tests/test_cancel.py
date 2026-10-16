"""Cancelling a task a machine is already running.

Once an attempt is ``started`` the function may be running on a node, and only the
worker can stop it. A cancel writes ``cancel_requested``, and the dispatcher asks the
worker to stop the attempt on every pass, as it does for one that ran past its time,
until the worker lets go. The attempt is ``cancelled`` once the worker says it stopped.
What the worker answered before the stop landed stands, and nothing that ends a
cancelled attempt writes another one down.
"""

from pathlib import Path

import pytest

from skyward.shared import codec, retry
from skyward.shared.frames import Done, Failed, Stopped
from skyward.shared.schemas import Task
from tests.test_retry import _Plane, _plane

pytestmark = pytest.mark.local


async def _cancelled(tmp_path: Path, decision: retry.Retry | None = None) -> tuple[_Plane, Task]:
    """A task a machine has started, which the caller then cancelled: the plane, and the task as it read before the cancel."""
    plane = await _plane(tmp_path / "skyward.sqlite")
    task = await plane.submit(decision)
    await plane.tasks.cancel(task.id, idempotency_key="cancel")
    return plane, task


def describe_cancelling_a_task_a_machine_is_running() -> None:
    async def asks_the_worker_to_stop_it_on_every_pass_until_it_lets_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        plane, task = await _cancelled(tmp_path)
        plane.runtimes.open(plane.compute, "pypi", "a private key")
        asked: list[tuple[str, str, str]] = []

        async def stop(compute_id: str, node_id: str, execution_id: str) -> None:
            asked.append((compute_id, node_id, execution_id))

        monkeypatch.setattr(plane.dispatcher, "_stop", stop)
        await plane.dispatcher.task(task.id)
        await plane.dispatcher.task(task.id)

        assert asked == [(plane.compute, plane.node_ids[0], task.executions[0].id)] * 2

    async def is_cancelled_once_the_worker_says_it_stopped(tmp_path: Path) -> None:
        plane, task = await _cancelled(tmp_path)

        await plane.dispatcher._settle(task, task.executions[0], Stopped(), plane.node_ids[0])

        settled = await plane.tasks.get(task.id)
        assert settled.state == "cancelled"
        assert dict((await plane.tasks.pressure(plane.compute)).holding) == {}, "the slot is free once the worker let go"
        assert await plane.said(task.id) == [("started", 1), ("cancelled", 1)]

    async def asked_again_it_leaves_the_stop_with_the_worker(tmp_path: Path) -> None:
        plane, task = await _cancelled(tmp_path)

        await plane.tasks.cancel(task.id, idempotency_key="again")

        asked = await plane.tasks.get(task.id)
        assert [e.state for e in asked.executions] == ["cancel_requested"], "code that may still be running is never declared cancelled"
        assert asked.state == "running"

    async def ends_as_the_worker_answered_when_it_finished_before_the_stop_landed(tmp_path: Path) -> None:
        plane, task = await _cancelled(tmp_path)

        await plane.dispatcher._settle(task, task.executions[0], Done(value=codec.dumps(42)), plane.node_ids[0])

        assert (await plane.tasks.get(task.id)).state == "succeeded"

    async def is_not_tried_again_when_it_fails_after_the_cancel(tmp_path: Path) -> None:
        plane, task = await _cancelled(tmp_path, retry.default)

        await plane.dispatcher._settle(task, task.executions[0], Failed(error="no", traceback="", retry=True), plane.node_ids[0])

        settled = await plane.tasks.get(task.id)
        assert [(e.ordinal, e.state) for e in settled.executions] == [(1, "failed")]

    async def is_not_tried_again_when_it_is_lost_after_the_cancel(tmp_path: Path) -> None:
        plane, task = await _cancelled(tmp_path, retry.default)

        await plane.dispatcher._lost(task, task.executions[0], RuntimeError("gone"), retry.Lost("node_gone", plane.node_ids[0]))

        settled = await plane.tasks.get(task.id)
        assert [(e.ordinal, e.state) for e in settled.executions] == [(1, "indeterminate")]
