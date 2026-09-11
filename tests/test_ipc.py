"""What happens to the worker's subprocess pool when a child dies under it.

A child killed mid-task — an OOM kill, a segfault in a native extension, a
container reclaiming memory — leaves a ``ProcessPoolExecutor`` broken for good.
The worker is not broken: it rebuilds the pool, gives the killed task the one
honest verdict there is, and takes the next task as if nothing happened.
"""

import os
from concurrent.futures import BrokenExecutor

import pytest

from skyward.shared import codec
from skyward.shared.frames import Done, Lost
from skyward.worker import ipc, worker

pytestmark = pytest.mark.local


def describe_a_pool_whose_child_dies_mid_task() -> None:
    def it_fails_that_task_alone_and_runs_the_next() -> None:
        async def scenario() -> int:
            with ipc.pool("process", reuse=True, workers=1) as pool:
                with pytest.raises(BrokenExecutor):
                    await pool.run(os._exit, 1)
                return await pool.run(len, "abc")

        assert worker.asyncio.run(scenario()) == 3


def describe_a_task_whose_subprocess_dies() -> None:
    def it_is_lost_and_the_next_one_is_done(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKYWARD_NODE", "nod_test")
        monkeypatch.setenv("SKYWARD_COMPUTE", "cmp_test")
        monkeypatch.setenv("SKYWARD_RANK", "0")
        monkeypatch.setenv("SKYWARD_PEERS", "10.0.0.1")
        monkeypatch.setenv("SKYWARD_PLUGINS", "[]")
        monkeypatch.setattr(worker, "MODE", "process")

        def die() -> None:
            os._exit(1)

        def answer() -> int:
            return 42

        arguments = codec.dumps(((), {}))

        async def scenario() -> tuple[object, object]:
            with ipc.pool("process", reuse=True, workers=1) as pool:
                monkeypatch.setattr(worker, "subprocesses", pool)
                first = await worker.execute("tsk_1", codec.dumps(die), arguments)
                second = await worker.execute("tsk_2", codec.dumps(answer), arguments)
                return first, second

        first, second = worker.asyncio.run(scenario())

        assert isinstance(first, Lost)
        assert isinstance(second, Done)
        assert codec.loads(second.value) == 42


def describe_a_loky_pool_left_idle() -> None:
    def it_keeps_its_worker_past_lokys_default_timeout() -> None:
        async def scenario() -> tuple[int, int]:
            with ipc.pool("loky", reuse=True, workers=1) as pool:
                before = await pool.run(os.getpid)
                await worker.asyncio.sleep(11)
                after = await pool.run(os.getpid)
                return before, after

        before, after = worker.asyncio.run(scenario())

        assert before == after
