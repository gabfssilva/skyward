"""What the SDK sends the daemon when a driver submits thousands of calls at once.

A campaign is one process asking for every run it wants and then waiting: six
thousand `task(...) > pool` in a row, all of the same function. What travels per
call is the arguments; what travels once is the function, and "once" has to survive
six thousand coroutines being inside the check for it together.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

import skyward as sky
from skyward.api import v1
from skyward.core.compute import Compute

pytestmark = pytest.mark.local


@sky.function
def train(rows: int) -> int:
    return rows


def describe_a_driver_that_submits_a_campaign_in_one_breath() -> None:
    async def it_uploads_the_function_once_however_many_calls_are_in_flight() -> None:
        """The check and the upload are several awaits apart.

        Every one of a thousand submissions passes ``function not in self._functions``
        before the first of them has finished registering it, so one upload of the code
        becomes a thousand — each with an excerpt of its own queued onto the thread pool
        behind it, and a thousand POSTs waiting on the connection pool they share.
        """
        pool, sent = _pool()

        await asyncio.gather(*(pool._submit(train(rows), dispatch="one") for rows in range(1000)))

        assert sent["PUT /v1/functions"] == 1, "the code went up once"
        assert sent["PUT /v1/functions/excerpt"] == 1, "and was read off the disk once"
        assert sent["POST /v1/tasks"] == 1000, "every call is still its own task"

    async def it_uploads_the_function_again_when_the_first_upload_failed() -> None:
        """A registration that died is not a registration, and the next call redoes it."""
        pool, sent = _pool(failures=1)

        with pytest.raises(ConnectionError):
            await pool._submit(train(1), dispatch="one")
        await pool._submit(train(2), dispatch="one")

        assert sent["PUT /v1/functions"] == 2


class _Counting:
    """A client that answers the way the daemon does and counts what it was asked."""

    def __init__(self, sent: dict[str, int], failures: int = 0) -> None:
        self.sent = sent
        self.failures = failures

    async def upload(self, path: str, body: bytes, headers: dict[str, str] | None = None) -> None:
        await asyncio.sleep(0)
        self.sent["PUT /v1/functions" if path.startswith("/v1/functions") else "PUT /v1/blobs"] += 1
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("the daemon hung up")

    async def call(self, method: str, path: str, *_: Any, **__: Any) -> Any:
        await asyncio.sleep(0)
        if path.endswith("/excerpt"):
            self.sent["PUT /v1/functions/excerpt"] += 1
            return None
        self.sent[f"{method} /v1/tasks"] += 1
        return v1.TaskResource(
            id="tsk_1",
            compute=v1.ComputeSummary(id="cmp_1", name=None),
            generation=1,
            function=v1.FunctionSummary(sha256="f" * 64, name="train", version=1),
            args_sha256="a" * 64,
            dispatch="one",
            state="queued",
            retry=None,
            executions=(),
            submitted_at=datetime.now(UTC),
            finished_at=None,
            rank=None,
            correlation_id=None,
            queue_timeout_seconds=None,
            run_timeout_seconds=None,
            result_sha256=None,
        )


def _pool(failures: int = 0) -> tuple[Compute, dict[str, int]]:
    sent = dict.fromkeys(("PUT /v1/functions", "PUT /v1/functions/excerpt", "PUT /v1/blobs", "POST /v1/tasks"), 0)
    pool = Compute(provider=sky.Container(), nodes=1)
    pool._client = _Counting(sent, failures)  # type: ignore[assignment]
    pool._id = "cmp_1"
    return pool, sent
