"""What the control plane says when it breaks in a way it did not plan for."""

import io

import pytest
from litestar.testing import AsyncTestClient

from skyward.server.application import mock
from skyward.server.http.app import create_app, with_real
from skyward.shared.observability import logger


class _Broken(mock.MockTasks):
    """A store that fails the way a store fails: not politely."""

    async def get(self, task_id: str):
        raise RuntimeError("the database went away")


@pytest.fixture
async def client():
    async with AsyncTestClient(app=create_app(with_real(tasks=_Broken()))) as client:
        yield client


async def test_a_break_nobody_planned_for_is_still_a_500(client: AsyncTestClient):
    response = await client.get("/v1/tasks/tsk_1")

    assert response.status_code == 500


async def test_a_500_says_what_it_was_for(client: AsyncTestClient):
    """Embedded, nothing else logs it: a 500 nobody records is a dead end for whoever hit it."""
    captured = io.StringIO()
    sink = logger.add(captured, level="ERROR")
    try:
        await client.get("/v1/tasks/tsk_1")
    finally:
        logger.remove(sink)

    assert "the database went away" in captured.getvalue()
    assert "/v1/tasks/tsk_1" in captured.getvalue()
