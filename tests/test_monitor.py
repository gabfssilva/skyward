"""``sky monitor`` in lines: what it prints for the computes it watches, and when it stops."""

from __future__ import annotations

import asyncio

import pytest

from skyward.api.v1 import LogEntryResource, Page
from skyward.cli.monitor import SAID, TYPES, _lines
from skyward.core.client import Client
from tests.conftest import cli

pytestmark = pytest.mark.local


def describe_watching_in_lines() -> None:
    @pytest.mark.parametrize("types", [(*TYPES, *SAID), TYPES], ids=["everything", "quiet"])
    async def several_computes_are_told_apart_by_name_and_the_lines_end_once_all_are_deleted(
        alone: str, capsys: pytest.CaptureFixture[str], types: tuple[str, ...]
    ) -> None:
        client = await Client.remote(alone)
        try:
            head = await _head(client)
            for name in ("first", "second"):
                assert (await asyncio.to_thread(cli, "compute", "create", "--provider", "fake", "--name", name, "--url", alone)).code == 0
            watching = asyncio.create_task(_lines(client, ("first", "second"), head, types))
            said = await _until(capsys, lambda lines: {"first", "second"} <= {line.split(" │ ")[0] for line in lines})

            for name in ("first", "second"):
                assert (await asyncio.to_thread(cli, "compute", "delete", name, "--url", alone)).code == 0
            async with asyncio.timeout(60):
                await watching
            said += capsys.readouterr().out.splitlines()
        finally:
            await client.close()

        assert all(line.split(" │ ")[0] in ("first", "second") for line in said), "each line starts with the compute it is about"
        assert {line.split(" │ ")[0] for line in said if line.endswith(" │ deleted")} == {"first", "second"}

    async def one_compute_needs_no_name_before_its_lines(alone: str, capsys: pytest.CaptureFixture[str]) -> None:
        client = await Client.remote(alone)
        try:
            head = await _head(client)
            assert (await asyncio.to_thread(cli, "compute", "create", "--provider", "fake", "--name", "only", "--url", alone)).code == 0
            watching = asyncio.create_task(_lines(client, ("only",), head))
            await _until(capsys, bool)

            assert (await asyncio.to_thread(cli, "compute", "delete", "only", "--url", alone)).code == 0
            async with asyncio.timeout(60):
                await watching
            said = capsys.readouterr().out.splitlines()
        finally:
            await client.close()

        assert not any(line.startswith("only │") for line in said)
        assert said[-1].endswith(" │ deleted")

    async def with_none_named_every_compute_is_watched_until_interrupted(alone: str, capsys: pytest.CaptureFixture[str]) -> None:
        client = await Client.remote(alone)
        try:
            head = await _head(client)
            watching = asyncio.create_task(_lines(client, (), head))
            assert (await asyncio.to_thread(cli, "compute", "create", "--provider", "fake", "--name", "later", "--url", alone)).code == 0

            said = await _until(capsys, lambda lines: any(line.startswith("later │") for line in lines))
            watching.cancel()
        finally:
            await client.close()

        assert said, "a compute created after the watch began is named by a read the first time it speaks"

    async def computes_already_deleted_leave_nothing_to_watch(alone: str) -> None:
        assert (await asyncio.to_thread(cli, "compute", "create", "--provider", "fake", "--name", "gone", "--url", alone)).code == 0
        assert (await asyncio.to_thread(cli, "compute", "delete", "gone", "--url", alone)).code == 0
        await _deleted(alone, "gone")

        ran = await asyncio.to_thread(cli, "monitor", "gone", "--url", alone)

        assert ran.code == 1
        assert "nothing to watch: gone is deleted" in ran.err


async def _head(client: Client) -> int | None:
    page = await client.call("GET", "/v1/events/log", Page[LogEntryResource], limit=1)
    return page.items[0].sequence if page.items else None


async def _until(capsys: pytest.CaptureFixture[str], holds: object, seconds: float = 60) -> list[str]:
    """The lines printed so far, once ``holds`` is true of them."""
    assert callable(holds)
    lines: list[str] = []
    async with asyncio.timeout(seconds):
        while not holds(lines):
            await asyncio.sleep(0.1)
            lines += capsys.readouterr().out.splitlines()
    return lines


async def _deleted(url: str, name: str) -> None:
    async with asyncio.timeout(60):
        while '"deleted"' not in (await asyncio.to_thread(cli, "compute", "get", name, "--url", url, "--output", "json")).out:
            await asyncio.sleep(0.2)
