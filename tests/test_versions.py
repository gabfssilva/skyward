"""One function across its uploads.

The daemon names code by the hash of its pickle, and a pickle changes for reasons
that have nothing to do with the code: the function moved down its file, or a run
captured a different id. So uploads are grouped into the function they are an
upload of, and a version counts what a person would count — the edits.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from litestar.testing import AsyncTestClient

from skyward.server.http.app import create_app, services
from skyward.server.persistence.db import connect
from skyward.shared import codec

pytestmark = pytest.mark.local

type Upload = dict[str, object]

SQUARE = "def area(r):\n    return r * r\n"
CUBE = "def area(r):\n    return r * r * r\n"


@pytest.fixture
async def http(tmp_path: Path) -> AsyncIterator[AsyncTestClient]:
    await connect(tmp_path / "skyward.sqlite")
    async with AsyncTestClient(app=create_app(services(), logging=False)) as client:
        yield client


def built(source: str, name: str, file: str = "train.py") -> Callable[..., object]:
    """A live function, defined in a file the way any user's module defines one."""
    namespace: dict[str, Callable[..., object]] = {}
    exec(compile(source, file, "exec"), namespace)
    return namespace[name]


def capturing(job: str) -> Callable[[], str]:
    def run() -> str:
        return job

    return run


async def upload(http: AsyncTestClient, function: Callable[..., object], name: str = "area") -> Upload:
    """What the SDK does with a callable: pickle it, name it by hash, put it."""
    blob = codec.dumps(function)
    sha = await codec.digest(blob)
    put = await http.put(f"/v1/functions/{sha}", content=blob, headers={"X-Skyward-Function-Name": name})
    assert put.status_code in (200, 201), put.text
    return (await http.get(f"/v1/functions/{sha}")).json()


def describe_one_function_across_its_uploads() -> None:
    async def moving_it_down_its_file_is_not_an_edit(http: AsyncTestClient) -> None:
        here = await upload(http, built(SQUARE, "area"))
        lower = await upload(http, built("\n\n\n" + SQUARE, "area"))

        assert here["sha256"] != lower["sha256"], "the pickle moved with the line numbers"
        assert (lower["lineage"], lower["version"]) == (here["lineage"], 1)

    async def a_run_that_captured_something_else_is_not_an_edit(http: AsyncTestClient) -> None:
        first = await upload(http, capturing("a" * 32), "run")
        second = await upload(http, capturing("b" * 32), "run")

        assert first["sha256"] != second["sha256"]
        assert (first["version"], second["version"]) == (1, 1)

    async def an_edit_is_the_next_version(http: AsyncTestClient) -> None:
        await upload(http, built(SQUARE, "area"))
        edited = await upload(http, built(CUBE, "area"))

        assert edited["version"] == 2

    async def going_back_to_older_code_is_a_new_version_and_not_the_old_number(http: AsyncTestClient) -> None:
        """So the newest upload always carries the highest version, and running the latest runs what was last sent."""
        await upload(http, built(SQUARE, "area"))
        await upload(http, built(CUBE, "area"))
        reverted = await upload(http, built("\n" + SQUARE, "area"))

        assert reverted["version"] == 3

    async def the_same_name_in_another_file_is_another_function(http: AsyncTestClient) -> None:
        mine = await upload(http, built(SQUARE, "area", file="geometry.py"))
        theirs = await upload(http, built(CUBE, "area", file="volumes.py"))

        assert mine["lineage"] != theirs["lineage"]
        assert (mine["version"], theirs["version"]) == (1, 1), "neither is an edit of the other"

    async def it_says_which_function_and_which_file(http: AsyncTestClient) -> None:
        found = await upload(http, built(SQUARE, "area"))

        assert (found["qualname"], found["origin"]) == ("area", "train.py")


def describe_listing_functions_rather_than_uploads() -> None:
    async def latest_is_one_row_per_function_with_its_highest_version(http: AsyncTestClient) -> None:
        await upload(http, built(SQUARE, "area"))
        await upload(http, built(CUBE, "area"))
        await upload(http, capturing("a" * 32), "run")
        newest = await upload(http, capturing("b" * 32), "run")

        listed = (await http.get("/v1/functions", params={"latest": "true"})).json()

        assert [(row["name"], row["version"]) for row in listed["items"]] == [("run", 1), ("area", 2)]
        assert listed["items"][0]["sha256"] == newest["sha256"], "the newest upload stands for its function"
        assert listed["total"] == 2

    async def a_lineage_is_every_upload_of_one_function_newest_first(http: AsyncTestClient) -> None:
        first = await upload(http, built(SQUARE, "area"))
        await upload(http, capturing("a" * 32), "run")
        second = await upload(http, built(CUBE, "area"))

        listed = (await http.get("/v1/functions", params={"lineage": str(first["lineage"])})).json()

        assert [row["sha256"] for row in listed["items"]] == [second["sha256"], first["sha256"]]
        assert [row["version"] for row in listed["items"]] == [2, 1]

    async def a_page_of_it_picks_up_where_the_last_one_ended(http: AsyncTestClient) -> None:
        uploads = [await upload(http, capturing(str(n) * 32), "run") for n in range(5)]

        first = (await http.get("/v1/functions", params={"limit": 2})).json()
        rest = (await http.get("/v1/functions", params={"limit": 10, "cursor": first["next_cursor"]})).json()

        seen = [row["sha256"] for row in first["items"] + rest["items"]]
        assert seen == [row["sha256"] for row in reversed(uploads)], "newest first, nothing repeated, nothing skipped"
        assert (first["total"], rest["total"]) == (5, 5)


def describe_a_function_written_in_the_console() -> None:
    async def its_versions_are_its_text(http: AsyncTestClient) -> None:
        """Its payload is the console's own closure, identical for every function written there, so it cannot be what tells two apart."""
        first = (await http.post("/v1/functions", json={"name": "area", "source": SQUARE})).json()
        other = (await http.post("/v1/functions", json={"name": "volume", "source": "def volume(r):\n    return r\n"})).json()
        edited = (await http.post("/v1/functions", json={"name": "area", "source": CUBE})).json()

        assert first["lineage"] != other["lineage"], "two functions, not two versions of the console's closure"
        assert (first["version"], other["version"], edited["version"]) == (1, 1, 2)
        assert edited["qualname"] == "area"
