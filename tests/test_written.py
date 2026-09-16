"""What a caller with no interpreter of its own can do.

The SDK pickles a live callable and sends the bytes. A browser has neither the
callable nor the pickler, so it sends the two things it does have — the text
somebody typed and the values they filled in — and the daemon turns both into
what everything below it already takes. Nothing here is a second way to run
something; it is a second way to say it, and it ends at the edge.
"""

import math
import os
from collections.abc import AsyncIterator
from pathlib import Path

import cloudpickle
import pytest
from litestar.testing import AsyncTestClient

from skyward.server.http.app import create_app, services
from skyward.server.http.controllers.tasks import pickled
from skyward.server.persistence.db import connect
from skyward.shared import codec
from skyward.shared.errors import SourceRejectedError
from skyward.shared.schemas import Call, TaskCreate
from skyward.worker.authored import authored

pytestmark = pytest.mark.local

AREA = "import math\n\n\ndef area(radius):\n    return math.pi * radius**2\n"


@pytest.fixture
async def http(tmp_path: Path) -> AsyncIterator[AsyncTestClient]:
    await connect(tmp_path / "skyward.sqlite")
    async with AsyncTestClient(app=create_app(services(), logging=False)) as client:
        yield client


def describe_a_function_written_as_text() -> None:
    def it_carries_its_text_and_no_bytecode() -> None:
        """Bytecode runs only on the Python that compiled it; a closure pickled by a 3.13 daemon crashed a 3.12 worker."""
        blob = cloudpickle.dumps(authored(AREA, "area"))

        assert b"CodeType" not in blob
        assert cloudpickle.loads(blob)(2) == pytest.approx(math.pi * 4)

    def a_keyword_the_function_takes_is_never_the_one_it_was_written_with() -> None:
        written = authored("def joined(source, name):\n    return source + name\n", "joined")

        assert written(source="a", name="b") == "ab"

    async def it_registers_the_pickle_the_sdk_would_have_uploaded(http: AsyncTestClient) -> None:
        written = await http.post("/v1/functions", json={"name": "area", "source": AREA})

        assert written.status_code == 201
        assert written.json()["sha256"] == await codec.digest(codec.dumps(authored(AREA, "area")))
        assert written.json()["codec"] == "cloudpickle+lz4", "the same blob, by the same name, as an upload"

    async def the_same_text_written_twice_is_the_same_function(http: AsyncTestClient) -> None:
        first = await http.post("/v1/functions", json={"name": "area", "source": AREA})
        again = await http.post("/v1/functions", json={"name": "area", "source": AREA})

        assert (first.status_code, again.status_code) == (201, 200)
        assert again.json()["sha256"] == first.json()["sha256"]

    async def editing_it_is_another_function_under_the_same_name(http: AsyncTestClient) -> None:
        first = await http.post("/v1/functions", json={"name": "area", "source": AREA})
        edited = await http.post("/v1/functions", json={"name": "area", "source": AREA.replace("**2", "**3")})

        assert edited.status_code == 201
        assert edited.json()["sha256"] != first.json()["sha256"], "a name takes in every version of its code"

        listed = (await http.get("/v1/functions")).json()
        assert [row["sha256"] for row in listed["items"]] == [edited.json()["sha256"], first.json()["sha256"]], "newest first"
        assert listed["total"] == 2

    async def the_text_is_read_back_with_it(http: AsyncTestClient) -> None:
        sha = (await http.post("/v1/functions", json={"name": "area", "source": AREA})).json()["sha256"]

        assert (await http.get(f"/v1/functions/{sha}")).json()["source"] == AREA

    async def a_function_the_sdk_pickled_has_no_text_to_read_back(http: AsyncTestClient) -> None:
        blob = codec.dumps(authored(AREA, "area"))
        sha = await codec.digest(blob)

        await http.put(f"/v1/functions/{sha}", content=blob, headers={"X-Skyward-Function-Name": "area"})

        assert (await http.get(f"/v1/functions/{sha}")).json()["source"] is None, "compiled bytecode has no source to give back"


def describe_text_that_is_not_a_function() -> None:
    async def one_that_does_not_parse_is_refused_where_it_was_written(http: AsyncTestClient) -> None:
        refused = await http.post("/v1/functions", json={"name": "area", "source": "def area(:\n"})

        assert refused.status_code == 422
        assert refused.json()["code"] == "source_rejected"
        assert refused.json()["details"]["line"] == 1, "so the line can be pointed at rather than described"

    async def one_that_defines_another_name_is_told_which(http: AsyncTestClient) -> None:
        refused = await http.post("/v1/functions", json={"name": "area", "source": "def volume(r):\n    return r\n"})

        assert refused.status_code == 422
        assert refused.json()["details"]["defines"] == ["volume"]

    async def nothing_is_stored_by_a_refusal(http: AsyncTestClient) -> None:
        await http.post("/v1/functions", json={"name": "area", "source": "def area(:\n"})

        assert (await http.get("/v1/functions")).json()["items"] == []


def describe_where_the_text_runs() -> None:
    def it_is_not_where_it_was_captured() -> None:
        """The daemon holds the source, never executes it — which is why a module it could not import is still registrable."""
        call = authored("import a_library_no_daemon_has\n\n\ndef go():\n    return 1\n", "go")

        with pytest.raises(ModuleNotFoundError):
            call()

    def it_runs_once_per_call_and_keeps_nothing_between() -> None:
        counting = "counted = []\n\n\ndef count():\n    counted.append(1)\n    return len(counted)\n"
        call = authored(counting, "count")

        assert (call(), call()) == (1, 1), "a module compiled per call is a module with no yesterday"

    def a_name_bound_to_something_else_is_refused_before_a_machine_is_asked() -> None:
        with pytest.raises(SourceRejectedError, match="defines no function"):
            authored("area = 3\n", "area")


def describe_arguments_written_as_json() -> None:
    async def they_become_the_pickle_everything_below_takes() -> None:
        submitted = await pickled(
            TaskCreate(compute="cmp_1", function="a" * 64, dispatch="one", call=Call(args=(2.0,), kwargs={"unit": "m", "places": 3})),
        )

        assert submitted.call is None, "the second way to say it ends at the edge"
        assert codec.loads(submitted.args_inline or b"") == ((2.0,), {"unit": "m", "places": 3})

    async def a_call_with_neither_is_a_call_with_no_arguments() -> None:
        submitted = await pickled(TaskCreate(compute="cmp_1", function="a" * 64, dispatch="one", call=Call()))

        assert codec.loads(submitted.args_inline or b"") == ((), {})

    async def a_caller_that_pickled_its_own_is_left_alone() -> None:
        already = TaskCreate(compute="cmp_1", function="a" * 64, dispatch="one", args_inline=os.urandom(32))

        assert await pickled(already) is already

    async def the_shape_of_them_is_refused_by_the_schema(http: AsyncTestClient) -> None:
        refused = await http.post(
            "/v1/tasks",
            json={"compute": "cmp_1", "function": "a" * 64, "dispatch": "one", "call": {"args": 7}},
            headers={"Idempotency-Key": "k"},
        )

        assert refused.status_code == 400, "arguments are a list and keyword arguments are an object, before anything is bought"
