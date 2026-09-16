"""The text of a function the SDK uploads, and what reaches the daemon of it.

The pickle a function travels as has no text in it, so the text is taken where it
exists — in the process that defined the function, off the file it came from — and
it is written out whole enough to read: what the function imports and uses from its
own module, above the function itself.
"""

from __future__ import annotations

import functools
import importlib
import sys
import textwrap
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest
from litestar.testing import AsyncTestClient

from skyward.core.excerpt import defined, excerpt
from skyward.server.http.app import create_app, services
from skyward.server.persistence.db import connect
from skyward.shared import codec

pytestmark = pytest.mark.local

type Load = Callable[[str], ModuleType]


@pytest.fixture
def load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Load]:
    """A module written to a real file and imported, the way a user's own code is."""
    monkeypatch.syspath_prepend(str(tmp_path))
    written: list[str] = []

    def loading(source: str) -> ModuleType:
        name = f"users_module_{abs(hash(tmp_path))}_{len(written)}"
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(source).lstrip())
        written.append(name)
        importlib.invalidate_caches()
        return importlib.import_module(name)

    yield loading
    for name in written:
        sys.modules.pop(name, None)


def describe_what_is_written_above_the_function() -> None:
    def it_is_the_module_the_function_needs_in_the_order_of_its_file(load: Load) -> None:
        module = load(
            """
            import json
            import collections as col
            from dataclasses import dataclass, asdict
            from statistics import median

            JOB_MARKER = "@@job "
            UNUSED = "nobody reads this"


            @dataclass(frozen=True)
            class Job:
                dataset: str
                rate: float


            def fill_job(job: Job) -> list[float]:
                return [median([job.rate, 0.5])]


            def job_event(job: Job, state: str) -> dict:
                return {"job": asdict(job), "state": state}


            def never_called():
                return 1


            def fill_job_encoded(job: Job) -> str:
                print(JOB_MARKER + json.dumps(job_event(job, "started")))
                return json.dumps(col.Counter(fill_job(job)))
            """
        )

        assert (
            excerpt(module.fill_job_encoded)
            == textwrap.dedent(
                """
            import collections as col
            import json
            from dataclasses import asdict
            from dataclasses import dataclass
            from statistics import median


            JOB_MARKER = '@@job '


            @dataclass(frozen=True)
            class Job:
                dataset: str
                rate: float


            def fill_job(job: Job) -> list[float]:
                return [median([job.rate, 0.5])]


            def job_event(job: Job, state: str) -> dict:
                return {"job": asdict(job), "state": state}


            def fill_job_encoded(job: Job) -> str:
                print(JOB_MARKER + json.dumps(job_event(job, "started")))
                return json.dumps(col.Counter(fill_job(job)))
            """
            ).lstrip()
        )

    def a_helper_of_a_helper_is_followed(load: Load) -> None:
        module = load(
            """
            def innermost():
                return 1

            def middle():
                return innermost()

            def outer():
                return middle()
            """
        )

        assert "def innermost" in (excerpt(module.outer) or "")

    def two_helpers_that_call_each_other_are_each_written_once(load: Load) -> None:
        module = load(
            """
            def ping(n):
                return pong(n - 1) if n else 0

            def pong(n):
                return ping(n - 1) if n else 1
            """
        )

        text = excerpt(module.ping) or ""
        assert (text.count("def ping"), text.count("def pong")) == (1, 1)


def describe_how_a_name_came_into_the_module() -> None:
    @pytest.mark.parametrize(
        ("line", "use"),
        [
            ("import json", "json.dumps(x)"),
            ("import collections as col", "col.Counter(x)"),
            ("from xml.dom import minidom", "minidom.parseString(x)"),
            ("from statistics import median", "median(x)"),
            ("from statistics import median as middle", "middle(x)"),
        ],
    )
    def it_is_written_the_way_it_was_imported(load: Load, line: str, use: str) -> None:
        module = load(f"{line}\n\n\ndef run(x):\n    return {use}\n")

        assert (excerpt(module.run) or "").splitlines()[0] == line

    def a_decorator_is_a_name_the_function_uses(load: Load) -> None:
        module = load("import functools\n\n\n@functools.cache\ndef run(x):\n    return x\n")

        assert (excerpt(module.run.__wrapped__) or "").startswith("import functools\n")

    def a_value_too_long_to_read_is_named_and_not_shown(load: Load) -> None:
        module = load("TABLE = list(range(1000))\n\n\ndef run(x):\n    return TABLE[x]\n")

        assert "TABLE = ...  # list" in (excerpt(module.run) or "")


def describe_a_function_handed_over_inside_something_else() -> None:
    def a_bound_method_is_read_as_the_method_it_binds(load: Load) -> None:
        module = load("import json\n\n\nclass Box:\n    def dumped(self, x):\n        return json.dumps(x)\n")

        assert excerpt(defined(module.Box().dumped)) == "import json\n\n\ndef dumped(self, x):\n    return json.dumps(x)\n"

    def a_partial_is_read_as_the_function_it_was_made_of(load: Load) -> None:
        module = load("import json\n\n\ndef dumped(indent, x):\n    return json.dumps(x, indent=indent)\n")

        written = defined(functools.partial(functools.partial(module.dumped), 2))

        assert written is module.dumped, "however deep the partials go"
        assert (excerpt(written) or "").endswith("def dumped(indent, x):\n    return json.dumps(x, indent=indent)\n")


def describe_a_function_with_no_file_behind_it() -> None:
    def it_has_no_text() -> None:
        namespace: dict[str, Callable[[], int]] = {}
        exec("def typed_at_a_prompt():\n    return 1\n", namespace)

        assert excerpt(namespace["typed_at_a_prompt"]) is None


@pytest.fixture
async def http(tmp_path: Path) -> AsyncIterator[AsyncTestClient]:
    await connect(tmp_path / "skyward.sqlite")
    async with AsyncTestClient(app=create_app(services(), logging=False)) as client:
        yield client


def describe_the_text_reaching_the_daemon() -> None:
    async def it_is_read_back_with_the_function_it_was_sent_for(http: AsyncTestClient, load: Load) -> None:
        module = load("import json\n\n\ndef run(x):\n    return json.dumps(x)\n")
        blob = codec.dumps(module.run)
        sha = await codec.digest(blob)
        await http.put(f"/v1/functions/{sha}", content=blob, headers={"X-Skyward-Function-Name": "run"})

        sent = await http.put(f"/v1/functions/{sha}/excerpt", json={"text": excerpt(module.run)})

        assert sent.status_code == 200, sent.text
        assert (await http.get(f"/v1/functions/{sha}")).json()["excerpt"] == "import json\n\n\ndef run(x):\n    return json.dumps(x)\n"

    async def a_function_the_daemon_does_not_have_is_refused(http: AsyncTestClient) -> None:
        refused = await http.put(f"/v1/functions/{'0' * 64}/excerpt", json={"text": "def run():\n    pass\n"})

        assert refused.status_code == 404
