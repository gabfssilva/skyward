"""What the daemon can tell about a pickled function without running it.

Unpickling is executing, and the daemon is not the machine anybody is paying to
run code on. So the payload is disassembled instead — and what comes out has to
be enough to say what a function is, which of its uploads are the same function,
and what a node needs to have for it to load at all.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from skyward.shared import codec
from skyward.shared.reading import read

pytestmark = pytest.mark.local

WIDTH = "import math\n\n\ndef area(radius):\n    return math.pi * radius**2\n"


def built(source: str, name: str) -> object:
    """A live callable out of source, the way any user's module would define one."""
    namespace: dict[str, object] = {}
    exec(compile(textwrap.dedent(source), "<a user's file>", "exec"), namespace)
    return namespace[name]


async def reading_of(source: str, name: str) -> object:
    return await read(codec.dumps(built(source, name)))


def describe_what_a_payload_says_about_itself() -> None:
    async def it_names_the_function_and_the_file_it_was_written_in() -> None:
        found = await reading_of("def area(radius):\n    return radius\n", "area")

        assert (found.qualname, found.origin) == ("area", "<a user's file>")

    async def it_says_the_function_that_was_sent_and_not_what_it_carries() -> None:
        found = await reading_of("def outer(x):\n    def inner(y):\n        return y\n    return inner(x)\n", "outer")

        assert found.qualname == "outer", "its nested function is rebuilt first, and is not the one that was sent"


def describe_which_uploads_are_the_same_function() -> None:
    async def one_that_moved_down_the_file_is_the_same_one() -> None:
        here = await reading_of("def area(r):\n    return r * r\n", "area")
        lower = await reading_of("\n\n\n\ndef area(r):\n    return r * r\n", "area")

        assert here.shape == lower.shape, "a line number is not a change to what it does"

    async def one_that_captured_something_else_is_the_same_one() -> None:
        def taking(job: str) -> object:
            def run() -> str:
                return job

            return run

        first, second = codec.dumps(taking("a" * 32)), codec.dumps(taking("b" * 32))

        assert first != second, "the blobs differ, which is why they are two rows"
        assert (await read(first)).shape == (await read(second)).shape, "and the same function, which is why they are one version"

    async def one_whose_body_changed_is_not() -> None:
        before = await reading_of("def area(r):\n    return r * r\n", "area")
        after = await reading_of("def area(r):\n    return r * r * r\n", "area")

        assert before.shape != after.shape

    async def one_that_calls_something_else_is_not() -> None:
        before = await reading_of("def area(r):\n    return one(r)\n", "area")
        after = await reading_of("def area(r):\n    return two(r)\n", "area")

        assert before.shape != after.shape, "the names it reaches for are part of what it does"


def describe_a_payload_that_cannot_be_read() -> None:
    async def bytes_that_are_not_a_payload_are_read_as_nothing() -> None:
        assert await read(b"not a pickle at all") == type(await read(b""))()

    async def a_truncated_one_is_read_as_nothing() -> None:
        whole = codec.dumps(built(WIDTH, "area"))

        assert (await read(whole[: len(whole) // 2])).shape is None


def describe_reading_a_payload_that_would_not_survive_being_loaded() -> None:
    async def it_reads_one_that_needs_a_library_the_daemon_does_not_have(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The point of the whole exercise: the daemon learns what a function is without loading anything it names."""
        (tmp_path / "a_library_the_daemon_lacks.py").write_text("class Model:\n    pass\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        try:
            library = importlib.import_module("a_library_the_daemon_lacks")
            blob = codec.dumps(library.Model)
        finally:
            sys.modules.pop("a_library_the_daemon_lacks", None)
            (tmp_path / "a_library_the_daemon_lacks.py").unlink()

        found = await read(blob)

        assert found == type(found)(), "a class sent by reference has no code of its own to read"
        assert "a_library_the_daemon_lacks" not in sys.modules, "and naming it was not importing it"
