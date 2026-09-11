"""What a node prints, and who gets to hear it.

The filtering happens on the node, so a line that is silenced is never shipped —
which is why the assertions are about what reaches this terminal.
"""

import sys
import time
from collections.abc import Callable, Generator
from contextlib import redirect_stdout
from pathlib import Path

import cloudpickle
import pytest

import skyward as sky
from skyward.worker import journal

cloudpickle.register_pickle_by_value(sys.modules[__name__])


@sky.function
def talkative(mark: str) -> int:
    print(f"<<{mark}>>")
    return 1


@sky.function
@sky.silent
def under_its_breath(mark: str) -> int:
    print(f"<<{mark}>>")
    return 1


@sky.function
@sky.stdout(only="head")
def head_only(mark: str) -> int:
    print(f"<<{mark}>> from {sky.instance_info().rank}")
    return 1


@sky.function
def captured_by_a_callback() -> list[str]:
    lines: list[str] = []

    with sky.redirect_output(lines.append):
        print("first")
        print("second")

    return [line for line in lines if line.strip()]


def waited(read: Callable[[], str], marker: str, seconds: float = 20.0) -> str:
    """Everything printed until *marker* shows up, or until the wait runs out."""
    seen = ""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        seen += read()
        if marker in seen:
            return seen
        time.sleep(0.1)
    return seen


@pytest.mark.compute
@pytest.mark.xdist_group("pool")
def describe_what_a_node_prints() -> None:
    def it_reaches_the_terminal_that_asked_for_the_work(pool: sky.Compute, capsys: pytest.CaptureFixture[str]) -> None:
        assert talkative("hello") >> pool == 1

        assert "<<hello>>" in waited(lambda: capsys.readouterr().err, "<<hello>>")

    def describe_when_the_function_is_silenced() -> None:
        def it_never_leaves_the_node(pool: sky.Compute, capsys: pytest.CaptureFixture[str]) -> None:
            assert under_its_breath("quiet") >> pool == 1
            assert talkative("after-quiet") >> pool == 1

            seen = waited(lambda: capsys.readouterr().err, "<<after-quiet>>")

            assert "<<quiet>>" not in seen, "a later line arrived, so the silenced one had its chance"

    def describe_when_only_the_head_may_speak() -> None:
        def the_other_ranks_are_dropped_on_the_node(pool: sky.Compute, capsys: pytest.CaptureFixture[str]) -> None:
            assert head_only("solo") @ pool == [1, 1]
            assert talkative("after-solo") >> pool == 1

            seen = ""
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline and not ("<<solo>>" in seen and "<<after-solo>>" in seen):
                seen += capsys.readouterr().err
                time.sleep(0.1)

            assert seen.count("<<solo>>") == 1, "two nodes ran it, one of them printed"


@pytest.mark.compute
@pytest.mark.xdist_group("pool")
def describe_redirecting_output_inside_the_function() -> None:
    def it_hands_the_lines_to_the_callback_instead(pool: sky.Compute) -> None:
        assert captured_by_a_callback() >> pool == ["first", "second"]


@sky.silent
def hushed_steps() -> Generator[int]:
    try:
        print("<<hushed before>>")
        yield 1
        print("<<hushed after>>")
        yield 2
    finally:
        print("<<hushed finally>>")


@sky.stdout(only="head")
def head_steps() -> Generator[int]:
    try:
        print("<<head before>>")
        yield 1
        print("<<head after>>")
        yield 2
    finally:
        print("<<head finally>>")


@pytest.fixture
def node_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    events = tmp_path / "events.jsonl"
    monkeypatch.setattr(journal, "EVENTS", str(events))
    monkeypatch.setattr(journal, "LOCK", str(tmp_path / "events.lock"))
    monkeypatch.setenv("SKYWARD_NODE", "nod_test")
    monkeypatch.setenv("SKYWARD_COMPUTE", "cmp_test")
    monkeypatch.setenv("SKYWARD_RANK", "0")
    monkeypatch.setenv("SKYWARD_PEERS", "10.0.0.1,10.0.0.2")
    return events


def printed(events: Path) -> list[str]:
    if not events.exists():
        return []
    return [event.content for event in map(journal.parse, events.read_text().splitlines()) if isinstance(event, journal.Console)]


@pytest.mark.local
def describe_a_filtered_generator() -> None:
    def describe_when_it_is_silenced() -> None:
        def it_says_nothing_across_its_yields_and_its_finally(node_journal: Path) -> None:
            pulled: list[int] = []
            with redirect_stdout(journal.Journal("stdout")):
                for item in hushed_steps():
                    print(f"<<consumer {item}>>")
                    pulled.append(item)

            assert pulled == [1, 2]
            assert printed(node_journal) == ["<<consumer 1>>", "<<consumer 2>>"]

        def it_says_nothing_in_its_finally_when_closed_early(node_journal: Path) -> None:
            with redirect_stdout(journal.Journal("stdout")):
                steps = hushed_steps()
                assert next(steps) == 1
                steps.close()
                print("<<closed>>")

            assert printed(node_journal) == ["<<closed>>"]

    def describe_when_only_the_head_may_speak() -> None:
        def it_speaks_on_the_head(node_journal: Path) -> None:
            with redirect_stdout(journal.Journal("stdout")):
                assert list(head_steps()) == [1, 2]

            assert printed(node_journal) == ["<<head before>>", "<<head after>>", "<<head finally>>"]

        def it_says_nothing_on_another_rank(node_journal: Path, monkeypatch: pytest.MonkeyPatch) -> None:
            monkeypatch.setenv("SKYWARD_RANK", "1")

            with redirect_stdout(journal.Journal("stdout")):
                assert list(head_steps()) == [1, 2]
                print("<<after>>")

            assert printed(node_journal) == ["<<after>>"]
