"""What a node prints, and who gets to hear it.

The filtering happens on the node, so a line that is silenced is never shipped —
which is why the assertions are about what reaches this terminal.
"""

import sys
import time
from collections.abc import Callable

import cloudpickle
import pytest

import skyward as sky

pytestmark = [pytest.mark.compute, pytest.mark.xdist_group("pool")]

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


def describe_redirecting_output_inside_the_function() -> None:
    def it_hands_the_lines_to_the_callback_instead(pool: sky.Compute) -> None:
        assert captured_by_a_callback() >> pool == ["first", "second"]
