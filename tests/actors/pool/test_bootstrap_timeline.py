from __future__ import annotations

from types import MappingProxyType

import pytest

from skyward.actors.messages import BootstrapCommand, BootstrapPhase, ConsoleOutput
from skyward.actors.pool.state import apply_stream_event

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _ni():
    from unittest.mock import MagicMock

    ni = MagicMock()
    ni.node = 0
    ni.instance.id = "i-1"
    return ni


_EMPTY = MappingProxyType({})


class TestApplyStreamEvent:
    def test_started_creates_timeline(self):
        tls = apply_stream_event(
            _EMPTY, "i-1", BootstrapPhase(instance=_ni(), event="started", phase="apt"),
        )
        tl = tls["i-1"]
        assert tl.phases == ("apt",)
        assert tl.active == "apt"
        assert tl.completed == frozenset()

    def test_completed_marks_phase(self):
        tls = apply_stream_event(
            _EMPTY, "i-1", BootstrapPhase(instance=_ni(), event="started", phase="apt"),
        )
        tls = apply_stream_event(
            tls, "i-1", BootstrapPhase(instance=_ni(), event="completed", phase="apt"),
        )
        assert "apt" in tls["i-1"].completed

    def test_next_phase_completes_previous_active(self):
        tls = apply_stream_event(
            _EMPTY, "i-1", BootstrapPhase(instance=_ni(), event="started", phase="apt"),
        )
        tls = apply_stream_event(
            tls, "i-1", BootstrapPhase(instance=_ni(), event="started", phase="uv"),
        )
        tl = tls["i-1"]
        assert tl.active == "uv"
        assert "apt" in tl.completed
        assert tl.phases == ("apt", "uv")

    def test_aggregate_bootstrap_phase_ignored(self):
        tls = apply_stream_event(
            _EMPTY, "i-1",
            BootstrapPhase(instance=_ni(), event="started", phase="bootstrap"),
        )
        assert "i-1" not in tls

    def test_command_and_output_update_output(self):
        tls = apply_stream_event(
            _EMPTY, "i-1", BootstrapPhase(instance=_ni(), event="started", phase="apt"),
        )
        tls = apply_stream_event(
            tls, "i-1", BootstrapCommand(instance=_ni(), command="x" * 200),
        )
        assert tls["i-1"].output == "x" * 80
        tls = apply_stream_event(
            tls, "i-1", ConsoleOutput(instance=_ni(), content="  installing…  "),
        )
        assert tls["i-1"].output == "installing…"

    def test_comment_and_blank_output_ignored(self):
        tls = apply_stream_event(
            _EMPTY, "i-1", BootstrapPhase(instance=_ni(), event="started", phase="apt"),
        )
        before = tls["i-1"]
        tls = apply_stream_event(tls, "i-1", ConsoleOutput(instance=_ni(), content="# hi"))
        tls = apply_stream_event(tls, "i-1", ConsoleOutput(instance=_ni(), content="   "))
        assert tls["i-1"] is before
