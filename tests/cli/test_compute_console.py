from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestConsoleCommandExists:
    def test_console_command_registered(self) -> None:
        from skyward.cli.compute import console_pool

        assert callable(console_pool)

    def test_console_stream_is_coroutine_function(self) -> None:
        import inspect

        from skyward.cli.compute import _console_stream

        assert inspect.iscoroutinefunction(_console_stream)
