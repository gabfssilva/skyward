from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestClientSubscribeMethod:
    def test_subscribe_method_exists(self) -> None:
        from skyward.daemon.client import DaemonClient

        client = DaemonClient()
        assert hasattr(client, "subscribe")

    def test_subscribe_is_async_generator(self) -> None:
        import inspect

        from skyward.daemon.client import DaemonClient

        client = DaemonClient()
        assert inspect.isasyncgenfunction(client.subscribe)
