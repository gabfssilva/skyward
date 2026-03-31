import asyncio

import pytest

from vscode.sidecar.bridge import Bridge


@pytest.fixture
def bridge() -> Bridge:
    return Bridge()


@pytest.mark.asyncio
async def test_handle_ping_without_daemon(bridge: Bridge) -> None:
    """Ping returns ok=False when daemon is not running."""
    result = await bridge.handle("daemon/ping", {})
    assert result == {"ok": False}


@pytest.mark.asyncio
async def test_handle_pools_list_without_daemon(bridge: Bridge) -> None:
    """List pools returns empty list when daemon is not running."""
    result = await bridge.handle("pools/list", {})
    assert result == {"pools": []}


@pytest.mark.asyncio
async def test_handle_unknown_method(bridge: Bridge) -> None:
    """Unknown methods raise ValueError."""
    with pytest.raises(ValueError, match="Unknown method"):
        await bridge.handle("unknown/method", {})


@pytest.mark.asyncio
async def test_subscribe_unsubscribe(bridge: Bridge) -> None:
    """Subscribe creates a task, unsubscribe cancels it."""
    result = await bridge.handle("pools/subscribe", {"pool": "train"})
    assert result == {"ok": True}
    assert "train" in bridge._subscriptions

    result = await bridge.handle("pools/unsubscribe", {"pool": "train"})
    assert result == {"ok": True}
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_close(bridge: Bridge) -> None:
    """Close cancels subscriptions and clears client."""
    await bridge.close()
    assert bridge._client is None
    assert len(bridge._subscriptions) == 0
