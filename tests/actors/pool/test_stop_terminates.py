"""Regression test: _shutdown_cluster must call provider.terminate with the
given iids, and is the final arbiter of whether pods actually die on
StopPool. If iids is empty, termination is silently skipped — which is the
main orphan-leak suspect path.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from skyward.actors.pool.actor import _shutdown_cluster


@pytest.mark.asyncio
async def test_shutdown_cluster_calls_terminate_with_iids() -> None:
    provider = MagicMock()
    provider.terminate = AsyncMock()
    provider.teardown = AsyncMock()
    cluster = MagicMock()

    await _shutdown_cluster(provider, cluster, ("pod-a", "pod-b"))

    provider.terminate.assert_awaited_once_with(cluster, ("pod-a", "pod-b"))
    provider.teardown.assert_awaited_once_with(cluster)


@pytest.mark.asyncio
async def test_shutdown_cluster_skips_terminate_on_empty_iids() -> None:
    """Silent-leak path: empty iids -> terminate NOT called; teardown still runs."""
    provider = MagicMock()
    provider.terminate = AsyncMock()
    provider.teardown = AsyncMock()
    cluster = MagicMock()

    await _shutdown_cluster(provider, cluster, ())

    provider.terminate.assert_not_called()
    provider.teardown.assert_awaited_once_with(cluster)


@pytest.mark.asyncio
async def test_shutdown_cluster_swallows_terminate_errors() -> None:
    """If terminate fails, _shutdown_cluster logs and continues to teardown."""
    provider = MagicMock()
    provider.terminate = AsyncMock(side_effect=RuntimeError("API down"))
    provider.teardown = AsyncMock()
    cluster = MagicMock()

    await _shutdown_cluster(provider, cluster, ("pod-a",))

    provider.terminate.assert_awaited_once()
    provider.teardown.assert_awaited_once()
