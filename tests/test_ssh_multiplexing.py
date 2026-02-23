from __future__ import annotations

from typing import Any
from unittest.mock import patch

import asyncssh
import pytest

import skyward as sky
from skyward import ComputePool, Worker

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180)]


def test_single_ssh_connection_per_node():
    connect_counts: dict[tuple[str, int], int] = {}
    original = asyncssh.connect

    async def spy(*args: Any, **kwargs: Any) -> asyncssh.SSHClientConnection:
        host = args[0] if args else kwargs.get("host", "")
        port = kwargs.get("port", 22)
        key = (str(host), int(port))
        connect_counts[key] = connect_counts.get(key, 0) + 1
        return await original(*args, **kwargs)

    with (
        patch("asyncssh.connect", spy),
        sky.App(console=False),
        ComputePool(
            provider=sky.Container(
                network="skyward",
                container_prefix="skyward-ssh-mux",
            ),
            nodes=2,
            worker=Worker(concurrency=1),
            vcpus=1,
            memory_gb=1,
        ),
    ):
        assert len(connect_counts) == 2, (
            f"Expected 2 endpoints, got {len(connect_counts)}: {connect_counts}"
        )
        assert all(n == 1 for n in connect_counts.values()), (
            f"Expected 1 connection per endpoint: {connect_counts}"
        )
