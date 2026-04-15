"""H3: ``ensure_server`` health probe / race-safe spawn logic."""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@pytest.mark.asyncio
async def test_probe_returns_false_for_missing_socket(tmp_path: Path) -> None:
    from skyward.server.driver.spawn import _probe

    assert await _probe(str(tmp_path / "nope.sock")) is False


@pytest.mark.asyncio
async def test_ensure_server_raises_timeout_when_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skyward.server.driver import spawn

    monkeypatch.setattr(spawn, "_spawn", lambda _sock: None)

    with pytest.raises(TimeoutError):
        await spawn.ensure_server(
            tmp_path / "sock",
            tmp_path / "lock",
            deadline_s=0.5,
            backoff_s=0.05,
        )
