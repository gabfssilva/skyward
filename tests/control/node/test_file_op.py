"""Node-level _run_file_op: op → transport method mapping + quoting."""

from __future__ import annotations

import pytest

from skyward.control.node.helpers import _run_file_op
from skyward.infra.ssh_transport import CommandResult

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class _FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def run(
        self, *command: str, timeout: float | None = None, check: bool = False,
    ) -> CommandResult:
        self.calls.append(("run", command))
        return CommandResult(exit_code=0, stdout="listing", stderr="")

    async def write_bytes(self, remote: str, content: bytes) -> None:
        self.calls.append(("write_bytes", (remote, content)))

    async def download(self, remote: str) -> bytes:
        self.calls.append(("download", (remote,)))
        return b"bytes"


async def test_ls_quotes_and_guards_path():
    t = _FakeTransport()
    result = await _run_file_op(t, 0, "ls", "/a b", b"", 5.0)  # type: ignore[arg-type]
    assert result.success is True
    assert result.listing == "listing"
    assert t.calls[0] == ("run", ("ls", "-la", "--", "'/a b'"))


async def test_rm_uses_rm_rf():
    t = _FakeTransport()
    result = await _run_file_op(t, 1, "rm", "/tmp/x", b"", 5.0)  # type: ignore[arg-type]
    assert result.success is True
    assert t.calls[0] == ("run", ("rm", "-rf", "--", "/tmp/x"))


async def test_upload_uses_write_bytes():
    t = _FakeTransport()
    result = await _run_file_op(t, 0, "upload", "/tmp/x", b"data", 5.0)  # type: ignore[arg-type]
    assert result.success is True
    assert t.calls[0] == ("write_bytes", ("/tmp/x", b"data"))


async def test_download_uses_download_and_returns_bytes():
    t = _FakeTransport()
    result = await _run_file_op(t, 0, "download", "/tmp/x", b"", 5.0)  # type: ignore[arg-type]
    assert result.success is True
    assert result.content == b"bytes"
    assert t.calls[0] == ("download", ("/tmp/x",))


async def test_transport_error_maps_to_failed_result():
    class _Boom(_FakeTransport):
        async def run(
            self, *command: str, timeout: float | None = None, check: bool = False,
        ) -> CommandResult:
            raise RuntimeError("transport failed: gone")

    result = await _run_file_op(_Boom(), 0, "ls", "/tmp", b"", 5.0)  # type: ignore[arg-type]
    assert result.success is False
    assert "gone" in (result.error or "")
