"""Node-level _run_file_op: op → transport message mapping + quoting."""

from __future__ import annotations

import pytest
from casty import ActorContext, ActorSystem, Behavior, Behaviors

from skyward.actors.node.helpers import _run_file_op
from skyward.infra.ssh_actor import (
    CommandResult,
    Download,
    DownloadResult,
    RunCommand,
    WriteBytes,
    WriteResult,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _probe(recorded: list[object]) -> Behavior:
    async def receive(ctx: ActorContext, msg: object) -> Behavior:
        recorded.append(msg)
        match msg:
            case RunCommand(reply_to=rt):
                rt.tell(CommandResult(exit_code=0, stdout="listing", stderr=""))
            case WriteBytes(reply_to=rt):
                rt.tell(WriteResult(success=True))
            case Download(reply_to=rt):
                rt.tell(DownloadResult(success=True, content=b"bytes"))
        return Behaviors.same()
    return Behaviors.receive(receive)


async def test_ls_quotes_and_guards_path():
    async with ActorSystem("nfo-ls") as system:
        recorded: list[object] = []
        tref = system.spawn(_probe(recorded), "t")
        result = await _run_file_op(tref, 0, "ls", "/a b", b"", 5.0)
        assert result.success is True
        assert result.listing == "listing"
        assert isinstance(recorded[0], RunCommand)
        assert recorded[0].command == ("ls", "-la", "--", "'/a b'")


async def test_rm_uses_rm_rf():
    async with ActorSystem("nfo-rm") as system:
        recorded: list[object] = []
        tref = system.spawn(_probe(recorded), "t")
        result = await _run_file_op(tref, 1, "rm", "/tmp/x", b"", 5.0)
        assert result.success is True
        assert recorded[0].command == ("rm", "-rf", "--", "/tmp/x")


async def test_upload_uses_write_bytes():
    async with ActorSystem("nfo-up") as system:
        recorded: list[object] = []
        tref = system.spawn(_probe(recorded), "t")
        result = await _run_file_op(tref, 0, "upload", "/tmp/x", b"data", 5.0)
        assert result.success is True
        assert isinstance(recorded[0], WriteBytes)
        assert recorded[0].content == b"data"


async def test_download_uses_download_and_returns_bytes():
    async with ActorSystem("nfo-dl") as system:
        recorded: list[object] = []
        tref = system.spawn(_probe(recorded), "t")
        result = await _run_file_op(tref, 0, "download", "/tmp/x", b"", 5.0)
        assert result.success is True
        assert result.content == b"bytes"
        assert isinstance(recorded[0], Download)
