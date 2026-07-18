"""File transfer and remote exec, both halves, without a machine.

The daemon half — which nodes an operation lands on, and what it sends them — is
driven against a fake SSH channel; the endpoint half against a fake ``Files``, so
the wire shape can be read off without a compute being live. What is deliberately
not covered is SFTP itself: :meth:`SshChannel.get` is asyncssh's client talking to
a real server, and faking that would be testing the fake.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from litestar.testing import AsyncTestClient

from skyward.application import ports
from skyward.application.runtimes import Files, Runtime, Runtimes
from skyward.runtime.source import Source
from skyward.runtime.ssh import Result
from skyward.server.app import create_app, with_real

pytestmark = pytest.mark.unit


def _noop(*_: object) -> None:
    pass


def _runtime() -> tuple[Runtimes, Runtime]:
    runtimes = Runtimes(_noop, _noop, _noop, _noop)
    return runtimes, runtimes.open("cmp", Source(argument="skyward"), "key")


class _FakeSsh:
    """One machine's link, remembering what it was asked to do."""

    def __init__(self, refuses: bool = False) -> None:
        self.commands: list[str] = []
        self.writes: list[tuple[str, bytes]] = []
        self.reads: list[str] = []
        self._refuses = refuses

    async def run(self, command: str, *, timeout: float | None = None) -> Result:
        self.commands.append(command)
        return Result(exit_code=0, stdout=f"ran {command}", stderr="")

    async def put(self, path: str, content: bytes) -> None:
        if self._refuses:
            raise OSError("permission denied")
        self.writes.append((path, content))

    async def get(self, path: str) -> AsyncIterator[bytes]:
        self.reads.append(path)
        yield b"first"
        yield b"second"


class _FakeNode:
    def __init__(self, rank: int, ready: bool = True, refuses: bool = False) -> None:
        self.tunnel = 1 if ready else None
        self._rank = rank
        self._ssh = _FakeSsh(refuses)


def _nodes(runtime: Runtime, monkeypatch: pytest.MonkeyPatch, **nodes: _FakeNode) -> None:
    monkeypatch.setattr(runtime, "nodes", dict(nodes))


async def test_selecting_every_node_reaches_all_the_ready_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime = _runtime()
    _nodes(runtime, monkeypatch, a=_FakeNode(0), b=_FakeNode(1), c=_FakeNode(2))

    ran = await runtime.run("all", "hostname")

    assert [node for node, _ in ran] == ["a", "b", "c"]


async def test_selecting_every_node_skips_one_that_is_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime = _runtime()
    _nodes(runtime, monkeypatch, a=_FakeNode(0), b=_FakeNode(1, ready=False))

    ran = await runtime.run("all", "hostname")

    assert [node for node, _ in ran] == ["a"]


async def test_selecting_a_rank_reaches_only_the_node_holding_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime = _runtime()
    a, b = _FakeNode(0), _FakeNode(1)
    _nodes(runtime, monkeypatch, a=a, b=b)

    ran = await runtime.run(1, "hostname")

    assert [node for node, _ in ran] == ["b"]
    assert a._ssh.commands == []
    assert b._ssh.commands == ["hostname"]


async def test_selecting_a_rank_that_is_not_ready_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime = _runtime()
    _nodes(runtime, monkeypatch, a=_FakeNode(0), b=_FakeNode(1, ready=False))

    with pytest.raises(RuntimeError, match="no ready node at rank 1"):
        await runtime.run(1, "hostname")


async def test_selecting_a_rank_nobody_holds_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime = _runtime()
    _nodes(runtime, monkeypatch, a=_FakeNode(0))

    with pytest.raises(RuntimeError, match="no ready node at rank 7"):
        await runtime.run(7, "hostname")


async def test_selecting_on_a_compute_with_nothing_ready_raises() -> None:
    _, runtime = _runtime()

    with pytest.raises(RuntimeError, match="no ready node to reach"):
        await runtime.run("all", "hostname")


async def test_run_tags_each_answer_with_the_node_that_gave_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime = _runtime()
    _nodes(runtime, monkeypatch, a=_FakeNode(0), b=_FakeNode(1))

    ran = await runtime.run("all", "nvidia-smi")

    assert ran == (
        ("a", Result(exit_code=0, stdout="ran nvidia-smi", stderr="")),
        ("b", Result(exit_code=0, stdout="ran nvidia-smi", stderr="")),
    )


async def test_put_writes_the_same_bytes_to_every_targeted_node(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime = _runtime()
    a, b = _FakeNode(0), _FakeNode(1)
    _nodes(runtime, monkeypatch, a=a, b=b)

    written = await runtime.put("all", "/opt/data.csv", b"rows")

    assert written == (("a", None), ("b", None))
    assert a._ssh.writes == [("/opt/data.csv", b"rows")]
    assert b._ssh.writes == [("/opt/data.csv", b"rows")]


async def test_put_reports_the_node_that_refused_without_losing_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime = _runtime()
    a, b = _FakeNode(0), _FakeNode(1, refuses=True)
    _nodes(runtime, monkeypatch, a=a, b=b)

    written = await runtime.put("all", "/opt/data.csv", b"rows")

    assert written == (("a", None), ("b", "permission denied"))
    assert a._ssh.writes == [("/opt/data.csv", b"rows")]


async def test_get_streams_one_node_s_copy_of_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime = _runtime()
    a, b = _FakeNode(0), _FakeNode(1)
    _nodes(runtime, monkeypatch, a=a, b=b)

    read = b"".join([chunk async for chunk in runtime.get(1, "/opt/model.pt")])

    assert read == b"firstsecond"
    assert b._ssh.reads == ["/opt/model.pt"]
    assert a._ssh.reads == []


async def test_get_refuses_a_rank_before_it_opens_a_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raise is the call's, not the first chunk's — by then a 200 has gone out."""
    _, runtime = _runtime()
    _nodes(runtime, monkeypatch, a=_FakeNode(0))

    with pytest.raises(RuntimeError, match="no ready node at rank 4"):
        runtime.get(4, "/opt/model.pt")


async def test_ls_quotes_the_path_so_a_shell_cannot_read_a_second_command_in_it(monkeypatch: pytest.MonkeyPatch) -> None:
    runtimes, runtime = _runtime()
    a = _FakeNode(0)
    _nodes(runtime, monkeypatch, a=a)

    await Files(runtimes).ls("cmp", "all", "/opt/my data; rm -rf /")

    assert a._ssh.commands == ["ls -la '/opt/my data; rm -rf /'"]


async def test_rm_removes_recursively(monkeypatch: pytest.MonkeyPatch) -> None:
    runtimes, runtime = _runtime()
    a = _FakeNode(0)
    _nodes(runtime, monkeypatch, a=a)

    await Files(runtimes).rm("cmp", "all", "/opt/checkpoints")

    assert a._ssh.commands == ["rm -rf /opt/checkpoints"]


async def test_a_file_written_to_the_compute_is_listed_and_then_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The round trip, on the one class the endpoints stand on."""
    runtimes, runtime = _runtime()
    a, b = _FakeNode(0), _FakeNode(1)
    _nodes(runtime, monkeypatch, a=a, b=b)
    files = Files(runtimes)

    written = await files.put("cmp", "all", "/opt/train.py", b"print('hi')")
    listed = await files.ls("cmp", 0, "/opt/train.py")
    removed = await files.rm("cmp", "all", "/opt/train.py")

    assert written == (("a", None), ("b", None))
    assert [node for node, _ in listed] == ["a"]
    assert [node for node, _ in removed] == ["a", "b"]
    assert a._ssh.writes == [("/opt/train.py", b"print('hi')")]
    assert a._ssh.commands == ["ls -la /opt/train.py", "rm -rf /opt/train.py"]
    assert b._ssh.commands == ["rm -rf /opt/train.py"]


async def test_files_on_a_compute_this_daemon_is_not_holding_raises() -> None:
    runtimes = Runtimes(_noop, _noop, _noop, _noop)

    with pytest.raises(RuntimeError, match="not live"):
        await Files(runtimes).ls("nobody", "all", "/opt")


class _FakeFiles:
    """The daemon half, faked: every call remembered, a fixed answer given."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, ports.Target | int, str]] = []
        self.written = bytearray()

    def _answer(self, node: str = "nod_0") -> tuple[tuple[str, Result], ...]:
        return ((node, Result(exit_code=0, stdout="total 0", stderr="")),)

    async def ls(self, compute_id: str, target: ports.Target, path: str) -> tuple[tuple[str, Result], ...]:
        self.calls.append(("ls", compute_id, target, path))
        return self._answer()

    async def rm(self, compute_id: str, target: ports.Target, path: str) -> tuple[tuple[str, Result], ...]:
        self.calls.append(("rm", compute_id, target, path))
        return self._answer()

    async def put(self, compute_id: str, target: ports.Target, path: str, content: bytes) -> tuple[tuple[str, str | None], ...]:
        self.calls.append(("put", compute_id, target, path))
        self.written.extend(content)
        return (("nod_0", None), ("nod_1", "permission denied"))

    def get(self, compute_id: str, rank: int, path: str) -> AsyncIterator[bytes]:
        self.calls.append(("get", compute_id, rank, path))
        return self._chunks()

    async def run(self, compute_id: str, target: ports.Target, command: str) -> tuple[tuple[str, Result], ...]:
        self.calls.append(("run", compute_id, target, command))
        return self._answer()

    async def _chunks(self) -> AsyncIterator[bytes]:
        yield b"weights"


@pytest.fixture
async def files() -> _FakeFiles:
    return _FakeFiles()


@pytest.fixture
async def client(files: _FakeFiles) -> AsyncIterator[AsyncTestClient]:
    async with AsyncTestClient(app=create_app(with_real(files=files))) as client:
        yield client


async def test_the_fake_files_satisfies_the_port(files: _FakeFiles) -> None:
    assert isinstance(files, ports.Files)


async def test_listing_defaults_to_rank_zero(client: AsyncTestClient, files: _FakeFiles) -> None:
    response = await client.get("/v1/computes/cmp/files", params={"path": "/opt"})

    assert response.status_code == 200
    assert response.json() == {"nod_0": {"exit_code": 0, "stdout": "total 0", "stderr": ""}}
    assert files.calls == [("ls", "cmp", 0, "/opt")]


async def test_listing_takes_the_rank_it_was_given(client: AsyncTestClient, files: _FakeFiles) -> None:
    await client.get("/v1/computes/cmp/files", params={"path": "/opt", "node": "2"})

    assert files.calls == [("ls", "cmp", 2, "/opt")]


async def test_removing_defaults_to_every_node(client: AsyncTestClient, files: _FakeFiles) -> None:
    response = await client.delete("/v1/computes/cmp/files", params={"path": "/opt/junk"})

    assert response.status_code == 200
    assert files.calls == [("rm", "cmp", "all", "/opt/junk")]


async def test_uploading_carries_the_body_and_defaults_to_every_node(client: AsyncTestClient, files: _FakeFiles) -> None:
    response = await client.put(
        "/v1/computes/cmp/files",
        content=b"print('hi')",
        params={"path": "/opt/train.py"},
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    assert response.json() == {"nod_0": None, "nod_1": "permission denied"}
    assert files.calls == [("put", "cmp", "all", "/opt/train.py")]
    assert bytes(files.written) == b"print('hi')"


async def test_downloading_streams_raw_bytes_from_one_node(client: AsyncTestClient, files: _FakeFiles) -> None:
    response = await client.get("/v1/computes/cmp/files/content", params={"path": "/opt/model.pt", "node": "1"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert response.content == b"weights"
    assert files.calls == [("get", "cmp", 1, "/opt/model.pt")]


async def test_downloading_defaults_to_rank_zero(client: AsyncTestClient, files: _FakeFiles) -> None:
    await client.get("/v1/computes/cmp/files/content", params={"path": "/opt/model.pt"})

    assert files.calls == [("get", "cmp", 0, "/opt/model.pt")]


async def test_downloading_from_every_node_at_once_is_refused(client: AsyncTestClient, files: _FakeFiles) -> None:
    response = await client.get("/v1/computes/cmp/files/content", params={"path": "/opt/model.pt", "node": "all"})

    assert response.status_code == 422
    assert response.json()["code"] == "capability_mismatch"
    assert files.calls == []


async def test_exec_runs_the_command_on_every_node_by_default(client: AsyncTestClient, files: _FakeFiles) -> None:
    response = await client.post("/v1/computes/cmp/exec", params={"command": "nvidia-smi -L"})

    assert response.status_code == 200
    assert response.json() == {"nod_0": {"exit_code": 0, "stdout": "total 0", "stderr": ""}}
    assert files.calls == [("run", "cmp", "all", "nvidia-smi -L")]


async def test_a_node_that_is_neither_all_nor_a_rank_is_refused(client: AsyncTestClient, files: _FakeFiles) -> None:
    response = await client.get("/v1/computes/cmp/files", params={"path": "/opt", "node": "head"})

    assert response.status_code == 422
    assert response.json()["code"] == "capability_mismatch"
    assert files.calls == []


async def test_a_refused_download_leaves_the_destination_alone(tmp_path: Path) -> None:
    """The rank is checked before the file is opened, and ``"wb"`` truncates on open.

    Otherwise ``download --node all`` empties whatever it was pointed at on its
    way to refusing — the caller loses a file to a command that never ran.
    """
    from skyward.cli.compute import download_path

    destination = tmp_path / "weights.pt"
    destination.write_bytes(b"the copy already here")

    with pytest.raises(SystemExit, match="takes a rank"):
        download_path("cmp", "/opt/model.pt", destination, node="all")

    assert destination.read_bytes() == b"the copy already here"
