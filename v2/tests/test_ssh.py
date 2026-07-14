"""The SSH channel, against a real SSH server.

asyncssh is a server as well as a client, so the machine on the other end is an
actual sshd speaking the actual protocol — no mock of a transport, which is the
one thing a mock could never be honest about.
"""

import asyncio
from pathlib import Path

import asyncssh
import pytest

from skyward2.runtime.ssh import SshChannel, SshUnavailableError


class Sshd(asyncssh.SSHServer):
    connections: list[asyncssh.SSHServerConnection] = []

    def connection_made(self, conn: asyncssh.SSHServerConnection) -> None:
        Sshd.connections.append(conn)

    def connection_requested(self, *_: object) -> bool:
        return True


async def shell(process: asyncssh.SSHServerProcess) -> None:
    command = await asyncio.create_subprocess_shell(
        str(process.command or "true"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await command.communicate()
    process.stdout.write(stdout.decode())
    process.stderr.write(stderr.decode())
    process.exit(command.returncode or 0)


@pytest.fixture
def key() -> asyncssh.SSHKey:
    return asyncssh.generate_private_key("ssh-ed25519")


@pytest.fixture
async def sshd(key: asyncssh.SSHKey, tmp_path: Path):
    Sshd.connections = []
    server = await asyncssh.create_server(
        Sshd,
        "127.0.0.1",
        0,
        server_host_keys=[asyncssh.generate_private_key("ssh-ed25519")],
        authorized_client_keys=asyncssh.import_authorized_keys(key.export_public_key().decode()),
        process_factory=shell,
        sftp_factory=lambda channel: asyncssh.SFTPServer(channel, chroot=str(tmp_path)),
    )
    yield next(iter(server.sockets)).getsockname()[1]
    server.close()


@pytest.fixture
async def channel(sshd: int, key: asyncssh.SSHKey):
    channel = SshChannel(
        "127.0.0.1",
        port=sshd,
        user="tester",
        private_key=key.export_private_key().decode(),
        connect_timeout=10.0,
    )
    await channel.connect()
    yield channel
    await channel.close()


async def echo_server() -> tuple[int, asyncio.Server]:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(await reader.read(64))
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return next(iter(server.sockets)).getsockname()[1], server


async def test_a_command_runs_and_reports_how_it_went(channel: SshChannel):
    assert (await channel.run("echo hello")).stdout.strip() == "hello"
    assert (await channel.run("exit 3")).exit_code == 3


async def test_output_arrives_a_line_at_a_time(channel: SshChannel):
    assert [line async for line in channel.stream("printf 'a\\nb\\nc\\n'")] == ["a", "b", "c"]


async def test_a_file_lands_on_the_machine(channel: SshChannel, tmp_path: Path):
    await channel.put("/bootstrap.sh", b"#!/bin/sh\necho hi\n")

    assert (tmp_path / "bootstrap.sh").read_bytes() == b"#!/bin/sh\necho hi\n"


async def test_the_wrong_key_fails_at_once_instead_of_burning_the_deadline(sshd: int):
    """More attempts cannot fix a bad credential.

    The deadline is generous because a machine takes minutes to accept SSH.
    Spending it on a typo is how a five-second failure becomes a five-minute
    mystery, so authentication is the one error that is not retried.
    """
    stranger = SshChannel(
        "127.0.0.1",
        port=sshd,
        user="tester",
        private_key=asyncssh.generate_private_key("ssh-ed25519").export_private_key().decode(),
        connect_timeout=60.0,
    )

    with pytest.raises(SshUnavailableError):
        async with asyncio.timeout(5.0):
            await stranger.connect()


async def test_an_unreachable_machine_is_retried_until_the_deadline():
    closed = SshChannel("127.0.0.1", port=1, connect_timeout=3.0)

    with pytest.raises(SshUnavailableError, match="unreachable"):
        await closed.connect()


async def test_a_dropped_link_is_redialled_and_the_command_that_was_waiting_still_runs(channel: SshChannel):
    Sshd.connections[-1].abort()

    assert (await channel.run("echo survived")).stdout.strip() == "survived"


async def test_a_forward_keeps_its_local_port_across_a_drop(channel: SshChannel):
    """The client on the other end of the tunnel must not have to be told.

    A casty member dialling 127.0.0.1:41000 keeps dialling 127.0.0.1:41000 while
    the link underneath is rebuilt. Handing out a new local port on every
    reconnect would mean rebuilding the cluster client every time a network
    hiccuped.
    """
    port, echo = await echo_server()
    local = await channel.forward(port)

    Sshd.connections[-1].abort()
    await channel.run("true")

    reader, writer = await asyncio.open_connection("127.0.0.1", local)
    writer.write(b"still here")
    await writer.drain()

    assert await reader.read(64) == b"still here"

    writer.close()
    echo.close()
