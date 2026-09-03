"""How the channel treats a machine that answers and says no, and one that hangs up.

A refused key used to be retried like a refused connection, for the whole connect
deadline. On a RunPod pod the key is written before sshd starts, so a refusal
there is final — and the old behaviour spent 240 seconds a machine discovering
nothing, then had the reconciler buy another machine to discover it again.

A link that drops mid-stream used to surface as the exception asyncssh died of,
thrown from inside the iteration. The tail following a node's journal only
expected the channel's own error, so the tail died silently and every phase the
bootstrap reached afterwards went unseen.
"""

import asyncio
from collections.abc import Callable, Coroutine

import asyncssh
import pytest

from skyward.server.application import ssh
from skyward.server.application.ssh import SshChannel, SshUnavailableError

pytestmark = pytest.mark.local


class _Doorman(asyncssh.SSHServer):
    """An SSH server that grants or refuses every public key by one rule."""

    def __init__(self, granted: Callable[[], bool]) -> None:
        self._granted = granted

    def begin_auth(self, username: str) -> bool:
        return True

    def public_key_auth_supported(self) -> bool:
        return True

    def validate_public_key(self, username: str, key: asyncssh.SSHKey) -> bool:
        return self._granted()


def describe_a_machine_that_refuses_the_key() -> None:
    async def it_gives_up_at_the_grace_not_the_connect_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ssh, "AUTH_GRACE", 0.3)
        acceptor = await _serving(granted=lambda: False)
        channel = _channel(acceptor.get_port(), connect_timeout=30.0)

        started = asyncio.get_running_loop().time()
        try:
            with pytest.raises(SshUnavailableError) as refused:
                await channel.connect()
        finally:
            await channel.close()
            acceptor.close()

        assert "does not trust this compute's key" in str(refused.value)
        assert asyncio.get_running_loop().time() - started < 5.0, "a final refusal must not run out the connect deadline"

    async def it_forgives_a_refusal_that_stops(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ssh, "AUTH_GRACE", 5.0)
        refusals = iter((False, False))
        acceptor = await _serving(granted=lambda: next(refusals, True))
        channel = _channel(acceptor.get_port(), connect_timeout=10.0)

        try:
            await channel.connect()
        finally:
            await channel.close()
            acceptor.close()


def describe_a_machine_that_hangs_up_mid_stream() -> None:
    async def it_ends_the_iteration_instead_of_raising() -> None:
        async def hang_up(process: asyncssh.SSHServerProcess[str]) -> None:
            process.stdout.write("one\n")
            await process.stdout.drain()
            process.channel.get_connection().abort()

        acceptor = await _serving(granted=lambda: True, process_factory=hang_up)
        channel = _channel(acceptor.get_port(), connect_timeout=10.0)

        try:
            await channel.connect()
            lines = [line async for line in channel.stream("tail -F journal")]
        finally:
            await channel.close()
            acceptor.close()

        assert lines == ["one"]


async def _serving(
    granted: Callable[[], bool],
    process_factory: Callable[[asyncssh.SSHServerProcess[str]], Coroutine[None, None, None]] | None = None,
) -> asyncssh.SSHAcceptor:
    return await asyncssh.listen(
        "127.0.0.1",
        0,
        server_host_keys=[asyncssh.generate_private_key("ssh-ed25519")],
        server_factory=lambda: _Doorman(granted),
        process_factory=process_factory,
    )


def _channel(port: int, connect_timeout: float) -> SshChannel:
    key = asyncssh.generate_private_key("ssh-ed25519")
    return SshChannel(
        "127.0.0.1",
        port=port,
        user="root",
        private_key=key.export_private_key().decode(),
        connect_timeout=connect_timeout,
        retry_delay=0.05,
    )
