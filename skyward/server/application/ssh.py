from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol, runtime_checkable

import asyncssh
from msgspec import Struct

from skyward.shared.observability import logger

logger = logger.bind(component="ssh")

RETRY_DELAY = 2.0
CONNECT_TIMEOUT = 10.0
CHUNK = 65536

AUTH_GRACE = 60.0
"""Seconds a machine may keep refusing the key before the refusal is believed.

A machine seconds old may answer SSH before the key it will eventually accept has
been written — cloud-init runs after sshd — so a refusal is retried. But one that
is still refusing after this long is not booting: it does not trust the key it was
given, no retry will change that, and the connect deadline would only bill the
compute for discovering nothing."""

type Channel = tuple[asyncssh.SSHReader[bytes], asyncssh.SSHWriter[bytes]]
"""One direct-tcpip connection to a node's port: bytes in, bytes out."""


class Result(Struct, frozen=True):
    """What one command said on one machine: its exit code and both its streams."""

    exit_code: int
    stdout: str
    stderr: str


class SshUnavailableError(RuntimeError):
    """The channel is closed, or gave up reconnecting. Nothing more will run on it."""


@runtime_checkable
class Ssh(Protocol):
    """A machine you can reach.

    The whole of what the node's capabilities are written against: bootstrap,
    launching the worker and tailing its events all take one of these and nothing
    else. It is what lets each of them be tested without a machine.
    """

    async def run(self, command: str, *, timeout: float | None = None) -> Result: ...

    async def put(self, path: str, content: bytes) -> None: ...

    def stream(self, command: str) -> AsyncIterator[str]: ...

    async def forward(self, remote_port: int, remote_host: str = "127.0.0.1") -> int: ...

    async def close(self) -> None: ...


class SshChannel:
    """One SSH connection, held open, that reconnects on its own.

    A dropped link is not a dead node. The connection is redialled underneath,
    the forwards are re-established **on the same local ports** — so a client
    dialling ``127.0.0.1:41000`` keeps dialling it across the outage — and calls
    made while the link is down simply wait rather than fail. Only when
    reconnection is exhausted does the channel go permanently unavailable, and
    only then does the node above learn that it lost a machine.

    A refused key is retried, briefly. On a machine seconds old the refusal is the
    likelier transient: sshd answers before cloud-init has written the key it will
    eventually accept, and a fast dial lands in that window every time. But a
    machine still refusing after :data:`AUTH_GRACE` of consecutive refusals is not
    booting — it does not trust the key it was handed — and it fails then, wearing
    the refusal as the reason, rather than spending the connect deadline retrying
    an answer that is final.
    """

    def __init__(
        self,
        host: str,
        port: int = 22,
        user: str = "root",
        private_key: str | None = None,
        password: str | None = None,
        *,
        connect_timeout: float = 300.0,
        reconnect_attempts: int = 30,
        retry_delay: float = RETRY_DELAY,
    ) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._private_key = private_key
        self._password = password
        self._connect_timeout = connect_timeout
        self._reconnect_attempts = reconnect_attempts
        self._retry_delay = retry_delay

        self._client_keys: list[asyncssh.SSHKey] | None = None
        self._conn: asyncssh.SSHClientConnection | None = None
        self._up = asyncio.Event()
        self._closed = False
        self._failure: str | None = None
        self._forwards: dict[tuple[str, int], int] = {}
        self._listeners: list[asyncssh.SSHListener] = []
        self._watcher: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        """Dial until the machine answers, or until a deadline says it never will.

        Two clocks. A machine that cannot be reached at all gets the whole connect
        timeout — it may still be booting. A machine that answers and refuses the
        key gets :data:`AUTH_GRACE` of consecutive refusals, and a refusal that
        outlasts it is terminal: the machine is up, it heard the key, and it said
        no. An error of any other kind resets that clock — a reboot mid-boot drops
        the connection, and the machine that comes back deserves the grace again.
        """
        last = ""
        refused: float | None = None
        clock = asyncio.get_running_loop().time
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(self._connect_timeout):
                while not self._closed:
                    try:
                        self._attach(await self._dial())
                        return
                    except asyncssh.PermissionDenied as exc:
                        last = str(exc)
                        refused = refused if refused is not None else clock()
                        if clock() - refused >= AUTH_GRACE:
                            self._fail(last)
                            raise SshUnavailableError(
                                f"{self._host}: the machine is up but has refused the key for {self._user} "
                                f"for {AUTH_GRACE:.0f}s — it does not trust this compute's key: {last}"
                            ) from exc
                        await asyncio.sleep(self._retry_delay)
                    except (OSError, asyncssh.Error) as exc:
                        last = str(exc)
                        refused = None
                        await asyncio.sleep(self._retry_delay)

        self._fail(last)
        raise SshUnavailableError(f"{self._host}: unreachable after {self._connect_timeout}s: {last}")

    async def run(self, command: str, *, timeout: float | None = None) -> Result:
        async def execute(conn: asyncssh.SSHClientConnection) -> Result:
            result = await conn.run(command, timeout=timeout, check=False)
            return Result(
                exit_code=result.exit_status or 0,
                stdout=str(result.stdout or ""),
                stderr=str(result.stderr or ""),
            )

        return await self._attempt(execute)

    async def put(self, path: str, content: bytes) -> None:
        async def write(conn: asyncssh.SSHClientConnection) -> None:
            async with conn.start_sftp_client() as sftp, sftp.open(path, "wb") as remote:
                await remote.write(content)

        await self._attempt(write)

    async def get(self, path: str) -> AsyncIterator[bytes]:
        """Read a file off the machine, a chunk at a time.

        Outside :meth:`_attempt`, which reruns an operation on the connection that
        replaces a dead one: rerunning this one would hand the caller the start of
        the file a second time, half way through writing it. A download that dies
        on a reconnect dies — the same contract :meth:`stream` has.
        """
        conn = await self._ready()
        async with conn.start_sftp_client() as sftp, sftp.open(path, "rb") as remote:
            while chunk := await remote.read(CHUNK):
                yield chunk

    async def _attempt[T](self, operation: Callable[[asyncssh.SSHClientConnection], Awaitable[T]]) -> T:
        """Run an operation, and run it again on the connection that replaces a dead one.

        A link drops before the watcher has noticed, and the operation is handed
        a corpse. Failing here would make every transient hiccup the caller's
        problem, when the channel is about to heal itself anyway — so the
        operation waits for the new connection and runs on that instead. It gives
        up only when the channel does.
        """
        while True:
            conn = await self._ready()
            try:
                return await operation(conn)
            except (asyncssh.ChannelOpenError, asyncssh.ConnectionLost, ConnectionResetError, BrokenPipeError):
                await self._replaced(conn)

    async def _replaced(self, stale: asyncssh.SSHClientConnection) -> None:
        while self._conn is stale:
            if self._closed or self._failure:
                raise SshUnavailableError(f"{self._host}: {self._failure or 'channel closed'}")
            await asyncio.sleep(0.05)

    async def stream(self, command: str) -> AsyncIterator[str]:
        """Run a command and yield its output a line at a time.

        Ends when the command ends, or when the link drops. Resuming across a
        reconnect is the caller's business, because only the caller knows where
        it got to.
        """
        conn = await self._ready()
        async with conn.create_process(command) as process:
            while line := await process.stdout.readline():
                yield line.rstrip("\n")

    async def forward(self, remote_port: int, remote_host: str = "127.0.0.1") -> int:
        """Open a local port onto a port on the machine, and keep it open."""
        conn = await self._ready()
        local_port = await self._listen(conn, remote_host, remote_port)
        self._forwards[remote_host, remote_port] = local_port
        return local_port

    async def open_connection(self, remote_host: str, remote_port: int) -> Channel:
        """Open one direct-tcpip channel to a port on the machine.

        Distinct from :meth:`forward`, which stands a local listener up for the
        life of the channel and re-establishes it across a reconnect: this is a
        single connection, opened on demand and torn down with the reader and
        writer it returns. One accepted socket, one channel — which is what a
        proxied TCP connection is made of, and why it is not tracked to survive a
        drop the way a forward is.
        """
        conn = await self._ready()
        return await conn.open_connection(remote_host, remote_port)

    async def open_shell(
        self,
        command: str | None = None,
        term: str = "xterm-256color",
        size: tuple[int, int] = (80, 24),
    ) -> Channel:
        """Open one interactive session on the machine, behind a pseudo-terminal.

        The PTY is what makes it a terminal rather than a pipe: it is what a login
        shell reads its job control from, what a REPL asks before it prints a prompt,
        and what ``ncurses`` sizes itself against — none of which a plain ``exec``
        channel offers. ``command`` omitted is the user's login shell.

        Reader and writer are the same pair :meth:`open_connection` hands back, so
        whatever pumps bytes through a forwarded socket pumps them through this. The
        error stream is folded into the output, because a terminal has one.
        """
        conn = await self._ready()
        columns, rows = size
        process = await conn.create_process(
            command,
            term_type=term,
            term_size=(columns, rows),
            encoding=None,
            stderr=asyncssh.STDOUT,
        )
        return process.stdout, process.stdin

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._watcher:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher

        self._drop_listeners()
        if self._conn:
            self._conn.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._conn.wait_closed(), timeout=5.0)
        self._up.set()

    async def _dial(self) -> asyncssh.SSHClientConnection:
        return await asyncssh.connect(
            self._host,
            port=self._port,
            username=self._user,
            client_keys=[asyncssh.import_private_key(self._private_key)] if self._private_key else None,
            password=self._password,
            known_hosts=None,
            connect_timeout=CONNECT_TIMEOUT,
            keepalive_interval=10,
            keepalive_count_max=3,
        )

    async def _ready(self) -> asyncssh.SSHClientConnection:
        await self._up.wait()
        if self._closed:
            raise SshUnavailableError(f"{self._host}: channel closed")
        if self._failure:
            raise SshUnavailableError(f"{self._host}: {self._failure}")
        assert self._conn is not None
        return self._conn

    def _attach(self, conn: asyncssh.SSHClientConnection) -> None:
        self._conn = conn
        self._up.set()
        self._watcher = asyncio.create_task(self._watch(conn))

    def _fail(self, reason: str) -> None:
        self._failure = reason
        self._up.set()

    async def _watch(self, conn: asyncssh.SSHClientConnection) -> None:
        await conn.wait_closed()
        if self._closed:
            return

        self._up.clear()
        self._drop_listeners()
        logger.warning("ssh: {} dropped, reconnecting", self._host)

        for _ in range(self._reconnect_attempts):
            await asyncio.sleep(self._retry_delay)
            if self._closed:
                return
            try:
                fresh = await self._dial()
                for (remote_host, remote_port), local_port in self._forwards.items():
                    await self._listen(fresh, remote_host, remote_port, local_port)
            except (OSError, asyncssh.Error):
                continue
            logger.info("ssh: {} back", self._host)
            self._attach(fresh)
            return

        self._fail("reconnection exhausted")

    async def _listen(
        self,
        conn: asyncssh.SSHClientConnection,
        remote_host: str,
        remote_port: int,
        local_port: int = 0,
    ) -> int:
        listener = await conn.forward_local_port("127.0.0.1", local_port, remote_host, remote_port)
        self._listeners.append(listener)
        return listener.get_port()

    def _drop_listeners(self) -> None:
        for listener in self._listeners:
            with contextlib.suppress(Exception):
                listener.close()
        self._listeners = []
