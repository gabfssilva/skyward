"""AsyncSSH-based transport for remote execution.

Service class pattern - dependencies bound at construction,
not passed on every call.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

import asyncssh
from loguru import logger

# =============================================================================
# Stream Event Types (parsed from JSONL)
# =============================================================================


@dataclass(frozen=True, slots=True)
class RawBootstrapConsole:
    """Raw console output from JSONL (without instance info)."""

    content: str
    stream: Literal["stdout", "stderr"] = "stdout"


@dataclass(frozen=True, slots=True)
class RawBootstrapPhase:
    """Raw phase event from JSONL (without instance info)."""

    event: Literal["started", "completed", "failed"]
    phase: str
    elapsed: float | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RawBootstrapCommand:
    """Raw command event from JSONL (without instance info)."""

    command: str


@dataclass(frozen=True, slots=True)
class RawMetricEvent:
    """Raw metric event from JSONL (without instance info)."""

    name: str
    value: float
    ts: float


@dataclass(frozen=True, slots=True)
class RawLogEvent:
    """Raw log event from JSONL (without instance info)."""

    content: str
    stream: Literal["stdout", "stderr"] = "stdout"


type RawStreamEvent = (
    RawBootstrapConsole
    | RawBootstrapPhase
    | RawBootstrapCommand
    | RawMetricEvent
    | RawLogEvent
)


# =============================================================================
# Exceptions
# =============================================================================


class BootstrapError(Exception):
    """Bootstrap failed."""

    def __init__(self, phase: str, error: str) -> None:
        self.phase = phase
        self.error = error
        super().__init__(f"Bootstrap phase '{phase}' failed: {error}")


# =============================================================================
# SSH Transport
# =============================================================================


@dataclass
class SSHTransport:
    """Async SSH transport using asyncssh.

    Service class that holds connection configuration.
    Dependencies (host, user, key) are bound at construction.

    Retry behavior is built-in to connect(). Configure via retry_* params:
    - retry_max_attempts: Max connection attempts (default: 60 = 5 min at 5s intervals)
    - retry_delay: Delay between attempts in seconds (default: 5.0)

    Example:
        >>> transport = SSHTransport(
        ...     host="10.0.0.1",
        ...     user="ubuntu",
        ...     key_path="~/.ssh/id_rsa",
        ... )
        >>> await transport.connect()  # Retries automatically
        >>> code, stdout, stderr = await transport.run("nvidia-smi")
        >>> await transport.close()

    As context manager:
        >>> async with SSHTransport(...) as t:
        ...     code, out, err = await t.run("ls")
    """

    host: str
    user: str
    key_path: str
    port: int = 22
    connect_timeout: float = 30.0
    retry_max_attempts: int = 150  # 5 min at 2s intervals
    retry_delay: float = 2.0

    _conn: asyncssh.SSHClientConnection | None = field(default=None, repr=False)

    async def connect(self) -> None:
        """Establish SSH connection with automatic retry.

        Uses exponential_base=1.0 (fixed delay) for predictable polling.
        Retries on any exception from asyncssh.connect().
        """
        if self._conn is not None:
            return

        from skyward.infra.retry import retry

        @retry(
            max_attempts=self.retry_max_attempts,
            base_delay=self.retry_delay,
            exponential_base=1.0,  # Fixed delay
            jitter=False,
        )
        async def do_connect() -> asyncssh.SSHClientConnection:
            return await asyncssh.connect(
                self.host,
                port=self.port,
                username=self.user,
                client_keys=[self.key_path],
                known_hosts=None,
                connect_timeout=self.connect_timeout,
            )

        self._conn = await do_connect()

    async def close(self) -> None:
        """Close SSH connection."""
        if self._conn is not None:
            self._conn.close()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._conn.wait_closed(), timeout=5.0)
            self._conn = None

    async def __aenter__(self) -> SSHTransport:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    @property
    def is_connected(self) -> bool:
        """Whether connection is established."""
        return self._conn is not None

    def _require_connection(self) -> asyncssh.SSHClientConnection:
        """Get connection or raise."""
        if self._conn is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._conn

    # -------------------------------------------------------------------------
    # Command Execution
    # -------------------------------------------------------------------------

    async def run(
        self,
        *command: str,
        timeout: float | None = None,
        check: bool = False,
    ) -> tuple[int, str, str]:
        """Execute command and return (exit_code, stdout, stderr).

        Args:
            command: Command and arguments.
            timeout: Execution timeout in seconds.
            check: If True, raise on non-zero exit code.

        Returns:
            Tuple of (exit_code, stdout, stderr).

        Raises:
            RuntimeError: If check=True and command fails.
        """
        conn = self._require_connection()
        cmd = " ".join(command)

        result = await conn.run(cmd, timeout=timeout, check=False)

        code = result.exit_status or 0
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        if check and code != 0:
            raise RuntimeError(f"Command failed ({code}): {stderr}")

        return code, str(stdout), str(stderr)

    async def run_stream(
        self,
        *command: str,
    ) -> AsyncIterator[tuple[str, str]]:
        """Execute command and stream output lines.

        Yields:
            Tuples of ("stdout" | "stderr", line).
        """
        conn = self._require_connection()
        cmd = " ".join(command)

        async with conn.create_process(cmd) as proc:
            # Read stdout and stderr concurrently
            async def read_stream(
                stream: asyncssh.SSHReader[str],
                name: str,
            ) -> AsyncIterator[tuple[str, str]]:
                async for line in stream:
                    yield (name, line.rstrip("\n"))

            stdout_iter = read_stream(proc.stdout, "stdout")
            stderr_iter = read_stream(proc.stderr, "stderr")

            # Simple alternating read
            async for item in self._merge_streams(stdout_iter, stderr_iter):
                yield item

    async def _merge_streams(
        self,
        *iterators: AsyncIterator[tuple[str, str]],
    ) -> AsyncIterator[tuple[str, str]]:
        """Merge multiple async iterators."""
        import asyncio

        pending: set[asyncio.Task[tuple[str, str] | None]] = set()
        iters = {id(it): it.__aiter__() for it in iterators}

        async def get_next(
            it_id: int, it: AsyncIterator[tuple[str, str]],
        ) -> tuple[str, str] | None:
            try:
                return await it.__anext__()
            except StopAsyncIteration:
                iters.pop(it_id, None)
                return None

        # Start initial fetches
        for it_id, it in iters.items():
            task = asyncio.create_task(get_next(it_id, it))
            task.it_id = it_id  # type: ignore[attr-defined]
            pending.add(task)

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                result = task.result()
                if result is not None:
                    yield result

                    # Schedule next from same iterator
                    it_id = task.it_id  # type: ignore[attr-defined]
                    if it_id in iters:
                        new_task = asyncio.create_task(get_next(it_id, iters[it_id]))
                        new_task.it_id = it_id  # type: ignore[attr-defined]
                        pending.add(new_task)

    # -------------------------------------------------------------------------
    # File Transfer
    # -------------------------------------------------------------------------

    async def upload(self, local: str, remote: str) -> None:
        """Upload file to remote host.

        Args:
            local: Local file path.
            remote: Remote destination path.
        """
        conn = self._require_connection()
        await asyncssh.scp(local, (conn, remote))

    async def download(self, remote: str, local: str) -> None:
        """Download file from remote host.

        Args:
            remote: Remote file path.
            local: Local destination path.
        """
        conn = self._require_connection()
        await asyncssh.scp((conn, remote), local)

    async def read_file(self, remote: str) -> str:
        """Read remote file contents."""
        _, stdout, _ = await self.run("cat", remote)
        return stdout

    async def write_file(self, remote: str, content: str) -> None:
        """Write content to remote file using SFTP."""
        conn = self._require_connection()

        async with conn.start_sftp_client() as sftp, sftp.open(remote, "w") as f:
            await f.write(content)

    async def write_bytes(self, remote: str, content: bytes) -> None:
        """Write binary content to remote file using SFTP."""
        conn = self._require_connection()

        async with conn.start_sftp_client() as sftp, sftp.open(remote, "wb") as f:
            await f.write(content)

    async def file_exists(self, remote: str) -> bool:
        """Check if remote file exists."""
        code, _, _ = await self.run("test", "-f", remote)
        return code == 0

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    async def wait_for_file(
        self,
        remote: str,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> bool:
        """Wait for remote file to exist.

        Returns:
            True if file appeared, False if timeout.
        """
        import asyncio

        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            if await self.file_exists(remote):
                return True
            await asyncio.sleep(poll_interval)
        return False

    async def tail_file(
        self,
        remote: str,
        follow: bool = True,
    ) -> AsyncIterator[str]:
        """Stream lines from remote file.

        Args:
            remote: Remote file path.
            follow: If True, use tail -f (stream indefinitely).

        Yields:
            Lines from the file.
        """
        cmd = ["tail", "-f", remote] if follow else ["tail", remote]
        async for stream, line in self.run_stream(*cmd):
            if stream == "stdout":
                yield line

    # -------------------------------------------------------------------------
    # Events Streaming (JSONL)
    # -------------------------------------------------------------------------

    async def stream_events(
        self,
        log_path: str = "/opt/skyward/events.jsonl",
        timeout: float = 600.0,
        wait_for_file_timeout: float = 120.0,
    ) -> AsyncIterator[RawStreamEvent]:
        """Stream events from JSONL log file via tail -F.

        Async version of v1's stream_events(). Uses tail -F to follow
        file rotation. Yields parsed events as they are emitted.

        The timeout is adaptive: it resets on each event received.
        This means long-running operations that produce output won't timeout.

        Args:
            log_path: Path to events.jsonl on remote.
            timeout: Maximum time between events before timeout.
            wait_for_file_timeout: Max time to wait for log file to exist.

        Yields:
            Parsed stream events (console, phase, command, metric, log).

        Raises:
            BootstrapError: If a phase fails.
            TimeoutError: If no events received within timeout.
            FileNotFoundError: If log file doesn't appear in time.
        """
        # Wait for log file to exist
        await self._wait_for_log_file(log_path, wait_for_file_timeout)

        # Start tail -F (capital F follows rotation)
        conn = self._require_connection()
        logger.debug("Starting tail -F {path}", path=log_path)

        async with conn.create_process(f"tail -F {log_path}") as proc:
            deadline = asyncio.get_event_loop().time() + timeout
            buffer = ""

            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise TimeoutError(f"Events stream timeout after {timeout}s of inactivity")

                try:
                    # Read with timeout
                    chunk = await asyncio.wait_for(
                        proc.stdout.read(4096),
                        timeout=min(1.0, remaining),
                    )

                    if not chunk:
                        # EOF - process ended
                        break

                    buffer += chunk

                    # Process complete lines
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue

                        event = _parse_jsonl_line(line)
                        if event is None:
                            continue

                        yield event

                        # Refresh deadline on event received
                        deadline = asyncio.get_event_loop().time() + timeout

                        # Check for failure
                        match event:
                            case RawBootstrapPhase(event="failed", phase=phase, error=error):
                                raise BootstrapError(phase, error or "unknown")

                except TimeoutError:
                    # Check if deadline exceeded
                    if asyncio.get_event_loop().time() >= deadline:
                        raise TimeoutError(
                            f"Events stream timeout after {timeout}s of inactivity"
                        ) from None
                    # Otherwise, just a read timeout - continue loop

    async def stream_bootstrap(
        self,
        log_path: str = "/opt/skyward/events.jsonl",
        timeout: float = 600.0,
    ) -> AsyncIterator[RawStreamEvent]:
        """Stream bootstrap events until completion.

        Convenience wrapper around stream_events() that stops when
        bootstrap phase completes successfully.

        Args:
            log_path: Path to events.jsonl on remote.
            timeout: Maximum time between events.

        Yields:
            Bootstrap events until completion.

        Raises:
            BootstrapError: If bootstrap fails.
            TimeoutError: If timeout exceeded.
        """
        async for event in self.stream_events(log_path, timeout):
            yield event

            # Stop on bootstrap completion
            match event:
                case RawBootstrapPhase(phase="bootstrap", event="completed"):
                    return

    async def _wait_for_log_file(self, log_path: str, timeout: float) -> None:
        """Wait for log file to exist on remote."""
        start = asyncio.get_event_loop().time()
        check_count = 0

        logger.debug("Waiting for log file {path}", path=log_path)

        while asyncio.get_event_loop().time() - start < timeout:
            check_count += 1
            if await self.file_exists(log_path):
                elapsed = asyncio.get_event_loop().time() - start
                logger.debug(
                    "Log file found after {elapsed:.1f}s ({checks} checks)",
                    elapsed=elapsed, checks=check_count,
                )
                return
            await asyncio.sleep(1.0)

        raise FileNotFoundError(f"Log file {log_path} not created within {timeout}s")


# =============================================================================
# JSONL Parsing
# =============================================================================


def _parse_jsonl_line(line: str) -> RawStreamEvent | None:
    """Parse a single JSONL line from events log.

    Args:
        line: Raw line from events.jsonl.

    Returns:
        Parsed event or None if line is invalid.
    """
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    match data.get("type"):
        case "console":
            return RawBootstrapConsole(
                content=data.get("content", ""),
                stream=data.get("stream", "stdout"),
            )
        case "phase":
            return RawBootstrapPhase(
                event=data.get("event", "started"),
                phase=data.get("phase", ""),
                elapsed=data.get("elapsed"),
                error=data.get("error"),
            )
        case "command":
            return RawBootstrapCommand(command=data.get("command", ""))
        case "metric":
            value = data.get("value")
            if value is None:
                return None
            try:
                return RawMetricEvent(
                    name=data.get("name", ""),
                    value=float(value),
                    ts=data.get("ts", 0.0),
                )
            except (ValueError, TypeError):
                return None
        case "log":
            return RawLogEvent(
                content=data.get("content", ""),
                stream=data.get("stream", "stdout"),
            )
        case _:
            return None
