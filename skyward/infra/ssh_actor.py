"""SSH transport as a Casty actor.

The transport actor tells this story:
  connecting → connected → reconnecting → connected → ... → failed → stopped

Fully encapsulates asyncssh — no raw connection leaks. Serves requests
(RunCommand, WriteFile, ForwardPort, etc.) via messages and pushes events
to subscribers.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.infra.ssh import RawStreamEvent
from skyward.observability.logger import logger

if TYPE_CHECKING:
    pass


# ── Requests ──────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class RunCommand:
    command: tuple[str, ...]
    timeout: float | None = None
    check: bool = False
    reply_to: ActorRef[CommandResult] = field(default=None)  # type: ignore[assignment]

@dataclass(frozen=True, slots=True)
class WriteFile:
    remote: str
    content: str
    reply_to: ActorRef[WriteResult] = field(default=None)  # type: ignore[assignment]

@dataclass(frozen=True, slots=True)
class WriteBytes:
    remote: str
    content: bytes
    reply_to: ActorRef[WriteResult] = field(default=None)  # type: ignore[assignment]

@dataclass(frozen=True, slots=True)
class Upload:
    local: str
    remote: str
    reply_to: ActorRef[UploadResult] = field(default=None)  # type: ignore[assignment]

@dataclass(frozen=True, slots=True)
class ForwardPort:
    remote_host: str
    remote_port: int
    reply_to: ActorRef[PortForwarded] = field(default=None)  # type: ignore[assignment]

@dataclass(frozen=True, slots=True)
class SubscribeEvents:
    start_line: int
    subscriber: ActorRef[StreamEvent]

@dataclass(frozen=True, slots=True)
class StopTransport:
    pass


# ── Replies ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str

@dataclass(frozen=True, slots=True)
class WriteResult:
    success: bool
    error: str | None = None

@dataclass(frozen=True, slots=True)
class UploadResult:
    success: bool
    error: str | None = None

@dataclass(frozen=True, slots=True)
class PortForwarded:
    local_port: int


# ── Events (pushed to parent/subscribers) ─────────────────────────────

@dataclass(frozen=True, slots=True)
class StreamEvent:
    lines_read: int
    event: RawStreamEvent

@dataclass(frozen=True, slots=True)
class ConnectionLost:
    error: str

@dataclass(frozen=True, slots=True)
class ConnectionRestored:
    local_port: int = 0

@dataclass(frozen=True, slots=True)
class ConnectionFailed:
    error: str

@dataclass(frozen=True, slots=True)
class PortReForwarded:
    old_port: int
    new_port: int


# ── Internal ──────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class _Connected:
    conn: Any  # asyncssh.SSHClientConnection

@dataclass(frozen=True, slots=True)
class _ConnectFailed:
    error: str

@dataclass(frozen=True, slots=True)
class _StreamedLine:
    lines_read: int
    event: RawStreamEvent

@dataclass(frozen=True, slots=True)
class _StreamEnded:
    error: str | None = None

@dataclass(frozen=True, slots=True)
class _ConnectionDropped:
    error: str

@dataclass(frozen=True, slots=True)
class _CommandDone:
    exit_code: int
    stdout: str
    stderr: str
    reply_to: ActorRef[CommandResult]

@dataclass(frozen=True, slots=True)
class _CommandError:
    error: str
    reply_to: ActorRef[CommandResult]

@dataclass(frozen=True, slots=True)
class _WriteDone:
    reply_to: ActorRef[WriteResult]

@dataclass(frozen=True, slots=True)
class _WriteError:
    error: str
    reply_to: ActorRef[WriteResult]

@dataclass(frozen=True, slots=True)
class _UploadDone:
    reply_to: ActorRef[UploadResult]

@dataclass(frozen=True, slots=True)
class _UploadError:
    error: str
    reply_to: ActorRef[UploadResult]

@dataclass(frozen=True, slots=True)
class _ForwardDone:
    local_port: int
    listener: Any  # asyncssh.SSHListener
    request: ForwardPort

@dataclass(frozen=True, slots=True)
class _ForwardError:
    error: str
    reply_to: ActorRef[PortForwarded]


# ── State ─────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class TransportState:
    conn: Any  # asyncssh.SSHClientConnection
    listeners: tuple[Any, ...] = ()  # asyncssh.SSHListener
    forward_requests: tuple[ForwardPort, ...] = ()
    subscribers: tuple[ActorRef[StreamEvent], ...] = ()
    stream_offset: int = 0


# ── Message union ─────────────────────────────────────────────────────

type TransportMsg = (
    RunCommand | WriteFile | WriteBytes | Upload
    | ForwardPort | SubscribeEvents | StopTransport
    | _Connected | _ConnectFailed
    | _StreamedLine | _StreamEnded | _ConnectionDropped
    | _CommandDone | _CommandError
    | _WriteDone | _WriteError
    | _UploadDone | _UploadError
    | _ForwardDone | _ForwardError
)


# ── Actor ─────────────────────────────────────────────────────────────

log = logger.bind(component="ssh_actor")


async def _default_connect(
    host: str, port: int, user: str, key_path: str, connect_timeout: float,
) -> Any:
    import asyncssh
    return await asyncssh.connect(
        host, port=port, username=user,
        client_keys=[key_path], known_hosts=None,
        connect_timeout=connect_timeout,
    )


def ssh_transport(
    host: str,
    user: str,
    key_path: str,
    port: int = 22,
    retry_max_attempts: int = 150,
    retry_delay: float = 2.0,
    connect_timeout: float = 30.0,
    parent: ActorRef | None = None,
    connect_fn: Callable[[], Awaitable[Any]] | None = None,
) -> Behavior[TransportMsg]:
    """SSH transport actor: connecting → connected → reconnecting → ... → stopped."""

    _connect = connect_fn or (
        lambda: _default_connect(host, port, user, key_path, connect_timeout)
    )

    # ── connecting ────────────────────────────────────────────────────

    def connecting(
        attempts: int = 0,
        pending: tuple[TransportMsg, ...] = (),
    ) -> Behavior[TransportMsg]:

        async def _try_connect() -> _Connected:
            conn = await _connect()
            return _Connected(conn=conn)

        async def setup(ctx: ActorContext[TransportMsg]) -> Behavior[TransportMsg]:
            ctx.pipe_to_self(
                _try_connect(),
                on_failure=lambda e: _ConnectFailed(error=str(e)),
            )
            return _connecting_receive(attempts, pending)

        return Behaviors.setup(setup)

    def _connecting_receive(
        attempts: int,
        pending: tuple[TransportMsg, ...],
    ) -> Behavior[TransportMsg]:
        async def receive(ctx: ActorContext[TransportMsg], msg: TransportMsg) -> Behavior[TransportMsg]:
            match msg:
                case _Connected(conn=conn):
                    log.info("SSH connected to {host}:{port}", host=host, port=port)
                    state = TransportState(conn=conn)
                    return _enter_connected(ctx, state, pending)

                case _ConnectFailed(error=error):
                    new_attempts = attempts + 1
                    if new_attempts >= retry_max_attempts:
                        log.error(
                            "SSH connection permanently failed after {n} attempts: {err}",
                            n=new_attempts, err=error,
                        )
                        if parent:
                            parent.tell(ConnectionFailed(error=error))
                        return Behaviors.stopped()

                    log.debug(
                        "SSH connect attempt {n} failed: {err}, retrying in {d}s",
                        n=new_attempts, err=error, d=retry_delay,
                    )

                    async def _retry() -> _Connected:
                        await asyncio.sleep(retry_delay)
                        conn = await _connect()
                        return _Connected(conn=conn)

                    ctx.pipe_to_self(
                        _retry(),
                        on_failure=lambda e: _ConnectFailed(error=str(e)),
                    )
                    return _connecting_receive(new_attempts, pending)

                case StopTransport():
                    return Behaviors.stopped()

                case _:
                    return _connecting_receive(attempts, (*pending, msg))

        return Behaviors.receive(receive)

    # ── enter connected ───────────────────────────────────────────────

    def _enter_connected(
        ctx: ActorContext[TransportMsg],
        state: TransportState,
        pending: tuple[TransportMsg, ...],
    ) -> Behavior[TransportMsg]:
        for msg in pending:
            ctx.self.tell(msg)
        return connected(state)

    # ── connected ─────────────────────────────────────────────────────

    def connected(state: TransportState) -> Behavior[TransportMsg]:
        async def receive(ctx: ActorContext[TransportMsg], msg: TransportMsg) -> Behavior[TransportMsg]:
            match msg:
                case RunCommand(command=cmd, timeout=timeout, check=chk, reply_to=rt):
                    async def _run_with_drop_detection(
                        _conn: Any = state.conn, _cmd: tuple[str, ...] = cmd,
                        _timeout: float | None = timeout, _chk: bool = chk,
                        _rt: ActorRef[CommandResult] = rt,
                    ) -> _CommandDone | _CommandError:
                        try:
                            code, out, err = await _do_run(_conn, _cmd, _timeout, _chk)
                            return _CommandDone(exit_code=code, stdout=out, stderr=err, reply_to=_rt)
                        except OSError as e:
                            _rt.tell(CommandResult(exit_code=-1, stdout="", stderr=str(e)))
                            ctx.self.tell(_ConnectionDropped(error=str(e)))
                            return _CommandError(error=str(e), reply_to=_rt)
                        except Exception as e:
                            return _CommandError(error=str(e), reply_to=_rt)

                    ctx.pipe_to_self(_run_with_drop_detection())
                    return Behaviors.same()

                case _CommandDone(exit_code=code, stdout=out, stderr=err, reply_to=rt):
                    rt.tell(CommandResult(exit_code=code, stdout=out, stderr=err))
                    return Behaviors.same()

                case _CommandError(error=_, reply_to=_):
                    return Behaviors.same()

                case _ConnectionDropped(error=error):
                    log.warning("Connection dropped: {err}", err=error)
                    if parent:
                        parent.tell(ConnectionLost(error=error))
                    return reconnecting(state)

                case WriteFile(remote=remote, content=content, reply_to=rt):
                    ctx.pipe_to_self(
                        _do_write_file(state.conn, remote, content),
                        mapper=lambda _: _WriteDone(reply_to=rt),
                        on_failure=lambda e: _WriteError(error=str(e), reply_to=rt),
                    )
                    return Behaviors.same()

                case WriteBytes(remote=remote, content=content, reply_to=rt):
                    ctx.pipe_to_self(
                        _do_write_bytes(state.conn, remote, content),
                        mapper=lambda _: _WriteDone(reply_to=rt),
                        on_failure=lambda e: _WriteError(error=str(e), reply_to=rt),
                    )
                    return Behaviors.same()

                case Upload(local=local, remote=remote, reply_to=rt):
                    ctx.pipe_to_self(
                        _do_upload(state.conn, local, remote),
                        mapper=lambda _: _UploadDone(reply_to=rt),
                        on_failure=lambda e: _UploadError(error=str(e), reply_to=rt),
                    )
                    return Behaviors.same()

                case _WriteDone(reply_to=rt):
                    rt.tell(WriteResult(success=True))
                    return Behaviors.same()

                case _WriteError(error=error, reply_to=rt):
                    rt.tell(WriteResult(success=False, error=error))
                    return Behaviors.same()

                case _UploadDone(reply_to=rt):
                    rt.tell(UploadResult(success=True))
                    return Behaviors.same()

                case _UploadError(error=error, reply_to=rt):
                    rt.tell(UploadResult(success=False, error=error))
                    return Behaviors.same()

                case ForwardPort(remote_host=rh, remote_port=rp, reply_to=rt):
                    ctx.pipe_to_self(
                        _do_forward(state.conn, rh, rp),
                        mapper=lambda result: _ForwardDone(
                            local_port=result[0], listener=result[1], request=msg,
                        ),
                        on_failure=lambda e: _ForwardError(error=str(e), reply_to=rt),
                    )
                    return Behaviors.same()

                case _ForwardDone(local_port=lp, listener=lis, request=req):
                    req.reply_to.tell(PortForwarded(local_port=lp))
                    new_state = replace(
                        state,
                        listeners=(*state.listeners, lis),
                        forward_requests=(*state.forward_requests, req),
                    )
                    return connected(new_state)

                case _ForwardError(error=error, reply_to=rt):
                    rt.tell(PortForwarded(local_port=-1))
                    return Behaviors.same()

                case SubscribeEvents(start_line=start, subscriber=sub):
                    new_state = replace(
                        state,
                        subscribers=(*state.subscribers, sub),
                    )
                    _start_stream(ctx, new_state, start)
                    return connected(new_state)

                case _StreamedLine(lines_read=lr, event=ev):
                    for sub in state.subscribers:
                        sub.tell(StreamEvent(lines_read=lr, event=ev))
                    return connected(replace(state, stream_offset=lr))

                case _StreamEnded(error=error):
                    if error and state.conn.is_closed():
                        log.warning("Connection lost (detected via stream): {err}", err=error)
                        if parent:
                            parent.tell(ConnectionLost(error=error))
                        return reconnecting(state)
                    if error:
                        log.warning("Event stream ended: {err}", err=error)
                    if state.subscribers and not state.conn.is_closed():
                        _start_stream(ctx, state, state.stream_offset)
                    return Behaviors.same()

                case StopTransport():
                    await _cleanup(state)
                    return Behaviors.stopped()

            return Behaviors.same()

        return Behaviors.receive(receive)

    # ── reconnecting ──────────────────────────────────────────────────

    def reconnecting(
        state: TransportState,
        attempts: int = 0,
        pending: tuple[TransportMsg, ...] = (),
    ) -> Behavior[TransportMsg]:
        if attempts >= retry_max_attempts:
            log.error("Reconnection permanently failed after {n} attempts", n=attempts)
            if parent:
                parent.tell(ConnectionFailed(error="max reconnection attempts"))
            return Behaviors.stopped()

        async def setup(ctx: ActorContext[TransportMsg]) -> Behavior[TransportMsg]:
            delay = retry_delay
            log.info("Reconnecting in {d}s (attempt {n})", d=delay, n=attempts + 1)

            async def _retry() -> _Connected:
                await asyncio.sleep(delay)
                conn = await _connect()
                return _Connected(conn=conn)

            ctx.pipe_to_self(
                _retry(),
                on_failure=lambda e: _ConnectFailed(error=str(e)),
            )
            return _reconnect_receive(state, attempts, pending)

        return Behaviors.setup(setup)

    def _reconnect_receive(
        state: TransportState,
        attempts: int,
        pending: tuple[TransportMsg, ...],
    ) -> Behavior[TransportMsg]:
        async def receive(ctx: ActorContext[TransportMsg], msg: TransportMsg) -> Behavior[TransportMsg]:
            match msg:
                case _Connected(conn=conn):
                    log.info("Reconnected to {host}:{port}", host=host, port=port)
                    new_state = TransportState(
                        conn=conn,
                        subscribers=state.subscribers,
                        stream_offset=state.stream_offset,
                    )
                    reforwarded = await _reforward_ports(conn, state.forward_requests)
                    new_state = replace(
                        new_state,
                        listeners=tuple(lis for _, lis in reforwarded),
                        forward_requests=state.forward_requests,
                    )
                    if parent:
                        new_port = reforwarded[0][0] if reforwarded else 0
                        parent.tell(ConnectionRestored(local_port=new_port))
                        for i, (rp, _) in enumerate(reforwarded):
                            old_port = state.listeners[i].get_port() if i < len(state.listeners) else -1
                            if old_port != rp:
                                parent.tell(PortReForwarded(old_port=old_port, new_port=rp))
                    if new_state.subscribers:
                        _start_stream(ctx, new_state, new_state.stream_offset)
                    return _enter_connected(ctx, new_state, pending)

                case _ConnectFailed():
                    return reconnecting(state, attempts + 1, pending)

                case StopTransport():
                    return Behaviors.stopped()

                case _StreamEnded():
                    return Behaviors.same()

                case _:
                    return _reconnect_receive(state, attempts, (*pending, msg))

        return Behaviors.receive(receive)

    # ── helpers ───────────────────────────────────────────────────────

    async def _do_run(
        conn: Any,
        command: tuple[str, ...],
        timeout: float | None,
        check: bool,
    ) -> tuple[int, str, str]:
        cmd = " ".join(command)
        result = await conn.run(cmd, timeout=timeout, check=False)
        code = result.exit_status or 0
        stdout = str(result.stdout or "")
        stderr = str(result.stderr or "")
        if check and code != 0:
            raise RuntimeError(f"Command failed ({code}): {stderr}")
        return code, stdout, stderr

    async def _do_write_file(conn: Any, remote: str, content: str) -> None:
        async with conn.start_sftp_client() as sftp, sftp.open(remote, "w") as f:
            await f.write(content)

    async def _do_write_bytes(conn: Any, remote: str, content: bytes) -> None:
        async with conn.start_sftp_client() as sftp, sftp.open(remote, "wb") as f:
            await f.write(content)

    async def _do_upload(conn: Any, local: str, remote: str) -> None:
        import asyncssh
        await asyncssh.scp(local, (conn, remote))

    async def _do_forward(
        conn: Any, remote_host: str, remote_port: int,
    ) -> tuple[int, Any]:
        listener = await conn.forward_local_port("", 0, remote_host, remote_port)
        local_port = listener.get_port()
        return local_port, listener

    def _start_stream(
        ctx: ActorContext[TransportMsg],
        state: TransportState,
        start_line: int,
    ) -> None:
        async def _stream_loop() -> _StreamEnded:
            from skyward.infra.ssh import _parse_jsonl_line

            try:
                for _ in range(120):
                    result = await state.conn.run(
                        "test -f /opt/skyward/events.jsonl", check=False,
                    )
                    if (result.exit_status or 0) == 0:
                        break
                    await asyncio.sleep(1.0)

                tail_offset = start_line + 1
                async with state.conn.create_process(
                    f"tail -n +{tail_offset} -F /opt/skyward/events.jsonl"
                ) as proc:
                    buffer = ""
                    lines_read = start_line

                    while True:
                        chunk = await asyncio.wait_for(
                            proc.stdout.read(4096), timeout=600.0,
                        )
                        if not chunk:
                            return _StreamEnded()

                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            lines_read += 1
                            line = line.strip()
                            if not line:
                                continue
                            event = _parse_jsonl_line(line)
                            if event is not None:
                                ctx.self.tell(_StreamedLine(
                                    lines_read=lines_read, event=event,
                                ))
            except asyncio.CancelledError:
                return _StreamEnded()

        ctx.pipe_to_self(
            _stream_loop(),
            mapper=lambda result: result,
            on_failure=lambda e: _StreamEnded(error=str(e)),
        )

    async def _reforward_ports(
        conn: Any,
        forward_requests: tuple[ForwardPort, ...],
    ) -> list[tuple[int, Any]]:
        results: list[tuple[int, Any]] = []
        for req in forward_requests:
            local_port, listener = await _do_forward(conn, req.remote_host, req.remote_port)
            results.append((local_port, listener))
        return results

    async def _cleanup(state: TransportState) -> None:
        for listener in state.listeners:
            with contextlib.suppress(Exception):
                listener.close()
        with contextlib.suppress(Exception):
            state.conn.close()
            await asyncio.wait_for(state.conn.wait_closed(), timeout=5.0)

    return connecting()
