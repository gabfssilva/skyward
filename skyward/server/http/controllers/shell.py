from __future__ import annotations

import asyncio
from contextlib import suppress

import asyncssh
import msgspec
from litestar import Controller, Request, get, post, websocket
from litestar.connection import WebSocket
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.response import Stream

from skyward.server.application import ports
from skyward.server.application.ssh import CHUNK, Pty
from skyward.server.http.exceptions import failures
from skyward.shared.errors import SkywardError
from skyward.shared.schemas import Error, Resize

BYTES = "application/octet-stream"

REFUSED = 4409
"""The close code for a session that could not be opened — the 409 of the paired halves.

Refusing the handshake would be the closer analogue, and it is the wrong one: a
browser is told nothing about a rejected upgrade, since the WebSocket API hands
``onerror`` no status and no body. So the socket is accepted, the refusal is sent as
the same :class:`Error` every other endpoint answers with, and only then is it closed
— the reason is in a frame, not in the close, which holds 123 bytes.
"""


class ShellController(Controller):
    path = "/computes/{compute:str}/shell"
    tags = ["shell"]

    @post(
        "/up",
        status_code=200,
        summary="The up half of an interactive session",
        description=(
            "The caller's keystrokes into a machine's terminal, as a streaming request body. **This request is the "
            "dispatch**: it opens the pseudo-terminal — at `node` if given, else at the lowest rank this daemon holds "
            "a link to — and pumps the body into it until the body ends.\n\n"
            "The machine does not have to be ready. Every machine that has answered SSH takes a terminal, which is "
            "how a bootstrap is watched while it is still happening; one that is still booting takes the session at "
            "the moment it answers, so the request may wait before the first byte comes back.\n\n"
            "The body is the keyboard. It has no length and closes only when the session does.\n\n"
            "Paired with `GET .../down` by the `cid` the caller mints — the two are one session, and HTTP/1.1 will not "
            "carry both directions on a single request."
        ),
        responses=failures(404, 409, 422),
    )
    async def up(
        self,
        compute_id: str,
        request: Request,
        shell: ports.Shell,
        cid: str = Parameter(query="cid", description="The session id, minted by the caller, shared with `down`."),
        node: int | None = Parameter(query="node", default=None, description="The rank to open the terminal on; omit for the lowest one held."),
        command: str | None = Parameter(query="command", default=None, description="What to run; omit for the login shell."),
        term: str = Parameter(query="term", default="xterm-256color", description="The terminal type to claim."),
        columns: int = Parameter(query="columns", default=80, description="The terminal width."),
        rows: int = Parameter(query="rows", default=24, description="The terminal height."),
    ) -> None:
        await shell.up(compute_id, cid, node, command, term, (columns, rows), request.stream())

    @get(
        "/down",
        media_type=BYTES,
        summary="The down half of an interactive session",
        description=(
            "What the terminal paints, as a raw byte stream — no framing, because a terminal has none, and the error "
            "stream is folded in because a terminal has one output. Waits for the matching `up` to open the session, "
            "then follows it until the shell exits.\n\n"
            "The wait is before the answer, not inside it: a session that cannot be opened — no machine at that rank, "
            "not this daemon's compute — is refused here with a status, rather than answered 200 and cut off part-way "
            "through the body.\n\n"
            "Not resumable. A dropped stream is a dead session; open another."
        ),
        responses={
            200: ResponseSpec(
                bytes,
                media_type=BYTES,
                description="Whatever the terminal paints, unframed, until the shell exits",
                generate_examples=False,
            ),
            **failures(404, 409, 422),
        },
    )
    async def down(
        self,
        compute_id: str,
        shell: ports.Shell,
        cid: str = Parameter(query="cid", description="The session id shared with `up`."),
    ) -> Stream:
        return Stream(await shell.down(cid), media_type=BYTES)

    @websocket("/attach")
    async def attach(
        self,
        compute: str,
        socket: WebSocket,
        shell: ports.Shell,
        computes: ports.Computes,
        node: int | None = Parameter(query="node", default=None, description="The rank to open the terminal on; omit for the lowest one held."),
        command: str | None = Parameter(query="command", default=None, description="What to run; omit for the login shell."),
        term: str = Parameter(query="term", default="xterm-256color", description="The terminal type to claim."),
        columns: int = Parameter(query="columns", default=80, description="The terminal width it opens at."),
        rows: int = Parameter(query="rows", default=24, description="The terminal height it opens at."),
    ) -> None:
        """One interactive session, both directions on one socket.

        The same terminal the paired halves open, for a caller that can hold a socket.
        Those two exist because HTTP/1.1 cannot carry a request body that is still
        being written alongside the response to it — which is exactly what a browser
        cannot work around, since a streaming request body needs HTTP/2. So the
        console uses this and the CLI uses those, and both reach the same pty.

        Binary frames are the terminal itself: keystrokes up, whatever it paints
        down, unframed, because a terminal has no frames and one output. Text frames
        up are a :class:`Resize` — the screen's new shape, which the halves can only
        say once and this can say whenever the window moves.

        The machine need not be ready: every machine that has answered SSH takes a
        terminal, and one still booting takes it the moment it does, so the socket
        may be open a while before the first byte comes back. One that cannot be
        opened at all is an :class:`Error` in a text frame and then a close — and so is
        a compute nobody has, which is why the name is looked up after the socket is
        accepted rather than by the dependency every other route resolves it with.
        """
        await socket.accept()
        try:
            pty = await shell.open(await computes.identify(compute), node, command, term, (columns, rows))
        except SkywardError as refused:
            await socket.send_json(Error(code=refused.code, message=refused.message, retryable=refused.retryable, details=refused.details or None))
            await socket.close(code=REFUSED, reason=refused.code)
            return

        await _carry(socket, pty)


async def _carry(socket: WebSocket, pty: Pty) -> None:
    """Both directions at once, until whichever ends first ends the other.

    A socket does not need the pairing the two halves do, so there is nothing here
    to reconcile: one task carries what is typed down to the machine, another
    carries what the terminal paints back up, and the session is over when either
    stops — the shell exiting, or the browser tab closing.
    """
    reader, writer = pty.channel
    both = (asyncio.create_task(_paint(socket, reader)), asyncio.create_task(_type(socket, pty)))
    try:
        await asyncio.wait(both, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in both:
            task.cancel()
        await asyncio.gather(*both, return_exceptions=True)
        with suppress(OSError, asyncssh.Error):
            writer.close()
        with suppress(Exception):
            await socket.close()


async def _paint(socket: WebSocket, reader: asyncssh.SSHReader[bytes]) -> None:
    """What the terminal paints, as binary frames, until the shell exits."""
    with suppress(OSError, asyncssh.Error):
        while data := await reader.read(CHUNK):
            await socket.send_bytes(data)


async def _type(socket: WebSocket, pty: Pty) -> None:
    """What the caller sends, until they stop sending.

    Binary is the keyboard and text is the screen's shape — the only two things a
    terminal takes from this side, which is why neither needs a tag to say which it
    is. A text frame that is not a shape is ignored: nothing else in this direction
    has anything to do with the session, and a frame the caller got wrong is not a
    reason to take their shell away.
    """
    _, writer = pty.channel
    while True:
        match await socket.receive():
            case {"type": "websocket.disconnect"}:
                return
            case {"bytes": bytes(typed)} if typed:
                writer.write(typed)
            case {"text": str(control)} if control:
                with suppress(msgspec.DecodeError, msgspec.ValidationError):
                    shape = msgspec.json.decode(control, type=Resize)
                    pty.resize((shape.columns, shape.rows))
            case _:
                pass
