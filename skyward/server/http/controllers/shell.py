from __future__ import annotations

from litestar import Controller, Request, get, post
from litestar.params import Parameter
from litestar.response import Stream

from skyward.server.application import ports

BYTES = "application/octet-stream"


class ShellController(Controller):
    path = "/computes/{compute_id:str}/shell"
    tags = ["shell"]

    @post(
        "/up",
        status_code=200,
        summary="The up half of an interactive session",
        description=(
            "The caller's keystrokes into a node's terminal, as a streaming request body. **This request is the "
            "dispatch**: it opens the pseudo-terminal — on `node` if given, else on the first ready node — and pumps "
            "the body into it until the body ends.\n\n"
            "The body is the keyboard. It has no length and closes only when the session does.\n\n"
            "Paired with `GET .../down` by the `cid` the caller mints — the two are one session, and HTTP/1.1 will not "
            "carry both directions on a single request."
        ),
    )
    async def up(
        self,
        compute_id: str,
        request: Request,
        shell: ports.Shell,
        cid: str = Parameter(query="cid", description="The session id, minted by the caller, shared with `down`."),
        node: str | None = Parameter(query="node", default=None, description="The node to open the terminal on; omit for the first ready one."),
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
            "Not resumable. A dropped stream is a dead session; open another."
        ),
    )
    async def down(
        self,
        compute_id: str,
        shell: ports.Shell,
        cid: str = Parameter(query="cid", description="The session id shared with `up`."),
    ) -> Stream:
        return Stream(shell.down(cid), media_type=BYTES)
