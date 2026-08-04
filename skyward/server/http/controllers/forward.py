from __future__ import annotations

from litestar import Controller, Request, get, post
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.response import Stream

from skyward.server.application import ports
from skyward.server.http.exceptions import failures

BYTES = "application/octet-stream"


class ForwardController(Controller):
    path = "/computes/{compute_id:str}/forward"
    tags = ["forward"]

    @post(
        "/up",
        status_code=200,
        summary="The up half of a forwarded connection",
        description=(
            "The caller's bytes into a node, as a streaming request body. **This request is the dispatch**: it opens the "
            "channel to a ready node — round-robin, unless `route` says otherwise — and pumps the body into it until the "
            "body ends.\n\n"
            "The body is the socket. It has no length and closes only when the connection does, which is why it is streamed "
            "rather than read: waiting for the end would be waiting for the caller to hang up.\n\n"
            "Paired with `GET .../down` by the `cid` the caller mints — the two are one connection, and HTTP/1.1 will not "
            "carry both directions on a single request."
        ),
        responses=failures(404, 422),
    )
    async def up(
        self,
        compute_id: str,
        request: Request,
        forwarder: ports.Forwarder,
        cid: str = Parameter(query="cid", description="The connection id, minted by the caller, shared with `down`."),
        port: int = Parameter(query="port", description="The port the service listens on inside the node."),
        route: ports.Route = Parameter(query="route", default="round_robin"),
    ) -> None:
        await forwarder.up(compute_id, cid, port, route, request.stream())

    @get(
        "/down",
        media_type=BYTES,
        summary="The down half of a forwarded connection",
        description=(
            "The node's bytes back to the caller, as a raw byte stream — no framing, because a byte proxy has no frames. "
            "Waits for the matching `up` to open the channel, then follows it until the node closes its side.\n\n"
            "Not resumable. A dropped stream is a dead connection; open another."
        ),
        responses={
            200: ResponseSpec(
                bytes,
                media_type=BYTES,
                description="Whatever the node sends back, unframed, until it closes its side",
                generate_examples=False,
            ),
            **failures(404, 422),
        },
    )
    async def down(
        self,
        compute_id: str,
        forwarder: ports.Forwarder,
        cid: str = Parameter(query="cid", description="The connection id shared with `up`."),
    ) -> Stream:
        return Stream(forwarder.down(cid), media_type=BYTES)
