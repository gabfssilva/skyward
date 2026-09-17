from __future__ import annotations

from litestar import Controller, Response, delete, get
from litestar.di import Provide
from litestar.params import Parameter

from skyward.api import v1
from skyward.server.application import ports
from skyward.server.application.reading import Block, Reader
from skyward.server.http import representation
from skyward.server.http.exceptions import failures
from skyward.server.http.include import node_blocks

NODE = "The rank the node holds, or its id. A rank reaches the node holding it, else the last one that did."


class NodeController(Controller):
    path = "/computes/{compute:str}/nodes"
    tags = ["nodes"]
    dependencies = {"blocks": Provide(node_blocks)}

    @get(
        summary="List a compute's nodes",
        description=(
            "The node holding each rank, by rank — what a compute carries as `nodes`. `include=replaced` adds the ones "
            "that held a rank before: a node that died stays until the provider confirms the machine is gone, which is "
            "what stops an instance from going missing with nobody knowing. `include` takes the blocks a compute's "
            "`nodes.` blocks are, without the prefix."
        ),
        responses=failures(404),
    )
    async def list(self, compute_id: str, reader: Reader, blocks: frozenset[Block]) -> v1.Page[v1.NodeResource]:
        return v1.Page(items=tuple(representation.node(each) for each in await reader.nodes(compute_id, blocks)), next_cursor=None, total=None)

    @get(
        "/{node:str}",
        summary="Read a node",
        description="One machine as the control plane knows it, reached by rank or by id.",
        responses=failures(404),
    )
    async def read(
        self,
        compute_id: str,
        reader: Reader,
        blocks: frozenset[Block],
        node: str = Parameter(description=NODE),
    ) -> v1.NodeResource:
        return representation.node(await reader.node(compute_id, node, blocks))

    @delete(
        "/{node:str}",
        status_code=202,
        summary="Drain and replace a node",
        description=(
            "Nodes are not directly creatable — they come from reconciling `spec.nodes`. But a node can be condemned: "
            "`desired: deleted` blocks new assignments, waits for the executions it is known to hold, then destroys the "
            "instance.\n\n"
            "If the compute still wants that capacity, the reconciler creates **another** node for the same `rank`, with "
            "a new `id`. The old node's tombstone remains."
        ),
        responses=failures(404, 409),
    )
    async def drain(
        self,
        compute_id: str,
        nodes: ports.Nodes,
        reader: Reader,
        node: str = Parameter(description=NODE),
        idempotency_key: str = Parameter(header="Idempotency-Key"),
    ) -> Response[v1.NodeResource]:
        condemned = (await reader.node(compute_id, node)).node
        await nodes.drain(compute_id, condemned.id, idempotency_key)
        return Response(representation.node(await reader.node(compute_id, condemned.id)), status_code=202)
