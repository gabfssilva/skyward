from __future__ import annotations

from litestar import Controller, Response, delete, get
from litestar.params import Parameter

from skyward.server.application import ports
from skyward.shared.schemas import Node, Page


class NodeController(Controller):
    path = "/computes/{compute_id:str}/nodes"
    tags = ["nodes"]

    @get(
        summary="List a compute's nodes",
        description=(
            "Includes tombstones by default. A node that died stays listed, with its `provider_binding` intact, until "
            "the provider confirms termination — that is what stops an instance from going missing with nobody knowing."
        ),
    )
    async def list(
        self,
        compute_id: str,
        nodes: ports.Nodes,
        include_terminal: bool = True,
        generation: int | None = None,
    ) -> Page[Node]:
        return await nodes.list(compute_id, include_terminal, generation)

    @get("/{node_id:str}", summary="Read a node")
    async def read(self, compute_id: str, node_id: str, nodes: ports.Nodes) -> Node:
        return await nodes.get(compute_id, node_id)

    @delete(
        "/{node_id:str}",
        status_code=202,
        summary="Drain and replace a node",
        description=(
            "Nodes are not directly creatable — they come from reconciling `spec.nodes`. But a node can be condemned: "
            "`desired: deleted` blocks new assignments, waits for the executions it is known to hold, then destroys the "
            "instance.\n\n"
            "If the compute still wants that capacity, the reconciler creates **another** node for the same `rank`, with "
            "a new `id`. The old node's tombstone remains."
        ),
    )
    async def drain(
        self,
        compute_id: str,
        node_id: str,
        nodes: ports.Nodes,
        idempotency_key: str = Parameter(header="Idempotency-Key"),
    ) -> Response[Node]:
        node = await nodes.drain(compute_id, node_id, idempotency_key)
        return Response(node, status_code=202)
