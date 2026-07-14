from __future__ import annotations

from typing import Any

import msgspec
from piccolo.columns import Column

from skyward2.application.errors import NotFoundError
from skyward2.application.provider import Machine
from skyward2.persistence.store import ident, now, once, packed, unpacked
from skyward2.persistence.tables import NodeRow
from skyward2.protocol.schemas import Error, Node, NodeDesired, NodeState, Page

LIVE: tuple[NodeState, ...] = ("requested", "provisioning", "connecting", "bootstrapping", "ready")
"""What counts as capacity: a machine we have, or one we are already buying.

A node in flight counts. That is the whole reason the row exists before the
machine does — a loop that counted only what is running would buy another one
every time it looked, for the two minutes it takes the first to boot.
"""

TERMINAL: tuple[NodeState, ...] = ("deleted",)
"""Given back, and stamped.

Nothing else is terminal. A node that is lost, failed or draining has stopped being
capacity but has not stopped costing anything — somebody still owes the provider a
terminate, and the row stays alive until it has been sent.
"""


class NodeStore:
    async def list(self, compute: str, include_terminal: bool, generation: int | None) -> Page[Node]:
        query = NodeRow.objects().where(NodeRow.compute_id == compute)

        if not include_terminal:
            query = query.where(NodeRow.state.not_in(list(TERMINAL)))
        if generation is not None:
            query = query.where(NodeRow.generation == generation)

        rows = await query.order_by(NodeRow.rank)
        return Page(items=tuple([await _to_node(row) for row in rows]))

    async def get(self, compute: str, node_id: str) -> Node:
        return await _to_node(await self._row(compute, node_id))

    async def drain(self, compute: str, node_id: str, idempotency_key: str) -> Node:
        """Ask for a node to go away. Do not make it go away.

        Same shape as deleting a compute: this writes ``draining``, and the node
        leaves once whatever is running on it has finished. Nothing here talks to a
        provider.
        """

        async def mark() -> str:
            row = await self._row(compute, node_id)
            row.desired = "deleted"
            row.state = "draining"
            row.revision += 1
            await row.save().run()
            return row.id

        _, _ = await once(f"node.drain:{compute}", idempotency_key, None, mark)
        return await self.get(compute, node_id)

    async def request(self, compute: str, generation: int) -> Node:
        """Write down that we want one more machine, before asking for it.

        The rank is the lowest free index among the nodes that are alive, so a
        compute that lost node 1 and replaced it gets its rank 1 back rather than
        growing a hole. Ranks are dense because the workers are handed the peer list
        and their own index into it, and an index into a list with a hole in it
        points at the wrong machine.
        """
        taken = {node.rank for node in await self.of(compute) if node.state in LIVE}
        rank = next(candidate for candidate in range(len(taken) + 1) if candidate not in taken)

        row = NodeRow(
            id=ident("nod"),
            compute_id=compute,
            machine_id=None,
            generation=generation,
            rank=rank,
            state="requested",
            created_at=now(),
        )
        await row.save().run()
        return await self.get(compute, row.id)

    async def launched(self, node_id: str, machine: Machine) -> None:
        """The provider answered with an id. Now the row can point at something."""
        await self._machine(node_id, machine, "provisioning")

    async def reachable(self, node_id: str, machine: Machine) -> None:
        """The machine has an address. Somebody can now try to log into it."""
        await self._machine(node_id, machine, "connecting")

    async def _machine(self, node_id: str, machine: Machine, state: NodeState) -> None:
        """The machine is kept whole, not filleted into columns.

        What it takes to log into a machine is the provider's business and differs
        between them — a port here, a root password there. The control plane has no
        use for the parts and every use for the whole, so it stores the whole.
        """
        await NodeRow.update({
            NodeRow.machine_id: machine.id,
            NodeRow.state: state,
            NodeRow.address: machine.private_host or machine.host,
            NodeRow.provider_binding: await packed(machine),
            NodeRow.revision: NodeRow.revision + 1,
        }).where(NodeRow.id == node_id).run()

    async def of(self, compute: str) -> tuple[Node, ...]:
        rows = await NodeRow.objects().where(NodeRow.compute_id == compute).order_by(NodeRow.rank)
        return tuple([await _to_node(row) for row in rows])

    async def observe(self, node_id: str, state: NodeState, error: Error | None = None) -> None:
        """What the node's own lifecycle reported. Written by nobody else."""
        row = await NodeRow.objects().where(NodeRow.id == node_id).first()
        if row is None:
            raise NotFoundError(f"no such node: {node_id}")

        changes: dict[Column | str, Any] = {
            NodeRow.state: state,
            NodeRow.last_error: await packed(error) if error else None,
            NodeRow.revision: NodeRow.revision + 1,
        }
        if state in TERMINAL and row.terminated_at is None:
            changes[NodeRow.terminated_at] = now()

        await NodeRow.update(changes).where(NodeRow.id == node_id).run()

    async def _row(self, compute: str, node_id: str) -> NodeRow:
        row = await NodeRow.objects().where(
            (NodeRow.compute_id == compute) & (NodeRow.id == node_id),
        ).first()
        if row is None:
            raise NotFoundError(f"no such node: {node_id}")
        return row


async def _to_node(row: NodeRow) -> Node:
    return Node(
        id=row.id,
        compute_id=row.compute_id,
        generation=row.generation,
        rank=row.rank,
        revision=row.revision,
        desired=msgspec.convert(row.desired, NodeDesired),
        state=msgspec.convert(row.state, NodeState),
        provider_binding=await unpacked(row.provider_binding, dict[str, Any]),
        machine=row.machine_id or None,
        created_at=row.created_at,
        address=row.address,
        accelerator=row.accelerator,
        price_per_hour=row.price_per_hour,
        last_error=await unpacked(row.last_error, Error) if row.last_error else None,
        terminated_at=row.terminated_at,
    )
