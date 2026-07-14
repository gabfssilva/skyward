from __future__ import annotations

from typing import Any

import msgspec
from piccolo.columns import Column

from skyward2.application.errors import NotFoundError
from skyward2.application.provider import Machine
from skyward2.persistence.store import ident, now, once, packed, unpacked
from skyward2.persistence.tables import NodeRow
from skyward2.protocol.schemas import Error, Node, NodeDesired, NodeState, Page

TERMINAL: tuple[NodeState, ...] = ("deleted", "failed", "lost")


class NodeStore:
    async def list(self, compute: str, include_terminal: bool, generation: int | None) -> Page[Node]:
        query = NodeRow.objects().where(NodeRow.compute_id == compute)

        if not include_terminal:
            query = query.where(NodeRow.state.not_in(list(TERMINAL)))
        if generation is not None:
            query = query.where(NodeRow.generation == generation)

        rows = await query.order_by(NodeRow.rank)
        return Page(items=tuple(_to_node(row) for row in rows))

    async def get(self, compute: str, node_id: str) -> Node:
        return _to_node(await self._row(compute, node_id))

    async def drain(self, compute: str, node_id: str, idempotency_key: str) -> Node:
        """Ask for a node to go away. Do not make it go away.

        Same shape as deleting a compute: this writes ``desired: deleted`` and the
        reconciler is the one that terminates the machine, once whatever is still
        running on it has been dealt with.
        """

        async def mark() -> str:
            row = await self._row(compute, node_id)
            row.desired = "deleted"
            row.revision += 1
            await row.save().run()
            return row.id

        _, _ = await once(f"node.drain:{compute}", idempotency_key, None, mark)
        return await self.get(compute, node_id)

    async def adopt(self, compute: str, generation: int, rank: int, machine: Machine) -> Node:
        """Record a machine the provider has and the store has not.

        The ordinary path in: `launch` returns machines and each becomes a row. The
        other path is a machine found by a reconcile that we do not remember
        creating — which means we created it and died before the commit. Both land
        here, because they are the same fact.
        """
        row = NodeRow(
            id=ident("nod"),
            compute_id=compute,
            machine_id=machine.id,
            generation=generation,
            rank=rank,
            state="provisioning",
            address=machine.private_host or machine.host,
            created_at=now(),
        )
        await row.save().run()
        return await self.get(compute, row.id)

    async def of(self, compute: str) -> tuple[Node, ...]:
        rows = await NodeRow.objects().where(NodeRow.compute_id == compute).order_by(NodeRow.rank)
        return tuple(_to_node(row) for row in rows)

    async def machines(self, compute: str) -> dict[str, str]:
        """Which node each machine is, for the join a reconcile does first."""
        rows = await NodeRow.select(NodeRow.machine_id, NodeRow.id).where(NodeRow.compute_id == compute)
        return {row["machine_id"]: row["id"] for row in rows}

    async def observe(self, node_id: str, state: NodeState, error: Error | None = None) -> None:
        """What the node's own lifecycle reported. Written by nobody else."""
        row = await NodeRow.objects().where(NodeRow.id == node_id).first()
        if row is None:
            raise NotFoundError(f"no such node: {node_id}")

        changes: dict[Column | str, Any] = {
            NodeRow.state: state,
            NodeRow.last_error: packed(error) if error else None,
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


def _to_node(row: NodeRow) -> Node:
    return Node(
        id=row.id,
        compute_id=row.compute_id,
        generation=row.generation,
        rank=row.rank,
        revision=row.revision,
        desired=msgspec.convert(row.desired, NodeDesired),
        state=msgspec.convert(row.state, NodeState),
        provider_binding={**unpacked(row.provider_binding, dict[str, Any]), "machine_id": row.machine_id},
        created_at=row.created_at,
        address=row.address,
        accelerator=row.accelerator,
        price_per_hour=row.price_per_hour,
        last_error=unpacked(row.last_error, Error) if row.last_error else None,
        terminated_at=row.terminated_at,
    )
