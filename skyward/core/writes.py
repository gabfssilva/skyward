"""What a watcher of the daemon changes: delete a compute, resize it, drain a node.

Each one is a single request to the daemon, and each is safe to send again. A
delete or a drain carries a fresh ``Idempotency-Key``, so a request the link
dropped under is asked once more without doing it twice. The writes against a
compute go through :func:`conditional`, which is also what the CLI and the pool
use: the loop that keeps a stale revision from refusing a change exists once.

Errors propagate. The caller decides what a refused write means to the person
looking at it.
"""

from __future__ import annotations

import uuid

import msgspec

from skyward.api.v1 import ComputeResource, NodeBounds, NodeResource, UpdateComputeResource
from skyward.core.client import Client
from skyward.core.errors import SkywardError

WRITE_ATTEMPTS = 5
"""How many times a conditional write is re-read and re-sent before it is a real conflict."""


async def conditional(client: Client, ref: str, method: str, body: bytes | None = None, headers: dict[str, str] | None = None) -> ComputeResource:
    """A write against a compute, guarded by the revision it was read at.

    ``If-Match`` is what keeps two writers from overwriting each other's intent,
    but the revision also moves on bookkeeping nobody asked for: the reconciler
    writes what it observed on every tick, and a lease renews itself on a timer.
    Either landing between the read and the write refuses a change nothing was
    racing, so the precondition is refreshed rather than handed back as a failure.
    """
    attempts = WRITE_ATTEMPTS
    while True:
        current = await client.call("GET", f"/v1/computes/{ref}", ComputeResource)
        try:
            return await client.call(
                method,
                f"/v1/computes/{current.id}",
                ComputeResource,
                body=body,
                headers={"If-Match": f'"{current.revision}"', **(headers or {})},
            )
        except SkywardError as error:
            attempts -= 1
            if error.code != "revision_conflict" or not attempts:
                raise


async def delete(client: Client, compute_id: str) -> None:
    await conditional(client, compute_id, "DELETE", headers={"Idempotency-Key": uuid.uuid4().hex})


async def scale(client: Client, compute: ComputeResource, minimum: int, maximum: int) -> None:
    """``initial`` goes back as it was: it is the size the pool opened at, which only its creation decides."""
    bounds = NodeBounds(initial=compute.spec.nodes.initial, min=minimum, max=maximum)
    await conditional(client, compute.id, "PATCH", msgspec.json.encode(UpdateComputeResource(nodes=bounds)))


async def drain(client: Client, compute_id: str, node_id: str) -> None:
    """The route answers 202 with the condemned node, so it is read as one rather than as an empty 204."""
    await client.call("DELETE", f"/v1/computes/{compute_id}/nodes/{node_id}", NodeResource, headers={"Idempotency-Key": uuid.uuid4().hex})


__all__ = ["WRITE_ATTEMPTS", "conditional", "delete", "drain", "scale"]
