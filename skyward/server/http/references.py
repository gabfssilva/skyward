"""Where a name becomes an id.

A caller reaches a compute by its name or by its id, on every route. Past the edge
nothing is keyed by a name: nodes, generations, tasks, events, metrics and the links
this daemon holds are all indexed by id, so a name handed through would read as a
compute with nothing in it — an empty page where the answer is a 404. Each route
takes ``compute_id`` instead, and one of these has already looked it up.
"""

from __future__ import annotations

from litestar.params import Parameter

from skyward.server.application import ports

COMPUTE = "The compute's name or id."


async def identified(computes: ports.Computes, compute: str = Parameter(description=COMPUTE)) -> str:
    """The id of the compute a path names."""
    return await computes.identify(compute)


async def narrowed(computes: ports.Computes, compute: str | None = Parameter(default=None, description=COMPUTE)) -> str | None:
    """The id of the compute a listing is narrowed to, when it is narrowed to one."""
    return None if compute is None else await computes.identify(compute)
