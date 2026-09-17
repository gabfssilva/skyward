"""What a caller asks a resource to carry beyond what it always does.

``include`` is a comma-separated list, because that is how a person types it into
a URL; Litestar only reads a list as a repeated key, so the list is read here. A
name nobody has is refused by name rather than ignored: a caller who misspelled a
block would otherwise read its absence as there being nothing to show.
"""

from collections.abc import Mapping

from litestar.exceptions import ValidationException
from litestar.params import Parameter

from skyward.server.application.reading import Block

COMPUTE: Mapping[str, Block] = {
    "nodes.metrics": "metrics",
    "nodes.phases": "phases",
    "nodes.running": "running",
    "nodes.tail": "tail",
    "nodes.replaced": "replaced",
    "tasks.latest": "latest",
    "tasks.pace": "pace",
    "utilization": "utilization",
}

NODE: Mapping[str, Block] = {
    "metrics": "metrics",
    "phases": "phases",
    "running": "running",
    "tail": "tail",
    "replaced": "replaced",
}


async def compute_blocks(
    include: str | None = Parameter(query="include", default=None, description=f"Blocks to carry beyond the default, comma-separated: {', '.join(COMPUTE)}."),
) -> frozenset[Block]:
    return _asked(include, COMPUTE)


async def node_blocks(
    include: str | None = Parameter(query="include", default=None, description=f"Blocks to carry beyond the default, comma-separated: {', '.join(NODE)}."),
) -> frozenset[Block]:
    return _asked(include, NODE)


def _asked(include: str | None, names: Mapping[str, Block]) -> frozenset[Block]:
    asked = [name.strip() for name in (include or "").split(",") if name.strip()]
    if unknown := [name for name in asked if name not in names]:
        raise ValidationException(f"include has no {', '.join(unknown)}; it takes {', '.join(names)}")
    return frozenset(names[name] for name in asked)
