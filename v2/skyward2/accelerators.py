"""The accelerator names, as the user spells them.

``sky.accelerators.A100(count=4)`` is the v1 spelling and it survives, but there
is no hand-written factory behind each name: the catalog already knows every
accelerator Skyward speaks of, and a module that listed them again would be a
second catalog to keep in step with the first. The attribute is resolved against
the catalog, so a name that exists here is a name a provider can be asked for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from skyward2.protocol.accelerators import CATALOG


@dataclass(frozen=True, slots=True)
class Accelerator:
    """A canonical accelerator name and how many of them a node should have."""

    name: str
    count: int = 1


def __getattr__(attribute: str) -> Callable[..., Accelerator]:
    name = attribute.lower().replace("_", "-")
    if name not in CATALOG:
        raise AttributeError(f"unknown accelerator: {attribute}")

    def accelerator(count: int = 1) -> Accelerator:
        return Accelerator(name, count)

    return accelerator


def __dir__() -> list[str]:
    return sorted(name.upper().replace("-", "_") for name in CATALOG)


__all__ = ["Accelerator"]
