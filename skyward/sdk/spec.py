"""What kind of machine, and how many.

A spec is one way to satisfy a pool: this provider, this accelerator, this
shape. A pool may be given several — "an A100 on AWS, or an A100 on Vast, take
whichever is cheaper" — and the control plane picks among the offers they match.
"""

from __future__ import annotations

from dataclasses import dataclass

from skyward.accelerators import Accelerator
from skyward.protocol.schemas import NodeBounds as Nodes
from skyward.sdk.provider import Provider

type NodeSpec = int | tuple[int, int] | Nodes


@dataclass(frozen=True, slots=True)
class Spec:
    provider: Provider
    accelerator: str | Accelerator | None = None
    cpus: int | None = None
    memory_gb: int | None = None
    region: str | None = None


__all__ = ["Accelerator", "NodeSpec", "Nodes", "Spec"]
