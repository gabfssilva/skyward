from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContainerSpecific:
    network: str
    base_image: str
    context: str
