"""Instance registry for monitoring.

Provides instance registry for tracking active instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .messages import (
    InstanceId,
    InstanceMetadata,
)


@dataclass
class InstanceRegistry:
    """Tracks active instances for monitoring.

    Shared state between monitors and handlers.
    """

    _instances: dict[InstanceId, InstanceMetadata] = field(default_factory=dict)

    def register(self, info: InstanceMetadata) -> None:
        self._instances[info.id] = info

    def unregister(self, instance_id: InstanceId) -> None:
        self._instances.pop(instance_id, None)

    @property
    def instances(self) -> list[InstanceMetadata]:
        return list(self._instances.values())

    @property
    def spot_instances(self) -> list[InstanceMetadata]:
        return [i for i in self._instances.values() if i.spot]

    def get(self, instance_id: InstanceId) -> InstanceMetadata | None:
        return self._instances.get(instance_id)


__all__ = [
    "InstanceRegistry",
]
