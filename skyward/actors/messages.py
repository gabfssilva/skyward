from __future__ import annotations

from dataclasses import dataclass

from skyward.messages import (
    ClusterId,
    InstanceId,
    ProviderName,
)


@dataclass(frozen=True, slots=True)
class Provision:
    cluster_id: ClusterId
    provider: ProviderName


@dataclass(frozen=True, slots=True)
class Replace:
    old_instance_id: InstanceId
    reason: str
