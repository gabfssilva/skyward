from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skyward.actors.messages import (
    DesiredCountChanged,
    DrainComplete,
    NodeId,
    NodeJoined,
    ReconcilerNodeLost,
)


@dataclass(frozen=True, slots=True)
class _ReconcileTick:
    pass


@dataclass(frozen=True, slots=True)
class _ProvisionResult:
    instances: tuple[Any, ...]
    cluster: Any


@dataclass(frozen=True, slots=True)
class _ProvisionError:
    error: str


@dataclass(frozen=True, slots=True)
class _TerminateResult:
    node_ids: tuple[NodeId, ...]


@dataclass(frozen=True, slots=True)
class _TerminateError:
    node_ids: tuple[NodeId, ...]
    error: str


type ReconcilerMsg = (
    DesiredCountChanged
    | ReconcilerNodeLost
    | NodeJoined
    | DrainComplete
    | _ReconcileTick
    | _ProvisionResult
    | _ProvisionError
    | _TerminateResult
    | _TerminateError
)
