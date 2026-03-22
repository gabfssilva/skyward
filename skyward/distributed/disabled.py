"""Registry that raises on every operation — used in standalone (non-cluster) mode."""

from __future__ import annotations

from typing import Any

from skyward.api.distributed import Consistency

_MSG = (
    "Distributed collections require cluster mode. "
    "Set Options(cluster=True) or remove distributed collection usage."
)


class DisabledRegistry:
    """Drop-in Registry that rejects all operations with a clear error."""

    __slots__ = ()

    def dict(self, name: str, *, consistency: Consistency | None = None) -> Any:
        raise RuntimeError(_MSG)

    def set(self, name: str, *, consistency: Consistency | None = None) -> Any:
        raise RuntimeError(_MSG)

    def counter(self, name: str, *, consistency: Consistency | None = None) -> Any:
        raise RuntimeError(_MSG)

    def queue(self, name: str) -> Any:
        raise RuntimeError(_MSG)

    def barrier(self, name: str, n: int) -> Any:
        raise RuntimeError(_MSG)

    def lock(self, name: str, timeout: float = 30) -> Any:
        raise RuntimeError(_MSG)

    def cleanup(self) -> None:
        pass
