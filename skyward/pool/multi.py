"""MultiPool - Parallel provisioning for multiple ComputePools."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING

from skyward.utils.conc import map_async

if TYPE_CHECKING:
    from skyward.pool.compute import ComputePool


@dataclass
class MultiPool:
    """Context manager for multiple pools with parallel provisioning.

    Provisions all pools concurrently, reducing total setup time from
    sum(t_i) to max(t_i).

    Example:
        with MultiPool(
            ComputePool(provider=AWS(), nodes=4),
            ComputePool(provider=AWS(), accelerator="A100"),
        ) as (cpu, gpu):
            r1 = process(data) >> cpu
            r2 = train(model) >> gpu
    """

    pools: tuple[ComputePool, ...]

    # Internal state
    _entered: list[ComputePool] = field(default_factory=list, init=False, repr=False)

    def __init__(self, *pools: ComputePool) -> None:
        object.__setattr__(self, "pools", pools)
        object.__setattr__(self, "_entered", [])

    def __enter__(self) -> tuple[ComputePool, ...]:
        """Provision all pools in parallel."""
        try:
            # map_async preserves order and propagates first exception
            for pool in map_async(self._enter_pool, self.pools):
                self._entered.append(pool)
            return self.pools
        except Exception:
            # Cleanup pools that were successfully entered
            self._cleanup_entered()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Shutdown all pools in parallel."""
        # Shutdown all entered pools (best-effort, errors swallowed)
        list(
            map_async(
                lambda p: self._exit_pool(p, exc_type, exc_val, exc_tb),
                self._entered,
            )
        )
        self._entered.clear()

    @staticmethod
    def _enter_pool(pool: ComputePool) -> ComputePool:
        """Enter a single pool (for map_async)."""
        pool.__enter__()
        return pool

    @staticmethod
    def _exit_pool(
        pool: ComputePool,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit a single pool, swallowing errors."""
        with contextlib.suppress(Exception):
            pool.__exit__(exc_type, exc_val, exc_tb)

    def _cleanup_entered(self) -> None:
        """Cleanup pools that were successfully entered (on failure)."""
        for pool in self._entered:
            with contextlib.suppress(Exception):
                pool.__exit__(None, None, None)
        self._entered.clear()
