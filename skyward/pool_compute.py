"""Adapter to make Pool compatible with Provider protocol.

The Provider protocol expects a Compute object with specific attributes.
This adapter wraps a Pool to provide those attributes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from skyward.accelerator import Accelerator
from skyward.image import Image
from skyward.spec import SpotLike

if TYPE_CHECKING:
    from skyward.pool import ComputePool
    from skyward.volume import Volume
    from skyward.worker.config import ResourceLimits, WorkerConfig


@dataclass(frozen=True, slots=True)
class _PoolCompute:
    """Adapter that makes a Pool look like a Compute for Provider protocol.

    Providers expect a Compute object with attributes like `nodes`, `image`,
    `accelerator`, etc. This adapter wraps Pool to provide those attributes.
    """

    pool: ComputePool
    fn: Callable[..., Any]
    nodes: int
    accelerator: Accelerator | list[Accelerator]
    image: Image
    cpu: int | None
    memory: str | None
    timeout: int
    spot: SpotLike
    volumes: list[Volume]

    # Worker isolation fields
    _workers_per_instance: int = 1
    _worker_configs: tuple[WorkerConfig, ...] | None = None
    _worker_limits: ResourceLimits | None = None
    _worker_partition_script: str = ""

    # Properties expected by providers

    @property
    def name(self) -> str:
        """Function name (placeholder for pool)."""
        return "pool"

    @property
    def is_cluster(self) -> bool:
        """True if multi-node."""
        return self.nodes > 1

    @property
    def wrapped_fn(self) -> Callable[..., Any]:
        """Return function (placeholder)."""
        return self.fn

    @property
    def head_port(self) -> int:
        """Port for distributed head node."""
        return 29500

    @property
    def placement_group(self) -> str | None:
        """Placement group name."""
        return None

    @property
    def workers_per_instance(self) -> int:
        """Number of workers per instance (for worker isolation)."""
        return self._workers_per_instance

    @property
    def worker_configs(self) -> tuple[WorkerConfig, ...] | None:
        """Worker configurations for isolation."""
        return self._worker_configs

    @property
    def worker_limits(self) -> ResourceLimits | None:
        """Resource limits for workers."""
        return self._worker_limits

    @property
    def worker_partition_script(self) -> str:
        """MIG/partition setup script."""
        return self._worker_partition_script

    # Compatibility properties (not used but may be checked)

    @property
    def _app(self) -> None:
        """No app for pool-based compute."""
        return None

    @property
    def _instances(self) -> None:
        """Instances managed by Pool, not _PoolCompute."""
        return None
