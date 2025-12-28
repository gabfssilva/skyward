"""Adapter to make Pool compatible with Provider protocol.

The Provider protocol expects a Compute object with specific attributes.
This adapter wraps a Pool to provide those attributes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from skyward.accelerator import Accelerator
from skyward.spec import SpotLike

if TYPE_CHECKING:
    from skyward.pool import ComputePool
    from skyward.volume import Volume


@dataclass(frozen=True, slots=True)
class _PoolCompute:
    """Adapter that makes a Pool look like a Compute for Provider protocol.

    Providers expect a Compute object with attributes like `nodes`, `pip`,
    `accelerator`, etc. This adapter wraps Pool to provide those attributes.
    """

    pool: ComputePool
    fn: Callable[..., Any]
    nodes: int
    accelerator: Accelerator | list[Accelerator]
    cpu: int | None
    memory: str | None
    python: str
    pip: tuple[str, ...]
    pip_extra_index_url: str | None
    apt: tuple[str, ...]
    env: frozenset[tuple[str, str]]
    timeout: int
    spot: SpotLike
    volumes: tuple[Volume, ...]

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
    def env_dict(self) -> dict[str, str]:
        """Environment variables as dict."""
        return dict(self.env)

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

    # Compatibility properties (not used but may be checked)

    @property
    def _app(self) -> None:
        """No app for pool-based compute."""
        return None

    @property
    def _instances(self) -> None:
        """Instances managed by Pool, not _PoolCompute."""
        return None
