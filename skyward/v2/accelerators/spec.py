"""Accelerator specification dataclass for v2.

Provides an immutable, type-safe accelerator configuration
following v2 patterns (frozen dataclass with slots).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Reuse v1 catalog for accelerator specs
from skyward.accelerators.catalog import SPECS


@dataclass(frozen=True, slots=True)
class Accelerator:
    """Immutable accelerator specification.

    Represents a GPU/TPU/accelerator configuration with memory,
    count, and optional metadata (CUDA versions, etc.).

    Args:
        name: Accelerator name (e.g., "H100", "A100", "T4").
        memory: VRAM size (e.g., "80GB", "40GB").
        count: Number of accelerators per node.
        metadata: Additional info (CUDA versions, form factor, etc.).

    Examples:
        >>> Accelerator("H100")
        Accelerator(name='H100', memory='', count=1, metadata=None)

        >>> Accelerator("A100", memory="40GB", count=4)
        Accelerator(name='A100', memory='40GB', count=4, metadata=None)

        >>> Accelerator.from_name("H100", count=8)
        Accelerator(name='H100', memory='80GB', count=8, metadata={'cuda': ...})
    """

    name: str
    memory: str = ""
    count: int = 1
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_name(cls, name: str, **overrides: Any) -> Accelerator:
        """Create an Accelerator from the catalog with optional overrides.

        Looks up the accelerator in the v1 catalog and applies
        any provided overrides.

        Args:
            name: Accelerator name from the catalog (e.g., "H100", "T4").
            **overrides: Override memory, count, or metadata.

        Returns:
            Accelerator instance with defaults from catalog.

        Raises:
            ValueError: If the accelerator name is not in the catalog.

        Examples:
            >>> Accelerator.from_name("T4")
            Accelerator(name='T4', memory='16GB', count=1, ...)

            >>> Accelerator.from_name("H100", count=4)
            Accelerator(name='H100', memory='80GB', count=4, ...)
        """
        if name not in SPECS:
            raise ValueError(
                f"Unknown accelerator: {name!r}. "
                f"Available: {', '.join(sorted(SPECS.keys())[:10])}..."
            )

        spec = SPECS[name]
        metadata = None
        if "cuda" in spec:
            metadata = {"cuda": spec["cuda"]}

        return cls(
            name=name,
            memory=overrides.get("memory", spec.get("memory", "")),
            count=overrides.get("count", spec.get("count", 1)),
            metadata=overrides.get("metadata", metadata),
        )

    def with_count(self, count: int) -> Accelerator:
        """Return a new Accelerator with a different count.

        Args:
            count: Number of accelerators per node.

        Returns:
            New Accelerator instance with updated count.

        Example:
            >>> h100 = Accelerator.from_name("H100")
            >>> h100_x4 = h100.with_count(4)
        """
        return Accelerator(
            name=self.name,
            memory=self.memory,
            count=count,
            metadata=self.metadata,
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.count == 1:
            return self.name
        return f"{self.count}x{self.name}"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        parts = [f"name={self.name!r}"]
        if self.memory:
            parts.append(f"memory={self.memory!r}")
        if self.count != 1:
            parts.append(f"count={self.count}")
        if self.metadata:
            parts.append(f"metadata={self.metadata!r}")
        return f"Accelerator({', '.join(parts)})"


__all__ = ["Accelerator"]
