"""Provider selection types."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skyward.types.protocols import ComputeSpec, Provider, ProviderConfig

__all__ = [
    "ProviderLiteral",
    "SelectionStrategy",
    "ProviderSelector",
    "SelectionLike",
    "SingleProvider",
    "ProviderLike",
]

type ProviderLiteral = Literal["aws", "verda", "digital_ocean", "vastai"]

type SelectionStrategy = Literal["first", "cheapest", "available"]
"""Built-in provider selection strategies."""

type ProviderSelector = Callable[[tuple[Provider, ...], ComputeSpec], Provider]
"""Callable that selects a provider from a list based on compute requirements."""

type SelectionLike = SelectionStrategy | ProviderSelector
"""Either a built-in strategy name or a custom selector function."""

type SingleProvider = ProviderConfig | ProviderLiteral

type ProviderLike = SingleProvider | Sequence[SingleProvider]
"""Single provider config or sequence of configs for multi-provider selection."""
