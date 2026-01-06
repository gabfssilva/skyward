"""Provider selection strategies.

This module provides strategies for selecting between multiple cloud providers
when provisioning compute resources. Selection can be based on:
- Order (first provider in list)
- Price (cheapest instance matching requirements)
- Availability (first provider with matching instances)

Example:
    from skyward import ComputePool, AWS, DigitalOcean

    # Multi-provider with cheapest selection
    pool = ComputePool(
        provider=[AWS(), DigitalOcean()],
        selection="cheapest",
        accelerator="A100",
    )

    # Custom selector
    def prefer_spot(providers, spec):
        for p in providers:
            if has_spot_capacity(p, spec):
                return p
        return providers[0]

    pool = ComputePool(
        provider=[AWS(), DigitalOcean()],
        selection=prefer_spot,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.types import ComputeSpec, Provider, ProviderConfig

from skyward.exceptions import NoMatchingInstanceError
from skyward.types import (
    ProviderLike,
    ProviderSelector,
    SelectionLike,
    SelectionStrategy,
    parse_memory_mb,
    select_instance,
)


class NoAvailableProviderError(Exception):
    """No provider has instances matching the requirements."""


class AllProvidersFailedError(Exception):
    """All providers failed to provision resources."""

    def __init__(self, errors: list[tuple[Provider, Exception]]) -> None:
        self.errors = errors
        names = ", ".join(p.name for p, _ in errors)
        super().__init__(f"All providers failed: {names}")


# Selection functions (pure)


def select_first(
    providers: tuple[Provider, ...],
    spec: ComputeSpec,
) -> Provider:
    """Returns first provider (default behavior)."""
    return providers[0]


def select_cheapest(
    providers: tuple[Provider, ...],
    spec: ComputeSpec,
) -> Provider:
    """Returns provider with cheapest matching instance.

    Compares the best price for the requested accelerator/resources
    across all providers and returns the one with the lowest price.
    """

    def get_best_price(provider: Provider) -> float:
        try:
            best = select_instance(
                provider.available_instances(),
                accelerator=spec.accelerator,
                cpu=spec.cpu or 1,
                memory_mb=parse_memory_mb(spec.memory),
            )
            return best.price_spot or best.price_on_demand or float("inf")
        except NoMatchingInstanceError:
            return float("inf")

    return min(providers, key=get_best_price)


def select_available(
    providers: tuple[Provider, ...],
    spec: ComputeSpec,
) -> Provider:
    """Returns first provider that has matching instances.

    Iterates through providers in order and returns the first one
    that has an instance type matching the requirements.
    """
    for provider in providers:
        try:
            select_instance(
                provider.available_instances(),
                accelerator=spec.accelerator,
                cpu=spec.cpu or 1,
                memory_mb=parse_memory_mb(spec.memory),
            )
            return provider
        except NoMatchingInstanceError:
            continue

    raise NoAvailableProviderError(
        f"No provider has instances matching: accelerator={spec.accelerator}, "
        f"cpu={spec.cpu or 1}, memory={spec.memory}"
    )


_STRATEGIES: dict[SelectionStrategy, ProviderSelector] = {
    "first": select_first,
    "cheapest": select_cheapest,
    "available": select_available,
}


def normalize_selector(selection: SelectionLike) -> ProviderSelector:
    """Normalize Literal or Callable to ProviderSelector.

    Args:
        selection: Either a strategy name ("first", "cheapest", "available")
            or a callable with signature (tuple[Provider, ...], ComputeSpec) -> Provider.

    Returns:
        A ProviderSelector callable.

    Raises:
        ValueError: If strategy name is unknown.
        TypeError: If selection is neither str nor callable.
    """
    match selection:
        case str() as strategy:
            if strategy not in _STRATEGIES:
                valid = ", ".join(f"'{s}'" for s in _STRATEGIES)
                raise ValueError(
                    f"Unknown selection strategy: {strategy!r}. Valid options: {valid}"
                )
            return _STRATEGIES[strategy]
        case selector if callable(selector):
            return selector
        case _:
            raise TypeError(
                f"Expected SelectionStrategy or Callable, got {type(selection).__name__}"
            )


def normalize_providers(provider: ProviderLike) -> tuple[ProviderConfig, ...]:
    """Normalize single provider or sequence to tuple of configs.

    Args:
        provider: Either a single ProviderConfig or a sequence of them.

    Returns:
        Tuple of ProviderConfig objects.
    """
    from skyward.types import ProviderConfig

    def normalize_single(single: ProviderLike) -> ProviderConfig:
        match single:
            case ProviderConfig():
                return single
            case str("aws"):
                from skyward.providers import AWS

                return AWS()
            case str("verda"):
                from skyward.providers import Verda

                return Verda()
            case str("digital_ocean"):
                from skyward.providers import DigitalOcean

                return DigitalOcean()
            case str("vastai"):
                from skyward.providers import VastAI

                return VastAI()

        raise Exception(f"Unknown provider: {single!r}")

    match provider:
        case list(providers):
            return tuple(normalize_single(provider) for provider in providers)
        case _:
            return (normalize_single(provider),)
