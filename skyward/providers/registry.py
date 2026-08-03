"""Every adapter skyward ships, keyed by kind.

Three adapters need an sdk of their own, and skyward does not install them with
the daemon. One whose sdk is absent is not registered — a kind skyward supports
but this installation cannot reach names the extra that would bring it.
"""

from collections.abc import Mapping
from importlib import import_module

from skyward.providers.container import ContainerProvider
from skyward.providers.fake import FakeProvider
from skyward.providers.hyperstack import HyperstackProvider
from skyward.providers.jarvislabs import JarvisLabsProvider
from skyward.providers.lambda_cloud import LambdaProvider
from skyward.providers.massed_compute import MassedComputeProvider
from skyward.providers.novita import NovitaProvider
from skyward.providers.runpod import RunPodProvider
from skyward.providers.scaleway import ScalewayProvider
from skyward.providers.tensordock import TensorDockProvider
from skyward.providers.vastai import VastAIProvider
from skyward.providers.verda import VerdaProvider
from skyward.providers.vultr import VultrProvider
from skyward.shared.errors import UnsupportedProviderError
from skyward.shared.provider import Catalog
from skyward.shared.schemas import ProviderKind

EXTRAS: Mapping[str, str] = {
    "aws": "AWSProvider",
    "gcp": "GCPProvider",
    "salad": "SaladProvider",
}


def _optional(module: str, symbol: str) -> type[Catalog] | None:
    try:
        return getattr(import_module(f"skyward.providers.{module}"), symbol)
    except ImportError:
        return None


ADAPTERS: tuple[type[Catalog], ...] = (
    ContainerProvider,
    FakeProvider,
    HyperstackProvider,
    JarvisLabsProvider,
    LambdaProvider,
    MassedComputeProvider,
    NovitaProvider,
    RunPodProvider,
    ScalewayProvider,
    TensorDockProvider,
    VastAIProvider,
    VerdaProvider,
    VultrProvider,
    *(adapter for module, symbol in EXTRAS.items() if (adapter := _optional(module, symbol))),
)

REGISTRY: dict[str, type[Catalog]] = {adapter.kind: adapter for adapter in ADAPTERS}


def adapter_for(kind: str) -> type[Catalog]:
    if kind in REGISTRY:
        return REGISTRY[kind]
    if kind in EXTRAS:
        raise UnsupportedProviderError(f"{kind} is not installed: pip install skyward[{kind}]", known=sorted(REGISTRY))
    raise UnsupportedProviderError(f"unknown provider kind: {kind}", known=sorted(REGISTRY))


def kinds() -> tuple[ProviderKind, ...]:
    return tuple(
        ProviderKind(
            kind=adapter.kind,
            credential_fields=adapter.credential_fields,
            offers_ttl_seconds=int(adapter.offers_ttl.total_seconds()),
        )
        for adapter in REGISTRY.values()
    )


__all__ = ["ADAPTERS", "EXTRAS", "REGISTRY", "adapter_for", "kinds"]
