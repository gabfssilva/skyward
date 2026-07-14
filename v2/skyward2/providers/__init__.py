from skyward2.application.errors import UnsupportedProviderError
from skyward2.application.provider import ProviderAdapter
from skyward2.protocol.schemas import ProviderKind
from skyward2.providers.fake import FakeProvider
from skyward2.providers.hyperstack import HyperstackProvider
from skyward2.providers.jarvislabs import JarvisLabsProvider
from skyward2.providers.lambda_cloud import LambdaProvider
from skyward2.providers.massed_compute import MassedComputeProvider
from skyward2.providers.novita import NovitaProvider
from skyward2.providers.runpod import RunPodProvider
from skyward2.providers.scaleway import ScalewayProvider
from skyward2.providers.tensordock import TensorDockProvider
from skyward2.providers.vastai import VastAIProvider
from skyward2.providers.verda import VerdaProvider
from skyward2.providers.vultr import VultrProvider

ADAPTERS: tuple[type[ProviderAdapter], ...] = (
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
)

REGISTRY: dict[str, type[ProviderAdapter]] = {adapter.kind: adapter for adapter in ADAPTERS}


def adapter_for(kind: str) -> type[ProviderAdapter]:
    if kind not in REGISTRY:
        raise UnsupportedProviderError(f"unknown provider kind: {kind}", known=sorted(REGISTRY))
    return REGISTRY[kind]


def kinds() -> tuple[ProviderKind, ...]:
    return tuple(
        ProviderKind(
            kind=adapter.kind,
            credential_fields=adapter.credential_fields,
            offers_ttl_seconds=int(adapter.offers_ttl.total_seconds()),
        )
        for adapter in REGISTRY.values()
    )


__all__ = ["ADAPTERS", "REGISTRY", "adapter_for", "kinds"]
