from skyward.application.errors import UnsupportedProviderError
from skyward.application.provider import Catalog
from skyward.protocol.schemas import ProviderKind
from skyward.providers.aws import AWSProvider
from skyward.providers.container import ContainerProvider
from skyward.providers.fake import FakeProvider
from skyward.providers.gcp import GCPProvider
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

ADAPTERS: tuple[type[Catalog], ...] = (
    AWSProvider,
    ContainerProvider,
    FakeProvider,
    GCPProvider,
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

REGISTRY: dict[str, type[Catalog]] = {adapter.kind: adapter for adapter in ADAPTERS}


def adapter_for(kind: str) -> type[Catalog]:
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
