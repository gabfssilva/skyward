from skyward2.application.errors import UnsupportedProviderError
from skyward2.application.provider import ProviderAdapter
from skyward2.protocol.schemas import ProviderKind
from skyward2.providers.fake import FakeProvider
from skyward2.providers.vastai import VastAIProvider

REGISTRY: dict[str, type[ProviderAdapter]] = {
    FakeProvider.kind: FakeProvider,
    VastAIProvider.kind: VastAIProvider,
}


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


__all__ = ["REGISTRY", "adapter_for", "kinds"]
