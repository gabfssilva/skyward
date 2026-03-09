"""Provider name → ProviderConfig registry for catalog-driven provider-less mode."""

from __future__ import annotations

from skyward.core.provider import ProviderConfig


def default_config(provider_type: str, region: str | None = None) -> ProviderConfig:
    """Create a default ProviderConfig for the given provider type.

    Lazy-imports each config class to avoid pulling in SDK dependencies.
    """
    match provider_type:
        case "aws":
            from skyward.providers.aws.config import AWS
            return AWS(region=region) if region else AWS()
        case "gcp":
            from skyward.providers.gcp.config import GCP
            return GCP(zone=region) if region else GCP()
        case "vastai":
            from skyward.providers.vastai.config import VastAI
            return VastAI()
        case "runpod":
            from skyward.providers.runpod.config import RunPod
            return RunPod()
        case "hyperstack":
            from skyward.providers.hyperstack.config import Hyperstack
            return Hyperstack(region=region) if region else Hyperstack()
        case "tensordock":
            from skyward.providers.tensordock.config import TensorDock
            return TensorDock()
        case "verda":
            from skyward.providers.verda.config import Verda
            return Verda()
        case _:
            raise ValueError(f"Unknown provider: {provider_type!r}")
