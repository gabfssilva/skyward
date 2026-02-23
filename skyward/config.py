"""TOML-based pool and provider configuration.

Loads ~/.skyward/defaults.toml (global) and skyward.toml (project),
merges them, and resolves named pools into ComputePool instances.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skyward.api.pool import ComputePool
    from skyward.api.spec import Image
    from skyward.providers.aws.config import AWS
    from skyward.providers.gcp.config import GCP
    from skyward.providers.runpod.config import RunPod
    from skyward.providers.vastai.config import VastAI
    from skyward.providers.verda.config import Verda

    type ProviderConfig = AWS | GCP | RunPod | VastAI | Verda

type RawConfig = dict[str, Any]

GLOBAL_CONFIG_PATH = Path.home() / ".skyward" / "defaults.toml"
PROJECT_CONFIG_NAME = "skyward.toml"


def _deep_merge(base: RawConfig, override: RawConfig) -> RawConfig:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_toml(path: Path) -> RawConfig:
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config(
    *,
    project_dir: Path | None = None,
    global_path: Path | None = None,
) -> RawConfig:
    global_cfg = _read_toml(global_path or GLOBAL_CONFIG_PATH)
    project_path = (project_dir or Path.cwd()) / PROJECT_CONFIG_NAME
    project_cfg = _read_toml(project_path)

    merged = _deep_merge(global_cfg, project_cfg)
    merged.setdefault("providers", {})
    merged.setdefault("pools", {})
    return merged


def _get_provider_map() -> dict[str, type]:
    from skyward.providers.aws.config import AWS
    from skyward.providers.gcp.config import GCP
    from skyward.providers.runpod.config import RunPod
    from skyward.providers.vastai.config import VastAI
    from skyward.providers.verda.config import Verda

    return {
        "aws": AWS,
        "gcp": GCP,
        "vastai": VastAI,
        "runpod": RunPod,
        "verda": Verda,
    }


def _build_provider(name: str, raw: RawConfig) -> ProviderConfig:
    raw = dict(raw)
    provider_type = raw.pop("type", None)
    if provider_type is None:
        raise ValueError(f"Provider '{name}' missing 'type' field")

    provider_map = _get_provider_map()
    cls = provider_map.get(provider_type)
    if cls is None:
        raise ValueError(
            f"Unknown provider type '{provider_type}'. "
            f"Valid: {', '.join(provider_map)}"
        )
    return cls(**raw)


def _build_image(raw: RawConfig) -> Image:
    from skyward.api.spec import Image

    return Image(**raw)


def resolve_pool(
    name: str,
    *,
    project_dir: Path | None = None,
    global_path: Path | None = None,
) -> ComputePool:
    from skyward.api.pool import ComputePool
    from skyward.api.spec import Image

    config = load_config(project_dir=project_dir, global_path=global_path)

    pools = config["pools"]
    if name not in pools:
        raise KeyError(f"Pool '{name}' not found. Available: {', '.join(pools) or 'none'}")

    raw_pool = dict(pools[name])

    provider_ref = raw_pool.pop("provider", None)
    if provider_ref is None:
        raise ValueError(f"Pool '{name}' missing 'provider' field")

    providers = config["providers"]
    if provider_ref not in providers:
        raise KeyError(
            f"Provider '{provider_ref}' not found. Available: {', '.join(providers) or 'none'}"
        )

    provider = _build_provider(provider_ref, providers[provider_ref])

    raw_image = raw_pool.pop("image", None)
    image = _build_image(raw_image) if raw_image else Image()

    raw_volumes = raw_pool.pop("volumes", None)
    volumes: tuple = ()
    if raw_volumes:
        from skyward.api.spec import Volume

        volumes = tuple(Volume(**v) for v in raw_volumes)

    return ComputePool(provider=provider, image=image, volumes=volumes, **raw_pool)
