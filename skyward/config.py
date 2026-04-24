"""TOML-based pool and provider configuration.

Loads ~/.skyward/defaults.toml (global) and skyward.toml (project),
merges them, and resolves named pools into ComputePool instances.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from skyward.api.logging import LogConfig
    from skyward.api.metrics import MetricsConfig
    from skyward.api.plugin import Plugin
    from skyward.api.spec import Nodes, Spec, Volume, Worker
    from skyward.core.pool import ComputePool
    from skyward.core.spec import Image
    from skyward.providers.aws.config import AWS
    from skyward.providers.gcp.config import GCP
    from skyward.providers.hyperstack.config import Hyperstack
    from skyward.providers.jarvislabs.config import JarvisLabs
    from skyward.providers.lambda_cloud.config import LambdaCloud
    from skyward.providers.massed_compute.config import MassedCompute
    from skyward.providers.novita.config import Novita
    from skyward.providers.runpod.config import RunPod
    from skyward.providers.scaleway.config import Scaleway
    from skyward.providers.tensordock.config import TensorDock
    from skyward.providers.vastai.config import VastAI
    from skyward.providers.verda.config import Verda
    from skyward.providers.vultr.config import Vultr
    from skyward.storage import Storage

    type ProviderConfig = AWS | GCP | Hyperstack | JarvisLabs | LambdaCloud | MassedCompute | Novita | RunPod | Scaleway | TensorDock | VastAI | Verda | Vultr

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
    from skyward.providers.container.config import Container
    from skyward.providers.gcp.config import GCP
    from skyward.providers.hyperstack.config import Hyperstack
    from skyward.providers.jarvislabs.config import JarvisLabs
    from skyward.providers.lambda_cloud.config import LambdaCloud
    from skyward.providers.massed_compute.config import MassedCompute
    from skyward.providers.novita.config import Novita
    from skyward.providers.runpod.config import RunPod
    from skyward.providers.scaleway.config import Scaleway
    from skyward.providers.tensordock.config import TensorDock
    from skyward.providers.vastai.config import VastAI
    from skyward.providers.verda.config import Verda
    from skyward.providers.vultr.config import Vultr

    return {
        "aws": AWS,
        "container": Container,
        "gcp": GCP,
        "hyperstack": Hyperstack,
        "jarvislabs": JarvisLabs,
        "lambda": LambdaCloud,
        "massed_compute": MassedCompute,
        "novita": Novita,
        "tensordock": TensorDock,
        "vastai": VastAI,
        "runpod": RunPod,
        "scaleway": Scaleway,
        "verda": Verda,
        "vultr": Vultr,
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
    from skyward.core.spec import Image, PipIndex

    raw = dict(raw)
    if raw_indexes := raw.get("pip_indexes"):
        raw["pip_indexes"] = [PipIndex(**idx) for idx in raw_indexes]
    if "metrics" in raw:
        raw["metrics"] = _build_metrics(raw["metrics"])
    return Image(**raw)


def _build_worker(raw: RawConfig) -> Worker:
    from skyward.api.spec import Worker

    return Worker(**raw)


def _build_nodes(raw: int | list[int] | RawConfig) -> int | tuple[int, int] | Nodes:
    from skyward.api.spec import Nodes

    match raw:
        case int():
            return raw
        case [int(min_n), int(max_n)]:
            return (min_n, max_n)
        case dict():
            return Nodes(**raw)
        case _:
            raise ValueError(
                f"Invalid nodes: {raw!r}. "
                "Expected int, [min, max], or {desired, min, max} table."
            )


def _build_logging(raw: bool | RawConfig) -> LogConfig | bool:
    match raw:
        case bool():
            return raw
        case dict():
            from skyward.api.logging import LogConfig

            return LogConfig(**raw)
        case _:
            raise ValueError(
                f"Invalid logging: {raw!r}. Expected true, false, or table."
            )


def _build_metrics(raw: bool | list[RawConfig]) -> MetricsConfig:
    from skyward.api.metrics import Metric

    match raw:
        case False:
            return None
        case True:
            from skyward.observability.metrics import Default

            return Default()
        case list():
            return tuple(Metric(**m) for m in raw)
        case _:
            raise ValueError(
                f"Invalid metrics: {raw!r}. "
                "Expected false, true, or array of tables."
            )


def _build_storage(raw: RawConfig) -> Storage:
    from skyward.storage import Storage

    return Storage(**raw)


def _build_volumes(raw_volumes: list[RawConfig]) -> tuple[Volume, ...]:
    from skyward.core.spec import Volume

    volumes: list[Volume] = []
    for v in raw_volumes:
        v = dict(v)
        if raw_storage := v.get("storage"):
            v["storage"] = _build_storage(raw_storage)
        volumes.append(Volume(**v))
    return tuple(volumes)


def _get_plugin_map() -> dict[str, Callable[..., Plugin]]:
    from skyward.plugins.accelerate import accelerate
    from skyward.plugins.cuml import cuml
    from skyward.plugins.jax import jax
    from skyward.plugins.joblib import joblib
    from skyward.plugins.keras import keras
    from skyward.plugins.mig import mig
    from skyward.plugins.mps import mps
    from skyward.plugins.sklearn import sklearn
    from skyward.plugins.torch import torch

    return {
        "accelerate": accelerate,
        "torch": torch,
        "jax": jax,
        "keras": keras,
        "cuml": cuml,
        "joblib": joblib,
        "sklearn": sklearn,
        "mig": mig,
        "mps": mps,
    }


def _build_plugins(raw: RawConfig) -> tuple[Plugin, ...]:
    plugin_map = _get_plugin_map()
    plugins: list[Plugin] = []
    for name, params in raw.items():
        factory = plugin_map.get(name)
        if factory is None:
            valid = ", ".join(sorted(plugin_map))
            raise ValueError(
                f"Unknown plugin '{name}'. Valid: {valid}"
            )
        match params:
            case True:
                plugins.append(factory())
            case dict():
                plugins.append(factory(**params))
            case _:
                raise ValueError(
                    f"Invalid config for plugin '{name}': {params!r}. "
                    "Expected true or a table of parameters."
                )
    return tuple(plugins)


def _build_pool_from_raw(name: str, raw_pool: RawConfig, config: RawConfig) -> ComputePool:
    from skyward.core.pool import ComputePool
    from skyward.core.spec import Image

    raw_pool = dict(raw_pool)

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
    volumes = _build_volumes(raw_volumes) if raw_volumes else ()

    raw_worker = raw_pool.pop("worker", None)
    worker = _build_worker(raw_worker) if raw_worker else None

    raw_nodes = raw_pool.pop("nodes", 1)
    nodes = _build_nodes(raw_nodes)

    raw_logging = raw_pool.pop("logging", None)
    logging = _build_logging(raw_logging) if raw_logging is not None else True

    raw_plugins = raw_pool.pop("plugins", None)
    plugins = _build_plugins(raw_plugins) if raw_plugins else ()

    if "accelerator" in raw_pool and isinstance(raw_pool["accelerator"], str):
        from skyward.accelerators import Accelerator

        raw_pool["accelerator"] = Accelerator.from_name(raw_pool["accelerator"])

    return ComputePool(
        provider=provider,
        image=image,
        volumes=volumes,
        worker=worker,
        nodes=nodes,
        logging=logging,
        plugins=plugins,
        **raw_pool,
    )


def resolve_pool(
    name: str,
    *,
    project_dir: Path | None = None,
    global_path: Path | None = None,
) -> ComputePool:
    config = load_config(project_dir=project_dir, global_path=global_path)

    pools = config["pools"]
    if name not in pools:
        raise KeyError(f"Pool '{name}' not found. Available: {', '.join(pools) or 'none'}")

    return _build_pool_from_raw(name, dict(pools[name]), config)


@dataclass(frozen=True, slots=True)
class PoolResolution:
    """Result of resolving a named pool from TOML."""
    pool: ComputePool
    specs: tuple[Spec, ...] = ()


def resolve_pool_config(
    name: str,
    *,
    project_dir: Path | None = None,
    global_path: Path | None = None,
) -> PoolResolution:
    """Resolve a named pool from ``skyward.toml``."""
    config = load_config(project_dir=project_dir, global_path=global_path)

    pools = config["pools"]
    if name not in pools:
        raise KeyError(f"Pool '{name}' not found. Available: {', '.join(pools) or 'none'}")

    pool = _build_pool_from_raw(name, dict(pools[name]), config)
    return PoolResolution(pool=pool, specs=pool._specs)
