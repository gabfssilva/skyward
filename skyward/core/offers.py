from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skyward.api.plugin import Plugin
from skyward.api.spec import Nodes
from skyward.core.model import Offer
from skyward.core.provider import ProviderConfig
from skyward.core.spec import Image, PoolSpec, Spec, Volume, Worker
from skyward.observability.logger import logger


@dataclass(frozen=True, slots=True)
class PoolConfig:
    image: Image
    worker: Worker
    ssh_timeout: int
    ssh_retry_interval: int
    provision_retry_delay: float
    max_provision_attempts: int
    volumes: tuple[Volume, ...]
    autoscale_cooldown: float
    autoscale_idle_timeout: float
    reconcile_tick_interval: float
    plugins: tuple[Plugin, ...]


async def select_offers(
    specs: list[Spec],
    config: PoolConfig,
) -> tuple[tuple[Offer, ...], ProviderConfig, Any, PoolSpec]:
    """Rank all offers across specs by price.

    Uses the OfferRepository (SQLite-backed catalog with on-demand live
    fetching) to query and rank offers from all configured providers.

    Returns (offers_sorted, provider_config, cloud_provider, pool_spec).
    """
    from skyward.offers import OfferRepository, to_offer

    provider_instances: dict[str, Any] = {}
    config_by_type: dict[str, ProviderConfig] = {}
    for s in specs:
        ptype = s.provider.type
        if ptype not in provider_instances:
            provider_instances[ptype] = await s.provider.create_provider()
            config_by_type[ptype] = s.provider

    repo = await OfferRepository.create(
        providers=list(provider_instances.values()),
    )

    ranked: list[tuple[float, Offer, ProviderConfig, PoolSpec]] = []

    for s in specs:
        accel = s.accelerator
        ptype = s.provider.type

        query = (
            repo.accelerator(accel.name).provider(ptype) if accel
            else repo.provider(ptype).cpu_only()
        )

        if accel and accel.memory:
            mem = accel.memory.upper().removesuffix("GB")
            if mem.isdigit():
                query = query.accelerator_memory(int(mem))

        if s.vcpus:
            query = query.vcpus(s.vcpus)
        if s.memory_gb:
            query = query.memory(s.memory_gb)
        if s.max_hourly_cost:
            query = query.max_price(s.max_hourly_cost)

        query = query.allocation(s.allocation)
        use_spot = s.allocation in ("spot", "spot-if-available")

        catalog_offers = await query.cheapest(20)
        if not catalog_offers:
            logger.warning(
                "No offers from {provider} for accelerator={acc}",
                provider=ptype, acc=accel.name if accel else "none",
            )
            continue

        provider_config = s.provider
        region = s.region or catalog_offers[0].region

        match s.nodes:
            case Nodes() as spec_nodes:
                pass
            case (min_n, max_n):
                spec_nodes = Nodes(min=min_n, max=max_n)
            case int(n):
                spec_nodes = Nodes(min=n)

        pool_spec = PoolSpec(
            nodes=spec_nodes,
            accelerator=accel,
            region=region,
            vcpus=s.vcpus,
            memory_gb=s.memory_gb,
            architecture=s.architecture,
            allocation=s.allocation,
            image=config.image,
            ttl=s.ttl,
            worker=config.worker,
            provider=provider_config.type,  # type: ignore[arg-type]
            max_hourly_cost=s.max_hourly_cost,
            ssh_timeout=float(config.ssh_timeout),
            ssh_retry_interval=float(config.ssh_retry_interval),
            provision_retry_delay=config.provision_retry_delay,
            max_provision_attempts=config.max_provision_attempts,
            volumes=config.volumes,
            autoscale_cooldown=config.autoscale_cooldown,
            autoscale_idle_timeout=config.autoscale_idle_timeout,
            reconcile_tick_interval=config.reconcile_tick_interval,
            plugins=config.plugins,
        )

        for co in catalog_offers:
            offer = to_offer(co)
            price_raw = offer.spot_price if use_spot else offer.on_demand_price
            price = price_raw if price_raw is not None else float("inf")
            ranked.append((price, offer, provider_config, pool_spec))

    repo.close()

    if not ranked:
        raise RuntimeError("No offers found across all specs")

    ranked.sort(key=lambda x: x[0])
    offers = tuple(r[1] for r in ranked)
    _, _, best_config, best_spec = ranked[0]

    cloud_provider = provider_instances[best_config.type]

    logger.info(
        "Selected: {provider} {instance} in {region} (${price}/hr)",
        provider=best_config.type, instance=offers[0].instance_type.name,
        region=best_spec.region, price=offers[0].spot_price or offers[0].on_demand_price or 0,
    )

    return offers, best_config, cloud_provider, best_spec
