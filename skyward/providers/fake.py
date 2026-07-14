from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

from skyward.protocol.schemas import Offer

CATALOG = (
    ("a100", 1, 12, 85.0, 1.10, 2.20),
    ("a100", 8, 96, 680.0, 8.80, 17.60),
    ("h100", 1, 20, 120.0, 2.40, 4.80),
    ("h100", 8, 160, 960.0, 19.20, 38.40),
    ("h100", 4, 80, 480.0, 6.00, None),
    (None, 0, 4, 16.0, 0.05, 0.12),
)


class FakeProvider:
    """A provider that costs nothing to query and never flakes.

    Exists so the cache, the TTL and the whole offers path can be tested without
    a network or an API key. Its catalog carries a spot-only flavor on purpose:
    Hyperstack, Verda and Massed Compute all publish some, and an offer with no
    on-demand price is the case that breaks naive price handling.
    """

    kind: ClassVar[str] = "fake"
    credential_fields: ClassVar[tuple[str, ...]] = ()
    offers_ttl: ClassVar[timedelta] = timedelta(minutes=5)

    def __init__(self, provider_id: str, name: str, config: Mapping[str, Any]) -> None:
        self._id = provider_id
        self._name = name
        self._region = str(config.get("region", "fake-1"))

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self:
        return cls(provider_id, name, config)

    async def offers(self) -> AsyncIterator[Offer]:
        now = datetime.now(UTC)
        for accelerator, count, cpus, memory_gb, spot, on_demand in CATALOG:
            label = f"{accelerator}x{count}" if accelerator else "cpu"
            yield Offer(
                id=f"fake-{label}",
                provider_id=self._id,
                provider_name=self._name,
                kind=self.kind,
                instance_type=f"fake.{label}",
                accelerator=accelerator,
                accelerator_count=count,
                cpus=cpus,
                memory_gb=memory_gb,
                region=self._region,
                spot_price=spot,
                on_demand_price=on_demand,
                available=4,
                fetched_at=now,
                expires_at=now + self.offers_ttl,
                specific={},
            )
