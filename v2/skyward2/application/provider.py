from collections.abc import AsyncIterator, Mapping
from datetime import timedelta
from typing import Any, ClassVar, Protocol, Self, runtime_checkable

from skyward2.protocol.schemas import Offer


@runtime_checkable
class ProviderAdapter(Protocol):
    """What a cloud provider must implement to be registered.

    Deliberately minimal: today it only has to price its hardware. Provisioning
    lands here later, and the split is on purpose — a provider can be listed and
    compared long before it can be trusted to run a compute.

    Credentials arrive through ``create``. An adapter never reads an environment
    variable, a config file or a keychain on its own: the controller owns the
    provider record, so it owns credential resolution. That is what allows two
    accounts of the same kind to coexist in one process.
    """

    kind: ClassVar[str]
    credential_fields: ClassVar[tuple[str, ...]]
    offers_ttl: ClassVar[timedelta]

    @classmethod
    def create(cls, provider_id: str, name: str, credentials: Mapping[str, str], config: Mapping[str, Any]) -> Self: ...

    def offers(self) -> AsyncIterator[Offer]:
        """Yield the whole catalog, unfiltered.

        Filtering and sorting happen against the cache, not against the API —
        a provider that filters server-side would have to be re-queried for
        every distinct question a user asks.

        Prices are per hour. A provider that bills per second normalizes here,
        so nothing downstream has to carry a billing unit around.
        """
        ...
