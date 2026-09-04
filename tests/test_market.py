"""Which offers a compute may buy: the spec names an account, and only that account's shelf is read."""

from typing import Any

import pytest

from skyward.server.application import market
from skyward.server.application.mock import OFFER
from skyward.shared.schemas import ComputeSpec, Image, NodeBounds, Page, ProviderRef, Spec, Worker

pytestmark = pytest.mark.local


class Shelf:
    """An offer cache that remembers what it was asked for."""

    def __init__(self) -> None:
        self.asked: list[dict[str, Any]] = []

    async def list(self, **query: Any) -> Page[Any]:
        self.asked.append(query)
        return Page(items=(OFFER,))


def describe_ranking_the_offers_for_a_spec() -> None:
    async def it_reads_the_shelf_of_the_account_the_spec_names() -> None:
        shelf = Shelf()
        spec = ComputeSpec(
            specs=(Spec(provider=ProviderRef(kind=OFFER.kind, name="production"), accelerator=OFFER.accelerator, accelerator_count=OFFER.accelerator_count),),
            nodes=NodeBounds(initial=1),
            image=Image(python="3.13"),
            worker=Worker(concurrency=1, executor="thread"),
        )

        ranked = await market.rank(spec, shelf)  # type: ignore[arg-type]

        assert [offer.id for offer in ranked] == [OFFER.id]
        assert [(asked["provider"], asked["kind"]) for asked in shelf.asked] == [("production", OFFER.kind)]
