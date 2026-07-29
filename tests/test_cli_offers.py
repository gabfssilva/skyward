from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("cyclopts")

from skyward.cli import offers as cli
from skyward.shared.schemas import Offer, Page

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def offer(
    *,
    provider: str = "aws-main",
    accelerator: str | None = "H100",
    count: int = 1,
    spot: float | None = None,
    on_demand: float | None = 3.0,
) -> Offer:
    return Offer(
        id=f"{provider}-{accelerator}-{count}-{spot}-{on_demand}",
        provider_id=provider,
        provider_name=provider,
        kind="aws",
        instance_type="p5.48xlarge",
        accelerator=accelerator,
        accelerator_count=count,
        cpus=192,
        memory_gb=2048.0,
        region="us-east-1",
        spot_price=spot,
        on_demand_price=on_demand,
        fetched_at=NOW,
        expires_at=NOW,
    )


class Recorder:
    """A stand-in for the SDK client that captures the query it was asked for."""

    def __init__(self, page: Page[Offer]) -> None:
        self.page = page
        self.query: dict[str, Any] = {}
        self.path = ""

    async def call(self, method: str, path: str, kind: object, **query: object) -> Page[Offer]:
        self.path = path
        self.query = dict(query)
        return self.page


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    captured = Recorder(Page(items=()))

    def fake_call(work: Any, *, url: object = None, database: object = None) -> object:
        return asyncio.run(work(captured))

    monkeypatch.setattr(cli, "call", fake_call)
    return captured


@pytest.mark.unit
def test_price_drops_trailing_zeros() -> None:
    assert cli._price(3.0) == "3"
    assert cli._price(0.1234) == "0.1234"
    assert cli._price(None) is None


@pytest.mark.unit
def test_accelerator_prefixes_the_count_only_when_plural() -> None:
    assert cli._accelerator(offer(count=1)) == "h100"
    assert cli._accelerator(offer(count=8)) == "8x h100"
    assert cli._accelerator(offer(accelerator=None)) is None


@pytest.mark.unit
def test_summary_aggregates_the_price_spread() -> None:
    offers = [offer(spot=1.0, on_demand=4.0), offer(spot=3.0, on_demand=4.0)]
    row = cli._summary(("h100", "aws-main"), offers)
    assert row == ("h100", "aws-main", 2, "1", "2", "3")


@pytest.mark.unit
def test_summary_survives_offers_with_no_price() -> None:
    row = cli._summary(("h100", "aws-main"), [offer(on_demand=None)])
    assert row == ("h100", "aws-main", 1, None, None, None)


@pytest.mark.unit
def test_list_sends_every_filter_as_a_query_param(recorder: Recorder) -> None:
    cli.list_offers(provider="aws-main", accelerator="H100", min_count=8, min_vram=80.0, max_price=5.0)
    assert recorder.path == "/v1/offers"
    assert recorder.query == {
        "provider": "aws-main",
        "accelerator": "H100",
        "min_count": 8,
        "min_vram": 80.0,
        "max_price": 5.0,
        "refresh": None,
    }


@pytest.mark.unit
def test_list_omits_refresh_unless_asked(recorder: Recorder) -> None:
    cli.list_offers(refresh=True)
    assert recorder.query["refresh"] is True


@pytest.mark.unit
def test_list_honours_the_limit(recorder: Recorder, capsys: pytest.CaptureFixture[str]) -> None:
    recorder.page = Page(items=tuple(offer(on_demand=float(index)) for index in range(1, 6)))
    cli.list_offers(limit=2)
    assert len(capsys.readouterr().out.strip().splitlines()) == 3


@pytest.mark.unit
def test_list_zero_limit_prints_everything(recorder: Recorder, capsys: pytest.CaptureFixture[str]) -> None:
    recorder.page = Page(items=tuple(offer(on_demand=float(index)) for index in range(1, 6)))
    cli.list_offers(limit=0)
    assert len(capsys.readouterr().out.strip().splitlines()) == 6


@pytest.mark.unit
def test_fetch_forces_a_refresh_and_counts_per_provider(recorder: Recorder, capsys: pytest.CaptureFixture[str]) -> None:
    recorder.page = Page(items=(offer(provider="aws-main"), offer(provider="aws-main"), offer(provider="vast")))
    cli.fetch_offers()

    assert recorder.query["refresh"] is True
    rows = capsys.readouterr().out.strip().splitlines()
    assert rows[0].split() == ["PROVIDER", "OFFERS"]
    assert rows[1].split() == ["aws-main", "2"]
    assert rows[2].split() == ["vast", "1"]


@pytest.mark.unit
def test_summary_orders_by_accelerator_then_cheapest(recorder: Recorder, capsys: pytest.CaptureFixture[str]) -> None:
    recorder.page = Page(
        items=(
            offer(provider="expensive", accelerator="A100", on_demand=9.0),
            offer(provider="cheap", accelerator="A100", on_demand=1.0),
            offer(provider="only", accelerator="H100", on_demand=5.0),
        ),
    )
    cli.summary_offers()

    rows = [line.split() for line in capsys.readouterr().out.strip().splitlines()]
    assert [row[0] for row in rows[1:]] == ["a100", "a100", "h100"]
    assert [row[1] for row in rows[1:]] == ["cheap", "expensive", "only"]


@pytest.mark.unit
def test_json_output_is_a_list_of_objects(recorder: Recorder, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    recorder.page = Page(items=(offer(),))
    cli.list_offers(output="json")

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["PROVIDER"] == "aws-main"
    assert payload[0]["ACCELERATOR"] == "h100"
    assert payload[0]["ON-DEMAND"] == "3"
