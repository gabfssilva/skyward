"""sky providers — what a registered account's row says about it."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytest.importorskip("cyclopts", reason="the sky CLI needs: pip install 'skyward[cli]'")

from skyward.cli import providers_app
from skyward.cli.providers import _check, _verdict
from skyward.protocol.schemas import Error, Provider

pytestmark = pytest.mark.unit

FETCHED = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
FAILED = Error(code="unsupported_provider", message="bad key", retryable=False)


def provider(
    *,
    offers_count: int = 0,
    offers_fetched_at: datetime | None = None,
    last_error: Error | None = None,
) -> Provider:
    return Provider(
        id="prv_1",
        name="prod",
        kind="aws",
        config={},
        offers_ttl_seconds=3600,
        created_at=FETCHED,
        offers_fetched_at=offers_fetched_at,
        offers_count=offers_count,
        last_error=last_error,
    )


def test_a_provider_that_never_fetched_is_unused():
    assert _verdict(provider()) == ("unused", "credentials never exercised")


def test_a_provider_with_offers_is_ok():
    assert _verdict(provider(offers_fetched_at=FETCHED, offers_count=12)) == ("ok", None)


def test_the_last_error_is_the_detail():
    assert _verdict(provider(last_error=FAILED)) == ("error", "bad key")


def test_an_error_wins_over_a_past_fetch():
    assert _verdict(provider(offers_fetched_at=FETCHED, last_error=FAILED))[0] == "error"


def test_the_check_row_is_in_column_order():
    assert _check(provider(offers_fetched_at=FETCHED, offers_count=3)) == ("prod", "aws", "ok", 3, FETCHED, None)


@pytest.mark.parametrize("name", ["list", "check"])
def test_the_commands_are_registered(name):
    assert providers_app[name] is not None
