"""Every adapter, against the real API it claims to speak.

Skipped unless the credential is in the environment. This is the only place the
environment is allowed to name a credential: the adapters themselves receive
theirs through `create()` and never go looking.
"""

import configparser
import os
from pathlib import Path

import pytest

from skyward2.protocol.accelerators import CATALOG
from skyward2.providers import REGISTRY

CREDENTIALS = {
    "aws": {"access_key_id": "AWS_ACCESS_KEY_ID", "secret_access_key": "AWS_SECRET_ACCESS_KEY"},
    "gcp": {"service_account_json": "GCP_SERVICE_ACCOUNT_JSON"},
    "hyperstack": {"api_key": "HYPERSTACK_API_KEY"},
    "jarvislabs": {"api_key": "JL_API_KEY"},
    "lambda": {"api_key": "LAMBDA_API_KEY"},
    "massed_compute": {"api_key": "MASSED_API_KEY"},
    "novita": {"api_key": "NOVITA_API_KEY"},
    "runpod": {"api_key": "RUNPOD_API_KEY"},
    "scaleway": {"secret_key": "SCW_SECRET_KEY"},
    "tensordock": {"api_token": "TENSORDOCK_API_TOKEN"},
    "vastai": {"api_key": "VAST_API_KEY"},
    "verda": {"client_id": "VERDA_CLIENT_ID", "client_secret": "VERDA_CLIENT_SECRET"},
    "vultr": {"api_key": "VULTR_API_KEY"},
}

PRICED_KINDS = sorted(set(CREDENTIALS) - {"tensordock"})

CONFIG = {"aws": {"regions": ["us-east-1"]}}

AWS_PROFILE = Path.home() / ".aws" / "credentials"


def credentials_for(kind: str) -> dict[str, str]:
    env = CREDENTIALS[kind]
    if kind == "aws" and not os.environ.get("AWS_ACCESS_KEY_ID"):
        return _aws_profile()

    missing = [var for var in env.values() if not os.environ.get(var)]
    if missing:
        pytest.skip(f"{kind}: no credential in {', '.join(missing)}")
    return {field: os.environ[var] for field, var in env.items()}


def _aws_profile() -> dict[str, str]:
    """Read the developer's AWS profile — in the test, never in the adapter.

    The adapter is forbidden from touching a credentials file: that is what lets
    two AWS accounts coexist in one process. So the test does the reading and
    hands the keys in through `create()`, exactly as the controller will.
    """
    if not AWS_PROFILE.exists():
        pytest.skip("aws: no AWS_ACCESS_KEY_ID and no ~/.aws/credentials")

    profile = configparser.ConfigParser()
    profile.read(AWS_PROFILE)
    if "default" not in profile:
        pytest.skip("aws: no default profile")

    return {
        "access_key_id": profile["default"]["aws_access_key_id"],
        "secret_access_key": profile["default"]["aws_secret_access_key"],
    }


def test_every_kind_declares_where_its_credential_comes_from():
    assert set(CREDENTIALS) | {"fake"} == set(REGISTRY)


@pytest.mark.sanity
@pytest.mark.parametrize("kind", PRICED_KINDS)
async def test_the_live_catalog_is_priced_and_plausible(kind):
    adapter = REGISTRY[kind].create("prv_live", "live", credentials_for(kind), CONFIG.get(kind, {}))

    offers = [offer async for offer in adapter.offers()]
    assert offers, f"{kind} returned an empty catalog"

    assert len({offer.id for offer in offers}) == len(offers), "offer ids must be unique within a provider"
    assert all(offer.kind == kind and offer.provider_id == "prv_live" for offer in offers)
    assert all(offer.expires_at > offer.fetched_at for offer in offers)

    accelerated = [offer for offer in offers if offer.accelerator_count > 0]
    assert accelerated, f"{kind} returned no accelerated offers"
    assert all(offer.accelerator for offer in accelerated), "an accelerated offer must name its accelerator"

    unknown = {offer.accelerator for offer in accelerated} - set(CATALOG)
    assert not unknown, f"{kind} speaks accelerators the catalog has never heard of: {sorted(unknown)}"

    assert all(offer.vram for offer in accelerated), "an accelerated offer must say how much VRAM a card has"
    assert all(1 <= offer.vram <= 400 for offer in accelerated), "vram is per card, in GB"

    prices = [price for offer in accelerated for price in (offer.spot_price, offer.on_demand_price) if price is not None]
    assert prices, f"{kind} priced nothing"
    assert all(0.01 < price < 500 for price in prices), f"{kind} prices are not hourly dollars: {sorted(prices)[:3]}"


@pytest.mark.sanity
async def test_tensordock_answers_even_though_its_marketplace_is_empty(kind="tensordock"):
    """TensorDock was absorbed by Voltage Park; the API is up, the supply is gone.

    Kept as its own case so an empty catalog reads as the documented state of the
    world rather than as a broken adapter.
    """
    adapter = REGISTRY[kind].create("prv_live", "live", credentials_for(kind), CONFIG.get(kind, {}))
    assert [offer async for offer in adapter.offers()] == []
