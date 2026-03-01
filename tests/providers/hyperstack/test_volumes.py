from __future__ import annotations

import asyncio

import pytest

from skyward.api.model import Cluster, Offer
from skyward.api.spec import PoolSpec
from skyward.providers.hyperstack.config import Hyperstack
from skyward.providers.hyperstack.provider import (
    HyperstackProvider,
    HyperstackSpecific,
    _constrain_region_for_volumes,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]

_DEFAULT_STORAGE_REGION = "CANADA-1"
_DEFAULT_STORAGE_ENDPOINT = "https://ca1.obj.nexgencloud.io"


def _dummy_cluster() -> Cluster[HyperstackSpecific]:
    return Cluster(
        id="test-cluster",
        status="ready",
        spec=PoolSpec(nodes=1, accelerator="A100", region="CANADA-1"),
        offer=Offer(
            id="1",
            instance_type=None,
            spot_price=None,
            on_demand_price=1.0,
            billing_unit="hour",
            specific={"flavor_name": "a100", "region": "CANADA-1"},
        ),
        ssh_key_path="/tmp/key",
        ssh_user="ubuntu",
        use_sudo=True,
        shutdown_command="sudo shutdown -h now",
        specific=HyperstackSpecific(
            environment_name="test-env",
            environment_id=1,
            key_name="test-key",
            flavor_name="a100",
            image_name="ubuntu",
            region="CANADA-1",
        ),
    )


class TestConstrainRegionForVolumes:
    def test_none_region_returns_canada(self):
        result = _constrain_region_for_volumes(None, _DEFAULT_STORAGE_REGION)
        assert result == _DEFAULT_STORAGE_REGION

    def test_canada_region_passes_through(self):
        result = _constrain_region_for_volumes("CANADA-1", _DEFAULT_STORAGE_REGION)
        assert result == "CANADA-1"

    def test_canada_region_case_insensitive(self):
        result = _constrain_region_for_volumes("canada-1", _DEFAULT_STORAGE_REGION)
        assert result == "canada-1"

    def test_wrong_region_raises(self):
        with pytest.raises(ValueError, match="Volumes require region CANADA-1"):
            _constrain_region_for_volumes("NORWAY-1", _DEFAULT_STORAGE_REGION)

    def test_tuple_with_canada_passes(self):
        result = _constrain_region_for_volumes(("CANADA-1", "NORWAY-1"), _DEFAULT_STORAGE_REGION)
        assert result == ("CANADA-1", "NORWAY-1")

    def test_tuple_without_canada_raises(self):
        with pytest.raises(ValueError, match="Volumes require region CANADA-1"):
            _constrain_region_for_volumes(("NORWAY-1", "US-WEST-1"), _DEFAULT_STORAGE_REGION)

    def test_custom_storage_region(self):
        result = _constrain_region_for_volumes("US-1", "US-1")
        assert result == "US-1"


class TestMountEndpoint:
    def test_returns_endpoint_with_credentials(self):
        provider = HyperstackProvider(Hyperstack())
        provider._access_key = "ak_test123"
        provider._secret_key = "sk_secret456"

        cluster = _dummy_cluster()
        endpoint = asyncio.get_event_loop().run_until_complete(
            provider.mount_endpoint(cluster),
        )
        assert endpoint.endpoint == _DEFAULT_STORAGE_ENDPOINT
        assert endpoint.access_key == "ak_test123"
        assert endpoint.secret_key == "sk_secret456"
        assert endpoint.path_style is True

    def test_raises_without_keys(self):
        provider = HyperstackProvider(Hyperstack())
        cluster = _dummy_cluster()

        with pytest.raises(RuntimeError, match="Access key not created"):
            asyncio.get_event_loop().run_until_complete(
                provider.mount_endpoint(cluster),
            )


class TestProviderAccessKeyState:
    def test_initial_state_is_none(self):
        provider = HyperstackProvider(Hyperstack())
        assert provider._access_key is None
        assert provider._secret_key is None
        assert provider._access_key_id is None
