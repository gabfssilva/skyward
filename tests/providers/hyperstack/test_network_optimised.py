"""Tests for Hyperstack network_optimised config option."""

from __future__ import annotations

import pytest

from skyward.providers.hyperstack.config import Hyperstack
from skyward.providers.hyperstack.provider import (
    _constrain_region_for_network,
    _to_offer,
)
from skyward.providers.hyperstack.types import EnvironmentFeatures

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]

_DEFAULT_REGIONS = ("CANADA-1", "US-1")


class TestConstrainRegionForNetwork:
    def test_none_region_returns_all_supported(self):
        result = _constrain_region_for_network(None, _DEFAULT_REGIONS)
        assert isinstance(result, tuple)
        assert set(result) == set(_DEFAULT_REGIONS)

    def test_canada_passes(self):
        result = _constrain_region_for_network("CANADA-1", _DEFAULT_REGIONS)
        assert result == "CANADA-1"

    def test_us1_passes(self):
        result = _constrain_region_for_network("US-1", _DEFAULT_REGIONS)
        assert result == "US-1"

    def test_case_insensitive(self):
        result = _constrain_region_for_network("canada-1", _DEFAULT_REGIONS)
        assert result == "canada-1"

    def test_unsupported_region_raises(self):
        with pytest.raises(ValueError, match="network_optimised=True requires"):
            _constrain_region_for_network("NORWAY-1", _DEFAULT_REGIONS)

    def test_tuple_all_supported_passes(self):
        result = _constrain_region_for_network(("CANADA-1", "US-1"), _DEFAULT_REGIONS)
        assert result == ("CANADA-1", "US-1")

    def test_tuple_with_unsupported_raises(self):
        with pytest.raises(ValueError, match="network_optimised=True requires"):
            _constrain_region_for_network(("CANADA-1", "NORWAY-1"), _DEFAULT_REGIONS)

    def test_tuple_all_unsupported_raises(self):
        with pytest.raises(ValueError, match="network_optimised=True requires"):
            _constrain_region_for_network(("NORWAY-1",), _DEFAULT_REGIONS)

    def test_custom_regions(self):
        result = _constrain_region_for_network("EU-1", ("EU-1", "US-2"))
        assert result == "EU-1"


class TestNetworkOptimisedConfig:
    def test_default_false(self):
        config = Hyperstack()
        assert config.network_optimised is False

    def test_explicit_true(self):
        config = Hyperstack(network_optimised=True)
        assert config.network_optimised is True

    def test_default_regions(self):
        config = Hyperstack()
        assert config.network_optimised_regions == ("CANADA-1", "US-1")

    def test_custom_regions(self):
        config = Hyperstack(network_optimised_regions=("EU-1", "US-2"))
        assert config.network_optimised_regions == ("EU-1", "US-2")

    def test_frozen(self):
        config = Hyperstack(network_optimised=True)
        with pytest.raises(AttributeError):
            config.network_optimised = False  # type: ignore[misc]


class TestObjectStorageConfig:
    def test_default_region(self):
        config = Hyperstack()
        assert config.object_storage_region == "CANADA-1"

    def test_default_endpoint(self):
        config = Hyperstack()
        assert config.object_storage_endpoint == "https://ca1.obj.nexgencloud.io"

    def test_custom_region(self):
        config = Hyperstack(object_storage_region="US-1")
        assert config.object_storage_region == "US-1"

    def test_custom_endpoint(self):
        config = Hyperstack(object_storage_endpoint="https://us1.obj.nexgencloud.io")
        assert config.object_storage_endpoint == "https://us1.obj.nexgencloud.io"


class TestOfferNetworkOptimisedField:
    _net_regions = frozenset({"CANADA-1", "US-1"})

    def test_canada_offer_is_network_optimised(self):
        flavor = {
            "id": 1, "name": "h100-80g", "cpu": 26,
            "ram": 200.0, "disk": 500, "gpu": "H100",
            "gpu_count": 1, "region_name": "CANADA-1",
        }
        offer = _to_offer(flavor, {"H100": 3.0}, self._net_regions)
        assert offer.specific["network_optimised"] is True

    def test_us1_offer_is_network_optimised(self):
        flavor = {
            "id": 2, "name": "a100-80g", "cpu": 26,
            "ram": 200.0, "disk": 500, "gpu": "A100",
            "gpu_count": 1, "region_name": "US-1",
        }
        offer = _to_offer(flavor, {"A100": 2.0}, self._net_regions)
        assert offer.specific["network_optimised"] is True

    def test_norway_offer_is_not_network_optimised(self):
        flavor = {
            "id": 3, "name": "a100-80g", "cpu": 26,
            "ram": 200.0, "disk": 500, "gpu": "A100",
            "gpu_count": 1, "region_name": "NORWAY-1",
        }
        offer = _to_offer(flavor, {"A100": 2.0}, self._net_regions)
        assert offer.specific["network_optimised"] is False


class TestEnvironmentFeaturesTypedDict:
    def test_full_features(self):
        features: EnvironmentFeatures = {
            "network_optimised": True,
            "green_status": "GREEN",
        }
        assert features["network_optimised"] is True
        assert features["green_status"] == "GREEN"

    def test_empty_features(self):
        features: EnvironmentFeatures = {}
        assert features.get("network_optimised", False) is False


class TestVolumeAndNetworkInteraction:
    def test_volumes_canada_then_network_passes(self):
        result = _constrain_region_for_network("CANADA-1", _DEFAULT_REGIONS)
        assert result == "CANADA-1"
