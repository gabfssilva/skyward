from __future__ import annotations

import pytest

from skyward.accelerators import Accelerator
from skyward.core.model import Cluster, Offer
from skyward.core.spec import Nodes, PoolSpec
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
        spec=PoolSpec(nodes=Nodes(desired=1), accelerator=Accelerator.from_name("A100"), region="CANADA-1"),
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


class TestMountPlan:
    @pytest.mark.asyncio()
    async def test_returns_fuse_plan_with_endpoint_and_credentials(self):
        from skyward.core.spec import Volume
        from skyward.providers.bootstrap.compose import resolve

        provider = HyperstackProvider(Hyperstack())
        provider._access_key = "ak_test123"
        provider._secret_key = "sk_secret456"

        cluster = _dummy_cluster()
        plan = await provider.mount_plan(cluster, (Volume(bucket="b", mount="/data"),))

        assert plan.deploy_hints == {}
        script = resolve(plan.bootstrap)
        assert _DEFAULT_STORAGE_ENDPOINT in script
        assert "ak_test123" in script
        assert "sk_secret456" in script
        assert "geesefs" in script

    @pytest.mark.asyncio()
    async def test_raises_without_keys(self):
        from skyward.core.spec import Volume

        provider = HyperstackProvider(Hyperstack())
        cluster = _dummy_cluster()

        with pytest.raises(RuntimeError, match="Access key not created"):
            await provider.mount_plan(cluster, (Volume(bucket="b", mount="/data"),))


