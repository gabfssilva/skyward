from __future__ import annotations

import pytest

from skyward.api.spec import Image, Nodes, PoolSpec, Spec
from skyward.providers.aws.config import AWS

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestDiskGbSpec:
    def test_spec_disk_gb_default_none(self) -> None:
        s = Spec(provider=AWS())
        assert s.disk_gb is None

    def test_spec_disk_gb_explicit(self) -> None:
        s = Spec(provider=AWS(), disk_gb=500)
        assert s.disk_gb == 500

    def test_pool_spec_disk_gb_default_none(self) -> None:
        ps = PoolSpec(nodes=Nodes(min=1), accelerator=None, region="us-east-1")
        assert ps.disk_gb is None

    def test_pool_spec_disk_gb_explicit(self) -> None:
        ps = PoolSpec(
            nodes=Nodes(min=1), accelerator=None,
            region="us-east-1", disk_gb=200,
        )
        assert ps.disk_gb == 200

    def test_pool_spec_built_with_disk_gb_from_spec(self) -> None:
        s = Spec(provider=AWS(), disk_gb=300, nodes=2)
        ps = PoolSpec(
            nodes=Nodes(min=2),
            accelerator=s.accelerator,
            region="us-east-1",
            disk_gb=s.disk_gb,
        )
        assert ps.disk_gb == 300

    def test_pool_spec_built_without_disk_gb(self) -> None:
        s = Spec(provider=AWS())
        ps = PoolSpec(
            nodes=Nodes(min=1),
            accelerator=s.accelerator,
            region="us-east-1",
            disk_gb=s.disk_gb,
        )
        assert ps.disk_gb is None
