from __future__ import annotations

import pytest

import skyward as sky
from skyward import ComputePool

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(120), pytest.mark.xdist_group("lifecycle")]


class TestPoolLifecycle:
    def test_pool_provisions_and_destroys(self):
        with ComputePool(
            provider=sky.Container(),
            nodes=1,
        ) as pool:
            @sky.compute
            def ping():
                return "pong"

            assert ping() >> pool == "pong"

    def test_pool_is_active_inside_context(self):
        with ComputePool(
            provider=sky.Container(),
            nodes=1,
        ) as pool:
            assert pool.is_active

    def test_pool_multi_node_provisions(self):
        with ComputePool(
            provider=sky.Container(),
            nodes=3,
        ) as pool:
            @sky.compute
            def whoami():
                return sky.instance_info().node  # pyright: ignore[reportOptionalMemberAccess]

            nodes = whoami() @ pool
            assert sorted(nodes) == [0, 1, 2]

    def test_multiple_tasks_on_same_pool(self):
        with ComputePool(
            provider=sky.Container(),
            nodes=1,
        ) as pool:
            @sky.compute
            def double(x):
                return x * 2

            for i in range(5):
                assert double(i) >> pool == i * 2

    def test_pool_with_custom_image(self):
        from skyward import Image

        with ComputePool(
            provider=sky.Container(),
            nodes=1,
            image=Image(pip=["requests"]),
        ) as pool:
            @sky.compute
            def check():
                import requests

                return bool(requests.__version__)

            assert check() >> pool is True

    def test_sky_singleton_resolves_active_pool(self):
        with ComputePool(
            provider=sky.Container(),
            nodes=1,
        ) as _:
            @sky.compute
            def ping():
                return "pong"

            result = ping() >> sky
            assert result == "pong"
