"""E2E test for plugin hooks (transform, decorate, around_app) via Container provider."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import pytest

import skyward as sky
from skyward.api.pool import ComputePool
from skyward.api.spec import Image
from skyward.plugins.plugin import Plugin

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180), pytest.mark.xdist_group("plugin-e2e")]

def plugin_test():
    def _decorate(fn: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import os
            os.environ["DECORATE_MARKER"] = "applied"
            return fn(*args, **kwargs)

        return wrapper

    def _around_app_lifecycle(info: Any) -> Any:
        @contextmanager
        def cm():
            import os

            os.environ["AROUND_APP_MARKER"] = "entered"
            try:
                yield
            finally:
                os.environ.pop("AROUND_APP_MARKER", None)

        return cm()

    return Plugin(
        name="test-hooks",
        transform=lambda img, c: replace(img, env={**img.env, "TRANSFORM_MARKER": "applied"}),
        decorate=_decorate,
        around_app=_around_app_lifecycle,
    )

@pytest.fixture(scope="module")
def plugin_pool():
    with sky.App(console=False), ComputePool(
        provider=sky.Container(container_prefix="skyward-plugin-e2e"),
        nodes=1,
        plugins=[plugin_test()],
    ) as p:
        yield p


class TestPluginHooksE2E:
    def test_transform_sets_env_var(self, plugin_pool: ComputePool) -> None:
        @sky.compute
        def check():
            import os

            return os.environ.get("TRANSFORM_MARKER")

        assert (check() >> plugin_pool) == "applied"

    def test_decorate_wraps_execution(self, plugin_pool: ComputePool) -> None:
        @sky.compute
        def check():
            import os

            return os.environ.get("DECORATE_MARKER")

        assert (check() >> plugin_pool) == "applied"

    def test_around_app_context_entered(self, plugin_pool: ComputePool) -> None:
        @sky.compute
        def check():
            import os

            return os.environ.get("AROUND_APP_MARKER")

        assert (check() >> plugin_pool) == "entered"

    def test_all_hooks_active(self, plugin_pool: ComputePool) -> None:
        @sky.compute
        def check_all():
            import os

            return {
                "transform": os.environ.get("TRANSFORM_MARKER"),
                "decorate": os.environ.get("DECORATE_MARKER"),
                "around_app": os.environ.get("AROUND_APP_MARKER"),
            }

        result = check_all() >> plugin_pool
        assert result == {
            "transform": "applied",
            "decorate": "applied",
            "around_app": "entered",
        }
