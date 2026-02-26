from __future__ import annotations

import pytest

import skyward as sky

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180), pytest.mark.xdist_group("keras")]


class TestKerasPlugin:
    def test_backend_is_set(self, keras_plugin_pool) -> None:
        @sky.compute
        def check_env():
            import os

            return os.environ.get("KERAS_BACKEND")

        results = check_env() @ keras_plugin_pool
        assert all(r == "torch" for r in results)

    def test_keras_import_works(self, keras_plugin_pool) -> None:
        @sky.compute
        def check_backend():
            import keras

            return keras.backend.backend()

        results = check_backend() @ keras_plugin_pool
        assert all(r == "torch" for r in results)
