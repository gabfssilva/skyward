from __future__ import annotations

import pytest

from skyward.actors.spy_adapter import pool_name_from_path

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


# ── pool_name_from_path ────────────────────────────────────────


class TestPoolNameFromPath:
    def test_extracts_from_session_pool_path(self):
        assert pool_name_from_path("/session/pool-train/node-0") == "train"

    def test_extracts_from_pool_only_path(self):
        assert pool_name_from_path("/session/pool-my-pool") == "my-pool"

    def test_extracts_with_deep_nested_path(self):
        assert pool_name_from_path("/session/pool-finetune/node-3/transport") == "finetune"

    def test_returns_none_without_pool(self):
        assert pool_name_from_path("/session/other-actor") is None

    def test_returns_none_for_root_path(self):
        assert pool_name_from_path("/") is None

    def test_returns_none_for_empty_string(self):
        assert pool_name_from_path("") is None

    def test_extracts_numeric_pool_name(self):
        assert pool_name_from_path("/session/pool-42/node-0") == "42"

    def test_extracts_pool_name_with_underscores(self):
        assert pool_name_from_path("/session/pool-my_pool/node-1") == "my_pool"

    def test_extracts_without_leading_slash(self):
        assert pool_name_from_path("session/pool-train/node-0") == "train"

    def test_extracts_without_leading_slash_pool_only(self):
        assert pool_name_from_path("session/pool-gpu") == "gpu"


# ── projection_actor ───────────────────────────────────────────


class TestProjectionActorExists:
    def test_projection_actor_is_callable(self):
        from skyward.actors.projection import projection_actor

        assert callable(projection_actor)

    def test_projection_actor_returns_behavior(self):
        from skyward.actors.projection import projection_actor
        from skyward.api.projection import SessionProjection

        projection = SessionProjection()
        behavior = projection_actor(projection)
        assert behavior is not None
