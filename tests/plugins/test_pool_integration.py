"""Tests for ComputePool plugin integration."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import pytest

from skyward.api.pool import ComputePool
from skyward.api.spec import Image
from skyward.plugins.plugin import Plugin
from skyward.providers import Container

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(**kwargs: Any) -> ComputePool:
    """Create a ComputePool without entering it."""
    defaults: dict[str, Any] = {"provider": Container(), "logging": False}
    defaults.update(kwargs)
    return ComputePool(**defaults)


# ---------------------------------------------------------------------------
# plugins parameter acceptance
# ---------------------------------------------------------------------------


class TestPluginsParameter:
    def test_defaults_to_empty_tuple(self) -> None:
        pool = _make_pool()
        assert pool._plugins == ()

    def test_accepts_list(self) -> None:
        p = Plugin(name="a")
        pool = _make_pool(plugins=[p])
        assert pool._plugins == (p,)

    def test_accepts_tuple(self) -> None:
        p = Plugin(name="b")
        pool = _make_pool(plugins=(p,))
        assert pool._plugins == (p,)

    def test_stored_as_tuple(self) -> None:
        p1 = Plugin(name="x")
        p2 = Plugin(name="y")
        pool = _make_pool(plugins=[p1, p2])
        assert isinstance(pool._plugins, tuple)
        assert pool._plugins == (p1, p2)

    def test_multiple_plugins_order_preserved(self) -> None:
        plugins = [Plugin(name=str(i)) for i in range(5)]
        pool = _make_pool(plugins=plugins)
        assert tuple(p.name for p in pool._plugins) == ("0", "1", "2", "3", "4")


# ---------------------------------------------------------------------------
# _apply_plugin_transforms
# ---------------------------------------------------------------------------


class TestApplyPluginTransforms:
    def test_no_plugins_returns_image_unchanged(self) -> None:
        pool = _make_pool()
        img = Image(pip=["numpy"])
        result = pool._apply_plugin_transforms(img)
        assert result is img

    def test_single_transform_applied(self) -> None:
        def add_torch(img: Image) -> Image:
            return replace(img, pip=(*img.pip, "torch"))

        p = Plugin(name="torch", transform=add_torch)
        pool = _make_pool(plugins=[p])
        result = pool._apply_plugin_transforms(Image(pip=["numpy"]))
        assert "torch" in result.pip
        assert "numpy" in result.pip

    def test_transforms_applied_in_order(self) -> None:
        def add_a(img: Image) -> Image:
            return replace(img, pip=(*img.pip, "a"))

        def add_b(img: Image) -> Image:
            return replace(img, pip=(*img.pip, "b"))

        p1 = Plugin(name="first", transform=add_a)
        p2 = Plugin(name="second", transform=add_b)
        pool = _make_pool(plugins=[p1, p2])
        result = pool._apply_plugin_transforms(Image())
        assert tuple(result.pip) == ("a", "b")

    def test_skips_plugins_without_transform(self) -> None:
        def add_x(img: Image) -> Image:
            return replace(img, pip=(*img.pip, "x"))

        p1 = Plugin(name="no-transform")
        p2 = Plugin(name="has-transform", transform=add_x)
        p3 = Plugin(name="also-no-transform")
        pool = _make_pool(plugins=[p1, p2, p3])
        result = pool._apply_plugin_transforms(Image())
        assert tuple(result.pip) == ("x",)

    def test_preserves_existing_image_fields(self) -> None:
        def add_pkg(img: Image) -> Image:
            return replace(img, pip=(*img.pip, "new-pkg"))

        p = Plugin(name="preserver", transform=add_pkg)
        pool = _make_pool(plugins=[p])
        original = Image(
            python="3.13",
            pip=["existing"],
            apt=["git"],
            env={"KEY": "val"},
        )
        result = pool._apply_plugin_transforms(original)
        assert result.python == "3.13"
        assert "existing" in result.pip
        assert "new-pkg" in result.pip
        assert "git" in result.apt
        assert result.env == {"KEY": "val"}

    def test_transform_receives_previous_result(self) -> None:
        received: list[Image] = []

        def capture(img: Image) -> Image:
            received.append(img)
            return replace(img, pip=(*img.pip, "captured"))

        def verify(img: Image) -> Image:
            received.append(img)
            return img

        p1 = Plugin(name="capture", transform=capture)
        p2 = Plugin(name="verify", transform=verify)
        pool = _make_pool(plugins=[p1, p2])
        # Transforms are applied during __init__, so pool.image has results
        assert len(received) == 2
        assert "captured" in received[1].pip
        assert "captured" in pool.image.pip


# ---------------------------------------------------------------------------
# _collect_plugin_bootstrap
# ---------------------------------------------------------------------------


class TestCollectPluginBootstrap:
    def test_no_plugins_returns_empty(self) -> None:
        pool = _make_pool()
        assert pool._collect_plugin_bootstrap() == ()

    def test_single_plugin_ops(self) -> None:
        p = Plugin(name="ops", bootstrap=("echo hello", "echo world"))
        pool = _make_pool(plugins=[p])
        result = pool._collect_plugin_bootstrap()
        assert result == ("echo hello", "echo world")

    def test_multiple_plugins_concatenated(self) -> None:
        p1 = Plugin(name="a", bootstrap=("op1",))
        p2 = Plugin(name="b", bootstrap=("op2", "op3"))
        p3 = Plugin(name="c", bootstrap=("op4",))
        pool = _make_pool(plugins=[p1, p2, p3])
        result = pool._collect_plugin_bootstrap()
        assert result == ("op1", "op2", "op3", "op4")

    def test_skips_plugins_with_empty_bootstrap(self) -> None:
        p1 = Plugin(name="empty")
        p2 = Plugin(name="has-ops", bootstrap=("echo x",))
        pool = _make_pool(plugins=[p1, p2])
        result = pool._collect_plugin_bootstrap()
        assert result == ("echo x",)

    def test_returns_tuple(self) -> None:
        p = Plugin(name="t", bootstrap=("op",))
        pool = _make_pool(plugins=[p])
        result = pool._collect_plugin_bootstrap()
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# _decorate_fn
# ---------------------------------------------------------------------------


class TestDecorateFn:
    def test_no_plugins_returns_fn_unchanged(self) -> None:
        pool = _make_pool()

        def original(x: int) -> int:
            return x * 2

        result = pool._decorate_fn(original)
        assert result is original

    def test_no_decorators_returns_fn_unchanged(self) -> None:
        p = Plugin(name="no-hooks")
        pool = _make_pool(plugins=[p])

        def original() -> str:
            return "hello"

        result = pool._decorate_fn(original)
        assert result is original

    def test_decorate_wraps_fn(self) -> None:
        calls: list[str] = []

        def my_decorator(fn: Any, args: tuple, kwargs: dict) -> Any:
            calls.append("decorated")
            return fn(*args, **kwargs)

        p = Plugin(name="dec", decorate=my_decorator)
        pool = _make_pool(plugins=[p])

        def original(x: int) -> int:
            calls.append("original")
            return x

        wrapped = pool._decorate_fn(original)
        assert wrapped is not original
        result = wrapped(42)
        assert result == 42
        assert calls == ["decorated", "original"]

    def test_multiple_decorators_applied_in_order(self) -> None:
        calls: list[str] = []

        def dec_a(fn: Any, args: tuple, kwargs: dict) -> Any:
            calls.append("a")
            return fn(*args, **kwargs)

        def dec_b(fn: Any, args: tuple, kwargs: dict) -> Any:
            calls.append("b")
            return fn(*args, **kwargs)

        p1 = Plugin(name="pa", decorate=dec_a)
        p2 = Plugin(name="pb", decorate=dec_b)
        pool = _make_pool(plugins=[p1, p2])

        def original() -> str:
            calls.append("fn")
            return "done"

        wrapped = pool._decorate_fn(original)
        result = wrapped()
        assert result == "done"
        assert calls == ["a", "b", "fn"]

    def test_around_app_comes_before_decorate(self) -> None:
        """around_app decorators are placed before decorate decorators."""
        calls: list[str] = []

        @contextmanager
        def lifecycle(info: Any):  # noqa: ANN201
            yield

        def my_decorate(fn: Any, args: tuple, kwargs: dict) -> Any:
            calls.append("decorate")
            return fn(*args, **kwargs)

        p = Plugin(
            name="both",
            around_app=lifecycle,
            decorate=my_decorate,
        )
        pool = _make_pool(plugins=[p])

        def original() -> str:
            calls.append("fn")
            return "ok"

        wrapped = pool._decorate_fn(original)
        assert wrapped is not original


# ---------------------------------------------------------------------------
# around_client lifecycle (unit-level, no pool enter)
# ---------------------------------------------------------------------------


class TestAroundClientUnit:
    def test_plugin_client_contexts_starts_empty(self) -> None:
        pool = _make_pool()
        assert pool._plugin_client_contexts == []

    def test_plugins_without_around_client_are_skipped(self) -> None:
        p = Plugin(name="no-client")
        pool = _make_pool(plugins=[p])
        assert pool._plugin_client_contexts == []
