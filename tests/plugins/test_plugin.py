"""Tests for the Plugin core type, builder methods, and helper functions."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


# ---------------------------------------------------------------------------
# Plugin creation
# ---------------------------------------------------------------------------


class TestPluginCreation:
    def test_minimal_plugin(self) -> None:
        from skyward.plugins.plugin import Plugin

        p = Plugin(name="test")
        assert p.name == "test"
        assert p.transform is None
        assert p.bootstrap is None
        assert p.decorate is None
        assert p.around_app is None
        assert p.around_client is None

    def test_create_factory(self) -> None:
        from skyward.plugins.plugin import Plugin

        p = Plugin.create("my-plugin")
        assert p.name == "my-plugin"
        assert p.transform is None
        assert p.bootstrap is None

    def test_direct_construction_with_all_fields(self) -> None:
        from skyward.plugins.plugin import Plugin

        transform = lambda img, cluster: img  # noqa: E731
        bootstrap_factory = lambda cluster: ("echo hello",)  # noqa: E731
        decorate = lambda fn: fn  # noqa: E731
        around_app = MagicMock()
        around_client = MagicMock()

        p = Plugin(
            name="full",
            transform=transform,
            bootstrap=bootstrap_factory,
            decorate=decorate,
            around_app=around_app,
            around_client=around_client,
        )
        assert p.name == "full"
        assert p.transform is transform
        assert p.bootstrap is bootstrap_factory
        assert p.decorate is decorate
        assert p.around_app is around_app
        assert p.around_client is around_client

    def test_frozen(self) -> None:
        from skyward.plugins.plugin import Plugin

        p = Plugin(name="frozen")
        with pytest.raises(FrozenInstanceError):
            p.name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Builder chain
# ---------------------------------------------------------------------------


class TestBuilderChain:
    def test_with_image_transform(self) -> None:
        from skyward.plugins.plugin import Plugin

        transform = lambda img, cluster: img  # noqa: E731
        p = Plugin.create("t").with_image_transform(transform)
        assert p.transform is transform
        assert p.name == "t"

    def test_with_bootstrap(self) -> None:
        from skyward.plugins.plugin import Plugin

        factory = lambda cluster: ("echo 1",)  # noqa: E731
        p = Plugin.create("b").with_bootstrap(factory)
        assert p.bootstrap is factory

    def test_with_decorator(self) -> None:
        from skyward.plugins.plugin import Plugin

        dec = lambda fn: fn  # noqa: E731
        p = Plugin.create("d").with_decorator(dec)
        assert p.decorate is dec

    def test_with_around_app(self) -> None:
        from skyward.plugins.plugin import Plugin

        hook = MagicMock()
        p = Plugin.create("a").with_around_app(hook)
        assert p.around_app is hook

    def test_with_around_client(self) -> None:
        from skyward.plugins.plugin import Plugin

        hook = MagicMock()
        p = Plugin.create("c").with_around_client(hook)
        assert p.around_client is hook

    def test_builder_returns_new_instance(self) -> None:
        from skyward.plugins.plugin import Plugin

        original = Plugin.create("immutable")
        with_transform = original.with_image_transform(lambda img, cluster: img)
        with_bootstrap = original.with_bootstrap(lambda cluster: ("echo x",))
        with_decorator = original.with_decorator(lambda fn: fn)

        assert original.transform is None
        assert original.bootstrap is None
        assert original.decorate is None
        assert with_transform is not original
        assert with_bootstrap is not original
        assert with_decorator is not original

    def test_full_builder_chain(self) -> None:
        from skyward.plugins.plugin import Plugin

        transform = lambda img, cluster: img  # noqa: E731
        bootstrap_factory = lambda cluster: ("echo setup",)  # noqa: E731
        dec = lambda fn: fn  # noqa: E731
        app_hook = MagicMock()
        client_hook = MagicMock()

        p = (
            Plugin.create("chained")
            .with_image_transform(transform)
            .with_bootstrap(bootstrap_factory)
            .with_decorator(dec)
            .with_around_app(app_hook)
            .with_around_client(client_hook)
        )
        assert p.name == "chained"
        assert p.transform is transform
        assert p.bootstrap is bootstrap_factory
        assert p.decorate is dec
        assert p.around_app is app_hook
        assert p.around_client is client_hook


# ---------------------------------------------------------------------------
# Transform composition
# ---------------------------------------------------------------------------


class TestTransformComposition:
    def test_single_transform(self) -> None:
        from skyward.plugins.plugin import Plugin

        calls: list[str] = []

        def transform(img: Any, cluster: Any) -> Any:
            calls.append("transformed")
            return img

        p = Plugin.create("t").with_image_transform(transform)
        assert p.transform is not None

        result = p.transform("image-stub", MagicMock())  # type: ignore[arg-type]
        assert calls == ["transformed"]
        assert result == "image-stub"

    def test_transform_can_modify_image(self) -> None:
        from dataclasses import dataclass, replace

        from skyward.plugins.plugin import Plugin

        @dataclass(frozen=True)
        class FakeImage:
            pip: tuple[str, ...] = ()

        def add_torch(img: FakeImage, cluster: Any) -> FakeImage:
            return replace(img, pip=(*img.pip, "torch"))

        p = Plugin.create("torch").with_image_transform(add_torch)  # type: ignore[arg-type]
        original = FakeImage(pip=("numpy",))
        assert p.transform is not None
        result = p.transform(original, MagicMock())  # type: ignore[arg-type]
        assert result.pip == ("numpy", "torch")
        assert original.pip == ("numpy",)

    def test_chained_transforms_compose(self) -> None:
        from dataclasses import dataclass, replace

        from skyward.plugins.plugin import Plugin

        @dataclass(frozen=True)
        class FakeImage:
            pip: tuple[str, ...] = ()

        def add_a(img: FakeImage, cluster: Any) -> FakeImage:
            return replace(img, pip=(*img.pip, "a"))

        def add_b(img: FakeImage, cluster: Any) -> FakeImage:
            return replace(img, pip=(*img.pip, "b"))

        p1 = Plugin.create("p1").with_image_transform(add_a)  # type: ignore[arg-type]
        p2 = Plugin.create("p2").with_image_transform(add_b)  # type: ignore[arg-type]

        cluster = MagicMock()
        img: Any = FakeImage()
        for plugin in [p1, p2]:
            if plugin.transform:
                img = plugin.transform(img, cluster)

        assert img.pip == ("a", "b")


# ---------------------------------------------------------------------------
# Decorator chaining (chain_decorators)
# ---------------------------------------------------------------------------


class TestChainDecorators:
    def test_empty_decorators_returns_fn(self) -> None:
        from skyward.plugins.plugin import chain_decorators

        def original(x: int) -> int:
            return x * 2

        wrapped = chain_decorators(original, [])
        assert wrapped is original

    def test_single_decorator(self) -> None:
        from skyward.plugins.plugin import chain_decorators

        calls: list[str] = []

        def original(x: int) -> int:
            calls.append("original")
            return x * 2

        def dec(fn: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                calls.append("decorator")
                return fn(*args, **kwargs)
            return wrapper

        wrapped = chain_decorators(original, [dec])
        result = wrapped(5)
        assert result == 10
        assert calls == ["decorator", "original"]

    def test_multiple_decorators_order(self) -> None:
        from skyward.plugins.plugin import chain_decorators

        calls: list[str] = []

        def original() -> str:
            calls.append("original")
            return "done"

        def dec_a(fn: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                calls.append("a-before")
                result = fn(*args, **kwargs)
                calls.append("a-after")
                return result
            return wrapper

        def dec_b(fn: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                calls.append("b-before")
                result = fn(*args, **kwargs)
                calls.append("b-after")
                return result
            return wrapper

        wrapped = chain_decorators(original, [dec_a, dec_b])
        result = wrapped()
        assert result == "done"
        assert calls == ["a-before", "b-before", "original", "b-after", "a-after"]

    def test_decorator_can_modify_args(self) -> None:
        from skyward.plugins.plugin import chain_decorators

        def original(x: int) -> int:
            return x

        def double_arg(fn: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return fn(args[0] * 2, **kwargs)
            return wrapper

        wrapped = chain_decorators(original, [double_arg])
        assert wrapped(5) == 10

    def test_decorator_can_modify_kwargs(self) -> None:
        from skyward.plugins.plugin import chain_decorators

        def original(x: int = 0) -> int:
            return x

        def set_kwarg(fn: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return fn(*args, x=42)
            return wrapper

        wrapped = chain_decorators(original, [set_kwarg])
        assert wrapped() == 42

    def test_decorator_can_short_circuit(self) -> None:
        from skyward.plugins.plugin import chain_decorators

        calls: list[str] = []

        def original() -> str:
            calls.append("original")
            return "original"

        def shortcircuit(fn: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return "intercepted"
            return wrapper

        wrapped = chain_decorators(original, [shortcircuit])
        assert wrapped() == "intercepted"
        assert calls == []

    def test_three_decorators(self) -> None:
        from skyward.plugins.plugin import chain_decorators

        calls: list[int] = []

        def original(x: int) -> int:
            return x

        def dec1(fn: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                calls.append(1)
                return fn(*args, **kwargs)
            return wrapper

        def dec2(fn: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                calls.append(2)
                return fn(*args, **kwargs)
            return wrapper

        def dec3(fn: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                calls.append(3)
                return fn(*args, **kwargs)
            return wrapper

        wrapped = chain_decorators(original, [dec1, dec2, dec3])
        assert wrapped(7) == 7
        assert calls == [1, 2, 3]

    def test_preserves_return_value(self) -> None:
        from skyward.plugins.plugin import chain_decorators

        def original() -> dict:
            return {"key": "value"}

        def passthrough(fn: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return fn(*args, **kwargs)
            return wrapper

        wrapped = chain_decorators(original, [passthrough, passthrough])
        assert wrapped() == {"key": "value"}


# ---------------------------------------------------------------------------
# around_app lifecycle (ensure_around_app integration)
# ---------------------------------------------------------------------------


class TestAroundAppLifecycle:
    def test_ensure_around_app_enters_context(self) -> None:
        from skyward.plugins.state import ensure_around_app, reset

        entered = False

        @contextmanager
        def lifecycle(info: Any):  # noqa: ANN201
            nonlocal entered
            entered = True
            yield

        reset()
        ensure_around_app("test", lifecycle, MagicMock())
        assert entered

    def test_ensure_around_app_is_idempotent(self) -> None:
        from skyward.plugins.state import ensure_around_app, reset

        call_count = 0

        @contextmanager
        def lifecycle(info: Any):  # noqa: ANN201
            nonlocal call_count
            call_count += 1
            yield

        reset()
        info = MagicMock()
        ensure_around_app("test", lifecycle, info)
        ensure_around_app("test", lifecycle, info)
        assert call_count == 1

    def test_multiple_plugins_all_entered(self) -> None:
        from skyward.plugins.state import ensure_around_app, is_setup, reset

        @contextmanager
        def lifecycle_a(info: Any):  # noqa: ANN201
            yield

        @contextmanager
        def lifecycle_b(info: Any):  # noqa: ANN201
            yield

        reset()
        info = MagicMock()
        ensure_around_app("plugin-a", lifecycle_a, info)
        ensure_around_app("plugin-b", lifecycle_b, info)
        assert is_setup("plugin-a")
        assert is_setup("plugin-b")


# ---------------------------------------------------------------------------
# Type alias accessibility
# ---------------------------------------------------------------------------


class TestTypeAliases:
    def test_image_transform_alias_exists(self) -> None:
        from skyward.plugins import plugin

        assert hasattr(plugin, "ImageTransform")

    def test_task_decorator_alias_exists(self) -> None:
        from skyward.plugins import plugin

        assert hasattr(plugin, "TaskDecorator")

    def test_app_lifecycle_alias_exists(self) -> None:
        from skyward.plugins import plugin

        assert hasattr(plugin, "AppLifecycle")

    def test_client_lifecycle_alias_exists(self) -> None:
        from skyward.plugins import plugin

        assert hasattr(plugin, "ClientLifecycle")


# ---------------------------------------------------------------------------
# Package __init__ re-export
# ---------------------------------------------------------------------------


class TestPackageExport:
    def test_plugin_importable_from_package(self) -> None:
        from skyward.plugins import Plugin

        p = Plugin(name="from-package")
        assert p.name == "from-package"

    def test_all_exports(self) -> None:
        import skyward.plugins

        assert "Plugin" in skyward.plugins.__all__
