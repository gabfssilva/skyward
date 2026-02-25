"""Plugin type for declarative third-party integrations."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skyward.api.model import Cluster
    from skyward.api.pool import ComputePool
    from skyward.api.runtime import InstanceInfo
    from skyward.api.spec import Image
    from skyward.providers.bootstrap import Op

type ImageTransform[S] = Callable[["Image", "Cluster[S]"], "Image"]
type BootstrapFactory[S] = Callable[["Cluster[S]"], tuple["Op", ...]]
type TaskDecorator = Callable[[Callable[..., Any], tuple, dict], Any]
type AppLifecycle = Callable[["InstanceInfo"], AbstractContextManager[None]]
type ClientLifecycle[S] = Callable[["ComputePool", "Cluster[S]"], AbstractContextManager[None]]


@dataclass(frozen=True, slots=True)
class Plugin:
    """Declarative third-party integration bundle.

    Bundles environment setup, bootstrap ops, worker lifecycle hooks,
    client-side hooks, and per-task wrapping into a single composable unit.

    Parameters
    ----------
    name
        Plugin identifier.
    transform
        (Image, Cluster[S]) -> Image transformer. Receives current Image and
        cluster metadata, returns modified copy.
    bootstrap
        Factory that receives Cluster[S] and returns extra shell ops appended
        after Image-driven bootstrap phases.
    decorate
        Per-task wrapper on remote worker: (fn, args, kwargs) -> T.
    around_app
        Worker lifecycle context manager: InstanceInfo -> ContextManager[None].
    around_client
        Client lifecycle context manager: (ComputePool, Cluster[S]) -> ContextManager[None].
    """

    name: str
    transform: ImageTransform[Any] | None = None
    bootstrap: BootstrapFactory[Any] | None = None
    decorate: TaskDecorator | None = None
    around_app: AppLifecycle | None = None
    around_client: ClientLifecycle[Any] | None = None

    @staticmethod
    def create(name: str) -> Plugin:
        return Plugin(name=name)

    def with_image_transform(self, transform: ImageTransform[Any]) -> Plugin:
        return replace(self, transform=transform)

    def with_bootstrap(self, factory: BootstrapFactory[Any]) -> Plugin:
        return replace(self, bootstrap=factory)

    def with_decorator(self, decorate: TaskDecorator) -> Plugin:
        return replace(self, decorate=decorate)

    def with_around_app(self, around: AppLifecycle) -> Plugin:
        return replace(self, around_app=around)

    def with_around_client(self, around: ClientLifecycle[Any]) -> Plugin:
        return replace(self, around_client=around)


def chain_decorators(
    fn: Callable[..., Any],
    decorators: list[TaskDecorator],
) -> Callable[..., Any]:
    """Chain plugin decorators around a function.

    First decorator in list = outermost (runs first).
    Empty list returns fn unchanged.
    """
    if not decorators:
        return fn

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        def call(idx: int, f: Callable[..., Any], a: tuple, kw: dict) -> Any:
            if idx >= len(decorators):
                return f(*a, **kw)
            return decorators[idx](
                lambda *a2, **kw2: call(idx + 1, f, a2, kw2), a, kw,
            )

        return call(0, fn, args, kwargs)

    return wrapped


def make_around_app_decorator(name: str, factory: AppLifecycle) -> TaskDecorator:
    """Create a task decorator that enters around_app once per worker."""

    def decorator(fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
        info = instance_info()
        ensure_around_app(name, factory, info)
        return fn(*args, **kwargs)

    return decorator


def instance_info() -> Any:
    """Lazy import of instance_info to avoid circular imports."""
    from skyward.api.runtime import instance_info as _instance_info

    return _instance_info()


def ensure_around_app(name: str, factory: AppLifecycle, info: object) -> None:
    """Lazy import of ensure_around_app to avoid circular imports."""
    from skyward.plugins.state import ensure_around_app as _ensure

    _ensure(name, factory, info)
