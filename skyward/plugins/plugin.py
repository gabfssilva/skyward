"""Plugin type for declarative third-party integrations."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from functools import reduce, wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skyward.api.model import Cluster
    from skyward.api.pool import ComputePool
    from skyward.api.runtime import InstanceInfo
    from skyward.api.spec import Image
    from skyward.providers.bootstrap import Op

type ImageTransform[S] = Callable[[Image, Cluster[S]], Image]
type BootstrapFactory[S] = Callable[[Cluster[S]], tuple[Op, ...]]
type TaskDecorator[**P, R] = Callable[[Callable[P, R]], Callable[P, R]]
type AppLifecycle = Callable[[InstanceInfo], AbstractContextManager[None]]
type ClientLifecycle[S] = Callable[[ComputePool, Cluster[S]], AbstractContextManager[None]]


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
        Classic Python decorator: (fn) -> fn. Wraps each @sky.compute
        function at execution time on the remote worker.
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

    def with_image_transform[S](self, transform: ImageTransform[S]) -> Plugin:
        return replace(self, transform=transform)

    def with_bootstrap[S](self, factory: BootstrapFactory[S]) -> Plugin:
        return replace(self, bootstrap=factory)

    def with_decorator[**P, R](self, decorate: TaskDecorator[P, R]) -> Plugin:
        return replace(self, decorate=decorate)

    def with_around_app(self, around: AppLifecycle) -> Plugin:
        return replace(self, around_app=around)

    def with_around_client[S](self, around: ClientLifecycle[S]) -> Plugin:
        return replace(self, around_client=around)


def chain_decorators[**P, R](
    fn: Callable[P, R],
    decorators: list[TaskDecorator[P, R]],
) -> Callable[P, R]:
    """Chain plugin decorators around a function.

    First decorator in list = outermost (runs first).
    Empty list returns fn unchanged.
    """
    return reduce(lambda f, d: d(f), reversed(decorators), fn)


def make_around_app_decorator(name: str, factory: AppLifecycle) -> TaskDecorator:
    """Create a task decorator that enters around_app once per worker."""

    def decorator[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            info = instance_info()
            ensure_around_app(name, factory, info)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def instance_info() -> Any:
    """Lazy import of instance_info to avoid circular imports."""
    from skyward.api.runtime import instance_info as _instance_info

    return _instance_info()


def ensure_around_app(name: str, factory: AppLifecycle, info: object) -> None:
    """Lazy import of ensure_around_app to avoid circular imports."""
    from skyward.plugins.state import ensure_around_app as _ensure

    _ensure(name, factory, info)
