"""Plugin type for declarative third-party integrations."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from functools import reduce
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
type ProcessLifecycle = Callable[[InstanceInfo], AbstractContextManager[None]]
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
        Entered once in the main worker process.
    around_process
        Subprocess lifecycle context manager: InstanceInfo -> ContextManager[None].
        Entered once per subprocess when executor="process". Lazy — enters on
        the first task execution in each subprocess, after env vars are propagated.
    around_client
        Client lifecycle context manager: (ComputePool, Cluster[S]) -> ContextManager[None].
    """

    name: str
    transform: ImageTransform[Any] | None = None
    bootstrap: BootstrapFactory[Any] | None = None
    decorate: TaskDecorator | None = None
    around_app: AppLifecycle | None = None
    around_process: ProcessLifecycle | None = None
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

    def with_around_process(self, around: ProcessLifecycle) -> Plugin:
        return replace(self, around_process=around)

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


