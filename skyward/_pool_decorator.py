"""Pool decorator for implicit pool context.

This module provides the @pool decorator that provisions resources
for the duration of a function, enabling the `>> sky` syntax.

Example:
    import skyward as sky

    @sky.compute
    def train(data):
        return model.fit(data)

    @sky.pool(provider=sky.AWS(), accelerator="A100", nodes=4)
    def main():
        result = train(data) >> sky
        return result

    main()  # provisions -> executes -> deprovisions
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import wraps
from typing import Literal, ParamSpec, TypeVar

from skyward._context import reset_current_pool, set_current_pool
from skyward.callback import Callback
from skyward.image import Image
from skyward.logging import LogConfig
from skyward.pool import ComputePool
from skyward.spec import AllocationLike
from skyward.types import Accelerator, Architecture, Auto, Memory, Provider
from skyward.volume import Volume

P = ParamSpec("P")
R = TypeVar("R")


def pool(
    provider: Provider,
    *,
    image: Image | None = None,
    nodes: int = 1,
    machine: str | None = None,
    accelerator: Accelerator | str | None = None,
    architecture: Architecture | None = None,
    cpu: int | None = None,
    memory: Memory | None = None,
    volume: dict[str, str] | Sequence[Volume] | None = None,
    allocation: AllocationLike = "spot-if-available",
    timeout: int = 3600,
    env: dict[str, str] | None = None,
    concurrency: int = 1,
    display: Literal["spinner", "quiet"] = "spinner",
    on_event: Callback | None = None,
    collect_metrics: bool = True,
    logging: LogConfig | bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that provisions a pool for the duration of the function.

    This enables the implicit `>> sky` syntax inside the decorated function.
    The pool is provisioned when the function is called and terminated when
    it returns (or raises).

    Args:
        provider: Cloud provider (AWS, DigitalOcean, Verda).
        image: Image specification (python, pip, apt, env).
        nodes: Number of nodes to provision. Default 1.
        machine: Direct instance type override (e.g., "p5.48xlarge").
        accelerator: Accelerator specification (e.g., "A100", Accelerator.NVIDIA.H100()).
        architecture: CPU architecture preference ("arm64", "x86_64", or Auto()).
        cpu: CPU cores per worker (for cgroups limits).
        memory: Memory per worker (e.g., "32GB").
        volume: Volumes to mount.
        allocation: Instance allocation strategy.
        timeout: Maximum execution time in seconds.
        env: Environment variables.
        concurrency: Number of concurrent tasks per instance.
        display: Display mode ("spinner" or "quiet").
        on_event: Callback for events.
        collect_metrics: Whether to collect metrics.
        logging: Logging configuration.

    Returns:
        Decorator that wraps the function with pool provisioning.

    Example:
        @sky.pool(provider=sky.AWS(), accelerator="A100", nodes=4)
        def train_pipeline():
            model = train(data) >> sky
            score = evaluate(model) >> sky
            return score

        result = train_pipeline()  # Pool is managed automatically
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            p = ComputePool(
                provider=provider,
                image=image or Image(),
                nodes=nodes,
                machine=machine,
                accelerator=accelerator,
                architecture=architecture or Auto(),
                cpu=cpu,
                memory=memory,
                volume=volume,
                allocation=allocation,
                timeout=timeout,
                env=env,
                concurrency=concurrency,
                display=display,
                on_event=on_event,
                collect_metrics=collect_metrics,
                logging=logging,
            )
            with p:
                token = set_current_pool(p)
                try:
                    return fn(*args, **kwargs)
                finally:
                    reset_current_pool(token)

        return wrapper

    return decorator
