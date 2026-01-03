"""JAX distributed training integration."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.cluster import InstanceInfo

__all__ = ["jax"]


def _jax_env_vars(pool: InstanceInfo) -> dict[str, str]:
    """Build JAX distributed environment variables."""
    coordinator_address = f"{pool.head_addr}:{pool.head_port}"
    total_processes = pool.total_nodes * pool.workers_per_node
    process_id = pool.node * pool.workers_per_node + pool.worker

    env = {
        "JAX_COORDINATOR_ADDRESS": coordinator_address,
        "JAX_NUM_PROCESSES": str(total_processes),
        "JAX_PROCESS_ID": str(process_id),
    }

    if pool.accelerators > 0:
        env["JAX_LOCAL_DEVICE_COUNT"] = str(pool.accelerators)

    return env


def _init_jax(pool: InstanceInfo) -> None:
    """Initialize JAX distributed runtime."""
    import jax

    if jax.distributed.is_initialized():
        return

    coordinator_address = f"{pool.head_addr}:{pool.head_port}"
    total_processes = pool.total_nodes * pool.workers_per_node
    process_id = pool.node * pool.workers_per_node + pool.worker

    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=total_processes,
        process_id=process_id,
    )


def jax[**P, R]() -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Configure JAX distributed training.

    Example:
        from skyward.integrations import jax

        @jax()
        @compute
        def train():
            import jax
            # jax.distributed already initialized
            ...
    """
    from skyward.pending import ComputeFunction

    def wrapper(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            from skyward.cluster import instance_info

            pool = instance_info()
            if pool is not None:
                for key, value in _jax_env_vars(pool).items():
                    os.environ[key] = value

                _init_jax(pool)

            return fn(*args, **kwargs)

        return inner

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        if isinstance(fn, ComputeFunction):
            return ComputeFunction(fn=wrapper(fn.fn), name=fn.name)
        return wrapper(fn)

    return decorator
