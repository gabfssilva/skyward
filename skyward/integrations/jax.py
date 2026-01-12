"""JAX distributed training integration."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable

__all__ = ["jax"]


def jax[**P, R]() -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Configure JAX distributed training.

    Example:
        import skyward as sky

        @sky.compute
        @sky.integrations.jax()
        def train():
            import jax
            # jax.distributed already initialized
            ...
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            import jax

            # Use v1's instance_info which reads from COMPUTE_POOL env var
            from skyward.cluster.info import instance_info

            pool = instance_info()

            if not pool or jax.distributed.is_initialized():
                return fn(*args, **kwargs)

            coordinator_address = f"{pool.head_addr}:{pool.head_port}"
            # Use workers_per_node for multi-GPU scenarios (like v1)
            total_processes = pool.total_nodes * pool.workers_per_node
            process_id = pool.node * pool.workers_per_node + pool.worker

            nccl_iface = pool.network.get("interface") if pool.total_nodes > 1 else None
            env = {
                "JAX_COORDINATOR_ADDRESS": coordinator_address,
                "JAX_NUM_PROCESSES": str(total_processes),
                "JAX_PROCESS_ID": str(process_id),
                "JAX_LOCAL_DEVICE_COUNT": str(pool.accelerators) if pool.accelerators > 0 else None,
                "NCCL_SOCKET_IFNAME": nccl_iface,
            }

            for key, value in env.items():
                if not value:
                    continue
                os.environ[key] = value

            jax.distributed.initialize(
                coordinator_address=coordinator_address,
                num_processes=total_processes,
                process_id=process_id,
            )

            return fn(*args, **kwargs)

        return wrapper

    return decorator
