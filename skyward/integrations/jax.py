"""JAX distributed training integration."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable


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

            from loguru import logger
            from skyward.api.runtime import instance_info

            log = logger.bind(integration="jax")
            pool = instance_info()

            log.debug("Pool info: {pool}", pool=pool)
            log.debug("Already initialized: {init}", init=jax.distributed.is_initialized())

            if not pool or jax.distributed.is_initialized():
                log.debug("Skipping distributed init (no pool or already initialized)")
                return fn(*args, **kwargs)

            coordinator_address = f"{pool.head_addr}:{pool.head_port}"
            total_processes = pool.total_nodes * pool.workers_per_node
            process_id = pool.node * pool.workers_per_node + pool.worker

            log.debug(
                "Coordinator: {addr}, processes={total}, process_id={pid}",
                addr=coordinator_address, total=total_processes, pid=process_id,
            )

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

            local_device_ids = list(range(pool.accelerators)) if pool.accelerators > 0 else None

            log.debug(
                "Calling jax.distributed.initialize(local_device_ids={ids})",
                ids=local_device_ids,
            )
            jax.distributed.initialize(
                coordinator_address=coordinator_address,
                num_processes=total_processes,
                process_id=process_id,
                local_device_ids=local_device_ids,
            )
            log.debug("jax.distributed.initialize() done")

            return fn(*args, **kwargs)

        return wrapper

    return decorator
