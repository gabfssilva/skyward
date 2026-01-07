"""PyTorch distributed training integration."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import Literal

__all__ = ["torch"]


def torch[**P, R](
    backend: Literal["nccl", "gloo"] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Configure PyTorch distributed training.

    Args:
        backend: Process group backend. Auto-detected if None (nccl for GPU, gloo for CPU).

    Example:
        import skyward as sky

        @sky.compute
        @sky.integrations.torch(backend="nccl")
        def train():
            import torch.distributed as dist
            assert dist.is_initialized()
            ...
    """
    from skyward.compute.pending import ComputeFunction

    def wrapper(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            import torch
            import torch.distributed as dist

            from skyward.cluster.info import instance_info

            pool = instance_info()

            if not pool or dist.is_initialized():
                return fn(*args, **kwargs)

            world_size = pool.total_nodes * pool.workers_per_node
            global_rank = pool.node * pool.workers_per_node + pool.worker

            nccl_iface = pool.network.get("interface") if pool.total_nodes > 1 else None
            env = {
                "MASTER_ADDR": pool.head_addr,
                "MASTER_PORT": str(pool.head_port),
                "WORLD_SIZE": str(world_size),
                "RANK": str(global_rank),
                "LOCAL_RANK": str(pool.worker),
                "LOCAL_WORLD_SIZE": str(pool.workers_per_node),
                "NODE_RANK": str(pool.node),
                "NCCL_SOCKET_IFNAME": nccl_iface,
            }

            for key, value in env.items():
                if value:
                    os.environ[key] = value

            be = backend or ("nccl" if torch.cuda.is_available() else "gloo")
            dist.init_process_group(backend=be, init_method="env://")

            return fn(*args, **kwargs)

        return inner

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        if isinstance(fn, ComputeFunction):
            return ComputeFunction(fn=wrapper(fn.fn), name=fn.name)
        return wrapper(fn)

    return decorator
