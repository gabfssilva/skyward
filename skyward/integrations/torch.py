"""PyTorch distributed training integration."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skyward.cluster import InstanceInfo

__all__ = ["torch"]


def _pytorch_env_vars(pool: InstanceInfo) -> dict[str, str]:
    """Build PyTorch distributed environment variables."""
    world_size = pool.total_nodes * pool.workers_per_node
    global_rank = pool.node * pool.workers_per_node + pool.worker

    env = {
        "MASTER_ADDR": pool.head_addr,
        "MASTER_PORT": str(pool.head_port),
        "WORLD_SIZE": str(world_size),
        "RANK": str(global_rank),
        "LOCAL_RANK": str(pool.worker),
        "LOCAL_WORLD_SIZE": str(pool.workers_per_node),
        "NODE_RANK": str(pool.node),
    }

    if pool.total_nodes > 1:
        env.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker")
        env.setdefault("NCCL_DEBUG", "WARN")

    return env


def _init_pytorch(pool: InstanceInfo) -> None:
    """Initialize PyTorch distributed process group."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")


def torch[**P, R](
    backend: Literal["nccl", "gloo"] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Configure PyTorch distributed training.

    Args:
        backend: Process group backend. Auto-detected if None (nccl for GPU, gloo for CPU).

    Example:
        from skyward.integrations import torch

        @torch(backend="nccl")
        @compute
        def train():
            import torch.distributed as dist
            assert dist.is_initialized()
            ...
    """
    from skyward.pending import ComputeFunction

    def wrapper(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            from skyward.cluster import instance_info

            pool = instance_info()
            if pool is not None:
                for key, value in _pytorch_env_vars(pool).items():
                    os.environ[key] = value

                import torch as torch_lib
                import torch.distributed as dist

                if not dist.is_initialized():
                    be = backend or ("nccl" if torch_lib.cuda.is_available() else "gloo")
                    dist.init_process_group(backend=be, init_method="env://")

            return fn(*args, **kwargs)

        return inner

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        if isinstance(fn, ComputeFunction):
            return ComputeFunction(fn=wrapper(fn.fn), name=fn.name)
        return wrapper(fn)

    return decorator
