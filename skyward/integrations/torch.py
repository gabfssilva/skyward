"""PyTorch distributed training integration."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import Literal


def torch[**P, R](
    _fn: Callable[P, R] | None = None,
    *,
    backend: Literal["nccl", "gloo"] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    """Configure PyTorch distributed training.

    Can be used with or without arguments:
        @sky.integrations.torch
        @sky.integrations.torch(backend="nccl")

    Args:
        backend: Process group backend. Auto-detected if None (nccl for GPU, gloo for CPU).
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            import torch  # type: ignore[reportMissingImports]
            import torch.distributed as dist  # type: ignore[reportMissingImports]

            from skyward.api.runtime import instance_info
            from skyward.observability.logger import logger

            log = logger.bind(integration="torch")
            pool = instance_info()

            if not pool or dist.is_initialized():
                log.debug("Skipping distributed init (no pool or already initialized)")
                return fn(*args, **kwargs)

            world_size = pool.total_nodes
            global_rank = pool.node

            env = {
                "MASTER_ADDR": pool.head_addr,
                "MASTER_PORT": str(pool.head_port),
                "WORLD_SIZE": str(world_size),
                "RANK": str(global_rank),
                "LOCAL_RANK": "0",
                "LOCAL_WORLD_SIZE": "1",
                "NODE_RANK": str(pool.node),
            }

            for key, value in env.items():
                if value:
                    os.environ[key] = value

            log.debug("Env vars set: {vars}", vars={k: v for k, v in env.items() if v})

            be = backend or ("nccl" if torch.cuda.is_available() else "gloo")  # type: ignore[reportAttributeAccessIssue]
            if backend is None:
                cuda = torch.cuda.is_available()  # type: ignore[reportAttributeAccessIssue]
                log.debug("Auto-detected backend: {be} (CUDA available={cuda})", be=be, cuda=cuda)
            log.debug(
                "Initializing process group: backend={be}, rank={rank}, world_size={ws}",
                be=be, rank=global_rank, ws=world_size,
            )
            dist.init_process_group(backend=be, init_method="env://")
            log.debug("Process group initialized")

            return fn(*args, **kwargs)

        return wrapper

    if _fn is not None:
        return decorator(_fn)

    return decorator
