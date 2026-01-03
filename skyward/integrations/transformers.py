"""Hugging Face Transformers distributed training integration."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import Literal

from skyward.integrations.torch import _pytorch_env_vars

__all__ = ["transformers"]


def transformers[**P, R](
    backend: Literal["nccl", "gloo"] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Configure Hugging Face Transformers distributed training.

    Uses PyTorch distributed backend internally.

    Args:
        backend: Process group backend. Auto-detected if None.

    Example:
        from skyward.integrations import transformers

        @transformers(backend="nccl")
        @compute
        def train():
            from transformers import Trainer
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

                import torch
                import torch.distributed as dist

                if not dist.is_initialized():
                    be = backend if backend else ("nccl" if torch.cuda.is_available() else "gloo")
                    dist.init_process_group(backend=be, init_method="env://")

            return fn(*args, **kwargs)

        return inner

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        if isinstance(fn, ComputeFunction):
            return ComputeFunction(fn=wrapper(fn.fn), name=fn.name)
        return wrapper(fn)

    return decorator
