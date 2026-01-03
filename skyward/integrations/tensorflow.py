"""TensorFlow distributed training integration."""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.cluster import InstanceInfo

__all__ = ["tensorflow"]


def _tensorflow_env_vars(pool: InstanceInfo) -> dict[str, str]:
    """Build TensorFlow distributed environment variables (TF_CONFIG)."""
    worker_addrs: list[str] = []
    for peer in sorted(pool.peers, key=lambda p: p.get("node", 0)):
        ip = peer.get("private_ip", peer.get("addr", ""))
        for worker_idx in range(pool.workers_per_node):
            port = pool.head_port + worker_idx
            worker_addrs.append(f"{ip}:{port}")

    task_index = pool.node * pool.workers_per_node + pool.worker

    tf_config = {
        "cluster": {"worker": worker_addrs},
        "task": {"type": "worker", "index": task_index},
    }

    return {"TF_CONFIG": json.dumps(tf_config)}


def _init_tensorflow(pool: InstanceInfo) -> None:
    """Initialize TensorFlow distributed configuration (env vars set separately)."""
    pass  # TF_CONFIG is set via env vars


def tensorflow[**P, R]() -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Configure TensorFlow distributed training.

    Example:
        from skyward.integrations import tensorflow

        @tensorflow()
        @compute
        def train():
            import tensorflow as tf
            # TF_CONFIG already set
            ...
    """
    from skyward.pending import ComputeFunction

    def wrapper(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            from skyward.cluster import instance_info

            pool = instance_info()
            if pool is not None:
                for key, value in _tensorflow_env_vars(pool).items():
                    os.environ[key] = value

            return fn(*args, **kwargs)

        return inner

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        if isinstance(fn, ComputeFunction):
            return ComputeFunction(fn=wrapper(fn.fn), name=fn.name)
        return wrapper(fn)

    return decorator
