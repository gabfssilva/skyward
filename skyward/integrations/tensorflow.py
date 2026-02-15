"""TensorFlow distributed training integration."""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable


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

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            from loguru import logger

            from skyward.api.runtime import instance_info

            log = logger.bind(integration="tensorflow")
            pool = instance_info()

            if not pool:
                log.debug("Skipping distributed init (no pool)")
                return fn(*args, **kwargs)

            # Build worker list - each node is one worker
            worker_addrs = [
                f"{pool.head_addr}:{pool.head_port + i}"
                for i in range(pool.total_nodes)
            ]

            task_index = pool.node

            tf_config = {
                "cluster": {"worker": worker_addrs},
                "task": {"type": "worker", "index": task_index},
            }

            os.environ["TF_CONFIG"] = json.dumps(tf_config)
            log.debug(
                "TF_CONFIG set: {workers} workers, task_index={idx}",
                workers=len(worker_addrs), idx=task_index,
            )

            return fn(*args, **kwargs)

        return wrapper

    return decorator
