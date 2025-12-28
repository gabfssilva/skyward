"""Framework detection and distributed initialization utilities."""

from __future__ import annotations

import functools
import json
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skyward.cluster import InstanceInfo

Framework = Literal["jax", "torch", "keras", "tensorflow", "transformers"]
Backend = Literal["jax", "torch", "tensorflow"] | None

__all__ = ["detect_framework", "wrap_with_distributed_init", "Framework", "Backend"]

logger = logging.getLogger("skyward.distributed")


def _pytorch_env_vars(pool: InstanceInfo) -> dict[str, str]:
    """Build PyTorch distributed environment variables."""
    env = {
        "MASTER_ADDR": pool.head_addr,
        "MASTER_PORT": str(pool.head_port),
        "WORLD_SIZE": str(pool.total_nodes),
        "RANK": str(pool.node),
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": str(pool.accelerators) if pool.accelerators > 0 else "1",
        "NODE_RANK": str(pool.node),
    }

    # NCCL optimizations for multi-node
    if pool.total_nodes > 1:
        env.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker")
        env.setdefault("NCCL_DEBUG", "WARN")

    return env


def _jax_env_vars(pool: InstanceInfo) -> dict[str, str]:
    """Build JAX distributed environment variables."""
    coordinator_address = f"{pool.head_addr}:{pool.head_port}"

    env = {
        "JAX_COORDINATOR_ADDRESS": coordinator_address,
        "JAX_NUM_PROCESSES": str(pool.total_nodes),
        "JAX_PROCESS_ID": str(pool.node),
    }

    if pool.accelerators > 0:
        env["JAX_LOCAL_DEVICE_COUNT"] = str(pool.accelerators)

    return env


def _tensorflow_env_vars(pool: InstanceInfo) -> dict[str, str]:
    """Build TensorFlow distributed environment variables (TF_CONFIG)."""
    worker_addrs = [
        f"{peer.get('private_ip', '')}:{pool.head_port}"
        for peer in sorted(pool.peers, key=lambda p: p.get("node", 0))
    ]

    tf_config = {
        "cluster": {"worker": worker_addrs},
        "task": {"type": "worker", "index": pool.node},
    }

    return {"TF_CONFIG": json.dumps(tf_config)}


_ENV_VAR_BUILDERS: dict[str, Callable[[InstanceInfo], dict[str, str]]] = {
    "torch": _pytorch_env_vars,
    "transformers": _pytorch_env_vars,
    "jax": _jax_env_vars,
    "tensorflow": _tensorflow_env_vars,
}


def _init_pytorch(pool: InstanceInfo) -> None:
    """Initialize PyTorch distributed process group."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    logger.info(
        f"PyTorch distributed initialized: rank {pool.node}/{pool.total_nodes}, "
        f"master={pool.head_addr}:{pool.head_port}"
    )


def _init_jax(pool: InstanceInfo) -> None:
    """Initialize JAX distributed runtime."""
    import jax

    coordinator_address = f"{pool.head_addr}:{pool.head_port}"

    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=pool.total_nodes,
        process_id=pool.node,
    )

    logger.info(
        f"JAX distributed initialized: process {pool.node}/{pool.total_nodes}, "
        f"coordinator={coordinator_address}"
    )


def _init_tensorflow(pool: InstanceInfo) -> None:
    """Log TensorFlow distributed configuration."""
    worker_addrs = [
        f"{peer.get('private_ip', '')}:{pool.head_port}"
        for peer in sorted(pool.peers, key=lambda p: p.get("node", 0))
    ]

    logger.info(
        f"TensorFlow distributed: worker {pool.node}/{pool.total_nodes}, "
        f"cluster={worker_addrs}"
    )


def _init_keras(pool: InstanceInfo) -> None:
    """Initialize Keras 3 DataParallel distribution."""
    if pool.total_nodes <= 1:
        return

    import keras

    devices = keras.distribution.list_devices()

    if not devices:
        logger.warning("No devices found for Keras DataParallel")
        return

    data_parallel = keras.distribution.DataParallel(devices=devices, auto_shard_dataset=False)
    keras.distribution.set_distribution(data_parallel)

    logger.info(f"Keras DataParallel set with {len(devices)} devices")


_FRAMEWORK_INITIALIZERS: dict[str, Callable[[InstanceInfo], None]] = {
    "torch": _init_pytorch,
    "transformers": _init_pytorch,
    "jax": _init_jax,
    "tensorflow": _init_tensorflow,
}


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    """Configuration for distributed training setup."""

    env_vars: Mapping[str, str]
    framework: Framework
    backend: Backend


def build_distributed_config(
    pool: InstanceInfo,
    framework: Framework,
    backend: Backend,
) -> DistributedConfig:
    """Build distributed training configuration."""
    effective_framework = backend if framework == "keras" and backend else framework
    builder = _ENV_VAR_BUILDERS.get(effective_framework)
    env_vars = builder(pool) if builder else {}

    return DistributedConfig(
        env_vars=env_vars,
        framework=framework,
        backend=backend,
    )


def apply_distributed_config(config: DistributedConfig, pool: InstanceInfo) -> None:
    """Apply distributed configuration: set env vars and initialize framework."""
    for key, value in config.env_vars.items():
        os.environ[key] = value

    effective_framework = (
        config.backend
        if config.framework == "keras" and config.backend
        else config.framework
    )

    initializer = _FRAMEWORK_INITIALIZERS.get(effective_framework)
    if initializer:
        initializer(pool)

    if config.framework == "keras":
        _init_keras(pool)


def detect_framework(
    pip: tuple[str, ...],
    env: frozenset[tuple[str, str]],
) -> tuple[Framework | None, Backend | None]:
    """Auto-detect ML framework from pip dependencies and environment variables.

    Analyzes pip dependencies to determine which ML framework is being used,
    and infers the backend for frameworks that need one (like Keras).

    Args:
        pip: Tuple of pip package specifications (e.g., ("keras==3.0", "jax[cuda]"))
        env: Frozenset of environment variable tuples (e.g., frozenset([("KERAS_BACKEND", "jax")]))

    Returns:
        Tuple of (framework, backend). Either can be None if not detected.

    Detection priority: keras > transformers > torch > jax > tensorflow

    Example:
        >>> detect_framework(("keras==3.0", "jax[cuda12]==0.8.1"), frozenset())
        ("keras", "jax")

        >>> detect_framework(("torch==2.0",), frozenset())
        ("torch", None)
    """
    env_dict = dict(env)
    # Normalize: lowercase, strip extras [...] and version specifiers
    pip_normalized = [
        p.lower().split("[")[0].split("=")[0].split("<")[0].split(">")[0] for p in pip
    ]

    has_keras = any(p.startswith("keras") for p in pip_normalized)
    has_torch = any(p.startswith(("torch", "pytorch")) for p in pip_normalized)
    has_jax = any(p.startswith("jax") for p in pip_normalized)
    has_tf = any(p.startswith("tensorflow") for p in pip_normalized)
    has_transformers = any(p.startswith("transformers") for p in pip_normalized)

    framework: Framework | None = None
    backend: Backend | None = None

    if has_keras:
        framework = "keras"
        # Get backend from env or infer from deps
        keras_backend = env_dict.get("KERAS_BACKEND")
        if keras_backend in ("jax", "torch", "tensorflow"):
            backend = keras_backend  # type: ignore[assignment]
        elif has_jax:
            backend = "jax"
        elif has_torch:
            backend = "torch"
        elif has_tf:
            backend = "tensorflow"
    elif has_transformers:
        framework = "transformers"
    elif has_torch:
        framework = "torch"
    elif has_jax:
        framework = "jax"
    elif has_tf:
        framework = "tensorflow"

    return framework, backend


def wrap_with_distributed_init[**P, R](
    fn: Callable[P, R],
    framework: Framework | None,
    backend: Backend,
) -> Callable[P, R]:
    """Wrap function to initialize distributed training environment before execution.

    This wrapper is applied before serializing the function for remote execution.
    It reads COMPUTE_POOL environment variable and configures framework-specific
    distributed training variables (MASTER_ADDR, JAX_COORDINATOR_ADDRESS, TF_CONFIG, etc.).

    Args:
        fn: Original function to wrap.
        framework: Detected ML framework (jax, torch, keras, tensorflow, transformers).
        backend: Backend for Keras (jax, torch, tensorflow).

    Returns:
        Wrapped function that initializes distributed environment before execution.

    Example:
        >>> wrapped = wrap_with_distributed_init(train_fn, "torch", None)
        >>> # When executed remotely, MASTER_ADDR, WORLD_SIZE, RANK are set automatically
    """
    if framework is None:
        return fn

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        from skyward.cluster import instance_info

        pool = instance_info()
        if pool is not None:
            _configure_framework_env(pool, framework, backend)
        return fn(*args, **kwargs)

    return wrapper


def _configure_framework_env(
    pool: InstanceInfo,
    framework: Framework,
    backend: Backend,
) -> None:
    """Configure framework-specific environment variables for distributed training."""
    config = build_distributed_config(pool, framework, backend)
    apply_distributed_config(config, pool)
