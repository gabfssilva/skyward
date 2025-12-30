"""Framework detection and distributed initialization utilities."""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skyward.cluster import InstanceInfo

Framework = Literal["jax", "torch", "keras", "tensorflow", "transformers"]
Backend = Literal["jax", "torch", "tensorflow"] | None

__all__ = [
    # Types
    "Framework",
    "Backend",
    # Decorators
    "keras",
    "torch",
    "jax",
    "tensorflow",
    "transformers",
    # Utilities
    "detect_framework",
]


def _pytorch_env_vars(pool: InstanceInfo) -> dict[str, str]:
    """Build PyTorch distributed environment variables."""
    # Global rank: each worker is a separate process
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

    # NCCL optimizations for multi-node
    if pool.total_nodes > 1:
        env.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker")
        env.setdefault("NCCL_DEBUG", "WARN")

    return env


def _jax_env_vars(pool: InstanceInfo) -> dict[str, str]:
    """Build JAX distributed environment variables."""
    coordinator_address = f"{pool.head_addr}:{pool.head_port}"

    # Global process ID: each worker is a separate JAX process
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


def _tensorflow_env_vars(pool: InstanceInfo) -> dict[str, str]:
    """Build TensorFlow distributed environment variables (TF_CONFIG)."""
    # Build addresses for all workers (each worker gets a unique port)
    worker_addrs: list[str] = []
    for peer in sorted(pool.peers, key=lambda p: p.get("node", 0)):
        ip = peer.get("private_ip", peer.get("addr", ""))
        for worker_idx in range(pool.workers_per_node):
            port = pool.head_port + worker_idx
            worker_addrs.append(f"{ip}:{port}")

    # Global task index: each worker is a separate TF worker
    task_index = pool.node * pool.workers_per_node + pool.worker

    tf_config = {
        "cluster": {"worker": worker_addrs},
        "task": {"type": "worker", "index": task_index},
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


def _init_jax(pool: InstanceInfo) -> None:
    """Initialize JAX distributed runtime."""
    print(f"[_init_jax] Importing jax...")
    import jax

    if jax.distributed.is_initialized():
        print(f"[_init_jax] Already initialized, skipping.")
        return

    coordinator_address = f"{pool.head_addr}:{pool.head_port}"
    # Global process ID: each worker is a separate JAX process
    total_processes = pool.total_nodes * pool.workers_per_node
    process_id = pool.node * pool.workers_per_node + pool.worker
    print(f"[_init_jax] coordinator_address={coordinator_address}, num_processes={total_processes}, process_id={process_id}")

    print(f"[_init_jax] Calling jax.distributed.initialize()...")
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=total_processes,
        process_id=process_id,
    )
    print(f"[_init_jax] jax.distributed.initialize() completed.")


def _init_tensorflow(pool: InstanceInfo) -> None:
    """Initialize TensorFlow distributed configuration (env vars set separately)."""
    pass  # TF_CONFIG is set via env vars


def _init_keras(pool: InstanceInfo) -> None:
    """Initialize Keras 3 DataParallel distribution."""
    print(f"[_init_keras] total_nodes={pool.total_nodes}")
    if pool.total_nodes <= 1:
        print(f"[_init_keras] Single node, skipping distribution setup.")
        return

    print(f"[_init_keras] Importing keras...")
    import keras

    # Set consistent seed across all nodes
    print(f"[_init_keras] Setting random seed for consistency...")
    keras.utils.set_random_seed(42)

    print(f"[_init_keras] Calling keras.distribution.list_devices()...")
    devices = keras.distribution.list_devices()
    print(f"[_init_keras] devices={devices}")

    if not devices:
        print(f"[_init_keras] No devices found, skipping distribution setup.")
        return

    print(f"[_init_keras] Creating DataParallel distribution...")
    data_parallel = keras.distribution.DataParallel(devices=devices, auto_shard_dataset=False)
    print(f"[_init_keras] Setting distribution...")
    keras.distribution.set_distribution(data_parallel)
    print(f"[_init_keras] Distribution set.")


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


# =============================================================================
# Decorators for explicit distributed configuration
# =============================================================================

# Import here to avoid circular imports
from skyward.pending import ComputeFunction


def keras[**P, R](
    backend: Backend = None,
) -> Callable[[Callable[P, R] | ComputeFunction[P, R]], Callable[P, R] | ComputeFunction[P, R]]:
    """Configure Keras 3 distributed training.

    Args:
        backend: Backend to use (jax, torch, tensorflow). Auto-detected if None.

    Example:
        @distributed.keras(backend="jax")
        @compute
        def train():
            import keras
            model = keras.Sequential([...])
            model.fit(...)
    """

    def wrapper(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            from skyward.cluster import instance_info

            print(f"[distributed.keras] Starting initialization, backend={backend}")

            pool = instance_info()
            print(f"[distributed.keras] pool={pool}")

            if pool is not None:
                # Set env vars for backend
                effective = backend if backend else "jax"
                print(f"[distributed.keras] effective backend={effective}")

                builder = _ENV_VAR_BUILDERS.get(effective)
                if builder:
                    env_vars = builder(pool)
                    print(f"[distributed.keras] Setting env vars: {env_vars}")
                    for key, value in env_vars.items():
                        os.environ[key] = value

                # Init backend
                print(f"[distributed.keras] Initializing backend {effective}...")
                initializer = _FRAMEWORK_INITIALIZERS.get(effective)
                if initializer:
                    print(f"[distributed.keras] Calling {initializer.__name__}...")
                    initializer(pool)
                    print(f"[distributed.keras] Backend initialized.")

                # Init Keras distribution
                print(f"[distributed.keras] Initializing Keras distribution...")
                _init_keras(pool)
                print(f"[distributed.keras] Keras distribution initialized.")

            print(f"[distributed.keras] Calling user function {fn.__name__}...")
            result = fn(*args, **kwargs)
            print(f"[distributed.keras] User function returned.")
            return result

        return inner

    def decorator(fn: Callable[P, R] | ComputeFunction[P, R]) -> Callable[P, R] | ComputeFunction[P, R]:
        if isinstance(fn, ComputeFunction):
            return ComputeFunction(fn=wrapper(fn.fn), name=fn.name)
        return wrapper(fn)

    return decorator


def torch[**P, R](
    backend: Literal["nccl", "gloo"] | None = None,
) -> Callable[[Callable[P, R] | ComputeFunction[P, R]], Callable[P, R] | ComputeFunction[P, R]]:
    """Configure PyTorch distributed training.

    Args:
        backend: Process group backend. Auto-detected if None (nccl for GPU, gloo for CPU).

    Example:
        @distributed.torch(backend="nccl")
        @compute
        def train():
            import torch.distributed as dist
            assert dist.is_initialized()
            ...
    """

    def wrapper(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            from skyward.cluster import instance_info

            pool = instance_info()
            if pool is not None:
                # Set env vars
                for key, value in _pytorch_env_vars(pool).items():
                    os.environ[key] = value

                # Init process group
                import torch
                import torch.distributed as dist

                if not dist.is_initialized():
                    be = backend if backend else ("nccl" if torch.cuda.is_available() else "gloo")
                    dist.init_process_group(backend=be, init_method="env://")

            return fn(*args, **kwargs)

        return inner

    def decorator(fn: Callable[P, R] | ComputeFunction[P, R]) -> Callable[P, R] | ComputeFunction[P, R]:
        if isinstance(fn, ComputeFunction):
            return ComputeFunction(fn=wrapper(fn.fn), name=fn.name)
        return wrapper(fn)

    return decorator


def jax[**P, R]() -> Callable[[Callable[P, R] | ComputeFunction[P, R]], Callable[P, R] | ComputeFunction[P, R]]:
    """Configure JAX distributed training.

    Example:
        @distributed.jax()
        @compute
        def train():
            import jax
            # jax.distributed already initialized
            ...
    """

    def wrapper(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            from skyward.cluster import instance_info

            pool = instance_info()
            if pool is not None:
                # Set env vars
                for key, value in _jax_env_vars(pool).items():
                    os.environ[key] = value

                # Init JAX distributed
                import jax

                # Global process ID: each worker is a separate JAX process
                total_processes = pool.total_nodes * pool.workers_per_node
                process_id = pool.node * pool.workers_per_node + pool.worker

                jax.distributed.initialize(
                    coordinator_address=f"{pool.head_addr}:{pool.head_port}",
                    num_processes=total_processes,
                    process_id=process_id,
                )

            return fn(*args, **kwargs)

        return inner

    def decorator(fn: Callable[P, R] | ComputeFunction[P, R]) -> Callable[P, R] | ComputeFunction[P, R]:
        if isinstance(fn, ComputeFunction):
            return ComputeFunction(fn=wrapper(fn.fn), name=fn.name)
        return wrapper(fn)

    return decorator


def tensorflow[**P, R]() -> Callable[[Callable[P, R] | ComputeFunction[P, R]], Callable[P, R] | ComputeFunction[P, R]]:
    """Configure TensorFlow distributed training.

    Example:
        @distributed.tensorflow()
        @compute
        def train():
            import tensorflow as tf
            # TF_CONFIG already set
            ...
    """

    def wrapper(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            from skyward.cluster import instance_info

            pool = instance_info()
            if pool is not None:
                # Set TF_CONFIG env var
                for key, value in _tensorflow_env_vars(pool).items():
                    os.environ[key] = value

            return fn(*args, **kwargs)

        return inner

    def decorator(fn: Callable[P, R] | ComputeFunction[P, R]) -> Callable[P, R] | ComputeFunction[P, R]:
        if isinstance(fn, ComputeFunction):
            return ComputeFunction(fn=wrapper(fn.fn), name=fn.name)
        return wrapper(fn)

    return decorator


def transformers[**P, R](
    backend: Literal["nccl", "gloo"] | None = None,
) -> Callable[[Callable[P, R] | ComputeFunction[P, R]], Callable[P, R] | ComputeFunction[P, R]]:
    """Configure Hugging Face Transformers distributed training.

    Uses PyTorch distributed backend internally.

    Args:
        backend: Process group backend. Auto-detected if None.

    Example:
        @distributed.transformers(backend="nccl")
        @compute
        def train():
            from transformers import Trainer
            ...
    """

    def wrapper(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            from skyward.cluster import instance_info

            pool = instance_info()
            if pool is not None:
                # Set env vars (uses PyTorch backend)
                for key, value in _pytorch_env_vars(pool).items():
                    os.environ[key] = value

                # Init process group
                import torch
                import torch.distributed as dist

                if not dist.is_initialized():
                    be = backend if backend else ("nccl" if torch.cuda.is_available() else "gloo")
                    dist.init_process_group(backend=be, init_method="env://")

            return fn(*args, **kwargs)

        return inner

    def decorator(fn: Callable[P, R] | ComputeFunction[P, R]) -> Callable[P, R] | ComputeFunction[P, R]:
        if isinstance(fn, ComputeFunction):
            return ComputeFunction(fn=wrapper(fn.fn), name=fn.name)
        return wrapper(fn)

    return decorator
