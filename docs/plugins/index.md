# What are Plugins?

Skyward's plugin system is the way you bring third-party frameworks into the compute pool. When you pass `plugins=[sky.plugins.torch()]` to a `ComputePool`, you are telling Skyward: install PyTorch on the remote workers, configure the distributed runtime before my function runs, and clean up when the worker stops. The plugin handles the environment setup, the lifecycle hooks, and the per-task wrapping — things you would otherwise do manually with `Image(pip=[...])`, environment variables, and boilerplate inside your `@sky.compute` functions.

The key insight is that plugins operate at the pool level, not at the function level. A single plugin declaration on the pool affects every task dispatched to it. This is different from the decorator pattern you might be used to, where each function explicitly opts in to framework setup. With plugins, the pool is the unit of configuration: once you declare that a pool uses PyTorch with NCCL, every function dispatched to that pool gets PyTorch's distributed environment configured automatically.

## The Plugin Dataclass

A `Plugin` is a frozen dataclass with five optional hooks. Each hook corresponds to a different phase in the pool and worker lifecycle. You do not need to implement all five — most plugins use two or three.

```python
@dataclass(frozen=True, slots=True)
class Plugin:
    name: str
    transform: ImageTransform | None = None
    bootstrap: BootstrapFactory | None = None
    decorate: TaskDecorator | None = None
    around_app: AppLifecycle | None = None
    around_client: ClientLifecycle | None = None
```

The hooks are:

**`transform`** modifies the `Image` before bootstrap. It receives the current `Image` and the `Cluster` metadata, and returns a new `Image` with additional pip packages, pip indexes, environment variables, or apt packages. This is how plugins install their dependencies on the remote worker. For example, the `torch` plugin appends `"torch"` to `pip` and adds PyTorch's CUDA wheel index. The `keras` plugin appends `"keras"` and sets `KERAS_BACKEND` in the environment. Since `Image` is a frozen dataclass, the transform returns a new copy via `replace()` — it never mutates the original.

**`bootstrap`** injects shell operations after the standard bootstrap phases (apt, pip, etc.). It receives the `Cluster` and returns a tuple of shell ops. The `huggingface` plugin uses this to run `huggingface-cli login` after pip packages are installed, so the worker is authenticated before any task runs. The `mps` plugin uses it to start the NVIDIA MPS daemon.

**`decorate`** wraps each `@sky.compute` function at execution time on the remote worker. It is a classic Python decorator: it takes a function and returns a function. This is for per-task logic that must run every time a function executes — things like logging, metrics collection, or framework-specific wrappers that depend on each call's arguments.

**`around_app`** is a context manager that runs once per worker process. It receives an `InstanceInfo` and returns a context manager. The context is entered when the first task arrives and stays active for the lifetime of the worker. This is designed for one-time, process-wide initialization — things that must happen exactly once and persist. The `torch` plugin uses this for `dist.init_process_group()`, the `jax` plugin for `jax.distributed.initialize()`, the `keras` plugin for `DataParallel` distribution setup, and the `cuml` plugin for `cuml.accel.install()`. All are irreversible, process-global operations that should not be repeated per task.

The state module (`skyward.plugins.state`) tracks which `around_app` hooks have been entered. It stores the context managers in a module-level dictionary and checks before entering — if the key already exists, it is a no-op. This makes the hook idempotent: even if multiple tasks execute on the same worker, each `around_app` is entered exactly once.

**`around_client`** is a context manager that runs on the client side, not the worker. It receives the `ComputePool` and the `Cluster`, and wraps the pool's entire active lifetime. The `joblib` and `sklearn` plugins use this to register the `SkywardBackend` as joblib's parallel backend, so that any `Parallel(n_jobs=-1)` call inside the `with` block dispatches work to the cluster instead of local processes.

## Builder API

You can construct plugins using the builder pattern instead of passing all hooks to the constructor:

```python
plugin = (
    Plugin.create("my-plugin")
    .with_image_transform(lambda img, cluster: replace(img, pip=(*img.pip, "my-lib")))
    .with_decorator(my_decorator)
    .with_around_app(my_lifecycle)
)
```

Each `.with_*` method returns a new `Plugin` instance (immutable — uses `replace()`). This is how the built-in plugins are implemented internally: the factory function (e.g., `sky.plugins.torch()`) defines the hooks as closures and chains them together with the builder.

## How Hooks Execute

The hooks run at different points in the pool lifecycle, and the order matters.

When the pool starts (`ComputePool.__enter__`):

1. **`transform`** hooks run first, in plugin order. Each transform receives the image returned by the previous one. The final image is used to generate the bootstrap script.
2. **`bootstrap`** hooks run after the standard bootstrap phases complete on each worker. The ops are appended in plugin order.
3. **`around_client`** hooks are entered on the client, in plugin order.

When a task executes on a worker:

4. **`around_app`** hooks are lazily entered on first task execution (idempotent — skipped if already active).
5. **`decorate`** hooks wrap the function. If multiple plugins have decorators, they are chained: the first plugin's decorator is outermost, the last is innermost. The chaining uses `functools.reduce` over `reversed(decorators)`, so the first plugin listed in `plugins=[...]` runs first and the last runs last.

When the pool stops (`ComputePool.__exit__`):

6. **`around_client`** contexts are exited in reverse order.
7. **`around_app`** contexts are exited in reverse order when the worker process shuts down.

## Plugin Composition

Plugins compose naturally because each hook is independent. You can stack multiple plugins and their effects combine:

```python
with sky.ComputePool(
    provider=sky.AWS(),
    accelerator="A100",
    nodes=4,
    plugins=[
        sky.plugins.torch(backend="nccl"),
        sky.plugins.huggingface(token="hf_xxx"),
    ],
) as pool:
    train() >> pool
```

The `torch` plugin adds PyTorch to pip and initializes DDP via `around_app`. The `huggingface` plugin adds transformers, datasets, and tokenizers to pip, sets `HF_TOKEN`, and runs `huggingface-cli login`. Their image transforms compose (PyTorch packages + HuggingFace packages), and their `around_app` hooks are entered independently in plugin order.

Order can matter. When using Keras with JAX, the JAX plugin should come first because its `around_app` initializes the distributed runtime that Keras's `decorate` depends on:

```python
plugins=[sky.plugins.jax(), sky.plugins.keras(backend="jax")]
```

The JAX plugin's `around_app` calls `jax.distributed.initialize()`, and Keras's `around_app` calls `keras.distribution.set_distribution(DataParallel(...))`. The distribution setup needs JAX's device mesh to already be visible, so JAX must initialize first. Since `around_app` hooks are entered in plugin order, listing JAX first ensures the correct sequence.

## Built-in Plugins

Skyward ships with eight plugins:

| Plugin | Primary Hooks | Purpose |
|--------|--------------|---------|
| [`torch`](torch.md) | `transform`, `around_app` | PyTorch installation and DDP initialization |
| [`jax`](jax.md) | `transform`, `around_app` | JAX installation and distributed initialization |
| [`keras`](keras.md) | `transform`, `around_app` | Keras backend configuration and DataParallel |
| [`huggingface`](huggingface.md) | `transform`, `bootstrap` | Transformers, datasets, tokenizers, and auth |
| [`joblib`](joblib.md) | `transform`, `around_client` | Distributed joblib parallel backend |
| [`sklearn`](sklearn.md) | `transform`, `around_client` | Scikit-learn with distributed joblib |
| [`cuml`](cuml.md) | `transform`, `around_app` | GPU-accelerated scikit-learn via RAPIDS cuML |
| [`mps`](mps.md) | `transform`, `bootstrap` | NVIDIA Multi-Process Service for GPU sharing |

## Custom Plugins

Building a custom plugin follows the same pattern as the built-in ones. Define your hooks as functions, then chain them with the builder:

```python
from dataclasses import replace
from skyward.plugins import Plugin

def my_framework() -> Plugin:
    def transform(image, cluster):
        return replace(image, pip=(*image.pip, "my-framework"))

    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            setup_my_framework()
            return fn(*args, **kwargs)
        return wrapper

    return (
        Plugin.create("my-framework")
        .with_image_transform(transform)
        .with_decorator(decorate)
    )
```

Use it like any built-in plugin:

```python
with sky.ComputePool(
    provider=sky.AWS(),
    plugins=[my_framework()],
) as pool:
    my_task() >> pool
```

## Next Steps

- [PyTorch](torch.md) — DDP initialization and CUDA wheel management
- [JAX](jax.md) — Distributed initialization with `around_app`
- [Keras](keras.md) — Backend-agnostic training with DataParallel
- [Distributed Training](../distributed-training.md) — How plugins fit into multi-node training
- [Getting Started](../getting-started.md) — First steps with Skyward
