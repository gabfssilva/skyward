# Keras

Keras 3 is backend-agnostic: the same model definition compiles and runs on JAX, PyTorch, or TensorFlow. But Keras decides which backend to use at import time, reading the `KERAS_BACKEND` environment variable the moment you write `import keras`. If the variable is not set by then, Keras falls back to its default, and there is no way to change the backend after the fact. This makes environment configuration critical, and it is precisely what the `keras` plugin handles.

`sky.plugins.keras()` ensures that `KERAS_BACKEND` is set correctly on every worker before any user code runs. It adds `keras` to the worker's pip dependencies, injects the environment variable into the bootstrap image, and re-sets it at worker startup as a safety net via the `around_app` hook. On multi-node JAX clusters, it goes further: it discovers all available devices, configures Keras's `DataParallel` distribution strategy, and synchronizes the random number generator across nodes so that weight initialization and dropout masks are reproducible across the cluster.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `"jax"` \| `"torch"` \| `"tensorflow"` | `"jax"` | The Keras backend to activate on the remote worker. |

The default is `"jax"` because JAX offers the tightest integration with Skyward's automatic distribution. When running on multiple nodes with the JAX backend, the plugin configures `DataParallel` distribution and RNG synchronization without any user code. The `"torch"` and `"tensorflow"` backends work equally well for model definition and single-node training, but multi-node distribution with those backends requires combining the Keras plugin with the corresponding framework plugin and using that framework's native distributed primitives.

## Why the backend must be set early

Keras reads `KERAS_BACKEND` exactly once, at import time. The following sequence silently produces wrong behavior:

```python
import keras                    # reads KERAS_BACKEND — too late to change
os.environ["KERAS_BACKEND"] = "jax"  # has no effect
```

The plugin addresses this at two levels. First, it adds `KERAS_BACKEND` to the `Image` environment dictionary via its `transform` hook. This means the variable is baked into the bootstrap script and is present in the shell environment before the worker process starts. Second, as a defensive measure, the plugin's `around_app` hook re-sets `os.environ["KERAS_BACKEND"]` when the worker initializes, before any user function runs. The two-layer approach — image-level and worker-lifecycle-level — guarantees the backend is always correct regardless of execution timing.

## Multi-node behavior by backend

### JAX backend

The JAX backend is the recommended choice for multi-node Keras training on Skyward. When the plugin detects that `total_nodes > 1` and the backend is `"jax"`, it performs three steps inside the `around_app` hook:

1. Calls `keras.distribution.list_devices()` to discover all JAX devices visible to this process (after `jax.distributed.initialize()` has been called by the JAX plugin).
2. Creates a `DataParallel` distribution with the discovered devices and calls `keras.distribution.set_distribution()` to activate it. This tells Keras to shard data and replicate model parameters across all devices automatically.
3. Calls `initialize_rng()` from Keras's internal JAX distribution library to synchronize random number generation across all nodes. Without this step, each node would initialize model weights differently, breaking gradient synchronization.

This means that on a 4-node cluster, `model.fit()` automatically distributes batches across all 4 nodes and averages gradients, with no changes to your training code.

Note that the JAX plugin must run *before* the Keras plugin so that `jax.distributed.initialize()` completes before Keras tries to list devices. Plugin order in the list matters — `around_app` hooks are entered in plugin order:

```python
plugins=[sky.plugins.jax(), sky.plugins.keras(backend="jax")]  # correct order
```

### PyTorch backend

With `backend="torch"`, the Keras plugin sets the environment variable and adds `keras` to pip, but does not configure any distributed strategy. PyTorch's distributed training relies on `torch.distributed.init_process_group()` and `DistributedDataParallel`, which the `torch` plugin handles. Keras models compiled with the PyTorch backend produce standard `torch.nn.Module` instances under the hood, so you can wrap them with DDP the same way you would with a native PyTorch model.

For multi-node training with the PyTorch backend, combine both plugins:

```python
plugins=[sky.plugins.torch(), sky.plugins.keras(backend="torch")]
```

The `torch` plugin initializes the process group; the `keras` plugin ensures the backend is set correctly. Your training code is responsible for wrapping the model with DDP and using `DistributedSampler`.

### TensorFlow backend

The `"tensorflow"` backend follows the same pattern as PyTorch: the Keras plugin handles backend configuration, and TensorFlow's native distribution mechanisms (`tf.distribute.MultiWorkerMirroredStrategy`) handle multi-node coordination. Combine with appropriate TensorFlow distributed setup in your training function.

### Single node

On a single node, no distribution configuration is needed regardless of backend. The Keras plugin sets `KERAS_BACKEND`, installs `keras`, and your model trains on whatever accelerator the node provides. This is the simplest configuration:

```python
plugins=[sky.plugins.keras()]
```

No companion framework plugin is required for single-node execution — the Keras plugin alone is sufficient.

## Usage

### JAX backend on multiple nodes (recommended)

This is the configuration with the best automatic distribution support. The JAX plugin initializes the distributed runtime, and the Keras plugin layers `DataParallel` on top:

```python
import skyward as sky

@sky.compute
def train():
    import keras
    from keras import layers

    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, y_train = sky.shard(x_train / 255.0, y_train, shuffle=True, seed=42)

    model.fit(x_train, y_train, epochs=5, batch_size=64)
    _, accuracy = model.evaluate(x_test / 255.0, y_test)
    return accuracy

with sky.ComputePool(
    provider=sky.AWS(),
    nodes=2,
    accelerator="T4",
    plugins=[sky.plugins.jax(), sky.plugins.keras(backend="jax")],
) as pool:
    results = train() @ pool
```

Each node trains on its shard of the data. The `DataParallel` distribution configured by the plugin handles parameter synchronization across the JAX device mesh.

### PyTorch backend on multiple nodes

When you prefer PyTorch as the execution engine — perhaps because your pipeline includes PyTorch-specific operations or custom CUDA kernels — use the `torch` backend with both plugins:

```python
with sky.ComputePool(
    provider=sky.AWS(),
    nodes=4,
    accelerator="A100",
    plugins=[sky.plugins.torch(), sky.plugins.keras(backend="torch")],
) as pool:
    results = train() @ pool
```

Your training function will need to handle distributed wrapping (DDP) and data partitioning (`DistributedSampler` or `sky.shard()`) explicitly, as the Keras plugin does not configure automatic distribution for the PyTorch backend.

### Single-node training

For experimentation, prototyping, or workloads that fit on a single GPU, the Keras plugin alone is enough:

```python
with sky.ComputePool(
    provider=sky.AWS(),
    accelerator="T4",
    plugins=[sky.plugins.keras()],
) as pool:
    result = train() >> pool
```

This uses the default JAX backend on one node. No distributed setup runs, and `model.fit()` behaves exactly as it would on your local machine.

## Plugin combinations

The Keras plugin is a backend configurator, not a distributed runtime. For multi-node training, always pair it with the plugin that matches the backend:

| Backend | Plugins | Distribution |
|---------|---------|-------------|
| `"jax"` | `sky.plugins.jax()` + `sky.plugins.keras(backend="jax")` | Automatic `DataParallel` with RNG sync |
| `"torch"` | `sky.plugins.torch()` + `sky.plugins.keras(backend="torch")` | Manual DDP wrapping required |
| `"tensorflow"` | `sky.plugins.keras(backend="tensorflow")` | Manual `tf.distribute` strategy required |
| Any (single node) | `sky.plugins.keras()` | None needed |

The JAX combination is unique in that distribution is fully automatic — the plugins handle everything. With PyTorch and TensorFlow, the Keras plugin provides backend configuration while the framework's native distributed APIs handle the rest.

## Hooks

The Keras plugin uses two of the five available plugin hooks:

**`transform`** modifies the worker's `Image` to add `keras` to pip dependencies and set `KERAS_BACKEND` in the environment dictionary. This ensures the variable is present in the shell environment from the moment the worker process starts.

**`around_app`** is a context manager entered once per worker process. It re-sets `KERAS_BACKEND` in `os.environ` as a safety net, and — for multi-node JAX — configures `DataParallel` distribution and synchronizes the RNG. Since `DataParallel` setup and RNG synchronization are one-time, process-global operations, `around_app` is the natural hook: it runs exactly once and persists for the worker's lifetime, matching the same pattern used by the JAX and PyTorch plugins.

The plugin does not use `bootstrap`, `decorate`, or `around_client` hooks.

## Further reading

- [Keras Training guide](../guides/keras-training.md) — step-by-step MNIST training with Keras and JAX on multiple nodes.
- [What are Plugins?](index.md) — How the plugin system works
- [PyTorch Distributed guide](../guides/pytorch-distributed.md) — relevant if using `backend="torch"` with DDP.
