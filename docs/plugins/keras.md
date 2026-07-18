# Keras

Keras 3 is backend-agnostic: the same model definition compiles and runs on JAX, PyTorch, or TensorFlow. But Keras reads `KERAS_BACKEND` once, when it is imported, and is stuck with what it finds. There is no way to change the backend after the fact.

`sky.plugins.Keras()` sets that variable in the worker process, before any task imports keras. The image cannot do it: the image's `env` reaches the bootstrap shell, which exits, and never the worker the tasks run in.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `"jax"` \| `"tensorflow"` \| `"torch"` | `"jax"` | The framework Keras runs on, and the package installed to carry it |

The backend value doubles as its package name, which is why one field installs both.

## How it works

### `image`

Appends `keras` and the backend's own package to the image's pip list. With the default that is `keras` and `jax`.

Note that this installs the plain `jax` wheel, not a CUDA build. For GPU work, pair with [`sky.plugins.Jax()`](jax.md), which adds the CUDA-specific wheel and its index.

### `setup`

Entered once in the worker process, before it takes its first task:

1. Sets `KERAS_BACKEND` to the configured backend.
2. On more than one node **and** the `jax` backend only: calls `keras.distribution.list_devices()`, and if it finds any, activates `keras.distribution.DataParallel(devices=devices, auto_shard_dataset=False)` and calls Keras's internal `initialize_rng()` to synchronize random state across nodes.

That second step is data-parallel training: every node runs the same graph over its own shard, and the only thing they must agree on is the random state. Forming the JAX process group is not this plugin's job — pair it with `Jax`, which is the collective and the one that knows the rendezvous.

`auto_shard_dataset=False` is deliberate: sharding is yours to do, with `sky.shard()`.

## Multi-node behaviour by backend

| Backend | Plugins | Distribution |
|---------|---------|--------------|
| `"jax"` | `Jax()` + `Keras(backend="jax")` | `DataParallel` with RNG sync, configured by the plugin |
| `"torch"` | `Torch()` + `Keras(backend="torch")` | Nothing automatic — wrap in DDP yourself |
| `"tensorflow"` | `Keras(backend="tensorflow")` | Nothing automatic — use `tf.distribute` yourself |
| Any, one node | `Keras()` | None needed |

With `backend="torch"` or `"tensorflow"` the plugin only sets the environment variable and installs the packages. Keras models on the PyTorch backend produce ordinary `torch.nn.Module` instances, so DDP wrapping works the same way it would on a native model.

## Usage

### JAX backend on multiple nodes

```python
import skyward as sky


@sky.function
def train():
    import keras
    from keras import layers

    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    x_train, y_train = sky.shard(x_train, y_train, shuffle=True, seed=42)

    model.fit(x_train, y_train, epochs=5, batch_size=64)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return float(accuracy)


with sky.Compute(
    provider=sky.AWS(),
    nodes=2,
    accelerator=sky.accelerators.T4(),
    plugins=[sky.plugins.Jax(), sky.plugins.Keras(backend="jax")],
) as compute:
    results = train() @ compute
```

Each node trains on its own shard; the `DataParallel` distribution handles parameter synchronization across the JAX device mesh.

### Single node

```python
with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.T4(),
    plugins=[sky.plugins.Keras()],
) as compute:
    result = train() >> compute
```

No distribution runs on one node, whatever the backend. `model.fit()` behaves exactly as it would locally.

## Further reading

- [Keras Training guide](../guides/keras-training.md) — MNIST across nodes with Keras and JAX
- [JAX plugin](jax.md) — the collective the `jax` backend needs
- [What are plugins?](index.md) — the hook model
