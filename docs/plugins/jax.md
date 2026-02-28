# JAX

JAX treats multiple machines as one big device mesh. After a single call to `jax.distributed.initialize()`, every node sees every accelerator across the cluster — `jax.devices()` returns the full set, and `jit` with sharding constraints distributes computation over it. The catch is that every process must call `initialize()` with the coordinator address, cluster size, and its own index, exactly once, before any distributed operation. `sky.plugins.jax()` takes care of this: it installs JAX with the correct CUDA wheels on the remote workers and calls `jax.distributed.initialize()` at startup with the topology from `instance_info()`.

## Parameters

The plugin accepts a single parameter:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cuda` | `str` | `"cu124"` | CUDA version suffix for the JAX wheel |

The `cuda` value becomes the extra specifier in the pip requirement — `jax[cu124]` — and controls which CUDA-specific wheels are pulled from Google's JAX release index. If your cluster runs CUDA 12.4, the default works. For other CUDA versions, pass the matching suffix (e.g., `"cu121"` for CUDA 12.1).

## How it works

### Image transform

The `transform` hook modifies the worker's `Image` before bootstrap. It does two things:

1. Appends `jax[{cuda}]` to the pip dependency list, where `{cuda}` is the configured CUDA suffix.
2. Adds Google's JAX CUDA release index (`https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`) as a pip index, scoped to the `jax` and `jaxlib` packages.

This means JAX and its CUDA bindings are installed from Google's official release channel during worker bootstrap. You do not need JAX installed locally — the plugin adds it to the remote environment.

### Worker lifecycle (`around_app`)

The `around_app` hook is a context manager that runs once when the worker process starts, before any task executes. It calls:

```python
jax.distributed.initialize(
    coordinator_address=f"{info.head_addr}:{info.head_port}",
    num_processes=info.total_nodes,
    process_id=info.node,
)
```

The values come from `instance_info()` — Skyward's runtime API that exposes the cluster topology to each worker. `head_addr` is the private IP of node 0 (the coordinator), `head_port` is the coordination port, `total_nodes` is the cluster size, and `node` is this process's index (0 through N-1).

After this call returns, JAX's global state is initialized. Every call to `jax.devices()` returns the full set of accelerators across all nodes, and JAX's compiler can partition computation across the entire mesh.

## Usage

```python
import skyward as sky

@sky.compute
def train():
    import jax
    import jax.numpy as jnp

    # jax.distributed is already initialized
    # all devices across all nodes are visible
    devices = jax.devices()
    print(f"Total devices: {len(devices)}")

    # distributed computation works out of the box
    mesh = jax.sharding.Mesh(jax.devices(), axis_names=("devices",))
    ...

with sky.ComputePool(
    provider=sky.AWS(),
    accelerator="A100",
    nodes=4,
    plugins=[sky.plugins.jax()],
) as pool:
    results = train() @ pool
```

The `@` operator broadcasts the function to all nodes. Each node executes `train()`, and by the time the function body runs, `jax.distributed.initialize()` has already been called by the plugin. The function sees the full device mesh and can use JAX's sharding primitives to partition computation.

## Combining with Keras

JAX is the recommended backend for multi-node Keras training. When using Keras with JAX, stack both plugins:

```python
with sky.ComputePool(
    provider=sky.AWS(),
    accelerator="A100",
    nodes=2,
    plugins=[
        sky.plugins.jax(),
        sky.plugins.keras(backend="jax"),
    ],
) as pool:
    results = train() @ pool
```

Order matters here. The JAX plugin's `around_app` initializes the distributed runtime, and the Keras plugin sets `KERAS_BACKEND=jax` so Keras uses JAX as its computation backend. Together, they give you multi-node Keras training where JAX handles the distributed device mesh and Keras provides the high-level model API.

The [Keras Training guide](../guides/keras-training.md) walks through a complete MNIST example using this combination.

## Next steps

- [Keras Training](../guides/keras-training.md) — JAX + Keras on multiple GPUs
- [What are Plugins?](index.md) — How the plugin system works
- [PyTorch Distributed](../guides/pytorch-distributed.md) — The PyTorch equivalent for comparison
