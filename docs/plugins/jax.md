# JAX

JAX treats multiple machines as one big device mesh. After a single call to `jax.distributed.initialize()`, every process sees every accelerator across the compute — `jax.devices()` returns the full set, and `jit` with sharding constraints distributes computation over it. The catch is that every process must call `initialize()` with the coordinator address, the number of processes, and its own index, exactly once, before any distributed operation.

`sky.plugins.Jax()` does that: it installs JAX with the CUDA wheels on the nodes and joins the cluster on each process's first task.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cuda` | `str` | `"cu124"` | The CUDA build to install JAX from, as a `jax[...]` extra |

The value becomes the extra in the pip requirement — `jax[cu124]` — and selects the CUDA-specific wheels pulled from Google's JAX release index. It is pinned rather than left to the default because the wheel has to match the driver the GPU images ship.

`Jax` is a **collective** plugin: a compute running one cannot be resized, because removing a rank does not shrink the job, it hangs it.

## How it works

### `image`

The `image` hook appends `jax[{cuda}]` to the image's pip list and adds Google's JAX CUDA release index (`https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`), scoped to the `jax` and `jaxlib` packages. You do not need JAX installed locally.

### `run`

On the first task in each process, the plugin calls:

```python
jax.distributed.initialize(
    coordinator_address=f"{info.head}:1234",
    num_processes=info.nodes,
    process_id=info.rank,
)
```

`info.head` is rank zero's address. A Skyward compute has no head; rank zero is a convention that satisfies a library which insists on being told where the rendezvous is.

This happens on the first task rather than at worker startup because `initialize()` is a collective, and the process that blocks in it must be the one that will run the collective code afterwards. Under `executor="process"` that is the child, not the worker. A module-global flag under a lock keeps it to once per process; every subsequent task calls straight through.

After the call, `jax.devices()` returns the accelerators of the whole compute and the compiler can partition across them.

## Usage

```python
import skyward as sky


@sky.function
def train():
    import jax

    devices = jax.devices()
    print(f"Total devices: {len(devices)}")

    mesh = jax.sharding.Mesh(devices, axis_names=("devices",))
    ...


with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.A100(),
    nodes=4,
    plugins=[sky.plugins.Jax()],
) as compute:
    results = train() @ compute
```

The `@` operator broadcasts to every node. By the time the function body runs, the process has joined the cluster.

## Combining with Keras

JAX is the recommended backend for multi-node Keras training. Stack both plugins, JAX first:

```python
with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.A100(),
    nodes=2,
    plugins=[
        sky.plugins.Jax(),
        sky.plugins.Keras(backend="jax"),
    ],
) as compute:
    results = train() @ compute
```

`Jax` forms the cluster; `Keras` sets `KERAS_BACKEND` and, on more than one node, configures Keras's `DataParallel` distribution. See the [Keras plugin](keras.md).

## Next steps

- [Keras](keras.md) — JAX + Keras across nodes
- [What are plugins?](index.md) — the hook model
- [PyTorch](torch.md) — the equivalent for torch
