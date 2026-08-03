# What are plugins?

Skyward's plugin system is how you bring a third-party framework into a compute. When you pass `plugins=[sky.plugins.Torch()]` to `Compute`, you are telling Skyward: install PyTorch on the nodes, form the process group before my function runs, and do it once. The plugin handles the image, the bootstrap phases, and the per-task setup — things you would otherwise do by hand with `Image(pip=[...])`, environment variables, and boilerplate inside every `@sky.function`.

Plugins operate at the compute level, not the function level. One declaration affects every task dispatched to that compute.

## A plugin is a value

A `Plugin` is a frozen [msgspec](https://jcristharif.com/msgspec/) `Struct`, not an object holding callbacks. It travels in the compute spec, is written to the daemon's database with it, and is rebuilt from its parameters on the node. That is why it cannot be a closure or a lambda: three processes on two machines have to agree about it, and what they agree about is its `kind` and its fields.

```python
class Torch(Plugin, frozen=True):
    kind: ClassVar[str] = "torch"
    collective: ClassVar[bool] = True

    backend: Literal["nccl", "gloo"] = "nccl"
    cuda: str = "cu128"
    version: str | None = None
```

Two class-level attributes describe the plugin rather than configure it:

- **`kind`** — its name on the wire, and how the node finds the class again. Unknown kinds are refused when the compute is created, not an hour later on a worker.
- **`collective`** — whether the plugin makes nodes depend on each other. A collective freezes the world when the last rank joins it, so the reconciler refuses to resize a compute running one: taking a rank away does not shrink the job, it hangs it at the next all-reduce. `Torch`, `Jax` and `Accelerate` are collectives.

Parameters are validated against the class's own fields when the compute is created, so a misspelt backend comes back from the call that made the pool.

## The five hooks

Every hook is optional and does nothing by default. Two run on the daemon, two on the node, one on the client.

| Hook | Signature | Where | When |
|---|---|---|---|
| `image` | `(Image) -> Image` | daemon | Once, when the compute is provisioned |
| `bootstrap` | `(Image, concurrency: int) -> tuple[str, ...]` | daemon | Script generation; phases appended after the image's own |
| `setup` | `(Info) -> ContextManager[None]` | node | Entered once before the worker takes a task, left when it stops |
| `run` | `(call, Info) -> T` | node | Around every task |
| `client` | `(Compute) -> ContextManager[None]` | client | Entered when the compute is ready, left before teardown |

**`image`** is a transform rather than a package list because plugins compose: each is handed what the ones before it asked for. It returns a new `Image` via `replace()` — pip packages, pip indexes, apt packages.

**`bootstrap`** runs on the daemon at script-generation time, so it may only *return* the shell phases the script will run; it never executes anything itself. `concurrency` is the worker's width — the one datum a phase needs that the image does not carry. `Mig` uses it to know how many ways to cut the GPU.

**`setup`** is the worker's own lifetime, in the worker process. This is where an environment variable a library reads at import time gets set (`Keras` and `KERAS_BACKEND`), or where a daemon every later child must inherit gets started (`Mps`).

**`run`** wraps one task, in the process that actually runs it. That distinction matters under a subprocess executor: the process that must hold the process group, or the patched scikit-learn, or the pinned CUDA device, is the child — not the worker that spawned it. Every built-in that does irreversible process-global work (`Torch`, `Jax`, `Accelerate`, `Cuml`, `Mig`) therefore does it here, once, guarded by a module-global flag under a lock. The first task pays for it; every task after calls straight through.

**`client`** is the only hook that does not travel. It runs in the process that opened the `with` block, and it is how a plugin reaches back into the live pool — `Joblib` and `Sklearn` use it to point joblib's parallel backend at the compute.

## Order

Plugins wrap in the order they are listed: the first is outermost, and therefore the one whose `run` executes first.

```python
plugins=[sky.plugins.Jax(), sky.plugins.Keras(backend="jax")]
```

`Jax.run` joins the distributed runtime before anything downstream of it sees a device list.

## Built-in plugins

| Plugin | Hooks | Purpose |
|--------|-------|---------|
| [`Torch`](torch.md) | `image`, `run` | PyTorch and its process group |
| [`Accelerate`](accelerate.md) | `image`, `run` | FSDP, DeepSpeed and mixed precision via Hugging Face Accelerate |
| [`Jax`](jax.md) | `image`, `run` | JAX and its distributed runtime |
| [`Keras`](keras.md) | `image`, `setup` | Keras 3 backend selection, plus DataParallel on JAX |
| `HuggingFace` | `image`, `setup` | `huggingface_hub` and `HF_TOKEN` |
| [`Joblib`](joblib.md) | `image`, `client` | joblib's parallel backend, fanned over the compute |
| [`Sklearn`](sklearn.md) | `image`, `client` | scikit-learn on the nodes, its joblib backend on the compute |
| [`Cuml`](cuml.md) | `image`, `run` | GPU scikit-learn via RAPIDS cuML |
| [`Mps`](mps.md) | `setup` | CUDA MPS, so one GPU serves several tasks at once |
| [`Mig`](mig.md) | `bootstrap`, `run` | NVIDIA MIG, one GPU slice per subprocess |

## Custom plugins

There is no builder API and no ad-hoc registration from a lambda — a plugin has to survive being serialized into the spec and rebuilt from its `kind`. A custom plugin is a subclass registered in `skyward.plugins.PLUGINS`:

```python
from typing import ClassVar

from msgspec.structs import replace

from skyward.worker.plugins import PLUGINS, Plugin
from skyward.shared.schemas import Image


class MyFramework(Plugin, frozen=True):
    kind: ClassVar[str] = "my-framework"

    version: str | None = None

    def image(self, image: Image) -> Image:
        package = f"my-framework=={self.version}" if self.version else "my-framework"
        return replace(image, pip=(*image.pip, package))


PLUGINS[MyFramework.kind] = MyFramework
```

The class has to be importable on the node too, since that is where it is rebuilt from its parameters.

## Next steps

- [PyTorch](torch.md) — the process group and CUDA wheel selection
- [JAX](jax.md) — the distributed runtime
- [Keras](keras.md) — backend-agnostic training
- [Distributed Training](../distributed-training.md) — how plugins fit into multi-node training
