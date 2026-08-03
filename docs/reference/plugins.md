# Plugins

A plugin is a frozen, serializable value included in the compute specification. The daemon writes its kind and parameters with the compute and reconstructs it on the worker. Plugins must not contain live clients, sockets, closures, or other process-local handles.

```python
import skyward as sky

with sky.Compute(
    provider=sky.AWS(),
    accelerator="A100",
    plugins=[sky.plugins.Torch(backend="nccl")],
) as compute:
    train(data) >> compute
```

The provider belongs on `Compute`; plugins belong in `plugins=[...]`. A plugin can transform the image, add bootstrap phases, prepare the worker, wrap each task, and install a client-side context.

## Built-in plugins

The worker registers these plugin kinds under `sky.plugins`:

| Plugin | Purpose | Main fields |
|---|---|---|
| `Torch` | PyTorch distributed initialization | `backend`, `cuda`, `version` |
| `HuggingFace` | Hugging Face installation and token setup | `token` |
| `Joblib` | Joblib parallel backend | `version` |
| `Jax` | JAX distributed initialization | `cuda` |
| `Keras` | Keras backend setup | `backend` |
| `Cuml` | RAPIDS cuML installation and initialization | `cuda` |
| `Sklearn` | scikit-learn client integration | `version` |
| `Accelerate` | Hugging Face Accelerate environment and collectives | `config`, `env` |
| `Mig` | NVIDIA MIG setup | `profile` |
| `Mps` | Apple Metal performance settings | `active_thread_percentage`, `pinned_memory_limit` |

Collective plugins such as `Torch`, `Jax`, and `Accelerate` make the worker topology part of the job. A compute using one cannot be elastically resized while the collective is active.

## Plugin lifecycle

The base plugin hooks have separate scopes:

- `image(image)` changes packages or image settings before provisioning;
- `bootstrap(image, concurrency)` adds generated node bootstrap phases;
- `setup(info)` runs once for the worker lifetime;
- `run(call, info)` wraps each task;
- `client(compute)` runs once in the client process while the compute is ready.

::: skyward.plugins.Plugin

::: skyward.plugins.Torch

::: skyward.plugins.HuggingFace

::: skyward.plugins.Joblib

::: skyward.plugins.Jax

::: skyward.plugins.Keras

::: skyward.plugins.Cuml

::: skyward.plugins.Sklearn

::: skyward.plugins.Accelerate

::: skyward.plugins.Mig

::: skyward.plugins.Mps
