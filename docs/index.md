<p align="center">
  <img src="logo_sky.png" alt="Skyward" width="400">
</p>

<p align="center">
  <strong>Cloud accelerators with a single decorator</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/skyward/"><img src="https://img.shields.io/pypi/v/skyward.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/skyward/"><img src="https://img.shields.io/pypi/pyversions/skyward.svg" alt="Python"></a>
  <a href="https://github.com/gabfssilva/skyward/actions"><img src="https://img.shields.io/github/actions/workflow/status/gabfssilva/skyward/tests.yml" alt="Tests"></a>
  <a href="https://github.com/gabfssilva/skyward/blob/main/LICENSE"><img src="https://img.shields.io/github/license/gabfssilva/skyward.svg" alt="License"></a>
</p>

---

Skyward is a Python library for ephemeral accelerator compute. Spin up cloud accelerators (GPUs, TPUs, Trainium, and more), run your ML training code, and tear them down automatically. No infrastructure to manage.

- **One decorator, any cloud.** `@compute` makes any function remotely executable. AWS, RunPod, VastAI, and Verda with a unified API.
- **Operators, not boilerplate.** `>>` executes on one node, `@` broadcasts to all, `&` runs in parallel. No job configs, no YAML.
- **Ephemeral by default.** Instances provision on demand and terminate automatically. Context managers guarantee cleanup.
- **Multi-accelerator out of the box.** GPUs, TPUs, Trainium — with PyTorch DDP, Keras 3, JAX, TensorFlow, and HuggingFace plugins.
- **Spot-aware.** Automatic spot instance selection, preemption detection, and replacement. Save 60-90% on compute costs.
- **Built on Casty.** Powered by a distributed, asyncio-based actor model for non-blocking orchestration.

## Quick example

```python
import skyward as sky

@sky.compute
def train(data):
    import torch
    model = create_model().cuda()
    return model.fit(data)

with sky.ComputePool(provider=sky.AWS(), accelerator=sky.accelerators.A100()) as pool:
    result = train(my_data) >> pool
# Accelerator terminated automatically
```

## Execution operators

| Operator | Syntax | Description |
|----------|--------|-------------|
| `>>` | `fn() >> pool` | Execute on single worker |
| `@` | `fn() @ pool` | Broadcast to ALL workers |
| `&` | `fn1() & fn2() >> pool` | Parallel execution |
| `gather()` | `gather(fn1(), fn2()) >> pool` | Dynamic parallel execution |

## Next steps

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting Started](getting-started.md)** — Installation, credentials, and first examples
- :material-lightbulb: **[Core Concepts](concepts.md)** — Programming model and ephemeral compute
- :material-cloud: **[Providers](providers.md)** — AWS, RunPod, VastAI, and Verda
- :material-chip: **[Accelerators](accelerators.md)** — Accelerator selection guide
- :material-server-network: **[Distributed Training](distributed-training.md)** — Multi-node with PyTorch, Keras, JAX
- :material-api: **[API Reference](reference/pool.md)** — Full autodoc of all public types

</div>
