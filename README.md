<p align="center">
  <img src="docs/logo_sky.png" alt="Skyward" width="400">
</p>

<p align="center">
  <strong>Cloud accelerators with a single decorator</strong>
</p>

<p align="center">
  <a href="https://github.com/gabfssilva/skyward/actions/workflows/python-package.yml"><img src="https://github.com/gabfssilva/skyward/actions/workflows/python-package.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/skyward/"><img src="https://img.shields.io/pypi/v/skyward" alt="PyPI"></a>
  <a href="https://pypi.org/project/skyward/"><img src="https://img.shields.io/pypi/pyversions/skyward" alt="Python"></a>
  <a href="https://github.com/gabfssilva/skyward/blob/main/LICENSE"><img src="https://img.shields.io/github/license/gabfssilva/skyward" alt="License"></a>
</p>

---

Skyward is a Python library for ephemeral accelerator compute. Spin up cloud accelerators, run your code, and tear them down automatically. No infrastructure to manage, no idle costs.

## Quick Example

```python
import skyward as sky

@sky.compute
def train(epochs: int) -> dict:
    import torch
    model = torch.nn.Linear(100, 10).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        loss = model(torch.randn(32, 100, device="cuda")).sum()
        loss.backward()
        optimizer.step()

    return {"final_loss": loss.item()}

with sky.ComputePool(provider=sky.AWS(), accelerator="T4", image=sky.Image(pip=["torch"])) as pool:
    result = train(epochs=100) >> pool
    print(result)
```

## Features

- **[One decorator, any cloud](https://gabfssilva.github.io/skyward/concepts/)** — `@compute` makes any function remotely executable. AWS, RunPod, VastAI, and Verda with a unified API.
- **[Operators, not boilerplate](https://gabfssilva.github.io/skyward/concepts/)** — `>>` executes on one node, `@` broadcasts to all, `&` runs in parallel. No job configs, no YAML.
- **[Ephemeral by default](https://gabfssilva.github.io/skyward/concepts/)** — Instances provision on demand and terminate automatically. Context managers guarantee cleanup.
- **[Multi-provider support](https://gabfssilva.github.io/skyward/providers/)** — AWS, RunPod, VastAI, Verda with automatic fallback and cost optimization.
- **[Distributed training](https://gabfssilva.github.io/skyward/distributed-training/)** — PyTorch DDP, Keras 3, JAX, TensorFlow, and HuggingFace integration decorators.
- **[Distributed collections](https://gabfssilva.github.io/skyward/distributed-collections/)** — Dict, set, counter, queue, barrier, and lock replicated across the cluster.
- **[Spot-aware](https://gabfssilva.github.io/skyward/providers/)** — Automatic spot instance selection, preemption detection, and replacement. Save 60-90% on compute costs.

## Install

```bash
uv add skyward
```

## Requirements

- Python 3.12+
- Cloud provider credentials ([setup guide](https://gabfssilva.github.io/skyward/getting-started/))

## Documentation

Full documentation at **[gabfssilva.github.io/skyward](https://gabfssilva.github.io/skyward/)**.

## License

MIT
