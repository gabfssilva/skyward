<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/gabfssilva/skyward/blob/main/docs/logo-dark.svg?raw=true">
    <img src="https://github.com/gabfssilva/skyward/blob/main/docs/logo.svg?raw=true" alt="Skyward" width="140">
  </picture>
</p>

<p align="center">
  <strong>Cloud accelerators with a single decorator</strong>
</p>

<p align="center">
  <a href="https://github.com/gabfssilva/skyward/actions/workflows/tests.yml"><img src="https://github.com/gabfssilva/skyward/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/skyward/"><img src="https://img.shields.io/pypi/v/skyward" alt="PyPI"></a>
  <a href="https://pypi.org/project/skyward/"><img src="https://img.shields.io/pypi/pyversions/skyward" alt="Python"></a>
  <a href="https://github.com/gabfssilva/skyward/blob/main/LICENSE"><img src="https://img.shields.io/github/license/gabfssilva/skyward" alt="License"></a>
</p>

---

<p align="center">
  <img src="https://github.com/gabfssilva/skyward/blob/main/docs/demo.gif?raw=true" alt="Skyward Demo" width="800">
</p>

Skyward is a Python library for ephemeral accelerator compute. Spin up cloud accelerators, run your code, and tear them down automatically. No infrastructure to manage, no idle costs.

## Quick Example

```python
# pi.py
import skyward as sky


@sky.app(
    provider=sky.Salad(priority="high"),
    accelerator=sky.accelerators.RTX_3090(),
    image=sky.Image(pip=["torch", "numpy"]),
)
def estimate_pi(n: int = 100_000_000) -> dict[str, str | float]:
    """Estimate pi from N random points, on a GPU."""
    import torch

    points = torch.rand(2, n, device="cuda")
    inside = (points[0] ** 2 + points[1] ** 2 <= 1).sum().item()

    return {"gpu": torch.cuda.get_device_name(), "pi": 4 * inside / n}
```

```bash
sky server start
sky run pi.py --n 500000000
```

`sky.app` declares the machines a function runs on. `sky run` provisions them, parses the command line against the function's signature, runs the function there, prints what it returned as JSON, and tears the machines down.

### Inside a program

`sky.Compute` is the same compute as a context manager, for a program that dispatches functions itself:

```python
import skyward as sky

@sky.function
def train(epochs: int) -> dict:
    import torch

    model = torch.nn.Linear(100, 10).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        loss = model(torch.randn(32, 100, device="cuda")).sum()
        loss.backward()
        optimizer.step()

    return { "final_loss": loss.item() }


with sky.Compute(
    provider=sky.AWS(), 
    accelerator=sky.accelerators.T4(), 
    image=sky.Image(pip=["torch"])
) as compute:
    result = train(epochs=100) >> compute
    print(result)
```

## Features

- **[A single API, any cloud](https://gabfssilva.github.io/skyward/concepts/)** — A unified declarative API to run functions on AWS, GCP, Hyperstack, RunPod, TensorDock, VastAI, Verda, and more.
- **[Operators, not boilerplate](https://gabfssilva.github.io/skyward/concepts/)** — `>>` executes on one node, `@` broadcasts to all, `&` runs in parallel. No job configs, no YAML.
- **[Ephemeral by default](https://gabfssilva.github.io/skyward/concepts/)** — Instances provision on demand and terminate automatically. Context managers guarantee cleanup.
- **[Multi-provider support](https://gabfssilva.github.io/skyward/providers/)** — AWS, GCP, Hyperstack, RunPod, TensorDock, VastAI, Verda with automatic fallback and cost optimization.
- **[Distributed training](https://gabfssilva.github.io/skyward/distributed-training/)** — PyTorch DDP, Keras 3, JAX, TensorFlow, and HuggingFace integration decorators.
- **[Distributed collections](https://gabfssilva.github.io/skyward/distributed-collections/)** — Dict, set, counter, queue, barrier, and lock replicated across the cluster.
- **[Spot-aware](https://gabfssilva.github.io/skyward/providers/)** — Automatic spot instance selection, preemption detection, and replacement. Save 60-90% on compute costs.

## Install

```bash
uv add "skyward[all]"
```

The base package is what a node installs. The SDK, the daemon, the `sky` command and each provider SDK are extras; [getting started](https://gabfssilva.github.io/skyward/getting-started/) lists them.

## Requirements

- Python 3.12+
- Cloud provider credentials ([setup guide](https://gabfssilva.github.io/skyward/getting-started/))

## Documentation

Full documentation at **[gabfssilva.github.io/skyward](https://gabfssilva.github.io/skyward/)**.

## License

MIT
