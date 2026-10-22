<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/gabfssilva/skyward/blob/main/docs/logo-dark.svg?raw=true">
    <img src="https://github.com/gabfssilva/skyward/blob/main/docs/logo.svg?raw=true" alt="Skyward" width="140">
  </picture>
</p>

<p align="center">
  Run Python functions on cloud accelerators with automatic provisioning and cleanup.
</p>

<p align="center">
  <a href="https://github.com/gabfssilva/skyward/actions/workflows/tests.yml"><img src="https://github.com/gabfssilva/skyward/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/skyward/"><img src="https://img.shields.io/pypi/v/skyward" alt="PyPI"></a>
  <a href="https://pypi.org/project/skyward/"><img src="https://img.shields.io/pypi/pyversions/skyward" alt="Python"></a>
  <a href="https://github.com/gabfssilva/skyward/blob/main/LICENSE"><img src="https://img.shields.io/github/license/gabfssilva/skyward" alt="License"></a>
</p>

<p align="center">
  <img src="https://github.com/gabfssilva/skyward/blob/main/docs/demo.gif?raw=true" alt="Skyward Demo" width="800">
</p>

```python
# pi.py
import skyward as sky


@sky.app(
    provider=sky.RunPod(),
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
sky run -f pi.py --n 500000000
```

## Installation

```bash
uv add "skyward[all]"
```

`all` includes every optional extra. For a smaller installation, combine only what you need, e.g. `uv add "skyward[client,server,cli,aws]"`. See [available extras](https://gabfssilva.github.io/skyward/getting-started/#optional-extras).

## Requirements

- Python 3.12+
- Cloud provider credentials ([setup guide](https://gabfssilva.github.io/skyward/getting-started/))

## Documentation

Full documentation at **[gabfssilva.github.io/skyward](https://gabfssilva.github.io/skyward/)**.

## License

[MIT](LICENSE)
