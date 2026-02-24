"""Torch CPU with Scoped Index.

Validates that bootstrap correctly resolves PyTorch from the CPU index
without breaking transitive dependencies like markupsafe.

Uses uv's explicit index support (``[[tool.uv.index]]`` + ``[tool.uv.sources]``)
to scope the PyTorch index to torch/torchvision only.
"""

import skyward as sky


@sky.compute
def matrix_multiply(size: int) -> dict:
    import torch

    a = torch.randn(size, size)
    b = torch.randn(size, size)
    c = a @ b
    return {"shape": list(c.shape), "device": str(c.device)}


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.AWS(),
        image=sky.Image(
            pip=["torch", "torchvision"],
            pip_indexes=[
                sky.PipIndex(
                    url="https://download.pytorch.org/whl/cpu",
                    packages=["torch", "torchvision"],
                ),
            ],
        ),
    ) as pool:
        result = matrix_multiply(512) >> pool
        print(result)
