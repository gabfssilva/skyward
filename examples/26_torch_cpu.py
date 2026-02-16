"""Torch CPU with Extra Index URL.

Validates that bootstrap correctly resolves PyTorch from the CPU index
without breaking transitive dependencies like markupsafe.

Uses --index-strategy unsafe-best-match to prevent uv from pulling
incompatible wheels from the PyTorch index for non-torch packages.
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
            pip_extra_index_url="https://download.pytorch.org/whl/cpu",
        ),
    ) as pool:
        result = matrix_multiply(512) >> pool
        print(result)
