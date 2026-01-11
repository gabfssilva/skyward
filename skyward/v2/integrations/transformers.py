"""Hugging Face Transformers distributed training integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

__all__ = ["transformers"]


def transformers[**P, R](
    backend: Literal["nccl", "gloo"] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Configure Hugging Face Transformers distributed training.

    Uses PyTorch distributed backend internally.

    Args:
        backend: Process group backend. Auto-detected if None.

    Example:
        from skyward.v2.integrations import transformers

        @transformers(backend="nccl")
        @compute
        def train():
            from transformers import Trainer
            ...
    """
    from skyward.v2.integrations.torch import torch

    return torch(backend=backend)
