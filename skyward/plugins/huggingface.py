"""Hugging Face plugin — transformers, datasets, tokenizers + auth."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from skyward.plugins.plugin import Plugin
from skyward.providers.bootstrap import phase_simple

if TYPE_CHECKING:
    from skyward.api.spec import Image


def huggingface(token: str) -> Plugin:
    """Hugging Face plugin with auth and core libraries.

    Parameters
    ----------
    token
        Hugging Face API token (HF_TOKEN).
    """

    def transform(image: Image) -> Image:
        return replace(
            image,
            pip=(*image.pip, "transformers", "datasets", "tokenizers"),
            env={**image.env, "HF_TOKEN": token},
        )

    return (
        Plugin.create("huggingface")
        .with_image_transform(transform)
        .with_bootstrap(
            phase_simple("huggingface", "huggingface-cli login --token $HF_TOKEN"),
        )
    )
