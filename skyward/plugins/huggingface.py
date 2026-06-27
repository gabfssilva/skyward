"""Hugging Face plugin — pre-download models into the worker cache."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from skyward.api.plugin import Plugin
from skyward.providers.bootstrap import phase
from skyward.providers.bootstrap.compose import SKYWARD_DIR

if TYPE_CHECKING:
    from skyward.api.model import Cluster
    from skyward.api.spec import Image
    from skyward.providers.bootstrap import Op


def huggingface(
    models: list[str],
    token: str | None = None,
) -> Plugin:
    """Hugging Face plugin that pre-downloads models during bootstrap.

    Installs ``huggingface_hub`` and downloads each requested model
    snapshot into the shared HF cache before the worker starts, so that
    ``from_pretrained`` calls inside ``@sky.function`` read from disk
    instead of fetching over the network.

    Parameters
    ----------
    models
        Repo ids to pre-download (e.g. ``["meta-llama/Llama-3.1-8B"]``).
    token
        Hugging Face token for gated or private repos. Exported as
        ``HF_TOKEN`` on every worker.

    Examples
    --------
    >>> with sky.Compute(
    ...     provider=sky.AWS(),
    ...     accelerator=sky.accelerators.A100(),
    ...     plugins=[sky.plugins.huggingface(models=["meta-llama/Llama-3.1-8B"])],
    ... ) as pool:
    ...     result = generate(prompt) >> pool
    """

    def transform(image: Image, cluster: Cluster[Any]) -> Image:
        env = {**image.env, "HF_XET_HIGH_PERFORMANCE": "1"}
        if token:
            env["HF_TOKEN"] = token
        return replace(
            image,
            pip=(*image.pip, "huggingface_hub"),
            env=env,
        )

    def bootstrap_factory(cluster: Cluster[Any]) -> tuple[Op, ...]:
        python_bin = f"{SKYWARD_DIR}/.venv/bin/python"
        repos = ", ".join(f'"{m}"' for m in models)
        snippet = (
            "from huggingface_hub import snapshot_download; "
            f'[print("cached", r, "->", snapshot_download(r), flush=True) for r in [{repos}]]'
        )
        return (phase("hf-download", f"{python_bin} -c '{snippet}'"),)

    return (
        Plugin.create("huggingface")
        .with_image_transform(transform)
        .with_bootstrap(bootstrap_factory)
    )
