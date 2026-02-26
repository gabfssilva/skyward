"""PyTorch plugin — environment + distributed training."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from skyward.accelerators import Accelerator
from skyward.api.spec import PipIndex
from skyward.plugins.plugin import Plugin

if TYPE_CHECKING:
    from skyward.api.model import Cluster
    from skyward.api.runtime import InstanceInfo
    from skyward.api.spec import Image


def torch(
    backend: Literal["nccl", "gloo"] | None = None,
    cuda: str = "cu128",
    version: str | Literal['latest'] = 'latest',
    vision: str | Literal['latest'] | None = None,
    audio: str | Literal['latest'] | None = None,
) -> Plugin:
    """PyTorch plugin with CUDA index and DDP initialization.

    Parameters
    ----------
    backend
        Process group backend. Auto-detected if None (nccl for GPU, gloo for CPU).
    cuda
        CUDA version suffix for PyTorch wheel index.
    """

    def transform(image: Image, cluster: Cluster[Any]) -> Image:
        torch_packages = []

        if version == 'latest':
            torch_packages.append('torch')
        elif version.startswith('>') or version.startswith('=='):
            torch_packages.append(f'torch{version}')
        else:
            torch_packages.append(f'torch=={version}')

        if vision == 'latest':
            torch_packages.append('torchvision')
        elif vision and (vision.startswith('>') or vision.startswith('==')):
            torch_packages.append(f'torchvision{vision}')
        elif vision:
            torch_packages.append(f'torchvision=={vision}')

        if audio == 'latest':
            torch_packages.append('torchaudio')
        elif audio and (audio.startswith('>') or audio.startswith('==')):
            torch_packages.append(f'torchaudio{vision}')
        elif audio:
            torch_packages.append(f'torchaudio=={vision}')

        pipindex = []

        match cluster.spec.accelerator:
            case Accelerator() as accelerator if accelerator.metadata and 'cuda' in accelerator.metadata:
                pipindex.append(PipIndex(
                    url=f"https://download.pytorch.org/whl/{cuda}",
                    packages=torch_packages,
                ))
            case _:
                pipindex.append(PipIndex(
                    url="https://download.pytorch.org/whl/cpu",
                    packages=torch_packages,
                ))

        return replace(
            image,
            pip=(*image.pip, *torch_packages),
            pip_indexes=(*image.pip_indexes, *pipindex),
        )

    @contextmanager
    def around(info: InstanceInfo) -> Iterator[None]:
        import torch as _torch  # type: ignore[reportMissingImports]
        import torch.distributed as dist  # type: ignore[reportMissingImports]

        from skyward.observability.logger import logger

        log = logger.bind(plugin="torch")

        if info.total_nodes < 2:
            yield
            return

        env = {
            "MASTER_ADDR": info.head_addr,
            "MASTER_PORT": str(info.head_port),
            "WORLD_SIZE": str(info.total_nodes),
            "RANK": str(info.node),
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
            "NODE_RANK": str(info.node),
        }
        for key, value in env.items():
            if value:
                os.environ[key] = value

        be = backend or ("nccl" if _torch.cuda.is_available() else "gloo")  # type: ignore[reportAttributeAccessIssue]
        log.debug(
            "Initializing process group: backend={be}, rank={rank}, world_size={ws}",
            be=be, rank=info.node, ws=info.total_nodes,
        )
        dist.init_process_group(backend=be, init_method="env://")
        try:
            yield
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    return (
        Plugin.create("torch")
        .with_image_transform(transform)
        .with_around_app(around)
    )
