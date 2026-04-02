"""Cross-provider Docker image catalog with typed metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type CudaVersion = Literal["12.6", "12.8", "12.9"] | str
type UbuntuVersion = Literal["22.04", "24.04"] | str
type CudaVariant = Literal["runtime", "devel", "cudnn-runtime"]
type PyTorchVersion = Literal["2.6", "2.7", "2.8"] | str

_CUDA_PATCHES: dict[str, str] = {
    "12.6": "12.6.3",
    "12.8": "12.8.1",
    "12.9": "12.9.1",
}

_PYTORCH_NGC_TAGS: dict[str, str] = {
    "2.6": "24.11",
    "2.7": "25.03",
    "2.8": "25.04",
}

_PYTORCH_CUDA: dict[str, str] = {
    "2.6": "12.6",
    "2.7": "12.8",
    "2.8": "12.9",
}

_RUNPOD_BASE_TAGS: dict[str, str] = {
    "12.6": "runpod/base:1.0.3-cuda1260-ubuntu2204",
    "12.8": "runpod/base:1.0.3-cuda1280-ubuntu2204",
    "12.9": "runpod/base:1.0.3-cuda1290-ubuntu2204",
}

_RUNPOD_PYTORCH_TAGS: dict[str, str] = {
    "2.6": "runpod/pytorch:2.6.0-py3.12-cuda12.6.3-devel-ubuntu24.04",
    "2.7": "runpod/pytorch:2.7.0-py3.13-cuda12.8.1-devel-ubuntu24.04",
    "2.8": "runpod/pytorch:2.8.0-py3.13-cuda12.8.1-devel-ubuntu24.04",
}

_RUNPOD_PYTORCH_CUDA: dict[str, str] = {
    "2.6": "12.6",
    "2.7": "12.8",
    "2.8": "12.8",
}


@dataclass(frozen=True, slots=True)
class DockerImage:
    """Typed Docker image reference with optional CUDA/Ubuntu metadata.

    Parameters
    ----------
    tag : str
        Full image tag (e.g. ``nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04``).
    cuda : str | None
        CUDA version, if applicable.
    ubuntu : str | None
        Ubuntu version, if applicable.
    """

    tag: str
    cuda: str | None = None
    ubuntu: str | None = None

    def __str__(self) -> str:
        return self.tag

    @classmethod
    def of(
        cls,
        tag: str,
        *,
        cuda: str | None = None,
        ubuntu: str | None = None,
    ) -> DockerImage:
        """Create a DockerImage from a custom tag with optional metadata.

        Parameters
        ----------
        tag : str
            Full image tag.
        cuda : str | None
            CUDA version metadata.
        ubuntu : str | None
            Ubuntu version metadata.

        Returns
        -------
        DockerImage
            New image instance.
        """
        return cls(tag=tag, cuda=cuda, ubuntu=ubuntu)


def cuda(
    version: CudaVersion,
    *,
    variant: CudaVariant = "runtime",
    ubuntu: UbuntuVersion = "24.04",
) -> DockerImage:
    """Build an NVIDIA CUDA base image tag.

    Parameters
    ----------
    version : CudaVersion
        CUDA major.minor version (e.g. ``"12.9"``).
    variant : CudaVariant
        Image variant: ``"runtime"``, ``"devel"``, or ``"cudnn-runtime"``.
    ubuntu : UbuntuVersion
        Ubuntu base version (e.g. ``"24.04"``).

    Returns
    -------
    DockerImage
        Image pointing to ``nvcr.io/nvidia/cuda:<patch>-<variant>-ubuntu<ubuntu>``.
    """
    patch = _CUDA_PATCHES.get(version, f"{version}.0")
    tag = f"nvcr.io/nvidia/cuda:{patch}-{variant}-ubuntu{ubuntu}"
    return DockerImage(tag=tag, cuda=version, ubuntu=ubuntu)


def ubuntu(version: UbuntuVersion = "24.04") -> DockerImage:
    """Build a plain Ubuntu image tag.

    Parameters
    ----------
    version : UbuntuVersion
        Ubuntu release version (e.g. ``"24.04"``).

    Returns
    -------
    DockerImage
        Image pointing to ``ubuntu:<version>``.
    """
    return DockerImage(tag=f"ubuntu:{version}", ubuntu=version)


def pytorch(version: PyTorchVersion, *, cuda: CudaVersion | None = None) -> DockerImage:
    """Build an NVIDIA NGC PyTorch image tag.

    Parameters
    ----------
    version : PyTorchVersion
        PyTorch major.minor version (e.g. ``"2.8"``).
    cuda : CudaVersion | None
        Optional CUDA version override; defaults to the version bundled with the NGC tag.

    Returns
    -------
    DockerImage
        Image pointing to ``nvcr.io/nvidia/pytorch:<ngc_tag>-py3``.
    """
    match _PYTORCH_NGC_TAGS.get(version):
        case str() as ngc_tag:
            tag = f"nvcr.io/nvidia/pytorch:{ngc_tag}-py3"
            resolved_cuda = cuda or _PYTORCH_CUDA.get(version)
        case _:
            tag = "nvcr.io/nvidia/pytorch:latest"
            resolved_cuda = cuda
    return DockerImage(tag=tag, cuda=resolved_cuda)


def runpod_base(*, cuda: CudaVersion = "12.9") -> DockerImage:
    """Build a RunPod base image tag.

    Parameters
    ----------
    cuda : CudaVersion
        CUDA version (e.g. ``"12.9"``).

    Returns
    -------
    DockerImage
        Image pointing to the matching ``runpod/base`` tag.
    """
    tag = _RUNPOD_BASE_TAGS.get(cuda, "runpod/base:latest")
    return DockerImage(tag=tag, cuda=cuda)


def runpod_pytorch(version: PyTorchVersion = "2.8", *, cuda: CudaVersion | None = None) -> DockerImage:
    """Build a RunPod PyTorch image tag.

    Parameters
    ----------
    version : PyTorchVersion
        PyTorch major.minor version (e.g. ``"2.8"``).
    cuda : CudaVersion | None
        Optional CUDA version override; defaults to the version bundled with the tag.

    Returns
    -------
    DockerImage
        Image pointing to the matching ``runpod/pytorch`` tag.
    """
    tag = _RUNPOD_PYTORCH_TAGS.get(version, "runpod/pytorch:latest")
    resolved_cuda = cuda or _RUNPOD_PYTORCH_CUDA.get(version)
    return DockerImage(tag=tag, cuda=resolved_cuda)


__all__ = [
    "CudaVariant",
    "CudaVersion",
    "DockerImage",
    "PyTorchVersion",
    "UbuntuVersion",
    "cuda",
    "pytorch",
    "runpod_base",
    "runpod_pytorch",
    "ubuntu",
]
