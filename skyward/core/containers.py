"""Cross-provider Docker image catalog.

``DockerImage`` is a ``str`` subclass: its value *is* the container image tag,
so ``Image(base=DockerImage.pytorch("2.8"))`` flows straight into the string
``base`` field with no coercion. The classmethods compute tags from static
lookup tables; ``cuda``/``ubuntu`` arguments only shape the computed tag.
"""

from __future__ import annotations

from typing import Literal

type CudaVersion = Literal["12.6", "12.8", "12.9"] | str
type UbuntuVersion = Literal["22.04", "24.04"] | str
type CudaVariant = Literal["runtime", "devel", "cudnn-runtime"]
type PyTorchVersion = Literal["2.6", "2.7", "2.8"] | str

_CUDA_PATCHES: dict[str, str] = {
    "12.0": "12.0.1",
    "12.1": "12.1.1",
    "12.2": "12.2.2",
    "12.3": "12.3.2",
    "12.4": "12.4.1",
    "12.5": "12.5.1",
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
    "12.8": "runpod/base:1.0.3-cuda1281-ubuntu2204",
    "12.9": "runpod/base:1.0.3-cuda1290-ubuntu2204",
}

_RUNPOD_PYTORCH_TAGS: dict[str, str] = {
    "2.6": "runpod/pytorch:2.6.0-py3.12-cuda12.6.3-devel-ubuntu24.04",
    "2.7": "runpod/pytorch:2.7.0-py3.13-cuda12.8.1-devel-ubuntu24.04",
    "2.8": "runpod/pytorch:2.8.0-py3.13-cuda12.8.1-devel-ubuntu24.04",
}


class DockerImage(str):
    """A container image tag, usable anywhere a ``str`` is (e.g. ``Image.base``).

    Instances are plain strings whose value is the fully qualified image tag.
    The classmethods build well-known tags from static tables.
    """

    __slots__ = ()

    @classmethod
    def of(
        cls,
        tag: str,
        *,
        cuda: str | None = None,
        ubuntu: str | None = None,
    ) -> DockerImage:
        """Wrap a custom image tag.

        Parameters
        ----------
        tag : str
            Full image tag.
        cuda : str | None
            CUDA version metadata; does not alter the tag.
        ubuntu : str | None
            Ubuntu version metadata; does not alter the tag.

        Returns
        -------
        DockerImage
            The tag as a ``DockerImage``.
        """
        return cls(tag)

    @classmethod
    def cuda(
        cls,
        version: CudaVersion,
        *,
        variant: CudaVariant = "cudnn-runtime",
        ubuntu: UbuntuVersion = "22.04",
        repository: Literal["nvidia", "dockerhub"] = "dockerhub",
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
        repository : Literal["nvidia", "dockerhub"]
            Image repository.

        Returns
        -------
        DockerImage
            ``nvidia/cuda:<patch>-<variant>-ubuntu<ubuntu>`` (optionally ``nvcr.io``-prefixed).
        """
        patch = _CUDA_PATCHES.get(version, f"{version}.0")
        tag = f"nvidia/cuda:{patch}-{variant}-ubuntu{ubuntu}"
        return cls(tag if repository == "dockerhub" else f"nvcr.io/{tag}")

    @classmethod
    def ubuntu(cls, version: UbuntuVersion = "24.04") -> DockerImage:
        """Build a plain Ubuntu image tag.

        Parameters
        ----------
        version : UbuntuVersion
            Ubuntu release version (e.g. ``"24.04"``).

        Returns
        -------
        DockerImage
            ``ubuntu:<version>``.
        """
        return cls(f"ubuntu:{version}")

    @classmethod
    def pytorch(cls, version: PyTorchVersion, *, cuda: CudaVersion | None = None) -> DockerImage:
        """Build an NVIDIA NGC PyTorch image tag.

        Parameters
        ----------
        version : PyTorchVersion
            PyTorch major.minor version (e.g. ``"2.8"``).
        cuda : CudaVersion | None
            CUDA version metadata; does not alter the tag.

        Returns
        -------
        DockerImage
            ``nvcr.io/nvidia/pytorch:<ngc_tag>-py3``.
        """
        match _PYTORCH_NGC_TAGS.get(version):
            case str() as ngc_tag:
                return cls(f"nvcr.io/nvidia/pytorch:{ngc_tag}-py3")
            case _:
                return cls("nvcr.io/nvidia/pytorch:latest")

    @classmethod
    def runpod_base(cls, *, cuda: CudaVersion = "12.9") -> DockerImage:
        """Build a RunPod base image tag.

        Parameters
        ----------
        cuda : CudaVersion
            CUDA version (e.g. ``"12.9"``).

        Returns
        -------
        DockerImage
            The matching ``runpod/base`` tag.
        """
        return cls(_RUNPOD_BASE_TAGS.get(cuda, "runpod/base:latest"))

    @classmethod
    def runpod_pytorch(cls, version: PyTorchVersion = "2.8", *, cuda: CudaVersion | None = None) -> DockerImage:
        """Build a RunPod PyTorch image tag.

        Parameters
        ----------
        version : PyTorchVersion
            PyTorch major.minor version (e.g. ``"2.8"``).
        cuda : CudaVersion | None
            CUDA version metadata; does not alter the tag.

        Returns
        -------
        DockerImage
            The matching ``runpod/pytorch`` tag.
        """
        return cls(_RUNPOD_PYTORCH_TAGS.get(version, "runpod/pytorch:latest"))


__all__ = ["DockerImage"]
