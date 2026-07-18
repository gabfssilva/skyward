"""Tests for the DockerImage str-subclass catalog."""

import pytest

from skyward.protocol.schemas import Image
from skyward.sdk.containers import DockerImage


def test_of_returns_tag_verbatim() -> None:
    assert DockerImage.of("myrepo/img:1.0") == "myrepo/img:1.0"
    assert DockerImage.of("myrepo/img:1.0", cuda="12.9", ubuntu="24.04") == "myrepo/img:1.0"


def test_cuda() -> None:
    assert DockerImage.cuda("12.9") == "nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04"
    assert DockerImage.cuda("12.8", variant="devel", ubuntu="24.04") == "nvidia/cuda:12.8.1-devel-ubuntu24.04"
    assert DockerImage.cuda("12.9", variant="runtime", ubuntu="24.04", repository="nvidia") == "nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04"
    assert DockerImage.cuda("13.0", variant="runtime") == "nvidia/cuda:13.0.0-runtime-ubuntu22.04"


def test_ubuntu() -> None:
    assert DockerImage.ubuntu() == "ubuntu:24.04"
    assert DockerImage.ubuntu("22.04") == "ubuntu:22.04"


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.6", "nvcr.io/nvidia/pytorch:24.11-py3"),
        ("2.7", "nvcr.io/nvidia/pytorch:25.03-py3"),
        ("2.8", "nvcr.io/nvidia/pytorch:25.04-py3"),
        ("9.9", "nvcr.io/nvidia/pytorch:latest"),
    ],
)
def test_pytorch(version: str, expected: str) -> None:
    assert DockerImage.pytorch(version) == expected


@pytest.mark.parametrize(
    ("cuda", "expected"),
    [
        ("12.6", "runpod/base:1.0.3-cuda1260-ubuntu2204"),
        ("12.8", "runpod/base:1.0.3-cuda1281-ubuntu2204"),
        ("12.9", "runpod/base:1.0.3-cuda1290-ubuntu2204"),
        ("9.9", "runpod/base:latest"),
    ],
)
def test_runpod_base(cuda: str, expected: str) -> None:
    assert DockerImage.runpod_base(cuda=cuda) == expected


def test_runpod_base_default() -> None:
    assert DockerImage.runpod_base() == "runpod/base:1.0.3-cuda1290-ubuntu2204"


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.6", "runpod/pytorch:2.6.0-py3.12-cuda12.6.3-devel-ubuntu24.04"),
        ("2.7", "runpod/pytorch:2.7.0-py3.13-cuda12.8.1-devel-ubuntu24.04"),
        ("2.8", "runpod/pytorch:2.8.0-py3.13-cuda12.8.1-devel-ubuntu24.04"),
        ("9.9", "runpod/pytorch:latest"),
    ],
)
def test_runpod_pytorch(version: str, expected: str) -> None:
    assert DockerImage.runpod_pytorch(version) == expected


def test_runpod_pytorch_default() -> None:
    assert DockerImage.runpod_pytorch() == "runpod/pytorch:2.8.0-py3.13-cuda12.8.1-devel-ubuntu24.04"


def test_is_str_subclass() -> None:
    assert isinstance(DockerImage.pytorch("2.8"), str)


def test_flows_into_image_base() -> None:
    img = DockerImage.pytorch("2.8")
    assert Image(base=img).base == "nvcr.io/nvidia/pytorch:25.04-py3"
    assert Image(base=img).base == img
