from __future__ import annotations

import pytest

from skyward.containers import DockerImage, cuda, pytorch, runpod_base, runpod_pytorch, ubuntu

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestDockerImage:
    def test_str_returns_tag(self) -> None:
        img = DockerImage(tag="nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04", cuda="12.9", ubuntu="24.04")
        assert str(img) == "nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04"

    def test_of_plain_tag(self) -> None:
        img = DockerImage.of("my-registry.io/custom:latest")
        assert str(img) == "my-registry.io/custom:latest"
        assert img.cuda is None
        assert img.ubuntu is None

    def test_of_with_metadata(self) -> None:
        img = DockerImage.of("my-registry.io/custom:latest", cuda="12.9", ubuntu="24.04")
        assert img.cuda == "12.9"
        assert img.ubuntu == "24.04"


class TestCuda:
    def test_known_version_default(self) -> None:
        img = cuda("12.9")
        assert str(img) == "nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04"
        assert img.cuda == "12.9"
        assert img.ubuntu == "22.04"

    def test_known_version_runtime(self) -> None:
        img = cuda("12.9", variant="runtime", ubuntu="24.04")
        assert str(img) == "nvidia/cuda:12.9.1-runtime-ubuntu24.04"

    def test_known_version_devel(self) -> None:
        img = cuda("12.9", variant="devel", ubuntu="24.04")
        assert str(img) == "nvidia/cuda:12.9.1-devel-ubuntu24.04"

    def test_known_version_cudnn_runtime(self) -> None:
        img = cuda("12.8", variant="cudnn-runtime", ubuntu="24.04")
        assert str(img) == "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04"

    def test_nvidia_repository(self) -> None:
        img = cuda("12.9", variant="runtime", ubuntu="24.04", repository="nvidia")
        assert str(img) == "nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04"

    def test_custom_ubuntu(self) -> None:
        img = cuda("12.9", ubuntu="20.04")
        assert str(img) == "nvidia/cuda:12.9.1-cudnn-runtime-ubuntu20.04"
        assert img.ubuntu == "20.04"

    def test_unknown_version_fallback(self) -> None:
        img = cuda("13.0")
        assert str(img) == "nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04"
        assert img.cuda == "13.0"


class TestUbuntu:
    def test_default(self) -> None:
        img = ubuntu()
        assert str(img) == "ubuntu:24.04"
        assert img.ubuntu == "24.04"
        assert img.cuda is None

    def test_custom_version(self) -> None:
        img = ubuntu("22.04")
        assert str(img) == "ubuntu:22.04"


class TestPyTorch:
    def test_default_cuda(self) -> None:
        img = pytorch("2.8")
        assert str(img) == "nvcr.io/nvidia/pytorch:25.04-py3"
        assert img.cuda == "12.9"

    def test_custom_cuda(self) -> None:
        img = pytorch("2.8", cuda="12.8")
        assert str(img) == "nvcr.io/nvidia/pytorch:25.04-py3"
        assert img.cuda == "12.8"

    def test_unknown_version(self) -> None:
        img = pytorch("3.0")
        assert str(img) == "nvcr.io/nvidia/pytorch:latest"
        assert img.cuda is None


class TestRunpodBase:
    def test_default(self) -> None:
        img = runpod_base()
        assert "runpod/base" in str(img)

    def test_custom_cuda(self) -> None:
        img = runpod_base(cuda="12.8")
        assert img.cuda == "12.8"


class TestRunpodPyTorch:
    def test_default(self) -> None:
        img = runpod_pytorch()
        assert "runpod/pytorch" in str(img)

    def test_custom_version(self) -> None:
        img = runpod_pytorch("2.7")
        assert "runpod/pytorch" in str(img)


class TestExports:
    def test_sky_containers_namespace(self) -> None:
        import skyward as sky
        img = sky.containers.cuda("12.9")
        assert isinstance(img, sky.DockerImage)

    def test_docker_image_of_in_provider_config(self) -> None:
        import skyward as sky
        img = sky.DockerImage.of("my-registry.io/custom:latest")
        config = sky.VastAI(docker_image=img)
        assert config.docker_image is img

    def test_catalog_image_in_provider_config(self) -> None:
        import skyward as sky
        img = sky.containers.cuda("12.9")
        config = sky.RunPod(container_image=img)
        assert config.container_image is img
