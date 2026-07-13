"""Unit tests for the Skyward kernelspec builder + installer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skyward.notebook import kernelspec

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def test_kernel_json_shape():
    spec = kernelspec.kernel_json("demo")
    assert spec["argv"] == ["python", "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    assert spec["display_name"] == "Skyward (demo)"
    assert spec["language"] == "python"
    assert spec["interrupt_mode"] == "message"
    prov = spec["metadata"]["kernel_provisioner"]
    assert prov["provisioner_name"] == "skyward"
    assert prov["config"]["session"] == "demo"


def test_install_and_remove(tmp_path, monkeypatch):
    kernels = tmp_path / "kernels"
    kernels.mkdir()

    from jupyter_client.kernelspec import KernelSpecManager

    def fake_install(self, source_dir, kernel_name=None, user=False, **kw):  # noqa: ANN001, ANN003
        dest = kernels / kernel_name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "kernel.json").write_text(
            (Path(source_dir) / "kernel.json").read_text()
        )
        return str(dest)

    removed: list[str] = []

    def fake_remove(self, name):  # noqa: ANN001
        removed.append(name)
        import shutil
        shutil.rmtree(kernels / name)

    monkeypatch.setattr(KernelSpecManager, "install_kernel_spec", fake_install)
    monkeypatch.setattr(KernelSpecManager, "remove_kernel_spec", fake_remove)

    name = kernelspec.install_kernelspec("demo")
    assert name == "skyward-demo"
    written = json.loads((kernels / "skyward-demo" / "kernel.json").read_text())
    assert written["metadata"]["kernel_provisioner"]["provisioner_name"] == "skyward"

    kernelspec.remove_kernelspec("demo")
    assert removed == ["skyward-demo"]
