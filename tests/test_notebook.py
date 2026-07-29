import json

import pytest

pytest.importorskip("jupyter_client", reason="the Skyward kernel needs: pip install 'skyward[notebook]'")

from jupyter_client.provisioning.provisioner_base import KernelProvisionerBase

from skyward.core.notebook import install_kernelspec, kernel_json, kernel_name, remove_kernelspec
from skyward.core.notebook.provisioner import CHANNELS, SkywardKernelProvisioner


@pytest.mark.unit
def test_kernel_json_names_the_provisioner_and_the_compute():
    spec = kernel_json("training")

    assert spec["display_name"] == "Skyward (training)"
    assert spec["language"] == "python"
    assert spec["interrupt_mode"] == "message"

    provisioner = spec["metadata"]["kernel_provisioner"]
    assert provisioner["provisioner_name"] == "skyward"
    assert provisioner["config"] == {"compute": "training"}


@pytest.mark.unit
def test_kernel_json_carries_the_url_only_when_there_is_one():
    assert kernel_json("training", "http://localhost:7590")["metadata"]["kernel_provisioner"]["config"] == {
        "compute": "training",
        "url": "http://localhost:7590",
    }


@pytest.mark.unit
def test_install_writes_the_spec_and_remove_deletes_it(tmp_path):
    name = install_kernelspec("training", "http://localhost:7590", tmp_path)
    assert name == kernel_name("training") == "skyward-training"

    written = tmp_path / name / "kernel.json"
    assert json.loads(written.read_text()) == kernel_json("training", "http://localhost:7590")

    assert remove_kernelspec("training", tmp_path) == name
    assert not (tmp_path / name).exists()


@pytest.mark.unit
def test_the_provisioner_is_one():
    assert issubclass(SkywardKernelProvisioner, KernelProvisionerBase)

    for name in ("pre_launch", "launch_kernel", "poll", "wait", "send_signal", "terminate", "kill", "cleanup"):
        assert callable(getattr(SkywardKernelProvisioner, name))

    provisioner = SkywardKernelProvisioner(kernel_id="k1", compute="training")
    assert provisioner.compute == "training"
    assert provisioner.url == ""
    assert not provisioner.has_process


@pytest.mark.unit
def test_every_zmq_channel_is_bridged():
    assert CHANNELS == ("shell_port", "iopub_port", "stdin_port", "control_port", "hb_port")
