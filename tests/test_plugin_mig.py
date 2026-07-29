"""The MIG plugin, as the spec sees it: a profile, the phases it asks for, its wire form."""

from skyward.worker.plugins.mig import Mig
from skyward.shared.schemas import Image, PluginRef


def test_bootstrap_enables_mig_and_cuts_one_slice_per_concurrent_slot():
    phases = Mig(profile="3g.40gb").bootstrap(Image(), 2)

    assert len(phases) == 2
    mig, dcgm = phases
    assert "nvidia-smi -mig 1" in mig
    assert mig.count("nvidia-smi mig -cgi 3g.40gb -C") == 2
    assert "nvidia-smi" in mig
    assert "dcgm" in dcgm


def test_the_plugin_travels_as_its_name_and_its_field():
    assert Mig(profile="1g.10gb").ref() == PluginRef(kind="mig", params={"profile": "1g.10gb"})
