"""The raw strings the eleven providers actually return, and what they must become.

Every case here was copied from a live API response. This is the test that keeps
`h100` and `h100-sxm` from becoming two different accelerators again.
"""

import pytest

from skyward2.protocol.accelerators import CATALOG, resolve
from skyward2.protocol.schemas import Offer
from skyward2.providers.gcp import LEGACY_VRAM

REAL_NAMES = [
    ("NVIDIA H100 80GB PCIe", "h100", 80),
    ("NVIDIA H100 PCIe", "h100", 80),
    ("H100-80G-PCIe", "h100", 80),
    ("H100-80G-PCIe-spot", "h100", 80),
    ("H100 SXM5 94GB", "h100", 94),
    ("8x H100 SXM5 80GB", "h100", 80),
    ("H100 NVL", "h100-nvl", 94),
    ("h100_nvl", "h100-nvl", 94),
    ("NVIDIA A100 80GB PCIe", "a100", 80),
    ("A100-40G-PCIe", "a100", 40),
    ("A100 SXM 80GB", "a100", 80),
    ("NVIDIA GeForce RTX 4090", "rtx-4090", 24),
    ("RTX 4090 24GB", "rtx-4090", 24),
    ("RTX-A6000", "rtx-a6000", 48),
    ("a6000", "rtx-a6000", 48),
    ("L40S", "l40s", 48),
    ("NVIDIA L4", "l4", 24),
    ("RTX PRO 6000", "rtx-pro-6000", 96),
    ("AMD Instinct MI300X", "mi300x", 192),
    ("Tesla V100", "v100", 32),
    ("RTX 4090 24GB (High frequency)", "rtx-4090", 24),
    ("RTX-PRO6000-SE", "rtx-pro-6000s", 96),
    ("DGX_A100", "a100", 80),
    ("pro_6000_blackwell", "rtx-pro-6000", 96),
    ("Q RTX 6000", "quadro-rtx-6000", 24),
    ("A100 (40 GB PCIe)", "a100", 40),
    ("Tesla V100 (16 GB)", "v100", 16),
    ("B200 (180 GB SXM6)", "b200", 180),
    ("RTX 6000 (24 GB)", "quadro-rtx-6000", 24),
    ("1x RTX PRO 6000 CC 96GB", "rtx-pro-6000", 96),
    ("AMD_MI325X", "mi325x", 256),
    ("A10G", "a10g", 24),
    ("T4g", "t4g", 16),
    ("L40S", "l40s", 48),
    ("nvidia-h100-80gb", "h100", 80),
    ("nvidia-h100-mega-80gb", "h100", 80),
    ("nvidia-tesla-t4", "t4", 16),
    ("nvidia-tesla-p100", "p100", 16),
    ("nvidia-l4", "l4", 24),
    ("nvidia-gb200", "gb200", 384),
]

GCP_LEGACY_NAMES = [
    ("nvidia-tesla-a100", "a100", 40, 80),
    ("nvidia-tesla-v100", "v100", 16, 32),
]


@pytest.mark.parametrize(("raw", "name", "vram"), REAL_NAMES)
def test_a_provider_name_resolves_to_the_shared_vocabulary(raw, name, vram):
    assert resolve(raw) == (name, vram)
    assert name in CATALOG


def test_a_reported_vram_beats_the_one_in_the_name():
    assert resolve("H100 80GB", vram=94) == ("h100", 94)


@pytest.mark.parametrize(("raw", "name", "actual_vram", "catalog_vram"), GCP_LEGACY_NAMES)
def test_gcp_names_a_smaller_card_than_the_catalog_default(raw, name, actual_vram, catalog_vram):
    """GCP's `nvidia-tesla-a100` is the 40GB part, and its V100 is the 16GB one.

    The catalog holds one VRAM per model, so only the provider can say this. The
    adapter passes it explicitly (LEGACY_VRAM); without it both would read as
    bigger cards than they are.
    """
    assert resolve(raw) == (name, catalog_vram), "the catalog default is the bigger card"
    assert LEGACY_VRAM[raw] == actual_vram
    assert resolve(raw, vram=LEGACY_VRAM[raw]) == (name, actual_vram)


def test_a_cpu_only_offer_has_no_accelerator():
    assert resolve(None) == (None, None)
    assert resolve("") == (None, None)


def test_an_unknown_gpu_survives_instead_of_disappearing():
    name, vram = resolve("NVIDIA X999 64GB SXM")
    assert name == "x999", "a GPU we have never seen must still be listed, under its own name"
    assert vram == 64
    assert name not in CATALOG


def test_the_offer_normalizes_whatever_the_adapter_hands_it():
    offer = Offer(
        id="o1",
        provider_id="prv_1",
        provider_name="p",
        kind="k",
        instance_type="8x H100",
        accelerator="NVIDIA H100 80GB SXM5",
        accelerator_count=8,
        cpus=96,
        memory_gb=1024.0,
        fetched_at=None,
        expires_at=None,
        spot_price=16.0,
        on_demand_price=20.0,
    )

    assert (offer.accelerator, offer.vram) == ("h100", 80)
    assert offer.price == 16.0
