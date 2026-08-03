"""The accelerator vocabulary every provider is normalized into.

A provider names its hardware however it likes: "NVIDIA H100 80GB SXM5",
"H100-80G-PCIe", "1x H100". Left alone, a question like "the cheapest H100
across my accounts" quietly misses the providers that spell it differently.

So the raw name is parsed once, here, into a canonical accelerator plus its
VRAM. Form factor (SXM/PCIe/NVL) is deliberately not part of the name — it is
what the raw string and VRAM already tell you, and folding it into the name is
what splits `h100` from `h100-sxm` in the first place. Ported from the v1
catalog (`skyward/accelerators/catalog.py`), which is the same vocabulary the
rest of Skyward already speaks.
"""

import re
from typing import NamedTuple


class Accelerator(NamedTuple):
    name: str
    vram: float
    manufacturer: str = ""
    architecture: str = ""
    cuda_min: str = ""
    cuda_max: str = ""


CATALOG: dict[str, Accelerator] = {
    "a10": Accelerator("a10", 24, "NVIDIA", "Ampere", "11.0", "13.1"),
    "a100": Accelerator("a100", 80, "NVIDIA", "Ampere", "11.0", "13.1"),
    "a100x": Accelerator("a100x", 80, "NVIDIA", "Ampere", "11.0", "13.1"),
    "a10g": Accelerator("a10g", 24, "NVIDIA", "Ampere", "11.0", "13.1"),
    "a16": Accelerator("a16", 16, "NVIDIA", "Ampere", "11.1", "13.1"),
    "a2": Accelerator("a2", 16, "NVIDIA", "Ampere", "11.0", "13.1"),
    "a30": Accelerator("a30", 24, "NVIDIA", "Ampere", "11.1", "13.1"),
    "a40": Accelerator("a40", 48, "NVIDIA", "Ampere", "11.1", "13.1"),
    "a800": Accelerator("a800", 80, "NVIDIA", "Ampere", "11.0", "13.1"),
    "b100": Accelerator("b100", 192, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "b200": Accelerator("b200", 192, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "b300": Accelerator("b300", 288, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "cmp-50hx": Accelerator("cmp-50hx", 10, "NVIDIA", "Turing", "11.0", "13.1"),
    "gaudi": Accelerator("gaudi", 32, "Intel", "Gaudi", "", ""),
    "gaudi2": Accelerator("gaudi2", 96, "Intel", "Gaudi", "", ""),
    "gaudi3": Accelerator("gaudi3", 128, "Intel", "Gaudi", "", ""),
    "gb200": Accelerator("gb200", 384, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "gb300": Accelerator("gb300", 288, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "gh200": Accelerator("gh200", 96, "NVIDIA", "Hopper", "11.8", "13.1"),
    "gtx-1050": Accelerator("gtx-1050", 2, "NVIDIA", "Pascal", "8.0", "12.6"),
    "gtx-1050ti": Accelerator("gtx-1050ti", 4, "NVIDIA", "Pascal", "8.0", "12.6"),
    "gtx-1060": Accelerator("gtx-1060", 6, "NVIDIA", "Pascal", "8.0", "12.6"),
    "gtx-1070": Accelerator("gtx-1070", 8, "NVIDIA", "Pascal", "8.0", "12.6"),
    "gtx-1070ti": Accelerator("gtx-1070ti", 8, "NVIDIA", "Pascal", "8.0", "12.6"),
    "gtx-1080": Accelerator("gtx-1080", 8, "NVIDIA", "Pascal", "8.0", "12.6"),
    "gtx-1080ti": Accelerator("gtx-1080ti", 11, "NVIDIA", "Pascal", "8.0", "12.6"),
    "gtx-1650": Accelerator("gtx-1650", 4, "NVIDIA", "Turing", "10.0", "13.1"),
    "gtx-1660": Accelerator("gtx-1660", 6, "NVIDIA", "Turing", "10.0", "13.1"),
    "gtx-1660super": Accelerator("gtx-1660super", 6, "NVIDIA", "Turing", "10.0", "13.1"),
    "gtx-1660ti": Accelerator("gtx-1660ti", 6, "NVIDIA", "Turing", "10.0", "13.1"),
    "h100": Accelerator("h100", 80, "NVIDIA", "Hopper", "11.8", "13.1"),
    "h100-nvl": Accelerator("h100-nvl", 94, "NVIDIA", "Hopper", "11.8", "13.1"),
    "h200": Accelerator("h200", 141, "NVIDIA", "Hopper", "11.8", "13.1"),
    "h200-nvl": Accelerator("h200-nvl", 141, "NVIDIA", "Hopper", "11.8", "13.1"),
    "inferentia1": Accelerator("inferentia1", 8, "AWS", "Inferentia", "", ""),
    "inferentia2": Accelerator("inferentia2", 32, "AWS", "Inferentia", "", ""),
    "instinct-mi25": Accelerator("instinct-mi25", 16, "AMD", "GCN", "", ""),
    "k80": Accelerator("k80", 12, "NVIDIA", "Kepler", "5.5", "10.2"),
    "l4": Accelerator("l4", 24, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "l40": Accelerator("l40", 48, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "l40s": Accelerator("l40s", 48, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "mi100": Accelerator("mi100", 32, "AMD", "CDNA", "", ""),
    "mi210": Accelerator("mi210", 64, "AMD", "CDNA2", "", ""),
    "mi250": Accelerator("mi250", 128, "AMD", "CDNA2", "", ""),
    "mi250x": Accelerator("mi250x", 128, "AMD", "CDNA2", "", ""),
    "mi300a": Accelerator("mi300a", 128, "AMD", "CDNA3", "", ""),
    "mi300b": Accelerator("mi300b", 192, "AMD", "CDNA3", "", ""),
    "mi300x": Accelerator("mi300x", 192, "AMD", "CDNA3", "", ""),
    "mi325x": Accelerator("mi325x", 256, "AMD", "CDNA3", "", ""),
    "mi355x": Accelerator("mi355x", 288, "AMD", "CDNA4", "", ""),
    "mi50": Accelerator("mi50", 16, "AMD", "CDNA", "", ""),
    "p100": Accelerator("p100", 16, "NVIDIA", "Pascal", "8.0", "12.6"),
    "p4": Accelerator("p4", 8, "NVIDIA", "Pascal", "8.0", "12.6"),
    "p40": Accelerator("p40", 24, "NVIDIA", "Pascal", "8.0", "12.6"),
    "quadro-p2000": Accelerator("quadro-p2000", 5, "NVIDIA", "Pascal", "8.0", "12.6"),
    "quadro-p4000": Accelerator("quadro-p4000", 8, "NVIDIA", "Pascal", "8.0", "12.6"),
    "quadro-rtx-4000": Accelerator("quadro-rtx-4000", 8, "NVIDIA", "Turing", "10.0", "13.1"),
    "quadro-rtx-5000": Accelerator("quadro-rtx-5000", 16, "NVIDIA", "Turing", "10.0", "13.1"),
    "quadro-rtx-6000": Accelerator("quadro-rtx-6000", 24, "NVIDIA", "Turing", "10.0", "13.1"),
    "quadro-rtx-8000": Accelerator("quadro-rtx-8000", 48, "NVIDIA", "Turing", "10.0", "13.1"),
    "radeon-pro-v520": Accelerator("radeon-pro-v520", 8, "AMD", "RDNA", "", ""),
    "radeon-pro-v710": Accelerator("radeon-pro-v710", 16, "AMD", "RDNA2", "", ""),
    "rtx-2000ada": Accelerator("rtx-2000ada", 16, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-2060": Accelerator("rtx-2060", 6, "NVIDIA", "Turing", "10.0", "13.1"),
    "rtx-2060super": Accelerator("rtx-2060super", 8, "NVIDIA", "Turing", "10.0", "13.1"),
    "rtx-2070": Accelerator("rtx-2070", 8, "NVIDIA", "Turing", "10.0", "13.1"),
    "rtx-2070super": Accelerator("rtx-2070super", 8, "NVIDIA", "Turing", "10.0", "13.1"),
    "rtx-2080": Accelerator("rtx-2080", 8, "NVIDIA", "Turing", "10.0", "13.1"),
    "rtx-2080super": Accelerator("rtx-2080super", 8, "NVIDIA", "Turing", "10.0", "13.1"),
    "rtx-2080ti": Accelerator("rtx-2080ti", 11, "NVIDIA", "Turing", "10.0", "13.1"),
    "rtx-3050": Accelerator("rtx-3050", 8, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-3060": Accelerator("rtx-3060", 12, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-3060laptop": Accelerator("rtx-3060laptop", 6, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-3060ti": Accelerator("rtx-3060ti", 8, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-3070": Accelerator("rtx-3070", 8, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-3070ti": Accelerator("rtx-3070ti", 8, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-3080": Accelerator("rtx-3080", 10, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-3080ti": Accelerator("rtx-3080ti", 12, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-3090": Accelerator("rtx-3090", 24, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-3090ti": Accelerator("rtx-3090ti", 24, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-4000ada": Accelerator("rtx-4000ada", 20, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-4060": Accelerator("rtx-4060", 8, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-4060ti": Accelerator("rtx-4060ti", 8, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-4070": Accelerator("rtx-4070", 12, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-4070laptop": Accelerator("rtx-4070laptop", 8, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-4070super": Accelerator("rtx-4070super", 12, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-4070ti": Accelerator("rtx-4070ti", 12, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-4070tisuper": Accelerator("rtx-4070tisuper", 16, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-4080": Accelerator("rtx-4080", 16, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-4080super": Accelerator("rtx-4080super", 16, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-4090": Accelerator("rtx-4090", 24, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-4090d": Accelerator("rtx-4090d", 24, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-4500ada": Accelerator("rtx-4500ada", 24, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-5000ada": Accelerator("rtx-5000ada", 32, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-5060": Accelerator("rtx-5060", 8, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rtx-5060ti": Accelerator("rtx-5060ti", 16, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rtx-5070": Accelerator("rtx-5070", 12, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rtx-5070ti": Accelerator("rtx-5070ti", 16, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rtx-5080": Accelerator("rtx-5080", 16, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rtx-5090": Accelerator("rtx-5090", 32, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rtx-5090laptop": Accelerator("rtx-5090laptop", 24, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rtx-5880ada": Accelerator("rtx-5880ada", 48, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-6000ada": Accelerator("rtx-6000ada", 48, "NVIDIA", "Ada Lovelace", "11.8", "13.1"),
    "rtx-a2000": Accelerator("rtx-a2000", 12, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-a4000": Accelerator("rtx-a4000", 16, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-a4500": Accelerator("rtx-a4500", 20, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-a5000": Accelerator("rtx-a5000", 24, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-a5000pro": Accelerator("rtx-a5000pro", 24, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-a6000": Accelerator("rtx-a6000", 48, "NVIDIA", "Ampere", "11.1", "13.1"),
    "rtx-pro-4000": Accelerator("rtx-pro-4000", 24, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rtx-pro-4500": Accelerator("rtx-pro-4500", 32, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rtx-pro-5000": Accelerator("rtx-pro-5000", 48, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rtx-pro-6000": Accelerator("rtx-pro-6000", 96, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rtx-pro-6000maxq": Accelerator("rtx-pro-6000maxq", 96, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rtx-pro-6000s": Accelerator("rtx-pro-6000s", 96, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rtx-pro-6000wk": Accelerator("rtx-pro-6000wk", 96, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rtx-pro-6000ws": Accelerator("rtx-pro-6000ws", 96, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rtx-pro-server-6000": Accelerator("rtx-pro-server-6000", 96, "NVIDIA", "Blackwell", "12.8", "13.1"),
    "rx-7800xt": Accelerator("rx-7800xt", 16, "AMD", "RDNA3", "", ""),
    "rx-7900xt": Accelerator("rx-7900xt", 20, "AMD", "RDNA3", "", ""),
    "rx-7900xtx": Accelerator("rx-7900xtx", 24, "AMD", "RDNA3", "", ""),
    "rx-9060xt": Accelerator("rx-9060xt", 16, "AMD", "RDNA4", "", ""),
    "rx-9070xt": Accelerator("rx-9070xt", 16, "AMD", "RDNA4", "", ""),
    "t4": Accelerator("t4", 16, "NVIDIA", "Turing", "10.0", "13.1"),
    "t4g": Accelerator("t4g", 16, "NVIDIA", "Turing", "10.0", "13.1"),
    "titan-rtx": Accelerator("titan-rtx", 24, "NVIDIA", "Turing", "10.0", "13.1"),
    "titan-v": Accelerator("titan-v", 12, "NVIDIA", "Volta", "9.0", "12.6"),
    "titan-xp": Accelerator("titan-xp", 12, "NVIDIA", "Pascal", "8.0", "12.6"),
    "tpu-v2": Accelerator("tpu-v2", 8, "Google", "TPU", "", ""),
    "tpu-v2-8": Accelerator("tpu-v2-8", 64, "Google", "TPU", "", ""),
    "tpu-v3": Accelerator("tpu-v3", 16, "Google", "TPU", "", ""),
    "tpu-v3-32": Accelerator("tpu-v3-32", 512, "Google", "TPU", "", ""),
    "tpu-v3-8": Accelerator("tpu-v3-8", 128, "Google", "TPU", "", ""),
    "tpu-v4": Accelerator("tpu-v4", 32, "Google", "TPU", "", ""),
    "tpu-v4-64": Accelerator("tpu-v4-64", 2048, "Google", "TPU", "", ""),
    "tpu-v5e": Accelerator("tpu-v5e", 16, "Google", "TPU", "", ""),
    "tpu-v5e-4": Accelerator("tpu-v5e-4", 64, "Google", "TPU", "", ""),
    "tpu-v5p": Accelerator("tpu-v5p", 95, "Google", "TPU", "", ""),
    "tpu-v5p-8": Accelerator("tpu-v5p-8", 760, "Google", "TPU", "", ""),
    "tpu-v6": Accelerator("tpu-v6", 32, "Google", "TPU", "", ""),
    "trainium1": Accelerator("trainium1", 32, "AWS", "Trainium", "", ""),
    "trainium2": Accelerator("trainium2", 64, "AWS", "Trainium", "", ""),
    "trainium3": Accelerator("trainium3", 128, "AWS", "Trainium", "", ""),
    "v100": Accelerator("v100", 32, "NVIDIA", "Volta", "9.0", "12.6"),
}

_INDEX = {re.sub(r"[^a-z0-9]", "", name): accelerator for name, accelerator in CATALOG.items()}

ALIASES = {
    "a2000": "rtx-a2000",
    "a4000": "rtx-a4000",
    "a4500": "rtx-a4500",
    "a5000": "rtx-a5000",
    "a6000": "rtx-a6000",
    "2000ada": "rtx-2000ada",
    "4000ada": "rtx-4000ada",
    "4500ada": "rtx-4500ada",
    "5000ada": "rtx-5000ada",
    "6000ada": "rtx-6000ada",
    "pro6000": "rtx-pro-6000",
    "pro6000blackwell": "rtx-pro-6000",
    "rtxpro6000cc": "rtx-pro-6000",
    "rtxpro6000se": "rtx-pro-6000s",
    "dgxa100": "a100",
    "h100mega": "h100",
    "rtx6000": "quadro-rtx-6000",
    "qrtx6000": "quadro-rtx-6000",
    "qrtx8000": "quadro-rtx-8000",
    "qrtx5000": "quadro-rtx-5000",
    "qrtx4000": "quadro-rtx-4000",
    "prov520": "radeon-pro-v520",
    "prov710": "radeon-pro-v710",
}

_QUALIFIER = re.compile(r"\s*\([^)]*\)\s*$")
_COUNT_PREFIX = re.compile(r"^\d+\s*x\s+", re.IGNORECASE)
_VENDOR = re.compile(r"^[-_\s]*(?:nvidia|amd|intel|geforce|tesla|instinct|radeon)\b[-_\s]*", re.IGNORECASE)
_VRAM = re.compile(r"(\d+(?:\.\d+)?)\s*(?:gb|g)\b", re.IGNORECASE)
_FORM_FACTOR = re.compile(r"[-_\s]*(?:pcie|sxm\d?|oam|nvlink|nvl|max-?q|ws|wk|hbm\d?e?|spot)\b", re.IGNORECASE)


def resolve(raw: str | None, vram: float | None = None) -> tuple[str | None, float | None]:
    """Turn a provider's GPU string into a canonical accelerator and its VRAM.

    Returns ``(None, None)`` for a CPU-only offer. An accelerator the catalog
    has never heard of comes back as its own squashed name, so a new GPU shows
    up in the listing instead of disappearing from it; the live sanity test is
    what tells us to add it to the catalog.

    The catalog is consulted before the VRAM is parsed out, because some GPUs
    have the memory glued into the model name: stripping "10G" out of "A10G"
    leaves "A".
    """
    if not raw or not raw.strip():
        return None, vram

    name = _COUNT_PREFIX.sub("", raw.strip().replace("_", " ").replace("/", " "))
    while (stripped := _VENDOR.sub("", name)) != name:
        name = stripped

    if accelerator := _lookup(name):
        return accelerator.name, vram or accelerator.vram

    if vram is None and (found := _VRAM.search(name)):
        vram = float(found.group(1))

    name = _QUALIFIER.sub("", _VRAM.sub("", name))

    if accelerator := _lookup(name):
        return accelerator.name, vram or accelerator.vram

    if accelerator := _lookup(_FORM_FACTOR.sub("", name)):
        return accelerator.name, vram or accelerator.vram

    key = re.sub(r"[^a-z0-9]", "", _FORM_FACTOR.sub("", name).lower())
    return (key, vram) if key else (None, vram)


def _lookup(name: str) -> Accelerator | None:
    """Match a cleaned-up name against the catalog, ignoring how it is spelled.

    The key is squashed to letters and digits, so "RTX-A6000", "RTX A6000" and
    "rtxa6000" are the same accelerator.
    """
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    if not key:
        return None
    if alias := ALIASES.get(key):
        key = re.sub(r"[^a-z0-9]", "", alias)
    return _INDEX.get(key)


__all__ = ["CATALOG", "Accelerator", "resolve"]
