"""The instruction set a machine runs, and the spellings providers use for it.

A wheel is built against an instruction set, so a node whose CPU does not match
the payload is not a slower node — it is a node where nothing imports. Two names
are worth having, because two is what the prebuilt-wheel ecosystem targets;
anything else a provider reports (i386, the mac variants, an architecture nobody
publishes wheels for) becomes nothing rather than a third name.

Nothing is answered on a guess. An offer whose provider does not say resolves to
``None``, and ``None`` is not a match for a request that named an architecture —
a machine sold on an unproven claim is a node that boots and then fails to run.
"""

from typing import Literal

type Architecture = Literal["x86_64", "arm64"]

ALIASES: dict[str, Architecture] = {
    "x86_64": "x86_64",
    "x86-64": "x86_64",
    "amd64": "x86_64",
    "x64": "x86_64",
    "arm64": "arm64",
    "aarch64": "arm64",
    "arm": "arm64",
}


def architecture(raw: str | None) -> Architecture | None:
    """Turn a provider's architecture string into a canonical one, or into nothing.

    A spelling the map has never seen comes back as ``None`` rather than as
    itself, which is the opposite of what :func:`skyward.protocol.accelerators.resolve`
    does with an unknown GPU. The asymmetry is deliberate: an unknown GPU is a
    listing nobody is harmed by, an unknown architecture is a claim the market
    would otherwise be allowed to satisfy a request with.
    """
    return ALIASES.get((raw or "").strip().lower())


__all__ = ["Architecture", "architecture"]
