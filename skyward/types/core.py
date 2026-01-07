"""Core type aliases for skyward."""

from __future__ import annotations

from typing import Literal

__all__ = [
    "Auto",
    "Architecture",
    "Megabytes",
    "Memory",
]


class Auto:
    """Sentinel class for automatic selection."""

    pass


type Architecture = Literal["arm64", "x86_64"] | Auto

type Megabytes = int

type Memory = (
    Literal[
        "512MiB",
        "1GiB",
        "2GiB",
        "4GiB",
        "8GiB",
        "16GiB",
        "32GiB",
        "64GiB",
        "128GiB",
        "256GiB",
        "512GiB",
    ]
    | Megabytes
    | Auto
)
