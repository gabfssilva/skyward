"""Verda-specific types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VerdaInstance:
    """Internal representation of a Verda instance."""

    id: str
    name: str
    ip: str = ""
    private_ip: str = ""
    status: str = ""
