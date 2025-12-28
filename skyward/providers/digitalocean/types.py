"""DigitalOcean-specific types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Droplet:
    """Internal representation of a DigitalOcean Droplet."""

    id: int
    name: str
    ip: str = ""
    private_ip: str = ""
