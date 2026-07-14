"""Who to rent from, and what it takes to log in.

The credentials are the user's to supply. Nothing here reads an environment
variable or a config file on their behalf — a program that hands Skyward a key
should be a program that can be read and see where the key came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Provider:
    kind: str
    credentials: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.kind)


def Container(image: str = "", name: str = "") -> Provider:  # noqa: N802
    """Local containers. The one provider that needs no credentials."""
    return Provider(kind="container", config={"image": image} if image else {}, name=name)
