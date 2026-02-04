"""Type definitions for distributed collections."""

from __future__ import annotations

from typing import Literal

type Consistency = Literal["strong", "eventual"]

__all__ = ["Consistency"]
