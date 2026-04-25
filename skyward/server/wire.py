"""Wire encoding for the HTTP server — cloudpickle + lz4."""

from __future__ import annotations

from typing import Any

import cloudpickle
import lz4.frame


def encode(value: Any) -> bytes:
    return lz4.frame.compress(cloudpickle.dumps(value))


def decode(payload: bytes) -> Any:
    return cloudpickle.loads(lz4.frame.decompress(payload))
