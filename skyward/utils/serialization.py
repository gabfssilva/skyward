"""Serialization utilities using cloudpickle with compression."""

from __future__ import annotations

import zlib
from typing import Any

import cloudpickle

from skyward.constants import COMPRESSED_MAGIC, COMPRESSION_LEVEL


def serialize(obj: Any, compress: bool = True) -> bytes:
    """Serialize an object to bytes using cloudpickle with optional compression.

    Args:
        obj: Any Python object to serialize.
        compress: Whether to compress the output (default True).

    Returns:
        Serialized (and optionally compressed) bytes.
    """
    pickled: bytes = cloudpickle.dumps(obj)

    if compress:
        compressed = zlib.compress(pickled, level=COMPRESSION_LEVEL)
        # Only use compression if it actually reduces size
        if len(compressed) < len(pickled):
            return COMPRESSED_MAGIC + compressed

    return pickled


def deserialize(data: bytes) -> Any:
    """Deserialize bytes back to a Python object.

    Automatically detects and handles compressed data.

    Args:
        data: Bytes to deserialize (may be compressed).

    Returns:
        The deserialized Python object.
    """
    if data.startswith(COMPRESSED_MAGIC):
        data = zlib.decompress(data[len(COMPRESSED_MAGIC) :])

    return cloudpickle.loads(data)
