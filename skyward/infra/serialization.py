"""Serialization utilities using cloudpickle with compression."""

from __future__ import annotations

import sys
import zlib
from typing import Any, Final

import cloudpickle

from skyward.observability.logger import logger

log = logger.bind(component="serialization")

COMPRESSED_MAGIC: Final = b"\x00CZ"
COMPRESSION_LEVEL: Final = 1

PYTHON_VERSION: Final = f"{sys.version_info.major}.{sys.version_info.minor}"


class PythonVersionMismatchError(RuntimeError):
    """Raised when local and remote Python versions don't match."""

    def __init__(self, local: str, remote: str) -> None:
        self.local = local
        self.remote = remote
        super().__init__(
            f"Python version mismatch: local={local}, remote={remote}. "
            f"Cloudpickle cannot safely serialize bytecode across versions. "
            f"Set Image(python='{local}') or use Image(python='auto')."
        )


def check_python_version(remote_version: str) -> None:
    """Validate that the remote Python version matches the local one.

    Should be called before serializing tasks to workers.

    Args:
        remote_version: The Python version running on the worker.

    Raises:
        PythonVersionMismatchError: If versions differ.
    """
    if remote_version != PYTHON_VERSION:
        raise PythonVersionMismatchError(local=PYTHON_VERSION, remote=remote_version)


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
        if len(compressed) < len(pickled):
            log.debug(
                "Serialized {raw} -> {compressed} bytes",
                raw=len(pickled), compressed=len(compressed),
            )
            return COMPRESSED_MAGIC + compressed

    log.debug("Serialized {size} bytes (uncompressed)", size=len(pickled))
    return pickled


def deserialize(data: bytes) -> Any:
    """Deserialize bytes back to a Python object.

    Automatically detects and handles compressed data.

    Args:
        data: Bytes to deserialize (may be compressed).

    Returns:
        The deserialized Python object.
    """
    log.debug("Deserializing {size} bytes", size=len(data))
    try:
        if data.startswith(COMPRESSED_MAGIC):
            data = zlib.decompress(data[len(COMPRESSED_MAGIC) :])

        return cloudpickle.loads(data)
    except Exception as e:
        log.error("Deserialization failed: {err}", err=e)
        raise
