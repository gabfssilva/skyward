"""8-byte length-prefixed cloudpickle framing over asyncio streams.

Wire format: [8 bytes big-endian length][cloudpickle payload]
"""

from __future__ import annotations

import asyncio
import struct

import cloudpickle

_HEADER = struct.Struct("!Q")  # 8-byte unsigned long long, big-endian
_MAX_MESSAGE_SIZE = 256 * 1024 * 1024  # 256 MB


def encode(msg: object) -> bytes:
    """Serialize a message with an 8-byte length prefix."""
    payload = cloudpickle.dumps(msg)
    return _HEADER.pack(len(payload)) + payload


def decode(data: bytes) -> object:
    """Deserialize a length-prefixed message (strips header)."""
    length = _HEADER.unpack_from(data, 0)[0]
    return cloudpickle.loads(data[_HEADER.size : _HEADER.size + length])


async def async_send(writer: asyncio.StreamWriter, msg: object) -> None:
    """Send a length-prefixed message over an asyncio stream."""
    writer.write(encode(msg))
    await writer.drain()


async def async_recv(
    reader: asyncio.StreamReader,
    max_size: int = _MAX_MESSAGE_SIZE,
) -> object:
    """Read a length-prefixed message from an asyncio stream."""
    header = await reader.readexactly(_HEADER.size)
    length = _HEADER.unpack_from(header, 0)[0]
    if length > max_size:
        raise ValueError(f"Message too large: {length} bytes (max {max_size})")
    payload = await reader.readexactly(length)
    return cloudpickle.loads(payload)
