"""Turning things into bytes, without stopping the world to do it.

Every serialization in skyward goes through one of these. That is the point of
the type: a ``cloudpickle.dumps`` written inline is a call that looks free and is
not — a two-hundred-megabyte argument spends a second of CPU inside the event
loop, and everything the daemon owes anybody else waits for it. The rule is
enforced by there being nowhere else to go.

Small values are encoded where they stand. A thread hop costs more than encoding
a hundred-byte struct, so offloading it would be paying to be slower — the
threshold is where the work starts to outweigh the handoff.
"""

from __future__ import annotations

import asyncio
import hashlib
from functools import cache
from typing import Any, Protocol, overload, runtime_checkable

import cloudpickle
import lz4.frame
import msgspec

THRESHOLD = 64 * 1024
"""Above this many bytes, the work is worth a thread; below it, the thread is the work."""


@runtime_checkable
class Codec[T](Protocol):
    """Bytes in, bytes out, and never on the caller's thread when it matters."""

    async def encode(self, value: T) -> bytes: ...

    async def decode(self, raw: bytes) -> T: ...


class Json[T]:
    """The wire format for anything the control plane understands.

    Structs, specs, statuses — small, frequent, and cheap. The decoder is built
    once per type because building one is not.

    The overloads are msgspec's own: a union alias is not a ``type[T]``, so a
    decoder for one is typed by the caller's annotation rather than inferred.
    """

    @overload
    def __init__(self, kind: type[T]) -> None: ...

    @overload
    def __init__(self: Json[Any], kind: Any) -> None: ...

    def __init__(self, kind: Any) -> None:
        self._encoder = msgspec.json.Encoder()
        self._decoder = msgspec.json.Decoder(kind)

    async def encode(self, value: T) -> bytes:
        return self._encoder.encode(value)

    async def decode(self, raw: bytes) -> T:
        if len(raw) < THRESHOLD:
            return self._decoder.decode(raw)
        return await asyncio.to_thread(self._decoder.decode, raw)


class Msgpack[T]:
    """The wire format for what a worker hands back.

    An outcome or a stream frame wraps the user's payload, so its size is the
    user's size — the same threshold decides whether decoding it is worth a
    thread.

    The overloads are msgspec's own: a union alias is not a ``type[T]``, so a
    decoder for one is typed by the caller's annotation rather than inferred.
    """

    @overload
    def __init__(self, kind: type[T]) -> None: ...

    @overload
    def __init__(self: Msgpack[Any], kind: Any) -> None: ...

    def __init__(self, kind: Any) -> None:
        self._decoder = msgspec.msgpack.Decoder(kind)

    async def decode(self, raw: bytes) -> T:
        if len(raw) < THRESHOLD:
            return self._decoder.decode(raw)
        return await asyncio.to_thread(self._decoder.decode, raw)


class Pickle[T]:
    """The wire format for anything only Python understands.

    Always threaded, in both directions. A user's argument is a numpy array or a
    model or a dataframe, and the moment it is none of those is not worth
    detecting. Unpickling is the more dangerous half: it runs imports, and the
    first ``torch`` in a payload is seconds of work that would otherwise land on
    the loop.
    """

    async def encode(self, value: T) -> bytes:
        return await asyncio.to_thread(dumps, value)

    async def decode(self, raw: bytes) -> T:
        return await asyncio.to_thread(lambda: loads(raw))


def dumps(value: object) -> bytes:
    """The same bytes, for a caller that is already off the loop.

    Code running inside a task is on a worker thread by construction, and paying
    for another one to reach the same two lines would be paying for the rule
    rather than obeying it.
    """
    return lz4.frame.compress(cloudpickle.dumps(value))


def loads[T](raw: bytes) -> T:
    return cloudpickle.loads(lz4.frame.decompress(raw))


@overload
def json[T](kind: type[T]) -> Json[T]: ...


@overload
def json(kind: Any) -> Json[Any]: ...


@cache
def json(kind: Any) -> Json[Any]:
    return Json(kind)


payload: Pickle[object] = Pickle()


async def digest(blob: bytes) -> str:
    """The name of a piece of content, which is a hash of all of it."""
    if len(blob) < THRESHOLD:
        return hashlib.sha256(blob).hexdigest()
    return await asyncio.to_thread(lambda: hashlib.sha256(blob).hexdigest())
